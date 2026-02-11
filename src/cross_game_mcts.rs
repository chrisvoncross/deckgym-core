//! Cross-Game Batched MCTS v3 — Thread-parallel + optimized batching.
//!
//! Architecture:
//!   - N game threads run MCTS simultaneously (CPU parallelism for simulation)
//!   - Central InferenceServer thread batches NN requests across ALL games
//!   - Adaptive collect window: waits for enough requests before firing GPU
//!
//! Key insight: The bottleneck is CPU game simulation (step_for_mcts +
//! opponent_turn_random), NOT GPU inference. Therefore:
//!   1. Thread parallelism is ESSENTIAL (utilizes all CPU cores)
//!   2. GPU batching helps but is secondary
//!   3. The inference server uses an adaptive window to maximize batch sizes
//!
//! v3 improvements over v1 (original async):
//!   - More parallel threads (64 default, was 16)
//!   - Adaptive collect window (up to 50ms, was fixed 2ms)
//!   - Pre-allocated inference buffers
//!   - Profiling: batch size tracking for optimization

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::actions::apply_action;
use crate::alphazero_mcts::{
    auto_play_forced, build_action_map, build_observation,
    MCTSConfig, MCTSEngine, NUM_ACTIONS, OBS_SIZE,
};
use crate::deck::Deck;
use crate::onnx_predictor::OnnxPredictor;
use crate::state::{GameOutcome, State};

// ═══════════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════════

struct NNRequest {
    obs_flat: Vec<f32>,
    mask_flat: Vec<f32>,
    batch_size: usize,
    response_tx: mpsc::Sender<NNResponse>,
}

struct NNResponse {
    policies_flat: Vec<f32>,
    values_flat: Vec<f32>,
}

/// Training data from a single completed game.
pub struct GameData {
    pub observations: Vec<f32>,
    pub policies: Vec<f32>,
    pub masks: Vec<f32>,
    pub game_value: f32,
    pub move_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════
// Inference Server — adaptive batching across all game threads
// ═══════════════════════════════════════════════════════════════════════

/// Run the inference server with adaptive collect window.
///
/// Instead of a fixed 2ms window (v1), this version adapts:
///   - Tracks how many game threads are active
///   - Waits until we have enough requests for a meaningful batch
///   - Uses a longer max timeout (50ms) to accumulate large batches
///   - Falls back to immediate processing if threads are finishing
fn run_inference_server(
    rx: mpsc::Receiver<NNRequest>,
    predictor: &mut OnnxPredictor,
    max_batch_size: usize,
    active_games: Arc<AtomicUsize>,
) {
    // Adaptive window: longer window = bigger batches = better GPU utilization.
    // But too long = threads idle waiting for results.
    // Sweet spot: wait until we have a decent fraction of active threads,
    // or until a timeout expires.
    let max_window = Duration::from_millis(50);
    let min_window = Duration::from_millis(1);

    let mut total_batches = 0u64;
    let mut total_samples = 0u64;
    let mut max_batch_seen = 0usize;

    loop {
        // Block-wait for first request
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => break,
        };

        let mut requests = vec![first];
        let mut total_in_batch = requests[0].batch_size;

        // Adaptive collect window based on active game count
        let n_active = active_games.load(Ordering::Relaxed);
        let target_requests = if n_active > 4 {
            n_active / 2 // Wait for at least half the active games
        } else {
            1 // Few games left, don't wait
        };

        // Use shorter window when few games are active
        let window = if n_active > 8 { max_window } else { min_window };
        let deadline = Instant::now() + window;

        while requests.len() < target_requests && total_in_batch < max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Drain anything already queued
                loop {
                    match rx.try_recv() {
                        Ok(req) => {
                            total_in_batch += req.batch_size;
                            requests.push(req);
                            if total_in_batch >= max_batch_size { break; }
                        }
                        Err(_) => break,
                    }
                }
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => {
                    total_in_batch += req.batch_size;
                    requests.push(req);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Concatenate into mega-batch
        let mut all_obs = Vec::with_capacity(total_in_batch * OBS_SIZE);
        let mut all_mask = Vec::with_capacity(total_in_batch * NUM_ACTIONS);
        for req in &requests {
            all_obs.extend_from_slice(&req.obs_flat);
            all_mask.extend_from_slice(&req.mask_flat);
        }

        // Single GPU call
        let (policies, values) = predictor.predict(&all_obs, &all_mask, total_in_batch);

        // Split and send back
        let mut offset = 0;
        for req in requests {
            let n = req.batch_size;
            let p_start = offset * NUM_ACTIONS;
            let p_end = (offset + n) * NUM_ACTIONS;
            let _ = req.response_tx.send(NNResponse {
                policies_flat: policies[p_start..p_end].to_vec(),
                values_flat: values[offset..offset + n].to_vec(),
            });
            offset += n;
        }

        // Stats
        total_batches += 1;
        total_samples += total_in_batch as u64;
        if total_in_batch > max_batch_seen {
            max_batch_seen = total_in_batch;
        }
    }

    if total_batches > 0 {
        log::info!(
            "📊 InferenceServer: {} batches, avg={:.0} samples, max={}, {:.0} samples/s",
            total_batches,
            total_samples as f64 / total_batches as f64,
            max_batch_seen,
            total_samples as f64, // rough
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Game thread — one complete self-play game
// ═══════════════════════════════════════════════════════════════════════

fn play_single_game(
    agent_config: &MCTSConfig,
    opp_config: &MCTSConfig,
    deck_a: &Deck,
    deck_b: &Deck,
    request_tx: mpsc::Sender<NNRequest>,
) -> Option<GameData> {
    let mut rng = StdRng::from_entropy();
    let mut state = State::initialize(deck_a, deck_b, &mut rng);
    let agent = 0usize;

    let agent_engine = MCTSEngine::new(agent_config.clone());
    let opp_engine = MCTSEngine::new(opp_config.clone());

    let mut observations: Vec<f32> = Vec::new();
    let mut policies: Vec<f32> = Vec::new();
    let mut masks: Vec<f32> = Vec::new();
    let mut move_number = 0usize;
    let max_total_moves = 500usize;
    let mut total_moves = 0usize;

    while !state.is_game_over() && total_moves < max_total_moves {
        auto_play_forced(&mut state, &mut rng);
        if state.is_game_over() { break; }

        let (actor, actions) = state.generate_possible_actions();
        if actions.is_empty() { break; }

        let is_agent = actor == agent;
        let engine = if is_agent { &agent_engine } else { &opp_engine };

        let tx = request_tx.clone();
        let mut predict_fn = move |obs: &[f32], mask: &[f32], batch_size: usize|
            -> (Vec<f32>, Vec<f32>)
        {
            let (resp_tx, resp_rx) = mpsc::channel();
            tx.send(NNRequest {
                obs_flat: obs.to_vec(),
                mask_flat: mask.to_vec(),
                batch_size,
                response_tx: resp_tx,
            }).expect("inference server disconnected");
            let resp = resp_rx.recv().expect("inference server dropped response");
            (resp.policies_flat, resp.values_flat)
        };

        let result = engine.search(&state, &mut rng, actor, move_number, &mut predict_fn);

        if is_agent {
            let obs = build_observation(&state, actor);
            let (mask_bool, _) = build_action_map(&actions, actor);
            let mask_f32: Vec<f32> = mask_bool
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();

            observations.extend_from_slice(&obs);
            policies.extend(result.policy.iter().map(|&p| p as f32));
            masks.extend_from_slice(&mask_f32);
            move_number += 1;
        }

        let (_, action_map) = build_action_map(&actions, actor);
        if let Some(&deckgym_idx) = action_map.get(&result.action) {
            apply_action(&mut rng, &mut state, &actions[deckgym_idx]);
        } else {
            apply_action(&mut rng, &mut state, &actions[0]);
        }
        total_moves += 1;
    }

    let game_value = match state.winner {
        Some(GameOutcome::Win(p)) if p == agent => 1.0,
        Some(GameOutcome::Win(_)) => -1.0,
        _ => 0.0,
    };

    if move_number == 0 { return None; }

    Some(GameData {
        observations,
        policies,
        masks,
        game_value,
        move_count: move_number,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

pub fn run_cross_game_selfplay(
    agent_config: &MCTSConfig,
    opp_config: &MCTSConfig,
    model_path: &Path,
    deck_a_path: &str,
    deck_b_path: &str,
    num_games: usize,
    num_parallel: usize,
    max_batch_size: usize,
) -> Vec<GameData> {
    let deck_a = Deck::from_file(deck_a_path).expect("Failed to load deck A");
    let deck_b = Deck::from_file(deck_b_path).expect("Failed to load deck B");

    let (req_tx, req_rx) = mpsc::channel();
    let active_games = Arc::new(AtomicUsize::new(0));
    let active_games_server = active_games.clone();

    let model_path_buf = model_path.to_path_buf();
    let eval_handle = thread::spawn(move || {
        let mut predictor =
            OnnxPredictor::new(&model_path_buf).expect("Failed to load ONNX model");
        log::info!(
            "🚀 InferenceServer started (max_batch={}, adaptive window)",
            max_batch_size
        );
        run_inference_server(req_rx, &mut predictor, max_batch_size, active_games_server);
        predictor.log_stats();
        log::info!("🏁 InferenceServer stopped");
    });

    let mut all_results: Vec<GameData> = Vec::with_capacity(num_games);
    let mut remaining = num_games;

    while remaining > 0 {
        let wave_size = remaining.min(num_parallel);
        remaining -= wave_size;

        active_games.store(wave_size, Ordering::Relaxed);

        thread::scope(|s| {
            let handles: Vec<_> = (0..wave_size)
                .map(|_| {
                    let tx = req_tx.clone();
                    let da = &deck_a;
                    let db = &deck_b;
                    let active = &active_games;
                    s.spawn(move || {
                        let result = play_single_game(agent_config, opp_config, da, db, tx);
                        active.fetch_sub(1, Ordering::Relaxed);
                        result
                    })
                })
                .collect();

            for h in handles {
                match h.join() {
                    Ok(Some(data)) => all_results.push(data),
                    Ok(None) => {}
                    Err(_) => log::warn!("⚠️ Game thread panicked"),
                }
            }
        });
    }

    drop(req_tx);
    let _ = eval_handle.join();

    all_results
}
