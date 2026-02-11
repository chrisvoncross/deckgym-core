//! Rust-side PPO Self-Play Loop — zero Python overhead, batched inference.
//!
//! Architecture:
//!   - N game environments managed entirely in Rust
//!   - Agent inference: batched obs sent to Python/GPU, actions returned
//!   - Opponent inference: batched ONNX in Rust (all opponents in ONE call)
//!   - Game simulation: parallel-ready (State is Send+Sync)
//!
//! Key optimization: opponent turns are BATCHED — instead of 64 individual
//! ONNX calls (one per env), we collect all opponent observations and make
//! ONE batched ONNX call per opponent "step". This reduces ONNX overhead
//! from ~320ms to ~5ms per batch step.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::actions::apply_action;
use crate::alphazero_mcts::{
    auto_play_forced, build_action_map, build_observation,
    NUM_ACTIONS,
};
use crate::deck::Deck;
use crate::onnx_predictor::OnnxPredictor;
use crate::state::{GameOutcome, State};

// ═══════════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════════

struct GameEnv {
    state: State,
    rng: StdRng,
    agent: usize,
    move_count: usize,
    total_actions: usize,
    done: bool,
    pending_actions: Vec<crate::actions::Action>,
    pending_action_map: HashMap<usize, usize>,
    pending_mask: [f32; NUM_ACTIONS],
}

pub struct GameTrajectory {
    pub observations: Vec<f32>,
    pub actions: Vec<i32>,
    pub log_probs: Vec<f32>,
    pub values: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<f32>,
    pub masks: Vec<f32>,
    pub game_value: f32,
    pub move_count: usize,
}

pub struct PendingBatch {
    pub obs_flat: Vec<f32>,
    pub mask_flat: Vec<f32>,
    pub n_pending: usize,
    pub env_indices: Vec<usize>,
}

// ═══════════════════════════════════════════════════════════════════════
// Batched opponent turns — ONE ONNX call for ALL envs per step
// ═══════════════════════════════════════════════════════════════════════

/// Play all opponents' turns simultaneously with batched ONNX inference.
///
/// Instead of calling ONNX 64 times (once per env), this collects ALL
/// opponent observations into ONE batch, makes ONE ONNX call, then
/// distributes actions. Repeats until all opponents have ended their turn.
///
/// Speedup: 64 individual calls (~320ms) → ~5 batched calls (~5ms)
fn batched_opponent_turns(
    envs: &mut [GameEnv],
    predictor: &mut OnnxPredictor,
) {
    let max_steps = 50;

    for _ in 0..max_steps {
        // Auto-play forced for all active envs
        for env in envs.iter_mut() {
            if !env.done && !env.state.is_game_over() {
                auto_play_forced(&mut env.state, &mut env.rng);
            }
        }

        // Find envs where opponent needs to act
        let mut opp_indices: Vec<usize> = Vec::new();
        let mut opp_action_data: Vec<(Vec<crate::actions::Action>, HashMap<usize, usize>)> =
            Vec::new();
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut mask_flat: Vec<f32> = Vec::new();

        for (i, env) in envs.iter().enumerate() {
            if env.done || env.state.is_game_over() {
                continue;
            }
            let (actor, actions) = env.state.generate_possible_actions();
            if actions.is_empty() || actor == env.agent {
                continue; // Agent's turn or no actions — skip
            }

            let opp = 1 - env.agent;
            let obs = build_observation(&env.state, opp);
            let (mask_bool, action_map) = build_action_map(&actions, opp);

            obs_flat.extend_from_slice(&obs);
            for &b in &mask_bool {
                mask_flat.push(if b { 1.0 } else { 0.0 });
            }

            opp_indices.push(i);
            opp_action_data.push((actions, action_map));
        }

        if opp_indices.is_empty() {
            break; // All opponents done
        }

        // ONE batched ONNX call for ALL opponents
        let n = opp_indices.len();
        let (policies, _values) = predictor.predict(&obs_flat, &mask_flat, n);

        // Apply best action for each opponent
        for (idx, &env_i) in opp_indices.iter().enumerate() {
            let policy = &policies[idx * NUM_ACTIONS..(idx + 1) * NUM_ACTIONS];
            let mask = &mask_flat[idx * NUM_ACTIONS..(idx + 1) * NUM_ACTIONS];

            // Greedy: pick highest-probability valid action
            let mut best_a = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for a in 0..NUM_ACTIONS {
                if mask[a] > 0.0 && policy[a] > best_p {
                    best_p = policy[a];
                    best_a = a;
                }
            }

            let (ref actions, ref action_map) = opp_action_data[idx];
            let env = &mut envs[env_i];

            if let Some(&deckgym_idx) = action_map.get(&best_a) {
                if deckgym_idx < actions.len() {
                    apply_action(&mut env.rng, &mut env.state, &actions[deckgym_idx]);
                }
            } else if !actions.is_empty() {
                apply_action(&mut env.rng, &mut env.state, &actions[0]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PPO Self-Play Coordinator
// ═══════════════════════════════════════════════════════════════════════

pub struct PPOSelfPlay {
    envs: Vec<GameEnv>,
    opp_predictor: OnnxPredictor,
    deck_files: Vec<String>,
    n_envs: usize,
}

impl PPOSelfPlay {
    pub fn new(
        n_envs: usize,
        deck_files: Vec<String>,
        opp_onnx_path: &Path,
    ) -> Result<Self, String> {
        let opp_predictor = OnnxPredictor::new(opp_onnx_path)
            .map_err(|e| format!("Failed to load opponent ONNX model: {}", e))?;

        Ok(PPOSelfPlay {
            envs: Vec::new(),
            opp_predictor,
            deck_files,
            n_envs,
        })
    }

    fn init_envs(&mut self) {
        self.envs.clear();
        let mut rng = StdRng::from_entropy();
        for _ in 0..self.n_envs {
            self.new_env(&mut rng);
        }
    }

    fn new_env(&mut self, rng: &mut StdRng) {
        let deck_a_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_b_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_a = Deck::from_file(deck_a_path).expect("Failed to load deck");
        let deck_b = Deck::from_file(deck_b_path).expect("Failed to load deck");

        let mut env_rng = StdRng::from_entropy();
        let state = State::initialize(&deck_a, &deck_b, &mut env_rng);
        let agent = rng.gen_range(0..2usize);

        self.envs.push(GameEnv {
            state,
            rng: env_rng,
            agent,
            move_count: 0,
            total_actions: 0,
            done: false,
            pending_actions: Vec::new(),
            pending_action_map: HashMap::new(),
            pending_mask: [0.0; NUM_ACTIONS],
        });
    }

    fn reset_env(&mut self, idx: usize) {
        let mut rng = StdRng::from_entropy();
        let deck_a_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_b_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_a = Deck::from_file(deck_a_path).expect("Failed to load deck");
        let deck_b = Deck::from_file(deck_b_path).expect("Failed to load deck");

        let mut env_rng = StdRng::from_entropy();
        let state = State::initialize(&deck_a, &deck_b, &mut env_rng);
        let agent = rng.gen_range(0..2usize);

        self.envs[idx] = GameEnv {
            state,
            rng: env_rng,
            agent,
            move_count: 0,
            total_actions: 0,
            done: false,
            pending_actions: Vec::new(),
            pending_action_map: HashMap::new(),
            pending_mask: [0.0; NUM_ACTIONS],
        };
    }

    /// Advance all envs to agent decision points using BATCHED opponent turns.
    pub fn get_pending_observations(&mut self) -> PendingBatch {
        // Phase 1: Play all opponent turns with batched ONNX
        // Retry loop handles games ending before agent gets to act
        for _ in 0..5 {
            // Reset games that ended
            for idx in 0..self.n_envs {
                if !self.envs[idx].done && self.envs[idx].state.is_game_over() {
                    self.reset_env(idx);
                }
            }

            // Batched opponent turns (ONE ONNX call per opponent "step")
            batched_opponent_turns(&mut self.envs, &mut self.opp_predictor);

            // Auto-play forced for all after opponent turns
            for env in self.envs.iter_mut() {
                if !env.done && !env.state.is_game_over() {
                    auto_play_forced(&mut env.state, &mut env.rng);
                }
            }

            // Check if any envs still need retry (game ended during opponent)
            let needs_retry = self.envs.iter().any(|e| {
                !e.done && e.state.is_game_over()
            });
            if !needs_retry {
                break;
            }
        }

        // Phase 2: Build observations for all envs at agent decision points
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut mask_flat: Vec<f32> = Vec::new();
        let mut env_indices: Vec<usize> = Vec::new();

        for idx in 0..self.n_envs {
            let env = &mut self.envs[idx];
            if env.done || env.state.is_game_over() {
                continue;
            }

            let (actor, actions) = env.state.generate_possible_actions();
            if actions.is_empty() || actor != env.agent {
                continue;
            }

            let obs = build_observation(&env.state, env.agent);
            let (mask_bool, action_map) = build_action_map(&actions, env.agent);
            let mask_f32: [f32; NUM_ACTIONS] = {
                let mut m = [0.0f32; NUM_ACTIONS];
                for (i, &b) in mask_bool.iter().enumerate() {
                    m[i] = if b { 1.0 } else { 0.0 };
                }
                m
            };

            obs_flat.extend_from_slice(&obs);
            mask_flat.extend_from_slice(&mask_f32);
            env_indices.push(idx);

            env.pending_actions = actions;
            env.pending_action_map = action_map;
            env.pending_mask = mask_f32;
        }

        PendingBatch {
            n_pending: env_indices.len(),
            obs_flat,
            mask_flat,
            env_indices,
        }
    }

    /// Run a complete batch of games with a predict_fn callback for agent inference.
    pub fn run_batch<F>(
        &mut self,
        num_games: usize,
        mut predict_fn: F,
    ) -> Vec<GameTrajectory>
    where
        F: FnMut(&[f32], &[f32], usize) -> (Vec<usize>, Vec<f32>, Vec<f32>),
    {
        self.init_envs();
        let mut all_trajectories: Vec<GameTrajectory> = Vec::with_capacity(num_games);
        let mut completed = 0usize;
        let start = Instant::now();

        // Per-env trajectory accumulators
        let mut env_obs: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_actions: Vec<Vec<i32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_log_probs: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_values: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_rewards: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_dones: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_masks: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();

        while completed < num_games {
            let pending = self.get_pending_observations();
            if pending.n_pending == 0 {
                break;
            }

            // Agent inference via Python/GPU callback
            let (actions, log_probs, values) = predict_fn(
                &pending.obs_flat,
                &pending.mask_flat,
                pending.n_pending,
            );

            // Apply actions and collect trajectories
            for (i, &idx) in pending.env_indices.iter().enumerate() {
                let semantic_action = actions[i];
                let log_prob = log_probs[i];
                let value = values[i];
                let env = &mut self.envs[idx];

                // Record observation before action
                let obs = build_observation(&env.state, env.agent);
                env_obs[idx].extend_from_slice(&obs);
                env_actions[idx].push(semantic_action as i32);
                env_log_probs[idx].push(log_prob);
                env_values[idx].push(value);
                env_masks[idx].extend_from_slice(&env.pending_mask);

                // Apply action
                if let Some(&deckgym_idx) = env.pending_action_map.get(&semantic_action) {
                    if deckgym_idx < env.pending_actions.len() {
                        apply_action(&mut env.rng, &mut env.state, &env.pending_actions[deckgym_idx]);
                    }
                } else if !env.pending_actions.is_empty() {
                    apply_action(&mut env.rng, &mut env.state, &env.pending_actions[0]);
                }

                env.total_actions += 1;
                env.move_count += 1;

                let done = env.state.is_game_over() || env.total_actions >= 500;
                let reward = if done {
                    match env.state.winner {
                        Some(GameOutcome::Win(p)) if p == env.agent => 1.0,
                        Some(GameOutcome::Win(_)) => -1.0,
                        _ => 0.0,
                    }
                } else {
                    0.0
                };

                env_rewards[idx].push(reward);
                env_dones[idx].push(if done { 1.0 } else { 0.0 });

                if done {
                    let mc = env.move_count;
                    let gv = reward;

                    let traj = GameTrajectory {
                        observations: std::mem::take(&mut env_obs[idx]),
                        actions: std::mem::take(&mut env_actions[idx]),
                        log_probs: std::mem::take(&mut env_log_probs[idx]),
                        values: std::mem::take(&mut env_values[idx]),
                        rewards: std::mem::take(&mut env_rewards[idx]),
                        dones: std::mem::take(&mut env_dones[idx]),
                        masks: std::mem::take(&mut env_masks[idx]),
                        game_value: gv,
                        move_count: mc,
                    };

                    all_trajectories.push(traj);
                    completed += 1;

                    if completed < num_games {
                        self.reset_env(idx);
                    } else {
                        env.done = true;
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        log::info!(
            "Rust PPO self-play: {} games in {:.1}s ({:.1} games/s)",
            completed, elapsed, completed as f64 / elapsed.max(0.001),
        );

        all_trajectories
    }
}
