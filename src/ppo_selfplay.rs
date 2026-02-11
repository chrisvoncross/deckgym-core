//! Rust-side PPO Self-Play Loop with League Training.
//!
//! Architecture (AlphaStar-inspired):
//!   - N game environments, each assigned an opponent TYPE:
//!     60% Self-Play (ONNX policy opponent — learns from population)
//!     25% Heuristic (priority-based bot — prevents strategy collapse)
//!     15% Random (uniform random — maximum diversity)
//!   - Agent inference: batched obs sent to Python/GPU
//!   - Opponent inference: batched ONNX / heuristic / random in Rust
//!
//! This prevents strategy collapse where the agent overfits to beating
//! its own past strategies but fails against different playstyles.

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

/// Opponent type for league training diversity.
#[derive(Clone, Copy, PartialEq)]
enum OpponentType {
    /// ONNX policy (learned, from population). Main self-play.
    SelfPlay,
    /// Priority heuristic (attack > evolve > energy > trainer > end).
    /// Prevents strategy collapse — fundamentally different playstyle.
    Heuristic,
    /// Uniform random action selection. Maximum diversity.
    Random,
}

struct GameEnv {
    state: State,
    rng: StdRng,
    agent: usize,
    move_count: usize,
    total_actions: usize,
    done: bool,
    opponent_type: OpponentType,
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
// Opponent implementations
// ═══════════════════════════════════════════════════════════════════════

/// Heuristic opponent: priority-based action selection.
/// Attack > Evolve > Energy > Trainer > Bench > Ability > EndTurn.
fn heuristic_select_action(
    actions: &[crate::actions::Action],
    action_map: &HashMap<usize, usize>,
    mask: &[f32],
    _rng: &mut StdRng,
) -> usize {
    // Semantic action priority groups (roughly matching the Python heuristic)
    // Attack actions: 4, 5 (ATTACK_0, ATTACK_1)
    for &a in &[4usize, 5] {
        if mask[a] > 0.0 { return a; }
    }
    // Evolve actions: 14-17
    for a in 14..=17 {
        if mask[a] > 0.0 { return a; }
    }
    // Energy actions: 6-9
    for a in 6..=9 {
        if mask[a] > 0.0 { return a; }
    }
    // Trainer/Item actions: 22-27
    for a in 22..=27 {
        if mask[a] > 0.0 { return a; }
    }
    // Bench actions: 10-13
    for a in 10..=13 {
        if mask[a] > 0.0 { return a; }
    }
    // Ability actions: 18-21
    for a in 18..=21 {
        if mask[a] > 0.0 { return a; }
    }
    // Retreat: 28, 34-36
    for &a in &[28usize, 34, 35, 36] {
        if mask[a] > 0.0 { return a; }
    }
    // EndTurn: 0
    if mask[0] > 0.0 { return 0; }
    // Fallback: first legal action
    for a in 0..NUM_ACTIONS {
        if mask[a] > 0.0 { return a; }
    }
    0
}

/// Random opponent: uniform random from legal actions.
fn random_select_action(mask: &[f32], rng: &mut StdRng) -> usize {
    let valid: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a] > 0.0).collect();
    if valid.is_empty() { return 0; }
    valid[rng.gen_range(0..valid.len())]
}

// ═══════════════════════════════════════════════════════════════════════
// League opponent turns — mixed ONNX / heuristic / random
// ═══════════════════════════════════════════════════════════════════════

/// Play all opponents' turns with league-style mixed opponents.
/// Self-Play envs use batched ONNX. Heuristic/Random envs use Rust-only logic.
fn league_opponent_turns(
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

        // Separate envs by opponent type
        let mut onnx_indices: Vec<usize> = Vec::new();
        let mut onnx_action_data: Vec<(Vec<crate::actions::Action>, HashMap<usize, usize>)> = Vec::new();
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut mask_flat: Vec<f32> = Vec::new();

        let mut heuristic_indices: Vec<usize> = Vec::new();
        let mut heuristic_data: Vec<(Vec<crate::actions::Action>, HashMap<usize, usize>, Vec<f32>)> = Vec::new();

        let mut random_indices: Vec<usize> = Vec::new();
        let mut random_masks: Vec<Vec<f32>> = Vec::new();

        for (i, env) in envs.iter().enumerate() {
            if env.done || env.state.is_game_over() { continue; }
            let (actor, actions) = env.state.generate_possible_actions();
            if actions.is_empty() || actor == env.agent { continue; }

            let opp = 1 - env.agent;
            let (mask_bool, action_map) = build_action_map(&actions, opp);
            let mask_f: Vec<f32> = mask_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

            match env.opponent_type {
                OpponentType::SelfPlay => {
                    let obs = build_observation(&env.state, opp);
                    obs_flat.extend_from_slice(&obs);
                    mask_flat.extend_from_slice(&mask_f);
                    onnx_indices.push(i);
                    onnx_action_data.push((actions, action_map));
                }
                OpponentType::Heuristic => {
                    heuristic_indices.push(i);
                    heuristic_data.push((actions, action_map, mask_f));
                }
                OpponentType::Random => {
                    random_indices.push(i);
                    random_masks.push(mask_f);
                }
            }
        }

        let any_active = !onnx_indices.is_empty() || !heuristic_indices.is_empty() || !random_indices.is_empty();
        if !any_active { break; }

        // ONNX opponents: batched inference
        if !onnx_indices.is_empty() {
            let n = onnx_indices.len();
            let (policies, _) = predictor.predict(&obs_flat, &mask_flat, n);

            for (idx, &env_i) in onnx_indices.iter().enumerate() {
                let policy = &policies[idx * NUM_ACTIONS..(idx + 1) * NUM_ACTIONS];
                let mask = &mask_flat[idx * NUM_ACTIONS..(idx + 1) * NUM_ACTIONS];

                let mut best_a = 0usize;
                let mut best_p = f32::NEG_INFINITY;
                for a in 0..NUM_ACTIONS {
                    if mask[a] > 0.0 && policy[a] > best_p {
                        best_p = policy[a];
                        best_a = a;
                    }
                }

                let (ref actions, ref action_map) = onnx_action_data[idx];
                let env = &mut envs[env_i];
                if let Some(&di) = action_map.get(&best_a) {
                    if di < actions.len() { apply_action(&mut env.rng, &mut env.state, &actions[di]); }
                } else if !actions.is_empty() {
                    apply_action(&mut env.rng, &mut env.state, &actions[0]);
                }
            }
        }

        // Heuristic opponents: priority-based (no ONNX)
        for (idx, &env_i) in heuristic_indices.iter().enumerate() {
            let (ref actions, ref action_map, ref mask) = heuristic_data[idx];
            let env = &mut envs[env_i];
            let sem = heuristic_select_action(actions, action_map, mask, &mut env.rng);
            if let Some(&di) = action_map.get(&sem) {
                if di < actions.len() { apply_action(&mut env.rng, &mut env.state, &actions[di]); }
            } else if !actions.is_empty() {
                apply_action(&mut env.rng, &mut env.state, &actions[0]);
            }
        }

        // Random opponents: uniform random (no ONNX)
        for (idx, &env_i) in random_indices.iter().enumerate() {
            let env = &mut envs[env_i];
            let sem = random_select_action(&random_masks[idx], &mut env.rng);
            let (_, actions) = env.state.generate_possible_actions();
            let (_, action_map) = build_action_map(&actions, 1 - env.agent);
            if let Some(&di) = action_map.get(&sem) {
                if di < actions.len() { apply_action(&mut env.rng, &mut env.state, &actions[di]); }
            } else if !actions.is_empty() {
                apply_action(&mut env.rng, &mut env.state, &actions[0]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PPO Self-Play Coordinator with League Training
// ═══════════════════════════════════════════════════════════════════════

pub struct PPOSelfPlay {
    envs: Vec<GameEnv>,
    opp_predictor: OnnxPredictor,
    deck_files: Vec<String>,
    n_envs: usize,
    // League training ratios (sum = 1.0)
    selfplay_ratio: f32,
    heuristic_ratio: f32,
    // random_ratio = 1.0 - selfplay_ratio - heuristic_ratio
}

impl PPOSelfPlay {
    pub fn new(
        n_envs: usize,
        deck_files: Vec<String>,
        opp_onnx_path: &Path,
        selfplay_ratio: f32,
        heuristic_ratio: f32,
    ) -> Result<Self, String> {
        let opp_predictor = OnnxPredictor::new(opp_onnx_path)
            .map_err(|e| format!("Failed to load opponent ONNX model: {}", e))?;

        log::info!(
            "League training: {:.0}% self-play, {:.0}% heuristic, {:.0}% random",
            selfplay_ratio * 100.0,
            heuristic_ratio * 100.0,
            (1.0 - selfplay_ratio - heuristic_ratio) * 100.0,
        );

        Ok(PPOSelfPlay {
            envs: Vec::new(),
            opp_predictor,
            deck_files,
            n_envs,
            selfplay_ratio,
            heuristic_ratio,
        })
    }

    /// Assign opponent type based on league ratios.
    fn assign_opponent_type(&self, rng: &mut StdRng) -> OpponentType {
        let r: f32 = rng.gen();
        if r < self.selfplay_ratio {
            OpponentType::SelfPlay
        } else if r < self.selfplay_ratio + self.heuristic_ratio {
            OpponentType::Heuristic
        } else {
            OpponentType::Random
        }
    }

    fn init_envs(&mut self) {
        self.envs.clear();
        let mut rng = StdRng::from_entropy();
        for _ in 0..self.n_envs {
            self.create_env(&mut rng);
        }
    }

    fn create_env(&mut self, rng: &mut StdRng) {
        let deck_a_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_b_path = &self.deck_files[rng.gen_range(0..self.deck_files.len())];
        let deck_a = Deck::from_file(deck_a_path).expect("Failed to load deck");
        let deck_b = Deck::from_file(deck_b_path).expect("Failed to load deck");

        let mut env_rng = StdRng::from_entropy();
        let state = State::initialize(&deck_a, &deck_b, &mut env_rng);
        let agent = rng.gen_range(0..2usize);
        let opp_type = self.assign_opponent_type(rng);

        self.envs.push(GameEnv {
            state,
            rng: env_rng,
            agent,
            move_count: 0,
            total_actions: 0,
            done: false,
            opponent_type: opp_type,
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
        let opp_type = self.assign_opponent_type(&mut rng);

        self.envs[idx] = GameEnv {
            state,
            rng: env_rng,
            agent,
            move_count: 0,
            total_actions: 0,
            done: false,
            opponent_type: opp_type,
            pending_actions: Vec::new(),
            pending_action_map: HashMap::new(),
            pending_mask: [0.0; NUM_ACTIONS],
        };
    }

    pub fn get_pending_observations(&mut self) -> PendingBatch {
        // Phase 1: League opponent turns (mixed ONNX / heuristic / random)
        for _ in 0..5 {
            for idx in 0..self.n_envs {
                if !self.envs[idx].done && self.envs[idx].state.is_game_over() {
                    self.reset_env(idx);
                }
            }
            league_opponent_turns(&mut self.envs, &mut self.opp_predictor);
            for env in self.envs.iter_mut() {
                if !env.done && !env.state.is_game_over() {
                    auto_play_forced(&mut env.state, &mut env.rng);
                }
            }
            let needs_retry = self.envs.iter().any(|e| !e.done && e.state.is_game_over());
            if !needs_retry { break; }
        }

        // Phase 2: Build observations for agent decision points
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut mask_flat: Vec<f32> = Vec::new();
        let mut env_indices: Vec<usize> = Vec::new();

        for idx in 0..self.n_envs {
            let env = &mut self.envs[idx];
            if env.done || env.state.is_game_over() { continue; }

            let (actor, actions) = env.state.generate_possible_actions();
            if actions.is_empty() || actor != env.agent { continue; }

            let obs = build_observation(&env.state, env.agent);
            let (mask_bool, action_map) = build_action_map(&actions, env.agent);
            let mask_f32: [f32; NUM_ACTIONS] = {
                let mut m = [0.0f32; NUM_ACTIONS];
                for (i, &b) in mask_bool.iter().enumerate() { m[i] = if b { 1.0 } else { 0.0 }; }
                m
            };

            obs_flat.extend_from_slice(&obs);
            mask_flat.extend_from_slice(&mask_f32);
            env_indices.push(idx);

            env.pending_actions = actions;
            env.pending_action_map = action_map;
            env.pending_mask = mask_f32;
        }

        PendingBatch { n_pending: env_indices.len(), obs_flat, mask_flat, env_indices }
    }

    pub fn run_batch<F>(
        &mut self, num_games: usize, mut predict_fn: F,
    ) -> Vec<GameTrajectory>
    where
        F: FnMut(&[f32], &[f32], usize) -> (Vec<usize>, Vec<f32>, Vec<f32>),
    {
        self.init_envs();
        let mut all_trajectories: Vec<GameTrajectory> = Vec::with_capacity(num_games);
        let mut completed = 0usize;
        let start = Instant::now();

        let mut env_obs: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_actions: Vec<Vec<i32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_log_probs: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_values: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_rewards: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_dones: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();
        let mut env_masks: Vec<Vec<f32>> = (0..self.n_envs).map(|_| Vec::new()).collect();

        // Track opponent type stats
        let mut sp_games = 0u32;
        let mut heur_games = 0u32;
        let mut rand_games = 0u32;

        while completed < num_games {
            let pending = self.get_pending_observations();
            if pending.n_pending == 0 { break; }

            let (actions, log_probs, values) = predict_fn(
                &pending.obs_flat, &pending.mask_flat, pending.n_pending,
            );

            for (i, &idx) in pending.env_indices.iter().enumerate() {
                let semantic_action = actions[i];
                let log_prob = log_probs[i];
                let value = values[i];
                let env = &mut self.envs[idx];

                let obs = build_observation(&env.state, env.agent);
                env_obs[idx].extend_from_slice(&obs);
                env_actions[idx].push(semantic_action as i32);
                env_log_probs[idx].push(log_prob);
                env_values[idx].push(value);
                env_masks[idx].extend_from_slice(&env.pending_mask);

                if let Some(&di) = env.pending_action_map.get(&semantic_action) {
                    if di < env.pending_actions.len() {
                        apply_action(&mut env.rng, &mut env.state, &env.pending_actions[di]);
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
                } else { 0.0 };

                env_rewards[idx].push(reward);
                env_dones[idx].push(if done { 1.0 } else { 0.0 });

                if done {
                    match env.opponent_type {
                        OpponentType::SelfPlay => sp_games += 1,
                        OpponentType::Heuristic => heur_games += 1,
                        OpponentType::Random => rand_games += 1,
                    }

                    let traj = GameTrajectory {
                        observations: std::mem::take(&mut env_obs[idx]),
                        actions: std::mem::take(&mut env_actions[idx]),
                        log_probs: std::mem::take(&mut env_log_probs[idx]),
                        values: std::mem::take(&mut env_values[idx]),
                        rewards: std::mem::take(&mut env_rewards[idx]),
                        dones: std::mem::take(&mut env_dones[idx]),
                        masks: std::mem::take(&mut env_masks[idx]),
                        game_value: reward,
                        move_count: env.move_count,
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
            "Rust PPO self-play: {} games in {:.1}s ({:.1} games/s) | League: {}sp + {}heur + {}rand",
            completed, elapsed, completed as f64 / elapsed.max(0.001),
            sp_games, heur_games, rand_games,
        );

        all_trajectories
    }
}
