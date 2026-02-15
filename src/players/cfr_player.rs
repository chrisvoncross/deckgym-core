//! Real-Time CFR Player — Meta/Pluribus-level game-time search.
//!
//! At every decision point, runs depth-limited External Sampling MCCFR
//! with DCFR discounting, warm-started from the neural network's policy.
//! This is the key difference between "good policy" (OnnxPlayer: argmax)
//! and "Nash equilibrium" (CfrPlayer: real-time search).
//!
//! Implements the same algorithm as `rebel/rust/src/cfr.rs` but integrated
//! into the deckgym Player trait for live gameplay and evaluation.
//!
//! References:
//!   - Brown & Sandholm (2019): "Solving Imperfect-Information Games via
//!     Discounted Regret Minimization" (DCFR)
//!   - Brown et al. (2020): "Combining Deep Reinforcement Learning and
//!     Search for Imperfect-Information Games" (ReBeL)
//!   - Brown & Sandholm (2019): "Superhuman AI for multiplayer poker" (Pluribus)

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::actions::{apply_action, Action};
use crate::alphazero_mcts::{
    auto_play_forced, build_action_map, build_observation, determinize_state, NUM_ACTIONS, OBS_SIZE,
};
use crate::onnx_predictor::OnnxPredictor;
use crate::state::{GameOutcome, State};
use crate::Deck;

use super::Player;

// ═══════════════════════════════════════════════════════════════════════
// CFR Configuration
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for real-time CFR search.
///
/// Default values are tuned for ~1-3 second decision time on a single
/// RTX 5060 Ti. Increase determinizations/iterations for stronger play
/// at the cost of latency.
#[derive(Debug, Clone)]
pub struct CfrConfig {
    /// Number of determinizations (samples of hidden information).
    /// More = better coverage of opponent's possible hands/decks.
    /// Pluribus uses 200; we use 16-32 for real-time play.
    pub num_determinizations: usize,
    /// Number of CFR iterations per solve.
    /// More = closer to Nash equilibrium.
    /// Training uses 4; real-time can afford 8-16.
    pub cfr_iterations: usize,
    /// Maximum search depth before falling back to NN value estimate.
    /// Deeper = more accurate but exponentially more expensive.
    pub depth_limit: usize,
    /// DCFR α parameter — positive regret discount exponent.
    pub dcfr_alpha: f64,
    /// DCFR β parameter — negative regret discount exponent.
    pub dcfr_beta: f64,
    /// DCFR γ parameter — strategy sum discount exponent.
    pub dcfr_gamma: f64,
}

impl Default for CfrConfig {
    fn default() -> Self {
        CfrConfig {
            num_determinizations: 16,
            cfr_iterations: 8,
            depth_limit: 3,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CFR Node — regret + strategy tracking per information set
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct CfrNode {
    cumulative_regret: [f64; NUM_ACTIONS],
    cumulative_strategy: [f64; NUM_ACTIONS],
    legal_mask: [bool; NUM_ACTIONS],
    num_legal: usize,
}

impl CfrNode {
    /// Create a new node, warm-started from NN policy priors.
    fn new(legal_mask: &[bool; NUM_ACTIONS], nn_policy: &[f32; NUM_ACTIONS]) -> Self {
        let num_legal = legal_mask.iter().filter(|&&m| m).count();
        let mut cumulative_regret = [0.0f64; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            if legal_mask[i] {
                // Warm-start: positive regret = NN prior → first strategy ≈ NN policy
                cumulative_regret[i] = nn_policy[i].max(1e-6) as f64;
            }
        }
        CfrNode {
            cumulative_regret,
            cumulative_strategy: [0.0; NUM_ACTIONS],
            legal_mask: *legal_mask,
            num_legal,
        }
    }

    /// Current strategy via regret matching (used during traversal).
    fn current_strategy(&self) -> [f64; NUM_ACTIONS] {
        let mut strategy = [0.0f64; NUM_ACTIONS];
        let mut positive_sum = 0.0;
        for i in 0..NUM_ACTIONS {
            if self.legal_mask[i] && self.cumulative_regret[i] > 0.0 {
                strategy[i] = self.cumulative_regret[i];
                positive_sum += strategy[i];
            }
        }
        if positive_sum > 0.0 {
            for s in &mut strategy { *s /= positive_sum; }
        } else {
            let uniform = 1.0 / self.num_legal.max(1) as f64;
            for i in 0..NUM_ACTIONS {
                if self.legal_mask[i] { strategy[i] = uniform; }
            }
        }
        strategy
    }

    /// Average strategy — converges to Nash equilibrium.
    fn average_strategy(&self) -> [f64; NUM_ACTIONS] {
        let mut avg = [0.0f64; NUM_ACTIONS];
        let total: f64 = self.cumulative_strategy.iter().sum();
        if total > 0.0 {
            for i in 0..NUM_ACTIONS { avg[i] = self.cumulative_strategy[i] / total; }
        } else {
            let uniform = 1.0 / self.num_legal.max(1) as f64;
            for i in 0..NUM_ACTIONS {
                if self.legal_mask[i] { avg[i] = uniform; }
            }
        }
        avg
    }

    /// DCFR discounting (Brown & Sandholm 2019).
    fn apply_dcfr_discount(&mut self, t: f64, alpha: f64, beta: f64, gamma: f64) {
        let pos_mult = t.powf(alpha) / (t.powf(alpha) + 1.0);
        let neg_mult = t.powf(beta) / (t.powf(beta) + 1.0);
        let strat_mult = (t / (t + 1.0)).powf(gamma);
        for i in 0..NUM_ACTIONS {
            if self.cumulative_regret[i] > 0.0 {
                self.cumulative_regret[i] *= pos_mult;
            } else {
                self.cumulative_regret[i] *= neg_mult;
            }
            self.cumulative_strategy[i] *= strat_mult;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CFR Solver — depth-limited External Sampling MCCFR
// ═══════════════════════════════════════════════════════════════════════

struct CfrSolver {
    config: CfrConfig,
    nodes: HashMap<u64, CfrNode>,
    predictor: Arc<Mutex<OnnxPredictor>>,
}

impl CfrSolver {
    fn new(config: CfrConfig, predictor: Arc<Mutex<OnnxPredictor>>) -> Self {
        CfrSolver {
            config,
            nodes: HashMap::with_capacity(256),
            predictor,
        }
    }

    /// Evaluate a leaf state with fresh NN inference (Bug 1 fix).
    ///
    /// Instead of recycling the root nn_value, we build the leaf state's
    /// observation and run a proper NN forward pass. This is what the
    /// Python CFR (cfr.py:198-201) does correctly.
    fn nn_leaf_value(&self, state: &mut State, agent: usize, traverser: usize) -> f64 {
        let obs = build_observation(state, agent);
        let (_actor, actions) = state.generate_possible_actions();
        let (mask_bool, _) = build_action_map(&actions, agent);
        let mask_f32: Vec<f32> = mask_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let value = {
            let mut pred = self.predictor.lock().expect("OnnxPredictor lock poisoned");
            let (_, values) = pred.predict(&obs, &mask_f32, 1);
            values[0] as f64
        };
        // Value is from agent's perspective; flip for opponent
        if agent == traverser { value } else { -value }
    }

    /// Solve for Nash equilibrium strategy at the current game state.
    ///
    /// Returns the averaged CFR strategy (77 floats, sums to ~1.0).
    fn solve(
        &mut self,
        state: &State,
        agent: usize,
        root_obs: &[f32; OBS_SIZE],
        legal_mask: &[bool; NUM_ACTIONS],
        nn_policy: &[f32; NUM_ACTIONS],
        rng: &mut StdRng,
    ) -> [f64; NUM_ACTIONS] {
        self.nodes.clear();

        for iteration in 0..self.config.cfr_iterations {
            for _det in 0..self.config.num_determinizations {
                let mut det_state = state.clone();
                let mut det_rng = StdRng::seed_from_u64(rng.gen());

                // Determinize hidden information (shuffle opponent hand + decks)
                determinize_state(&mut det_state, &mut det_rng);

                self.traverse(
                    &mut det_state,
                    &mut det_rng,
                    agent,
                    agent,
                    0,
                    nn_policy,
                );
            }

            // DCFR discounting after each iteration
            let t = (iteration + 1) as f64;
            for node in self.nodes.values_mut() {
                node.apply_dcfr_discount(
                    t,
                    self.config.dcfr_alpha,
                    self.config.dcfr_beta,
                    self.config.dcfr_gamma,
                );
            }
        }

        // Return average strategy from root node (Bug 2 fix: use obs + mask)
        let root_key = Self::make_key(root_obs, legal_mask);
        if let Some(node) = self.nodes.get(&root_key) {
            node.average_strategy()
        } else {
            // Fallback: return NN policy as f64
            let mut fallback = [0.0f64; NUM_ACTIONS];
            for i in 0..NUM_ACTIONS {
                fallback[i] = nn_policy[i] as f64;
            }
            fallback
        }
    }

    /// Recursive External Sampling MCCFR traversal.
    ///
    /// Bug 1 fix: at depth_limit, runs fresh NN inference on the leaf state
    /// instead of recycling the root nn_value.
    /// Bug 2 fix: info-set key uses observation + legal_mask for uniqueness.
    fn traverse(
        &mut self,
        state: &mut State,
        rng: &mut StdRng,
        agent: usize,
        traverser: usize,
        depth: usize,
        nn_policy: &[f32; NUM_ACTIONS],
    ) -> f64 {
        // Terminal check
        if state.is_game_over() {
            return match state.winner {
                Some(GameOutcome::Win(p)) if p == traverser => 1.0,
                Some(GameOutcome::Win(_)) => -1.0,
                _ => 0.0,
            };
        }

        // Depth limit → fresh NN value estimate for this LEAF state (Bug 1 fix)
        if depth >= self.config.depth_limit {
            return self.nn_leaf_value(state, agent, traverser);
        }

        // Get legal actions at this state
        let (_actor, actions) = state.generate_possible_actions();
        if actions.is_empty() {
            return 0.0;
        }

        let (legal_mask, action_map) = build_action_map(&actions, agent);
        let legal_actions: Vec<usize> = (0..NUM_ACTIONS)
            .filter(|&i| legal_mask[i])
            .collect();

        if legal_actions.is_empty() {
            return 0.0;
        }

        // Get or create CFR node (Bug 2 fix: key includes observation)
        let obs = build_observation(state, agent);
        let key = Self::make_key(&obs, &legal_mask);
        if !self.nodes.contains_key(&key) {
            self.nodes.insert(key, CfrNode::new(&legal_mask, nn_policy));
        }

        let strategy = self.nodes.get(&key).unwrap().current_strategy();

        if _actor == traverser {
            // Traverser's node: explore ALL actions, compute counterfactual values
            let mut action_values = [0.0f64; NUM_ACTIONS];
            let mut node_value = 0.0f64;

            for &sem_action in &legal_actions {
                let mut child_state = state.clone();
                let mut child_rng = StdRng::seed_from_u64(rng.gen());

                let game_over = step_for_cfr(
                    &mut child_state,
                    &mut child_rng,
                    agent,
                    sem_action,
                    &actions,
                    &action_map,
                );

                let child_val = if game_over {
                    match child_state.winner {
                        Some(GameOutcome::Win(p)) if p == traverser => 1.0,
                        Some(GameOutcome::Win(_)) => -1.0,
                        _ => 0.0,
                    }
                } else {
                    self.traverse(
                        &mut child_state,
                        &mut child_rng,
                        agent,
                        traverser,
                        depth + 1,
                        nn_policy,
                    )
                };

                action_values[sem_action] = child_val;
                node_value += strategy[sem_action] * child_val;
            }

            // Update regrets and strategy sum
            let node = self.nodes.get_mut(&key).unwrap();
            for &sem_action in &legal_actions {
                let regret = action_values[sem_action] - node_value;
                node.cumulative_regret[sem_action] += regret;
                node.cumulative_strategy[sem_action] += strategy[sem_action];
            }

            node_value
        } else {
            // Opponent's node: sample ONE action (External Sampling)
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen_action = legal_actions[0];
            for &sem_action in &legal_actions {
                cumulative += strategy[sem_action];
                if r < cumulative {
                    chosen_action = sem_action;
                    break;
                }
            }

            let mut child_state = state.clone();
            let mut child_rng = StdRng::seed_from_u64(rng.gen());

            let game_over = step_for_cfr(
                &mut child_state,
                &mut child_rng,
                agent,
                chosen_action,
                &actions,
                &action_map,
            );

            if game_over {
                match child_state.winner {
                    Some(GameOutcome::Win(p)) if p == traverser => 1.0,
                    Some(GameOutcome::Win(_)) => -1.0,
                    _ => 0.0,
                }
            } else {
                self.traverse(
                    &mut child_state,
                    &mut child_rng,
                    agent,
                    traverser,
                    depth + 1,
                    nn_policy,
                )
            }
        }
    }

    /// FNV-1a hash of observation + action mask → info-set key (Bug 2 fix).
    ///
    /// Matches Python CFR's _info_state_key(obs, mask): quantizes observation
    /// to int16 (4 decimal places) and hashes both obs + mask together.
    /// This prevents different game states with the same legal moves from
    /// colliding in the node table.
    fn make_key(obs: &[f32; OBS_SIZE], legal_mask: &[bool; NUM_ACTIONS]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        // Hash quantized observation (matches Python: (obs * 10000).astype(int16))
        for &v in obs.iter() {
            let q = (v * 10000.0) as i16;
            let bytes = q.to_le_bytes();
            for &b in &bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        // Hash legal mask
        for i in 0..NUM_ACTIONS {
            if legal_mask[i] {
                hash ^= (i as u64).wrapping_add(0x9e3779b97f4a7c15);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Game stepping for CFR — apply action + forced + opponent random
// ═══════════════════════════════════════════════════════════════════════

/// Apply a semantic action, auto-play forced actions, handle opponent turn.
/// Returns true if the game is over after this step.
fn step_for_cfr(
    state: &mut State,
    rng: &mut StdRng,
    agent: usize,
    semantic_action: usize,
    actions: &[Action],
    action_map: &HashMap<usize, usize>,
) -> bool {
    // Apply agent's chosen action
    if let Some(&deckgym_idx) = action_map.get(&semantic_action) {
        if deckgym_idx < actions.len() {
            apply_action(rng, state, &actions[deckgym_idx]);
        }
    } else {
        // Invalid action — try END_TURN as fallback
        if let Some(&end_idx) = action_map.get(&0) {
            if end_idx < actions.len() {
                apply_action(rng, state, &actions[end_idx]);
            }
        }
    }

    // Auto-play forced actions
    auto_play_forced(state, rng);

    // Opponent turn (random policy for CFR simulations)
    if !state.is_game_over() {
        opponent_turn_random_cfr(state, rng, agent);
    }

    // Auto-play forced for agent's next turn
    if !state.is_game_over() {
        auto_play_forced(state, rng);
    }

    state.is_game_over()
}

/// Play opponent's turn with random actions until it's agent's turn again.
fn opponent_turn_random_cfr(state: &mut State, rng: &mut StdRng, agent: usize) {
    let max_actions = 10u32;
    let mut count = 0u32;

    while !state.is_game_over() && count < max_actions {
        auto_play_forced(state, rng);
        if state.is_game_over() {
            break;
        }

        let (actor, actions) = state.generate_possible_actions();
        if actions.is_empty() || actor == agent {
            break;
        }

        let idx = rng.gen_range(0..actions.len());
        apply_action(rng, state, &actions[idx]);
        count += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CfrPlayer — the Player trait implementation
// ═══════════════════════════════════════════════════════════════════════

/// Real-time CFR search player.
///
/// At every decision point:
///   1. Builds observation from game state
///   2. Runs NN inference to get policy prior + value estimate
///   3. Runs depth-limited MCCFR search warm-started from NN
///   4. Samples action from the Nash equilibrium strategy
///
/// This is the Pluribus approach: NN provides the "blueprint" strategy,
/// CFR refines it in real-time for the specific game situation.
pub struct CfrPlayer {
    pub deck: Deck,
    predictor: Arc<Mutex<OnnxPredictor>>,
    config: CfrConfig,
}

impl CfrPlayer {
    /// Create a new CfrPlayer, loading the ONNX model from disk.
    pub fn new(
        deck: Deck,
        model_path: &Path,
        config: CfrConfig,
    ) -> Result<Self, ort::Error> {
        let predictor = OnnxPredictor::new(model_path)?;
        Ok(Self {
            deck,
            predictor: Arc::new(Mutex::new(predictor)),
            config,
        })
    }

    /// Create a CfrPlayer sharing an already-loaded predictor.
    pub fn with_shared_predictor(
        deck: Deck,
        predictor: Arc<Mutex<OnnxPredictor>>,
        config: CfrConfig,
    ) -> Self {
        Self {
            deck,
            predictor,
            config,
        }
    }
}

impl Player for CfrPlayer {
    fn decision_fn(
        &mut self,
        rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action {
        // Trivial case: only one legal action
        if possible_actions.len() == 1 {
            return possible_actions[0].clone();
        }

        let agent = possible_actions[0].actor;

        // 1. Build observation + action map
        let obs = build_observation(state, agent);
        let (mask_bool, action_map) = build_action_map(possible_actions, agent);
        let mask_f32: Vec<f32> = mask_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // 2. Run NN inference for policy prior (value now computed per-leaf)
        let nn_policy = {
            let mut pred = self.predictor.lock().expect("OnnxPredictor lock poisoned");
            let (policies, _values) = pred.predict(&obs, &mask_f32, 1);
            let mut policy = [0.0f32; NUM_ACTIONS];
            policy.copy_from_slice(&policies[..NUM_ACTIONS]);
            policy
        };

        // 3. Run real-time CFR search (predictor shared for leaf evaluation)
        let mut solver = CfrSolver::new(self.config.clone(), self.predictor.clone());
        let cfr_strategy = solver.solve(
            state,
            agent,
            &obs,
            &mask_bool,
            &nn_policy,
            rng,
        );

        // 4. Sample action from Nash equilibrium strategy
        //    (sampling > argmax for game-theoretic optimality in imperfect info)
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut chosen_sem = 0usize;
        for sem in 0..NUM_ACTIONS {
            if mask_bool[sem] {
                cumulative += cfr_strategy[sem];
                if r < cumulative {
                    chosen_sem = sem;
                    break;
                }
                chosen_sem = sem; // fallback to last legal action
            }
        }

        // 5. Map semantic action → deckgym action
        if let Some(&deckgym_idx) = action_map.get(&chosen_sem) {
            possible_actions[deckgym_idx].clone()
        } else {
            log::warn!(
                "CfrPlayer: semantic action {} not in action_map, using fallback",
                chosen_sem
            );
            possible_actions[0].clone()
        }
    }

    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }
}

impl std::fmt::Debug for CfrPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CfrPlayer(dets={}, iters={}, depth={})",
            self.config.num_determinizations,
            self.config.cfr_iterations,
            self.config.depth_limit,
        )
    }
}
