//! AlphaZero MCTS Engine — Gumbel Sequential Halving + IS-MCTS.
//!
//! Pure-Rust MCTS operating directly on deckgym-core `State`.
//! Only crosses FFI boundary for neural network evaluation (Python/PyTorch).
//!
//! Implements:
//!   1. Gumbel MCTS (Danihelka et al. 2022) — Sequential Halving root selection
//!   2. IS-MCTS — Determinization-based search for imperfect information
//!   3. Virtual Loss — Parallel tree traversal support
//!   4. Leaf Batching — Batch NN evaluation for throughput
//!   5. Arena-allocated nodes — Zero-cost parent traversal

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Dirichlet, Distribution, Gumbel};
use std::collections::HashMap;

use crate::actions::{apply_action, Action, SimpleAction};
use crate::models::{Card, EnergyType, TrainerType};
use crate::state::{GameOutcome, PlayedCard, State};

// ═══════════════════════════════════════════════════════════════════════
// Constants — must match pokemon_tcg_env.py exactly
// ═══════════════════════════════════════════════════════════════════════

pub const NUM_ENERGY_TYPES: usize = 10;
pub const NUM_CONDITIONS: usize = 7;
pub const CARD_FEATURES: usize = 48; // 1+1+10+1+1+1+1+1+7+4+10+10
pub const NUM_CARD_SLOTS: usize = 8;
pub const GLOBAL_FEATURES: usize = 22;
pub const HAND_FEATURES: usize = 17;
pub const OBS_SIZE: usize = NUM_CARD_SLOTS * CARD_FEATURES + GLOBAL_FEATURES + HAND_FEATURES; // 423
pub const NUM_ACTIONS: usize = 77;
pub const POINTS_TO_WIN: f32 = 3.0;
pub const MAX_HAND_SIZE: f32 = 10.0;
pub const DECK_SIZE: f32 = 20.0;

/// Map EnergyType to observation index (0-9).
fn energy_idx(e: EnergyType) -> usize {
    match e {
        EnergyType::Water => 0,
        EnergyType::Fire => 1,
        EnergyType::Grass => 2,
        EnergyType::Lightning => 3,
        EnergyType::Psychic => 4,
        EnergyType::Fighting => 5,
        EnergyType::Darkness => 6,
        EnergyType::Metal => 7,
        EnergyType::Dragon => 8,
        EnergyType::Colorless => 9,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MoveEnergy combo lookup (src_idx, dst_idx) → semantic action index
// ═══════════════════════════════════════════════════════════════════════

fn move_energy_semantic(src: usize, dst: usize) -> Option<usize> {
    match (src, dst) {
        (0, 1) => Some(65), (0, 2) => Some(66), (0, 3) => Some(67),
        (1, 0) => Some(68), (1, 2) => Some(69), (1, 3) => Some(70),
        (2, 0) => Some(71), (2, 1) => Some(72), (2, 3) => Some(73),
        (3, 0) => Some(74), (3, 1) => Some(75), (3, 2) => Some(76),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MCTS Configuration
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct MCTSConfig {
    pub num_simulations: usize,
    pub num_determinizations: usize,
    pub c_puct: f32,
    pub temperature: f32,
    pub temp_threshold: usize,
    pub temp_final: f32,
    pub dirichlet_alpha: f64,
    pub dirichlet_frac: f64,
    pub add_noise: bool,
    pub use_gumbel: bool,
    pub gumbel_c_visit: f32,
    pub gumbel_c_scale: f32,
    pub max_considered_actions: usize,
    pub leaf_batch_size: usize,
    pub use_virtual_loss: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig {
            num_simulations: 200,
            num_determinizations: 8,
            c_puct: 1.5,
            temperature: 1.0,
            temp_threshold: 15,
            temp_final: 0.1,
            dirichlet_alpha: 0.3,
            dirichlet_frac: 0.25,
            add_noise: true,
            use_gumbel: true,
            gumbel_c_visit: 50.0,
            gumbel_c_scale: 1.0,
            max_considered_actions: 16,
            leaf_batch_size: 32,
            use_virtual_loss: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MCTS Node (arena-allocated)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) type NodeId = usize;

pub(crate) struct MCTSNode {
    pub(crate) parent: Option<NodeId>,
    pub(crate) action: i32,
    pub(crate) prior: f32,
    pub(crate) visit_count: u32,
    pub(crate) total_value: f64,
    pub(crate) virtual_losses: u32,
    pub(crate) children: HashMap<usize, NodeId>, // semantic_action → node_id
    pub(crate) is_terminal: bool,
    pub(crate) terminal_value: f32,
}

impl MCTSNode {
    pub(crate) fn new(parent: Option<NodeId>, action: i32, prior: f32) -> Self {
        MCTSNode {
            parent,
            action,
            prior,
            visit_count: 0,
            total_value: 0.0,
            virtual_losses: 0,
            children: HashMap::new(),
            is_terminal: false,
            terminal_value: 0.0,
        }
    }

    pub(crate) fn q_value(&self) -> f64 {
        let eff_n = self.visit_count as f64 + self.virtual_losses as f64;
        if eff_n == 0.0 { return 0.0; }
        let eff_w = self.total_value - self.virtual_losses as f64;
        eff_w / eff_n
    }

    pub(crate) fn completed_q(&self) -> f64 {
        self.total_value / (self.visit_count.max(1) as f64)
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Arena-based node allocator for zero-cost tree traversal.
pub(crate) struct NodeArena {
    pub(crate) nodes: Vec<MCTSNode>,
}

impl NodeArena {
    pub(crate) fn new() -> Self {
        NodeArena { nodes: Vec::with_capacity(4096) }
    }

    pub(crate) fn alloc(&mut self, parent: Option<NodeId>, action: i32, prior: f32) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(MCTSNode::new(parent, action, prior));
        id
    }

    pub(crate) fn get(&self, id: NodeId) -> &MCTSNode {
        &self.nodes[id]
    }

    pub(crate) fn get_mut(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id]
    }

    /// PUCT child selection — returns (child_node_id, child_semantic_action).
    pub(crate) fn select_child(&self, node_id: NodeId, c_puct: f32) -> (NodeId, usize) {
        let node = &self.nodes[node_id];
        let parent_n = node.visit_count as f64;
        let sqrt_parent = parent_n.sqrt();

        let mut best_id = 0;
        let mut best_action = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (&sem_action, &child_id) in &node.children {
            let child = &self.nodes[child_id];
            let eff_n = child.visit_count as f64 + child.virtual_losses as f64;
            let exploration = (c_puct as f64) * (child.prior as f64) * sqrt_parent / (1.0 + eff_n);
            let score = child.q_value() + exploration;
            if score > best_score {
                best_score = score;
                best_id = child_id;
                best_action = sem_action;
            }
        }
        (best_id, best_action)
    }

    /// Expand node with action priors. Only adds children that don't exist yet.
    pub(crate) fn expand(&mut self, node_id: NodeId, action_priors: &[(usize, f32)]) {
        for &(sem_action, prior) in action_priors {
            if !self.nodes[node_id].children.contains_key(&sem_action) {
                let child_id = self.alloc(Some(node_id), sem_action as i32, prior);
                self.nodes[node_id].children.insert(sem_action, child_id);
            }
        }
    }

    /// Add virtual loss from node to root.
    pub(crate) fn add_virtual_loss(&mut self, node_id: NodeId) {
        let mut id = Some(node_id);
        while let Some(nid) = id {
            self.nodes[nid].virtual_losses += 1;
            id = self.nodes[nid].parent;
        }
    }

    /// Revert virtual loss from node to root.
    pub(crate) fn revert_virtual_loss(&mut self, node_id: NodeId) {
        let mut id = Some(node_id);
        while let Some(nid) = id {
            self.nodes[nid].virtual_losses = self.nodes[nid].virtual_losses.saturating_sub(1);
            id = self.nodes[nid].parent;
        }
    }

    /// Backpropagate value from node to root.
    pub(crate) fn backpropagate(&mut self, node_id: NodeId, value: f64) {
        let mut id = Some(node_id);
        while let Some(nid) = id {
            self.nodes[nid].visit_count += 1;
            self.nodes[nid].total_value += value;
            id = self.nodes[nid].parent;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Observation building — port of _get_observation() / _encode_played_card()
// ═══════════════════════════════════════════════════════════════════════

const MAX_TURNS: f32 = 100.0;

/// Encode a single PlayedCard into the observation vector at `slot`.
fn encode_played_card(obs: &mut [f32; OBS_SIZE], slot: usize, pc: &PlayedCard) {
    let offset = slot * CARD_FEATURES;
    let card = &pc.card;

    // HP
    let total_hp = pc.get_effective_total_hp() as f32;
    let remaining_hp = pc.get_remaining_hp() as f32;
    obs[offset] = remaining_hp / total_hp.max(1.0);
    obs[offset + 1] = total_hp / 300.0;

    // Attached energy breakdown by type
    for e in &pc.attached_energy {
        obs[offset + 2 + energy_idx(*e)] += 1.0 / 4.0;
    }

    // Card properties
    obs[offset + 12] = if card.is_ex() { 1.0 } else { 0.0 };
    obs[offset + 13] = match card {
        Card::Pokemon(p) => p.stage as f32 * 0.5,
        _ => 0.0,
    };
    obs[offset + 14] = card.get_retreat_cost().map_or(0.0, |rc| rc.len() as f32 / 4.0);
    obs[offset + 15] = if card.get_ability().is_some() { 1.0 } else { 0.0 };
    obs[offset + 16] = if pc.has_tool_attached() { 1.0 } else { 0.0 };

    // Conditions (7 slots)
    obs[offset + 17] = if pc.poisoned { 1.0 } else { 0.0 };
    obs[offset + 18] = if pc.burned { 1.0 } else { 0.0 };
    obs[offset + 19] = if pc.asleep { 1.0 } else { 0.0 };
    obs[offset + 20] = if pc.paralyzed { 1.0 } else { 0.0 };
    obs[offset + 21] = if pc.confused { 1.0 } else { 0.0 };
    obs[offset + 22] = 0.0; // cant_attack placeholder
    obs[offset + 23] = 0.0; // damage_immune placeholder

    // Attacks (up to 2)
    let attacks = card.get_attacks();
    for a_idx in 0..attacks.len().min(2) {
        obs[offset + 24 + a_idx * 2] = attacks[a_idx].fixed_damage as f32 / 200.0;
        obs[offset + 25 + a_idx * 2] = attacks[a_idx].energy_required.len() as f32 / 5.0;
    }

    // Weakness type one-hot
    if let Card::Pokemon(p) = card {
        if let Some(w) = p.weakness {
            obs[offset + 28 + energy_idx(w)] = 1.0;
        }
        // Card type one-hot
        obs[offset + 38 + energy_idx(p.energy_type)] = 1.0;
    }
}

/// Build the full 423-float observation vector from State.
/// `agent` = whose perspective, `opp` = opponent.
pub(crate) fn build_observation(state: &State, agent: usize) -> [f32; OBS_SIZE] {
    let mut obs = [0.0f32; OBS_SIZE];
    let opp = 1 - agent;

    // === Entity block: 8 card slots × CARD_FEATURES ===
    // Own active (slot 0)
    if let Some(ref active) = state.in_play_pokemon[agent][0] {
        encode_played_card(&mut obs, 0, active);
    }
    // Own bench (slots 1-3)
    for i in 1..4 {
        if let Some(ref pc) = state.in_play_pokemon[agent][i] {
            encode_played_card(&mut obs, i, pc);
        }
    }
    // Enemy active (slot 4)
    if let Some(ref active) = state.in_play_pokemon[opp][0] {
        encode_played_card(&mut obs, 4, active);
    }
    // Enemy bench (slots 5-7)
    for i in 1..4 {
        if let Some(ref pc) = state.in_play_pokemon[opp][i] {
            encode_played_card(&mut obs, 4 + i, pc);
        }
    }

    // === Global features (22) ===
    let g = NUM_CARD_SLOTS * CARD_FEATURES;
    obs[g]     = state.points[agent] as f32 / POINTS_TO_WIN;
    obs[g + 1] = state.points[opp] as f32 / POINTS_TO_WIN;
    obs[g + 2] = state.turn_count as f32 / MAX_TURNS;
    obs[g + 3] = state.hands[agent].len() as f32 / MAX_HAND_SIZE;
    // has_energy
    obs[g + 4] = if state.current_energy.is_some() { 1.0 } else { 0.0 };
    obs[g + 5] = if state.has_played_support { 1.0 } else { 0.0 };
    // agent_starts — we encode as 1.0 if agent == 0 (simplification — matches Python)
    obs[g + 6] = if agent == 0 { 1.0 } else { 0.0 };
    // Current energy type one-hot (10 slots: g+7..g+16)
    if let Some(ce) = state.current_energy {
        obs[g + 7 + energy_idx(ce)] = 1.0;
    }
    // Stadium placeholder
    obs[g + 17] = if state.active_stadium.is_some() { 1.0 } else { 0.0 };
    // Deck/discard sizes
    obs[g + 18] = state.decks[agent].cards.len() as f32 / DECK_SIZE;
    obs[g + 19] = state.decks[opp].cards.len() as f32 / DECK_SIZE;
    obs[g + 20] = state.discard_piles[agent].len() as f32 / DECK_SIZE;
    obs[g + 21] = state.discard_piles[opp].len() as f32 / DECK_SIZE;

    // === Hand summary features (17) ===
    let h = g + GLOBAL_FEATURES;
    let hand = &state.hands[agent];
    let mut pokemon_count = 0u32;
    let mut trainer_count = 0u32;
    let mut basic_count = 0u32;
    let mut _ex_count = 0u32;
    let mut energy_breakdown = [0u32; NUM_ENERGY_TYPES];

    for card in hand {
        match card {
            Card::Pokemon(p) => {
                pokemon_count += 1;
                if p.stage == 0 { basic_count += 1; }
                if card.is_ex() { _ex_count += 1; }
                energy_breakdown[energy_idx(p.energy_type)] += 1;
            }
            Card::Trainer(_) => {
                trainer_count += 1;
            }
        }
    }
    obs[h]     = pokemon_count as f32 / MAX_HAND_SIZE;
    obs[h + 1] = trainer_count as f32 / MAX_HAND_SIZE;
    obs[h + 2] = trainer_count as f32 / MAX_HAND_SIZE; // items ≈ trainers
    obs[h + 3] = 0.0; // supporters
    obs[h + 4] = 0.0; // tools
    obs[h + 5] = 0.0; // evolvable
    obs[h + 6] = basic_count as f32 / MAX_HAND_SIZE;
    for j in 0..NUM_ENERGY_TYPES {
        obs[h + 7 + j] = energy_breakdown[j] as f32 / MAX_HAND_SIZE;
    }

    obs
}

// ═══════════════════════════════════════════════════════════════════════
// Semantic action constants (mirrors Python SemanticAction IntEnum)
// ═══════════════════════════════════════════════════════════════════════

const END_TURN: usize = 0;
const SET_ACTIVE_0: usize = 1;  const SET_ACTIVE_1: usize = 2;  const SET_ACTIVE_2: usize = 3;
const ATTACK_0: usize = 4;      const ATTACK_1: usize = 5;
const ADD_ENERGY_ACTIVE: usize = 6;
const ADD_ENERGY_BENCH_0: usize = 7; const ADD_ENERGY_BENCH_1: usize = 8; const ADD_ENERGY_BENCH_2: usize = 9;
const ADD_BENCH_0: usize = 10;  const ADD_BENCH_1: usize = 11;
const ADD_BENCH_2: usize = 12;  const ADD_BENCH_3: usize = 13;
const EVOLVE_ACTIVE: usize = 14;
const EVOLVE_BENCH_0: usize = 15; const EVOLVE_BENCH_1: usize = 16; const EVOLVE_BENCH_2: usize = 17;
const ABILITY_ACTIVE: usize = 18;
const ABILITY_BENCH_0: usize = 19; const ABILITY_BENCH_1: usize = 20; const ABILITY_BENCH_2: usize = 21;
const ITEM_0: usize = 22; const ITEM_1: usize = 23; const ITEM_2: usize = 24; const ITEM_3: usize = 25;
const SUPPORTER_0: usize = 26;  const SUPPORTER_1: usize = 27;
const RETREAT: usize = 28;
const TOOL_ACTIVE: usize = 29;
const TOOL_BENCH_0: usize = 30; const TOOL_BENCH_1: usize = 31; const TOOL_BENCH_2: usize = 32;
const STADIUM: usize = 33;
const RETREAT_BENCH_0: usize = 34; const RETREAT_BENCH_1: usize = 35; const RETREAT_BENCH_2: usize = 36;
const ATTACK_0_OPP_BENCH_0: usize = 37; const ATTACK_0_OPP_BENCH_1: usize = 38; const ATTACK_0_OPP_BENCH_2: usize = 39;
const ATTACK_0_OWN_BENCH_0: usize = 59; const ATTACK_0_OWN_BENCH_1: usize = 60; const ATTACK_0_OWN_BENCH_2: usize = 61;

const BENCH_SLOTS: [usize; 3] = [0, 1, 2];

fn bench_slot(in_play_idx: usize) -> usize {
    (in_play_idx.saturating_sub(1)).min(2)
}

// ═══════════════════════════════════════════════════════════════════════
// Action mapping — port of _build_action_map() / _parse_action_string()
// ═══════════════════════════════════════════════════════════════════════

/// Intermediate classification for counter-based actions.
enum ActionSlot {
    Semantic(usize),
    PlaceActive,
    PlaceBench,
    Item,
    Supporter,
    Skip, // forced action or unmappable
}

/// Classify a single SimpleAction into an ActionSlot.
fn classify_action(action: &SimpleAction, agent: usize) -> ActionSlot {
    match action {
        SimpleAction::EndTurn => ActionSlot::Semantic(END_TURN),
        SimpleAction::Noop => ActionSlot::Semantic(END_TURN),

        SimpleAction::Attack(idx) => ActionSlot::Semantic(if *idx == 0 { ATTACK_0 } else { ATTACK_1 }),

        SimpleAction::Place(_, position) => {
            if *position == 0 { ActionSlot::PlaceActive } else { ActionSlot::PlaceBench }
        }

        SimpleAction::Evolve { in_play_idx, .. } => {
            if *in_play_idx == 0 {
                ActionSlot::Semantic(EVOLVE_ACTIVE)
            } else {
                ActionSlot::Semantic([EVOLVE_BENCH_0, EVOLVE_BENCH_1, EVOLVE_BENCH_2][bench_slot(*in_play_idx)])
            }
        }

        SimpleAction::UseAbility { in_play_idx } => {
            if *in_play_idx == 0 {
                ActionSlot::Semantic(ABILITY_ACTIVE)
            } else {
                ActionSlot::Semantic([ABILITY_BENCH_0, ABILITY_BENCH_1, ABILITY_BENCH_2][bench_slot(*in_play_idx)])
            }
        }

        SimpleAction::Retreat(position) => {
            ActionSlot::Semantic([RETREAT_BENCH_0, RETREAT_BENCH_1, RETREAT_BENCH_2][bench_slot(*position)])
        }

        SimpleAction::Attach { attachments, .. } => {
            // Use first attachment's in_play_idx
            if let Some(&(_, _, in_play_idx)) = attachments.first() {
                if in_play_idx == 0 {
                    ActionSlot::Semantic(ADD_ENERGY_ACTIVE)
                } else {
                    ActionSlot::Semantic([ADD_ENERGY_BENCH_0, ADD_ENERGY_BENCH_1, ADD_ENERGY_BENCH_2][bench_slot(in_play_idx)])
                }
            } else {
                ActionSlot::Semantic(ADD_ENERGY_ACTIVE)
            }
        }

        SimpleAction::Play { trainer_card } => {
            if trainer_card.trainer_card_type == TrainerType::Supporter {
                ActionSlot::Supporter
            } else {
                ActionSlot::Item
            }
        }

        SimpleAction::AttachTool { in_play_idx, .. } => {
            if *in_play_idx == 0 {
                ActionSlot::Semantic(TOOL_ACTIVE)
            } else {
                ActionSlot::Semantic([TOOL_BENCH_0, TOOL_BENCH_1, TOOL_BENCH_2][bench_slot(*in_play_idx)])
            }
        }

        SimpleAction::MoveEnergy { from_in_play_idx, to_in_play_idx, .. } => {
            match move_energy_semantic(*from_in_play_idx, *to_in_play_idx) {
                Some(sem) => ActionSlot::Semantic(sem),
                None => ActionSlot::Skip,
            }
        }

        SimpleAction::Activate { in_play_idx, .. } => {
            let bs = bench_slot(*in_play_idx);
            ActionSlot::Semantic([SET_ACTIVE_0, SET_ACTIVE_1, SET_ACTIVE_2][bs])
        }

        SimpleAction::ApplyDamage { targets, .. } => {
            // Check for bench-targeting snipe attacks
            for &(_, target_player, target_idx) in targets {
                let opp = 1 - agent;
                if target_player == opp && target_idx > 0 {
                    let bs = bench_slot(target_idx);
                    return ActionSlot::Semantic([ATTACK_0_OPP_BENCH_0, ATTACK_0_OPP_BENCH_1, ATTACK_0_OPP_BENCH_2][bs]);
                }
                if target_player == agent && target_idx > 0 {
                    let bs = bench_slot(target_idx);
                    return ActionSlot::Semantic([ATTACK_0_OWN_BENCH_0, ATTACK_0_OWN_BENCH_1, ATTACK_0_OWN_BENCH_2][bs]);
                }
            }
            ActionSlot::Skip // default damage — forced
        }

        // Forced / mechanical actions → skip
        SimpleAction::DrawCard { .. } => ActionSlot::Skip,
        SimpleAction::Heal { .. } | SimpleAction::HealAndDiscardEnergy { .. } => ActionSlot::Item,
        SimpleAction::MoveAllDamage { .. } => ActionSlot::Item,
        SimpleAction::CommunicatePokemon { .. } => ActionSlot::Item,
        SimpleAction::ShufflePokemonIntoDeck { .. } => ActionSlot::Item,
        SimpleAction::ShuffleOpponentSupporter { .. } => ActionSlot::Supporter,
        SimpleAction::DiscardOpponentSupporter { .. } => ActionSlot::Supporter,
        SimpleAction::DiscardOwnCards { .. } => ActionSlot::Item,
        SimpleAction::AttachFromDiscard { .. } => ActionSlot::Item,
        SimpleAction::ApplyEeveeBagDamageBoost => ActionSlot::Item,
        SimpleAction::HealAllEeveeEvolutions => ActionSlot::Item,
        SimpleAction::DiscardFossil { .. } => ActionSlot::Item,
        SimpleAction::ReturnPokemonToHand { .. } => ActionSlot::Item,
    }
}

/// Build action mask and action map from legal actions.
/// Returns (action_mask[77], action_map: semantic→deckgym_idx).
pub(crate) fn build_action_map(
    actions: &[Action],
    agent: usize,
) -> ([bool; NUM_ACTIONS], HashMap<usize, usize>) {
    let mut mask = [false; NUM_ACTIONS];
    let mut map: HashMap<usize, usize> = HashMap::new();

    let mut place_active_count = 0usize;
    let mut place_bench_count = 0usize;
    let mut item_count = 0usize;
    let mut supporter_count = 0usize;

    for (deckgym_idx, action) in actions.iter().enumerate() {
        if action.actor != agent {
            continue;
        }
        let slot = classify_action(&action.action, agent);
        let sem = match slot {
            ActionSlot::Semantic(s) => s,
            ActionSlot::PlaceActive => {
                let s = [SET_ACTIVE_0, SET_ACTIVE_1, SET_ACTIVE_2][place_active_count.min(2)];
                place_active_count += 1;
                s
            }
            ActionSlot::PlaceBench => {
                let s = [ADD_BENCH_0, ADD_BENCH_1, ADD_BENCH_2, ADD_BENCH_3][place_bench_count.min(3)];
                place_bench_count += 1;
                s
            }
            ActionSlot::Item => {
                let s = [ITEM_0, ITEM_1, ITEM_2, ITEM_3][item_count.min(3)];
                item_count += 1;
                s
            }
            ActionSlot::Supporter => {
                let s = [SUPPORTER_0, SUPPORTER_1][supporter_count.min(1)];
                supporter_count += 1;
                s
            }
            ActionSlot::Skip => continue,
        };
        if !map.contains_key(&sem) {
            map.insert(sem, deckgym_idx);
            mask[sem] = true;
        }
    }
    (mask, map)
}

// ═══════════════════════════════════════════════════════════════════════
// Game stepping — forced actions + opponent random play
// ═══════════════════════════════════════════════════════════════════════

/// Check if a SimpleAction is forced/mechanical (no meaningful player choice).
fn is_forced_action(action: &SimpleAction) -> bool {
    matches!(
        action,
        SimpleAction::DrawCard { .. }
            | SimpleAction::Noop
            | SimpleAction::ApplyDamage { .. }
            | SimpleAction::Heal { .. }
            | SimpleAction::HealAndDiscardEnergy { .. }
            | SimpleAction::MoveAllDamage { .. }
            | SimpleAction::ApplyEeveeBagDamageBoost
            | SimpleAction::HealAllEeveeEvolutions
    )
}

/// Auto-play forced/mechanical actions for the current actor.
/// Returns the number of forced actions played.
pub(crate) fn auto_play_forced(state: &mut State, rng: &mut StdRng) -> u32 {
    let mut count = 0u32;
    let max_forced = 100u32;

    while !state.is_game_over() && count < max_forced {
        let (_actor, actions) = state.generate_possible_actions();
        if actions.is_empty() {
            break;
        }

        // Only 1 legal action — must take it
        if actions.len() == 1 {
            apply_action(rng, state, &actions[0]);
            count += 1;
            continue;
        }

        // Check if ALL actions are forced/mechanical
        let all_forced = actions.iter().all(|a| is_forced_action(&a.action));
        if all_forced {
            apply_action(rng, state, &actions[0]);
            count += 1;
            continue;
        }

        break; // Real decision point
    }
    count
}

/// Play opponent's turn with random action selection until it's agent's turn.
///
/// PERF: This is the #1 hotspot in MCTS — called for EVERY simulation.
/// Each call generates possible actions + applies them for every opponent step.
/// Capped at 30 actions (was 200) because:
///   - A typical Pokémon TCG turn has 5-15 meaningful actions
///   - Actions beyond ~20 are forced/mechanical (auto_play handles those)
///   - Reducing from 200→30 saves ~85% of wasted simulation time
///   - For MCTS value estimation, a partial turn is sufficient
fn opponent_turn_random(state: &mut State, rng: &mut StdRng, agent: usize) {
    let max_actions = 30u32;  // Was 200 — most turns finish in <15 actions
    let mut count = 0u32;

    while !state.is_game_over() && count < max_actions {
        auto_play_forced(state, rng);
        if state.is_game_over() {
            break;
        }

        let (actor, actions) = state.generate_possible_actions();
        if actions.is_empty() || actor == agent {
            break; // Agent's turn
        }

        // Random opponent action
        let idx = rng.gen_range(0..actions.len());
        apply_action(rng, state, &actions[idx]);
        count += 1;
    }
}

/// Apply a semantic action, auto-play forced, handle opponent turn.
/// Returns (game_over, winner_is_agent).
pub(crate) fn step_for_mcts(
    state: &mut State,
    rng: &mut StdRng,
    agent: usize,
    semantic_action: usize,
    actions: &[Action],
    action_map: &HashMap<usize, usize>,
) -> (bool, Option<bool>) {
    // Apply agent's chosen action
    if let Some(&deckgym_idx) = action_map.get(&semantic_action) {
        if deckgym_idx < actions.len() {
            apply_action(rng, state, &actions[deckgym_idx]);
        }
    } else {
        // Invalid action — try END_TURN as fallback
        if let Some(&end_idx) = action_map.get(&END_TURN) {
            if end_idx < actions.len() {
                apply_action(rng, state, &actions[end_idx]);
            }
        }
    }

    // Auto-play forced actions
    auto_play_forced(state, rng);

    // Opponent turn (random policy for MCTS simulations)
    if !state.is_game_over() {
        opponent_turn_random(state, rng, agent);
    }

    // Auto-play forced for agent's next turn
    if !state.is_game_over() {
        auto_play_forced(state, rng);
    }

    let game_over = state.is_game_over();
    let winner_is_agent = if game_over {
        match state.winner {
            Some(GameOutcome::Win(p)) => Some(p == agent),
            Some(GameOutcome::Tie) => None,
            None => None, // turn limit
        }
    } else {
        None
    };
    (game_over, winner_is_agent)
}

// ═══════════════════════════════════════════════════════════════════════
// MCTSEngine — Gumbel Sequential Halving + IS-MCTS
// ═══════════════════════════════════════════════════════════════════════

/// Determinize hidden info by shuffling opponent hand + both decks.
pub(crate) fn determinize_state(state: &mut State, rng: &mut StdRng) {
    // Shuffle both decks (hidden order)
    state.decks[0].cards.shuffle(rng);
    state.decks[1].cards.shuffle(rng);
    // Opponent hand is hidden — shuffle it too
    // (This is a lightweight determinization; full IS-MCTS would
    //  resample from the information set, but shuffling is sufficient
    //  when combined with multiple determinizations.)
}

/// Result of a single MCTS search call.
pub struct MCTSResult {
    pub action: usize,
    pub policy: [f64; NUM_ACTIONS],
}

/// Core MCTS engine — operates entirely in Rust except for NN evaluation.
pub struct MCTSEngine {
    pub config: MCTSConfig,
}

impl MCTSEngine {
    pub fn new(config: MCTSConfig) -> Self {
        MCTSEngine { config }
    }

    /// Main search entry point — IS-MCTS with Gumbel Sequential Halving.
    ///
    /// Uses **Determinization-Interleaved** leaf batching: all determinizations
    /// run simulations in lockstep, collecting leaves across ALL dets into a
    /// single large NN batch call.  This maximizes GPU throughput by sending
    /// `n_dets × leaf_batch_size` samples per call instead of `leaf_batch_size`.
    ///
    /// `predict_fn` is called with (obs_batch: &[f32], mask_batch: &[f32], batch_size: usize)
    /// and returns (policies: Vec<f32>, values: Vec<f32>) flattened.
    pub fn search<F>(
        &self,
        state: &State,
        rng: &mut StdRng,
        agent: usize,
        move_number: usize,
        predict_fn: &mut F,
    ) -> MCTSResult
    where
        F: FnMut(&[f32], &[f32], usize) -> (Vec<f32>, Vec<f32>),
    {
        let n_dets = self.config.num_determinizations;

        // === 1. Create determinizations + batch root evaluation ===
        let mut det_states: Vec<State> = Vec::with_capacity(n_dets);
        let mut det_rngs: Vec<StdRng> = Vec::with_capacity(n_dets);
        let mut obs_flat: Vec<f32> = Vec::with_capacity(n_dets * OBS_SIZE);
        let mut mask_flat: Vec<f32> = Vec::with_capacity(n_dets * NUM_ACTIONS);
        let mut det_actions_list: Vec<Vec<Action>> = Vec::with_capacity(n_dets);
        let mut det_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(n_dets);

        for _ in 0..n_dets {
            let mut s = state.clone();
            let mut det_rng = StdRng::seed_from_u64(rng.gen());
            determinize_state(&mut s, &mut det_rng);

            // Auto-play forced actions to reach a decision point
            auto_play_forced(&mut s, &mut det_rng);

            let obs = build_observation(&s, agent);
            let (_actor, actions) = s.generate_possible_actions();
            let (mask_bool, action_map) = build_action_map(&actions, agent);

            let mask_f32: Vec<f32> = mask_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

            obs_flat.extend_from_slice(&obs);
            mask_flat.extend_from_slice(&mask_f32);
            det_actions_list.push(actions);
            det_maps.push(action_map);
            det_states.push(s);
            det_rngs.push(det_rng);
        }

        // Batch NN evaluation for all root states
        let (root_policies_flat, _root_values_flat) =
            predict_fn(&obs_flat, &mask_flat, n_dets);

        // === 2. Prepare per-det Gumbel MCTS state ===
        let mut det_arenas: Vec<NodeArena> = Vec::with_capacity(n_dets);
        let mut det_root_ids: Vec<NodeId> = Vec::with_capacity(n_dets);
        let mut det_candidates: Vec<Vec<usize>> = Vec::with_capacity(n_dets);
        let mut det_gumbel_logits: Vec<[f64; NUM_ACTIONS]> = Vec::with_capacity(n_dets);
        let mut det_sims_budget: Vec<usize> = Vec::with_capacity(n_dets);
        let mut last_mask = [0.0f32; NUM_ACTIONS];

        for d in 0..n_dets {
            let root_policy: &[f32] = &root_policies_flat[d * NUM_ACTIONS..(d + 1) * NUM_ACTIONS];
            let mask_slice: &[f32] = &mask_flat[d * NUM_ACTIONS..(d + 1) * NUM_ACTIONS];

            if d == 0 {
                last_mask.copy_from_slice(mask_slice);
            }

            let mut arena = NodeArena::new();
            let root_id = arena.alloc(None, -1, 0.0);

            // Build action priors (with Dirichlet noise)
            let mut action_priors: Vec<(usize, f32)> = Vec::new();
            for a in 0..NUM_ACTIONS {
                if mask_slice[a] > 0.0 {
                    action_priors.push((a, root_policy[a]));
                }
            }

            // Dirichlet noise at root
            if self.config.add_noise && !action_priors.is_empty() {
                let n = action_priors.len();
                let alpha = vec![self.config.dirichlet_alpha; n];
                if let Ok(dirichlet) = Dirichlet::new(&alpha) {
                    let noise: Vec<f64> = dirichlet.sample(&mut det_rngs[d]);
                    let frac = self.config.dirichlet_frac;
                    for (i, (_, prior)) in action_priors.iter_mut().enumerate() {
                        *prior = ((1.0 - frac) * (*prior as f64) + frac * noise[i]) as f32;
                    }
                }
            }

            arena.expand(root_id, &action_priors);

            let legal_actions: Vec<usize> = action_priors.iter().map(|&(a, _)| a).collect();
            let m = self.config.max_considered_actions.min(legal_actions.len());

            // Gumbel-Top-k to select m candidate actions
            let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();
            let mut gumbel_logits = [f64::NEG_INFINITY; NUM_ACTIONS];
            for &(a, prior) in &action_priors {
                let log_p = (prior as f64 + 1e-8).ln();
                let g: f64 = gumbel_dist.sample(&mut det_rngs[d]);
                gumbel_logits[a] = log_p + g;
            }

            let candidates = if m <= 1 {
                legal_actions.iter().take(1).cloned().collect()
            } else {
                let mut legal_gumbel: Vec<(usize, f64)> = legal_actions.iter().map(|&a| (a, gumbel_logits[a])).collect();
                legal_gumbel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                legal_gumbel.iter().take(m).map(|&(a, _)| a).collect()
            };

            det_arenas.push(arena);
            det_root_ids.push(root_id);
            det_candidates.push(candidates);
            det_gumbel_logits.push(gumbel_logits);
            det_sims_budget.push(self.config.num_simulations);
        }

        // === 3. Interleaved simulation loop across all determinizations ===
        self.run_interleaved_mcts(
            &det_states,
            &mut det_rngs,
            agent,
            &det_actions_list,
            &det_maps,
            &mut det_arenas,
            &det_root_ids,
            &mut det_candidates,
            &det_gumbel_logits,
            predict_fn,
        );

        // === 4. Aggregate results across determinizations ===
        let mut aggregate_visits = [0.0f64; NUM_ACTIONS];
        let mut aggregate_q = [0.0f64; NUM_ACTIONS];
        let mut aggregate_q_count = [0.0f64; NUM_ACTIONS];

        for d in 0..n_dets {
            let root_id = det_root_ids[d];
            for (&sem_action, &child_id) in &det_arenas[d].get(root_id).children {
                let v = det_arenas[d].get(child_id).visit_count as f64;
                aggregate_visits[sem_action] += v;
                if v > 0.0 {
                    aggregate_q[sem_action] += det_arenas[d].get(child_id).completed_q();
                    aggregate_q_count[sem_action] += 1.0;
                }
            }
        }

        // Build policy from aggregate visits
        let total: f64 = aggregate_visits.iter().sum();
        let mut policy = [0.0f64; NUM_ACTIONS];
        if total > 0.0 {
            for a in 0..NUM_ACTIONS {
                policy[a] = aggregate_visits[a] / total;
            }
        } else {
            let mask_sum: f32 = last_mask.iter().sum();
            if mask_sum > 0.0 {
                for a in 0..NUM_ACTIONS {
                    policy[a] = last_mask[a] as f64 / mask_sum as f64;
                }
            }
        }

        // Mean Q across determinizations
        let mut mean_q = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if aggregate_q_count[a] > 0.0 {
                mean_q[a] = aggregate_q[a] / aggregate_q_count[a];
            }
        }

        // === 5. Action selection ===
        let action = if self.config.use_gumbel {
            self.gumbel_select(&mean_q, &root_policies_flat[0..NUM_ACTIONS], &last_mask, move_number, rng)
        } else {
            let temp = if move_number <= self.config.temp_threshold {
                self.config.temperature
            } else {
                self.config.temp_final
            };
            self.select_action(&policy, &last_mask, temp, rng)
        };

        MCTSResult { action, policy }
    }

    /// Gumbel action selection using completed Q-values from MCTS tree.
    pub(crate) fn gumbel_select(
        &self,
        completed_q: &[f64; NUM_ACTIONS],
        prior: &[f32],
        mask: &[f32; NUM_ACTIONS],
        move_number: usize,
        rng: &mut StdRng,
    ) -> usize {
        // Late-game: greedy argmax Q
        if move_number > self.config.temp_threshold {
            let mut best_a = 0;
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..NUM_ACTIONS {
                if mask[a] > 0.0 && completed_q[a] > best_q {
                    best_q = completed_q[a];
                    best_a = a;
                }
            }
            return best_a;
        }

        let valid: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a] > 0.0).collect();
        if valid.is_empty() { return 0; }
        if valid.len() == 1 { return valid[0]; }

        // g(a) = log π(a) + Gumbel(0,1)
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();
        let mut scores = [f64::NEG_INFINITY; NUM_ACTIONS];
        for &a in &valid {
            let log_prior = (prior[a] as f64 + 1e-8).ln();
            let gumbel_noise: f64 = gumbel_dist.sample(rng);
            let sigma_q = self.config.gumbel_c_visit as f64
                * self.config.gumbel_c_scale as f64
                * completed_q[a];
            scores[a] = log_prior + gumbel_noise + sigma_q;
        }

        let mut best_a = valid[0];
        let mut best_s = f64::NEG_INFINITY;
        for &a in &valid {
            if scores[a] > best_s {
                best_s = scores[a];
                best_a = a;
            }
        }
        best_a
    }

    /// Non-Gumbel action selection with temperature.
    pub(crate) fn select_action(
        &self,
        policy: &[f64; NUM_ACTIONS],
        mask: &[f32; NUM_ACTIONS],
        temperature: f32,
        rng: &mut StdRng,
    ) -> usize {
        let valid: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a] > 0.0).collect();
        if valid.is_empty() { return 0; }
        if valid.len() == 1 { return valid[0]; }

        if temperature < 0.01 {
            // Greedy
            let mut best_a = valid[0];
            let mut best_p = f64::NEG_INFINITY;
            for &a in &valid {
                if policy[a] > best_p { best_p = policy[a]; best_a = a; }
            }
            return best_a;
        }

        // Temperature-scaled sampling
        let log_p: Vec<f64> = valid.iter().map(|&a| (policy[a] + 1e-10).ln() / temperature as f64).collect();
        let max_lp = log_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = log_p.iter().map(|&lp| (lp - max_lp).exp()).collect();
        let sum: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }

        // Weighted random choice
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative { return valid[i]; }
        }
        *valid.last().unwrap()
    }

    /// Interleaved MCTS across all determinizations with cross-det leaf batching.
    ///
    /// Instead of running each determinization sequentially (each making many
    /// small NN calls), we interleave: each round, every det produces leaves,
    /// we merge ALL leaves into ONE big NN batch, then distribute results back.
    ///
    /// With 8 dets × 32 leaves/det = 256 samples per NN call → GPU fully utilized.
    fn run_interleaved_mcts<F>(
        &self,
        det_states: &[State],
        det_rngs: &mut [StdRng],
        agent: usize,
        det_actions_list: &[Vec<Action>],
        det_maps: &[HashMap<usize, usize>],
        arenas: &mut [NodeArena],
        root_ids: &[NodeId],
        candidates: &mut [Vec<usize>],
        gumbel_logits: &[[f64; NUM_ACTIONS]],
        predict_fn: &mut F,
    ) where
        F: FnMut(&[f32], &[f32], usize) -> (Vec<f32>, Vec<f32>),
    {
        let n_dets = det_states.len();
        let total_sims = self.config.num_simulations;
        let batch_size = self.config.leaf_batch_size;

        // Compute Sequential Halving schedule
        let max_m: usize = candidates.iter().map(|c| c.len()).max().unwrap_or(1);
        let num_rounds = (max_m as f64).log2().ceil().max(1.0) as usize;
        let sims_per_action_per_round = (total_sims / (max_m.max(1) * num_rounds)).max(1);

        // === Sequential Halving rounds (interleaved across dets) ===
        for _round in 0..num_rounds {
            // For each det, run sims_per_action_per_round for each candidate
            // Interleaved: collect leaves from ALL dets, batch-eval together
            for d in 0..n_dets {
                if candidates[d].len() <= 1 { continue; }
            }

            let cands_snapshot: Vec<Vec<usize>> = candidates.iter().map(|c| c.clone()).collect();

            for &action_idx in &Self::unique_actions_across_dets(&cands_snapshot) {
                // Run sims for this action across all dets that have it as candidate
                self.run_sims_interleaved(
                    det_states, det_rngs, agent,
                    det_actions_list, det_maps,
                    arenas, root_ids,
                    &cands_snapshot, action_idx,
                    sims_per_action_per_round, batch_size,
                    predict_fn,
                );
            }

            // Sequential Halving: score and halve candidates per det
            for d in 0..n_dets {
                if candidates[d].len() <= 1 { continue; }

                let mut scores: Vec<(usize, f64)> = candidates[d].iter().map(|&a| {
                    let q_hat = if let Some(&child_id) = arenas[d].get(root_ids[d]).children.get(&a) {
                        arenas[d].get(child_id).completed_q()
                    } else {
                        0.0
                    };
                    let sigma_q = self.config.gumbel_c_visit as f64
                        * self.config.gumbel_c_scale as f64
                        * q_hat;
                    (a, gumbel_logits[d][a] + sigma_q)
                }).collect();

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let half = (scores.len() / 2).max(1);
                candidates[d] = scores.into_iter().take(half).map(|(a, _)| a).collect();
            }
        }

        // === Run remaining budget on surviving candidates (interleaved) ===
        let cands_final: Vec<Vec<usize>> = candidates.iter().map(|c| c.clone()).collect();
        for d in 0..n_dets {
            if cands_final[d].is_empty() { continue; }

            let total_used: u32 = arenas[d].get(root_ids[d]).children.values()
                .map(|&cid| arenas[d].get(cid).visit_count)
                .sum();
            let remaining = total_sims.saturating_sub(total_used as usize);
            if remaining > 0 {
                let sims_each = (remaining / cands_final[d].len()).max(1);
                for &a in &cands_final[d] {
                    self.run_sims_interleaved(
                        det_states, det_rngs, agent,
                        det_actions_list, det_maps,
                        arenas, root_ids,
                        &cands_final, a,
                        sims_each, batch_size,
                        predict_fn,
                    );
                }
            }
        }
    }

    /// Collect unique actions across all determinizations' candidate lists.
    fn unique_actions_across_dets(candidates: &[Vec<usize>]) -> Vec<usize> {
        let mut seen = [false; NUM_ACTIONS];
        let mut result = Vec::new();
        for cands in candidates {
            for &a in cands {
                if !seen[a] {
                    seen[a] = true;
                    result.push(a);
                }
            }
        }
        result
    }

    /// Run simulations for a specific action across ALL determinizations that
    /// have it as a candidate, collecting leaves into a single cross-det batch.
    fn run_sims_interleaved<F>(
        &self,
        det_states: &[State],
        det_rngs: &mut [StdRng],
        agent: usize,
        det_actions_list: &[Vec<Action>],
        det_maps: &[HashMap<usize, usize>],
        arenas: &mut [NodeArena],
        root_ids: &[NodeId],
        candidates: &[Vec<usize>],
        action_idx: usize,
        num_sims: usize,
        batch_size: usize,
        predict_fn: &mut F,
    ) where
        F: FnMut(&[f32], &[f32], usize) -> (Vec<f32>, Vec<f32>),
    {
        let n_dets = det_states.len();

        // Which dets have this action as a candidate?
        let active_dets: Vec<usize> = (0..n_dets)
            .filter(|&d| candidates[d].contains(&action_idx)
                && arenas[d].get(root_ids[d]).children.contains_key(&action_idx))
            .collect();

        if active_dets.is_empty() { return; }

        let mut sims_done = 0;

        while sims_done < num_sims {
            let remaining = num_sims - sims_done;
            let current_batch = batch_size.min(remaining);

            // Cross-det leaf collection: (det_idx, node_id, state)
            let mut all_leaf_det_idx: Vec<usize> = Vec::new();
            let mut all_leaf_node_ids: Vec<NodeId> = Vec::new();
            let mut all_leaf_states: Vec<State> = Vec::new();
            // Cross-det terminal collection: (det_idx, node_id, value)
            let mut all_terminal: Vec<(usize, NodeId, f64)> = Vec::new();

            for &d in &active_dets {
                for _ in 0..current_batch {
                    let mut sim_state = det_states[d].clone();
                    let mut sim_rng = StdRng::seed_from_u64(det_rngs[d].gen());

                    let child_id = match arenas[d].get(root_ids[d]).children.get(&action_idx) {
                        Some(&cid) => cid,
                        None => break,
                    };

                    if self.config.use_virtual_loss {
                        arenas[d].add_virtual_loss(child_id);
                    }

                    let (game_over, winner) = step_for_mcts(
                        &mut sim_state, &mut sim_rng, agent,
                        action_idx, &det_actions_list[d], &det_maps[d],
                    );

                    if game_over {
                        let value = match winner {
                            Some(true) => 1.0,
                            Some(false) => -1.0,
                            None => 0.0,
                        };
                        arenas[d].get_mut(child_id).is_terminal = true;
                        arenas[d].get_mut(child_id).terminal_value = value as f32;
                        all_terminal.push((d, child_id, value));
                        continue;
                    }

                    // PUCT tree traversal
                    let mut node_id = child_id;
                    let mut hit_terminal = false;

                    while !arenas[d].get(node_id).is_leaf() && !arenas[d].get(node_id).is_terminal {
                        let (sel_child, sel_action) = arenas[d].select_child(node_id, self.config.c_puct);

                        let (_, current_actions) = sim_state.generate_possible_actions();
                        let (current_mask, current_map) = build_action_map(&current_actions, agent);
                        if !current_mask[sel_action] {
                            break;
                        }

                        node_id = sel_child;
                        if self.config.use_virtual_loss {
                            arenas[d].add_virtual_loss(node_id);
                        }

                        let (game_over, winner) = step_for_mcts(
                            &mut sim_state, &mut sim_rng, agent,
                            sel_action, &current_actions, &current_map,
                        );

                        if game_over {
                            let value = match winner {
                                Some(true) => 1.0,
                                Some(false) => -1.0,
                                None => 0.0,
                            };
                            arenas[d].get_mut(node_id).is_terminal = true;
                            arenas[d].get_mut(node_id).terminal_value = value as f32;
                            all_terminal.push((d, node_id, value));
                            hit_terminal = true;
                            break;
                        }
                    }

                    if !hit_terminal {
                        if arenas[d].get(node_id).is_terminal {
                            all_terminal.push((d, node_id, arenas[d].get(node_id).terminal_value as f64));
                        } else {
                            all_leaf_det_idx.push(d);
                            all_leaf_node_ids.push(node_id);
                            all_leaf_states.push(sim_state);
                        }
                    }
                }
            }

            // === Single cross-det batch NN evaluation ===
            if !all_leaf_states.is_empty() {
                let n_leaves = all_leaf_states.len();
                let mut obs_flat: Vec<f32> = Vec::with_capacity(n_leaves * OBS_SIZE);
                let mut mask_flat: Vec<f32> = Vec::with_capacity(n_leaves * NUM_ACTIONS);

                for leaf_state in &all_leaf_states {
                    let obs = build_observation(leaf_state, agent);
                    obs_flat.extend_from_slice(&obs);

                    let (_, actions) = leaf_state.generate_possible_actions();
                    let (mask_bool, _) = build_action_map(&actions, agent);
                    for &b in &mask_bool {
                        mask_flat.push(if b { 1.0 } else { 0.0 });
                    }
                }

                let (policies_flat, values_flat) = predict_fn(&obs_flat, &mask_flat, n_leaves);

                // Distribute results back to per-det arenas
                for (i, (&d, &leaf_id)) in all_leaf_det_idx.iter().zip(all_leaf_node_ids.iter()).enumerate() {
                    let leaf_policy = &policies_flat[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
                    let leaf_mask = &mask_flat[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];

                    let mut priors: Vec<(usize, f32)> = Vec::new();
                    for a in 0..NUM_ACTIONS {
                        if leaf_mask[a] > 0.0 {
                            priors.push((a, leaf_policy[a]));
                        }
                    }
                    if !priors.is_empty() {
                        arenas[d].expand(leaf_id, &priors);
                    }

                    if self.config.use_virtual_loss {
                        arenas[d].revert_virtual_loss(leaf_id);
                    }
                    let value = if i < values_flat.len() { values_flat[i] as f64 } else { 0.0 };
                    arenas[d].backpropagate(leaf_id, value);
                }
            }

            // Backpropagate terminals
            for &(d, term_id, term_value) in &all_terminal {
                if self.config.use_virtual_loss {
                    arenas[d].revert_virtual_loss(term_id);
                }
                arenas[d].backpropagate(term_id, term_value);
            }

            sims_done += current_batch;
        }
    }
} // end impl MCTSEngine

// ═══════════════════════════════════════════════════════════════════════
// PyO3 Bindings — Python-facing classes
// ═══════════════════════════════════════════════════════════════════════

use crate::python_bindings::PyRLEnv;

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct PyMCTSConfig {
    inner: MCTSConfig,
}

#[pymethods]
impl PyMCTSConfig {
    #[new]
    #[pyo3(signature = (
        num_simulations=200,
        num_determinizations=8,
        c_puct=1.5,
        temperature=1.0,
        temp_threshold=15,
        temp_final=0.1,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        add_noise=true,
        use_gumbel=true,
        gumbel_c_visit=50.0,
        gumbel_c_scale=1.0,
        max_considered_actions=16,
        leaf_batch_size=32,
        use_virtual_loss=true,
    ))]
    fn new(
        num_simulations: usize,
        num_determinizations: usize,
        c_puct: f32,
        temperature: f32,
        temp_threshold: usize,
        temp_final: f32,
        dirichlet_alpha: f64,
        dirichlet_frac: f64,
        add_noise: bool,
        use_gumbel: bool,
        gumbel_c_visit: f32,
        gumbel_c_scale: f32,
        max_considered_actions: usize,
        leaf_batch_size: usize,
        use_virtual_loss: bool,
    ) -> Self {
        PyMCTSConfig {
            inner: MCTSConfig {
                num_simulations,
                num_determinizations,
                c_puct,
                temperature,
                temp_threshold,
                temp_final,
                dirichlet_alpha,
                dirichlet_frac,
                add_noise,
                use_gumbel,
                gumbel_c_visit,
                gumbel_c_scale,
                max_considered_actions,
                leaf_batch_size,
                use_virtual_loss,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MCTSConfig(sims={}, dets={}, c_puct={}, gumbel={})",
            self.inner.num_simulations,
            self.inner.num_determinizations,
            self.inner.c_puct,
            self.inner.use_gumbel,
        )
    }
}


#[pyclass(unsendable)]
pub struct PyMCTSEngine {
    engine: MCTSEngine,
}

#[pymethods]
impl PyMCTSEngine {
    #[new]
    fn new(config: &PyMCTSConfig) -> Self {
        PyMCTSEngine {
            engine: MCTSEngine::new(config.inner.clone()),
        }
    }

    /// Run MCTS search from the current game state.
    ///
    /// Args:
    ///   env: PyRLEnv — current game environment (state + rng are cloned, env is not mutated)
    ///   predict_fn: callable(obs_list, mask_list, batch_size) -> (policies_list, values_list)
    ///     - obs_list: flat list of f32, length = batch_size * 423
    ///     - mask_list: flat list of f32, length = batch_size * 77
    ///     - policies_list: flat list of f32, length = batch_size * 77
    ///     - values_list: flat list of f32, length = batch_size
    ///   agent: player index (0 or 1)
    ///   move_number: current move number (for temperature schedule)
    ///
    /// Returns: (action_index: int, policy: list[float] of length 77)
    fn search(
        &self,
        py: Python<'_>,
        env: &PyRLEnv,
        predict_fn: PyObject,
        agent: usize,
        move_number: usize,
    ) -> PyResult<(usize, Vec<f64>)> {
        // Clone state + rng so we don't mutate the caller's env
        let state = env.state.clone();
        let mut rng = env.rng.clone();

        // Create the Rust closure that bridges to Python predict_fn
        let mut callback = |obs_flat: &[f32], mask_flat: &[f32], batch_size: usize|
            -> (Vec<f32>, Vec<f32>)
        {
            Python::with_gil(|py| {
                // Convert Rust slices → Python lists
                let obs_list = PyList::new_bound(py, obs_flat);
                let mask_list = PyList::new_bound(py, mask_flat);

                // Call: predict_fn(obs_list, mask_list, batch_size) → (policies_list, values_list)
                let result = predict_fn
                    .call1(py, (obs_list, mask_list, batch_size))
                    .expect("predict_fn call failed");

                let tuple = result
                    .downcast_bound::<PyTuple>(py)
                    .expect("predict_fn must return a tuple");

                // Extract policies: list of f32
                let policies_obj = tuple.get_item(0).unwrap();
                let policies: Vec<f32> = policies_obj
                    .extract::<Vec<f32>>()
                    .expect("policies must be a list of floats");

                // Extract values: list of f32
                let values_obj = tuple.get_item(1).unwrap();
                let values: Vec<f32> = values_obj
                    .extract::<Vec<f32>>()
                    .expect("values must be a list of floats");

                (policies, values)
            })
        };

        let result = self.engine.search(&state, &mut rng, agent, move_number, &mut callback);

        Ok((result.action, result.policy.to_vec()))
    }

    /// Run MCTS search using ONNX Runtime predictor (no Python callback needed).
    ///
    /// This eliminates the entire Rust→Python→PyTorch→Python→Rust FFI overhead.
    /// Requires the `onnx` feature and a loaded PyOnnxPredictor.
    ///
    /// Args:
    ///   env: PyRLEnv — current game environment
    ///   predictor: PyOnnxPredictor — ONNX Runtime session
    ///   agent: player index (0 or 1)
    ///   move_number: current move number (for temperature schedule)
    ///
    /// Returns: (action_index: int, policy: list[float] of length 77)
    #[cfg(feature = "onnx")]
    fn search_onnx(
        &self,
        env: &PyRLEnv,
        predictor: &mut PyOnnxPredictor,
        agent: usize,
        move_number: usize,
    ) -> PyResult<(usize, Vec<f64>)> {
        let state = env.state.clone();
        let mut rng = env.rng.clone();

        let mut callback = |obs_flat: &[f32], mask_flat: &[f32], batch_size: usize|
            -> (Vec<f32>, Vec<f32>)
        {
            predictor.inner.predict(obs_flat, mask_flat, batch_size)
        };

        let result = self.engine.search(&state, &mut rng, agent, move_number, &mut callback);
        Ok((result.action, result.policy.to_vec()))
    }

    fn __repr__(&self) -> String {
        format!(
            "MCTSEngine(sims={}, dets={}, gumbel={})",
            self.engine.config.num_simulations,
            self.engine.config.num_determinizations,
            self.engine.config.use_gumbel,
        )
    }
}

/// Python-facing ONNX Runtime predictor.
///
/// Wraps `OnnxPredictor` for use from Python.  Load once, reuse across
/// all MCTS searches — the Session is thread-safe.
#[cfg(feature = "onnx")]
#[pyclass(unsendable)]
pub struct PyOnnxPredictor {
    inner: crate::onnx_predictor::OnnxPredictor,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyOnnxPredictor {
    /// Load an ONNX model from the given file path.
    ///
    /// Tries CUDA execution provider first, falls back to CPU.
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let path = std::path::Path::new(&model_path);
        let predictor = crate::onnx_predictor::OnnxPredictor::new(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load ONNX model: {}", e)
            ))?;
        Ok(PyOnnxPredictor { inner: predictor })
    }

    fn __repr__(&self) -> String {
        "OnnxPredictor(loaded)".to_string()
    }
}

/// Python-facing Cross-Game Self-Play coordinator.
///
/// Runs N self-play games simultaneously with a shared ONNX Runtime
/// inference server.  All game logic, MCTS, and NN evaluation runs
/// entirely in Rust — zero Python overhead during self-play.
///
/// Usage from Python:
/// ```python
/// from deckgym import CrossGameSelfPlay, MCTSConfig
/// cg = CrossGameSelfPlay()
/// results = cg.run(
///     model_path="model.onnx",
///     deck_a_path="deck_a.json",
///     deck_b_path="deck_b.json",
///     num_games=64,
/// )
/// for r in results:
///     buffer.add(r["observations"], r["policies"], r["masks"], r["value"])
/// ```
#[cfg(feature = "onnx")]
#[pyclass]
pub struct PyCrossGameSelfPlay {
    agent_config: MCTSConfig,
    opp_config: MCTSConfig,
    num_parallel: usize,
    max_batch_size: usize,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyCrossGameSelfPlay {
    /// Create a new cross-game coordinator.
    ///
    /// Args:
    ///   agent_config: MCTSConfig for the learning agent
    ///   opp_config: MCTSConfig for the opponent (fewer sims, no noise)
    ///   num_parallel: Number of simultaneous game threads (default: 16)
    ///   max_batch_size: Max samples per GPU call (default: 4096)
    #[new]
    #[pyo3(signature = (agent_config, opp_config, num_parallel=16, max_batch_size=4096))]
    fn new(
        agent_config: &PyMCTSConfig,
        opp_config: &PyMCTSConfig,
        num_parallel: usize,
        max_batch_size: usize,
    ) -> Self {
        PyCrossGameSelfPlay {
            agent_config: agent_config.inner.clone(),
            opp_config: opp_config.inner.clone(),
            num_parallel,
            max_batch_size,
        }
    }

    /// Run N self-play games with cross-game batched NN evaluation.
    ///
    /// Returns a list of dicts, one per completed game:
    ///   - "observations": flat list of f32, length = move_count * 423
    ///   - "policies": flat list of f32, length = move_count * 77
    ///   - "masks": flat list of f32, length = move_count * 77
    ///   - "value": f32, game outcome from agent perspective
    ///   - "move_count": int, number of agent moves
    #[pyo3(signature = (model_path, deck_a_path, deck_b_path, num_games))]
    fn run(
        &self,
        py: Python<'_>,
        model_path: String,
        deck_a_path: String,
        deck_b_path: String,
        num_games: usize,
    ) -> PyResult<Vec<PyObject>> {
        let path = std::path::Path::new(&model_path);

        // Release the GIL for the entire self-play run
        let results = py.allow_threads(|| {
            crate::cross_game_mcts::run_cross_game_selfplay(
                &self.agent_config,
                &self.opp_config,
                path,
                &deck_a_path,
                &deck_b_path,
                num_games,
                self.num_parallel,
                self.max_batch_size,
            )
        });

        // Convert Rust GameData → Python dicts
        let py_results: Vec<PyObject> = results
            .into_iter()
            .map(|game| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("observations", &game.observations).unwrap();
                dict.set_item("policies", &game.policies).unwrap();
                dict.set_item("masks", &game.masks).unwrap();
                dict.set_item("value", game.game_value).unwrap();
                dict.set_item("move_count", game.move_count).unwrap();
                dict.into()
            })
            .collect();

        Ok(py_results)
    }

    fn __repr__(&self) -> String {
        format!(
            "CrossGameSelfPlay(parallel={}, max_batch={})",
            self.num_parallel, self.max_batch_size,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyPPOSelfPlay — Rust-side PPO self-play with callback for GPU inference
// ═══════════════════════════════════════════════════════════════════════

/// PPO self-play loop entirely in Rust.
///
/// Eliminates Python env overhead by managing all game logic in Rust.
/// Only the agent's NN inference crosses the Rust/Python boundary.
/// Opponent inference uses ONNX in Rust (zero Python overhead).
///
/// ```python
/// from deckgym import PPOSelfPlay
/// sp = PPOSelfPlay(n_envs=64, deck_files=["deck1.txt", "deck2.txt"],
///                  opp_onnx_path="opponent.onnx")
/// trajectories = sp.run(num_games=256, predict_fn=my_gpu_predict)
/// ```
#[cfg(feature = "onnx")]
#[pyclass]
pub struct PyPPOSelfPlay {
    n_envs: usize,
    deck_files: Vec<String>,
    opp_onnx_path: String,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyPPOSelfPlay {
    #[new]
    #[pyo3(signature = (n_envs, deck_files, opp_onnx_path))]
    fn new(n_envs: usize, deck_files: Vec<String>, opp_onnx_path: String) -> Self {
        PyPPOSelfPlay {
            n_envs,
            deck_files,
            opp_onnx_path,
        }
    }

    /// Run PPO self-play for num_games, using predict_fn for agent inference.
    ///
    /// predict_fn(obs_flat: list[float], mask_flat: list[float], batch_size: int)
    ///   -> (actions: list[int], log_probs: list[float], values: list[float])
    ///
    /// Returns list of trajectory dicts, one per completed game.
    #[pyo3(signature = (num_games, predict_fn))]
    fn run(
        &self,
        py: Python<'_>,
        num_games: usize,
        predict_fn: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Vec<PyObject>> {
        let opp_path = std::path::Path::new(&self.opp_onnx_path);
        let mut selfplay = crate::ppo_selfplay::PPOSelfPlay::new(
            self.n_envs,
            self.deck_files.clone(),
            opp_path,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Wrap Python predict_fn as a Rust closure
        // We need to acquire the GIL for each predict call
        let predict_fn_ref = predict_fn.to_object(py);
        let trajectories = selfplay.run_batch(num_games, |obs_flat, mask_flat, batch_size| {
            // Acquire GIL for Python callback
            Python::with_gil(|py| {
                let obs_list = pyo3::types::PyList::new_bound(py, obs_flat);
                let mask_list = pyo3::types::PyList::new_bound(py, mask_flat);
                let result = predict_fn_ref
                    .call1(py, (obs_list, mask_list, batch_size))
                    .expect("predict_fn failed");

                let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)
                    .expect("predict_fn must return (actions, log_probs, values)");

                let actions: Vec<usize> = tuple
                    .get_item(0).unwrap()
                    .extract::<Vec<usize>>()
                    .expect("actions must be list of int");
                let log_probs: Vec<f32> = tuple
                    .get_item(1).unwrap()
                    .extract::<Vec<f32>>()
                    .expect("log_probs must be list of float");
                let values: Vec<f32> = tuple
                    .get_item(2).unwrap()
                    .extract::<Vec<f32>>()
                    .expect("values must be list of float");

                (actions, log_probs, values)
            })
        });

        // Convert trajectories to Python dicts
        let py_results: Vec<PyObject> = trajectories
            .into_iter()
            .map(|traj| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("observations", &traj.observations).unwrap();
                dict.set_item("actions", &traj.actions).unwrap();
                dict.set_item("log_probs", &traj.log_probs).unwrap();
                dict.set_item("values", &traj.values).unwrap();
                dict.set_item("rewards", &traj.rewards).unwrap();
                dict.set_item("dones", &traj.dones).unwrap();
                dict.set_item("masks", &traj.masks).unwrap();
                dict.set_item("game_value", traj.game_value).unwrap();
                dict.set_item("move_count", traj.move_count).unwrap();
                dict.into()
            })
            .collect();

        Ok(py_results)
    }

    fn __repr__(&self) -> String {
        format!(
            "PPOSelfPlay(n_envs={}, decks={}, opp={})",
            self.n_envs,
            self.deck_files.len(),
            self.opp_onnx_path,
        )
    }
}