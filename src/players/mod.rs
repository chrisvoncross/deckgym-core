mod attach_attack_player;
mod end_turn_player;
mod evolution_rusher_player;
mod expectiminimax_player;
mod human_player;
mod mcts_player;
#[cfg(all(feature = "python", feature = "onnx"))]
pub mod cfr_player;
#[cfg(all(feature = "python", feature = "onnx"))]
pub mod onnx_player;
mod random_player;
mod value_function_player;
pub mod value_functions;
mod weighted_random_player;

pub use attach_attack_player::AttachAttackPlayer;
pub use end_turn_player::EndTurnPlayer;
pub use evolution_rusher_player::EvolutionRusherPlayer;
pub use expectiminimax_player::{ExpectiMiniMaxPlayer, ValueFunction};
pub use human_player::HumanPlayer;
pub use mcts_player::MctsPlayer;
#[cfg(all(feature = "python", feature = "onnx"))]
pub use cfr_player::{CfrConfig, CfrPlayer};
#[cfg(all(feature = "python", feature = "onnx"))]
pub use onnx_player::OnnxPlayer;
pub use random_player::RandomPlayer;
pub use value_function_player::ValueFunctionPlayer;
pub use value_functions::*;
pub use weighted_random_player::WeightedRandomPlayer;

use crate::{actions::Action, Deck, State};
use rand::rngs::StdRng;
use std::fmt::Debug;

pub trait Player: Debug {
    fn get_deck(&self) -> Deck;
    fn decision_fn(
        &mut self,
        rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action;
}

/// Enum for allowed player strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerCode {
    AA,
    ET,
    R,
    H,
    W,
    M,
    V,
    E { max_depth: usize },
    ER, // Evolution Rusher
    /// Neural network player via ONNX model.
    /// Only available with features `python` + `onnx`.
    Onnx { model_path: String },
    /// Real-time CFR search player (Pluribus-level).
    /// Only available with features `python` + `onnx`.
    Cfr {
        model_path: String,
        num_determinizations: usize,
        cfr_iterations: usize,
        depth_limit: usize,
    },
}
/// Custom parser function enforcing case-insensitivity.
///
/// Supports:
///   - `onnx:/path/to/model.onnx` for neural network players
///   - `cfr:model.onnx:16:8:3` for real-time CFR search (dets:iters:depth)
///   - `cfr:model.onnx` for CFR with default params (16 dets, 8 iters, depth 3)
pub fn parse_player_code(s: &str) -> Result<PlayerCode, String> {
    // Check for "cfr:" prefix (case-insensitive) — must check before "onnx:"
    if s.len() > 4 && s[..4].eq_ignore_ascii_case("cfr:") {
        let rest = &s[4..];
        let parts: Vec<&str> = rest.split(':').collect();
        if parts.is_empty() || parts[0].is_empty() {
            return Err(
                "cfr: requires a model path, e.g., 'cfr:model.onnx' or 'cfr:model.onnx:16:8:3'"
                    .into(),
            );
        }
        let model_path = parts[0].to_string();
        let num_determinizations = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
        let cfr_iterations = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let depth_limit = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        return Ok(PlayerCode::Cfr {
            model_path,
            num_determinizations,
            cfr_iterations,
            depth_limit,
        });
    }

    // Check for "onnx:" prefix (case-insensitive)
    if s.len() > 5 && s[..5].eq_ignore_ascii_case("onnx:") {
        let model_path = s[5..].to_string();
        if model_path.is_empty() {
            return Err("onnx: requires a model path, e.g., 'onnx:models/gen50.onnx'".into());
        }
        return Ok(PlayerCode::Onnx { model_path });
    }

    let lower = s.to_ascii_lowercase();

    // Check if it starts with 'e' followed by digits (e.g., e2, e4)
    if lower.starts_with('e') && lower.len() > 1 {
        let rest = &lower[1..];
        if let Ok(max_depth) = rest.parse::<usize>() {
            return Ok(PlayerCode::E { max_depth });
        }
        // If it starts with 'e' but not followed by valid number, check if it's 'er'
        if lower == "er" {
            return Ok(PlayerCode::ER);
        }
        return Err(format!("Invalid player code: {s}. Use 'e<number>' for ExpectiMiniMax with depth, e.g., 'e2', 'e5'"));
    }

    match lower.as_str() {
        "aa" => Ok(PlayerCode::AA),
        "et" => Ok(PlayerCode::ET),
        "r" => Ok(PlayerCode::R),
        "h" => Ok(PlayerCode::H),
        "w" => Ok(PlayerCode::W),
        "m" => Ok(PlayerCode::M),
        "v" => Ok(PlayerCode::V),
        "e" => Ok(PlayerCode::E { max_depth: 3 }), // Default depth
        "er" => Ok(PlayerCode::ER),
        _ => Err(format!("Invalid player code: {s}")),
    }
}

pub fn parse_player_code_generic(s: String) -> Result<PlayerCode, String> {
    parse_player_code(s.as_ref())
}

pub fn fill_code_array(maybe_players: Option<Vec<PlayerCode>>) -> Vec<PlayerCode> {
    match maybe_players {
        Some(mut player_codes) => {
            if player_codes.is_empty() || player_codes.len() > 2 {
                panic!("Invalid number of players");
            } else if player_codes.len() == 1 {
                player_codes.push(PlayerCode::R);
            }
            player_codes
        }
        None => vec![PlayerCode::R, PlayerCode::R],
    }
}

pub fn create_players(
    deck_a: Deck,
    deck_b: Deck,
    players: Vec<PlayerCode>,
) -> Vec<Box<dyn Player>> {
    let player_a: Box<dyn Player> = get_player(deck_a.clone(), &players[0]);
    let player_b: Box<dyn Player> = get_player(deck_b.clone(), &players[1]);
    vec![player_a, player_b]
}

pub fn get_player(deck: Deck, player: &PlayerCode) -> Box<dyn Player> {
    match player {
        PlayerCode::AA => Box::new(AttachAttackPlayer { deck }),
        PlayerCode::ET => Box::new(EndTurnPlayer { deck }),
        PlayerCode::R => Box::new(RandomPlayer { deck }),
        PlayerCode::H => Box::new(HumanPlayer { deck }),
        PlayerCode::W => Box::new(WeightedRandomPlayer { deck }),
        PlayerCode::M => Box::new(MctsPlayer::new(deck, 100)),
        PlayerCode::V => Box::new(ValueFunctionPlayer { deck }),
        PlayerCode::E { max_depth } => Box::new(ExpectiMiniMaxPlayer {
            deck,
            max_depth: *max_depth,
            write_debug_trees: false,
            value_function: Box::new(value_functions::baseline_value_function),
        }),
        PlayerCode::ER => Box::new(EvolutionRusherPlayer { deck }),
        #[cfg(all(feature = "python", feature = "onnx"))]
        PlayerCode::Onnx { model_path } => {
            let path = std::path::Path::new(model_path);
            Box::new(
                OnnxPlayer::new(deck, path)
                    .unwrap_or_else(|e| panic!("Failed to load ONNX model '{}': {}", model_path, e)),
            )
        }
        #[cfg(not(all(feature = "python", feature = "onnx")))]
        PlayerCode::Onnx { .. } => {
            panic!(
                "OnnxPlayer requires features 'python' + 'onnx'. \
                 Rebuild with: maturin develop --features 'python,onnx' --release"
            )
        }
        #[cfg(all(feature = "python", feature = "onnx"))]
        PlayerCode::Cfr {
            model_path,
            num_determinizations,
            cfr_iterations,
            depth_limit,
        } => {
            let path = std::path::Path::new(model_path);
            let config = CfrConfig {
                num_determinizations: *num_determinizations,
                cfr_iterations: *cfr_iterations,
                depth_limit: *depth_limit,
                ..CfrConfig::default()
            };
            Box::new(
                CfrPlayer::new(deck, path, config)
                    .unwrap_or_else(|e| panic!("Failed to load CFR model '{}': {}", model_path, e)),
            )
        }
        #[cfg(not(all(feature = "python", feature = "onnx")))]
        PlayerCode::Cfr { .. } => {
            panic!(
                "CfrPlayer requires features 'python' + 'onnx'. \
                 Rebuild with: maturin develop --features 'python,onnx' --release"
            )
        }
    }
}
