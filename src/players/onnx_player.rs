//! ONNX Neural Network Player — uses a trained PokemonTransformer model to play.
//!
//! Wraps `OnnxPredictor` + `build_observation` + `build_action_map` behind the
//! `Player` trait so the model can compete against any deckgym bot via `simulate()`.
//!
//! The ONNX session is shared via `Arc<Mutex<>>` so multiple games can reuse
//! the same loaded model (avoids expensive per-game model loading).

use std::path::Path;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;

use crate::actions::Action;
use crate::alphazero_mcts::{build_action_map, build_observation, NUM_ACTIONS};
use crate::onnx_predictor::OnnxPredictor;
use crate::{Deck, State};

use super::Player;

/// Neural network player backed by an ONNX model.
///
/// Uses argmax policy selection (greedy / deterministic).
/// For evaluation — not training.
pub struct OnnxPlayer {
    pub deck: Deck,
    predictor: Arc<Mutex<OnnxPredictor>>,
}

impl OnnxPlayer {
    /// Create a new OnnxPlayer, loading the model from disk.
    ///
    /// Expensive — prefer `with_shared_predictor` when running multiple games.
    pub fn new(deck: Deck, model_path: &Path) -> Result<Self, ort::Error> {
        let predictor = OnnxPredictor::new(model_path)?;
        Ok(Self {
            deck,
            predictor: Arc::new(Mutex::new(predictor)),
        })
    }

    /// Create an OnnxPlayer sharing an already-loaded predictor.
    ///
    /// Cheap — use this in simulation loops.
    pub fn with_shared_predictor(deck: Deck, predictor: Arc<Mutex<OnnxPredictor>>) -> Self {
        Self { deck, predictor }
    }
}

impl Player for OnnxPlayer {
    fn decision_fn(
        &mut self,
        _rng: &mut StdRng,
        state: &State,
        possible_actions: &[Action],
    ) -> Action {
        if possible_actions.len() == 1 {
            return possible_actions[0].clone();
        }

        let agent = possible_actions[0].actor;

        // Encode state → observation vector
        let obs = build_observation(state, agent);

        // Map legal actions → semantic mask + index map
        let (mask_bool, action_map) = build_action_map(possible_actions, agent);
        let mask_f32: Vec<f32> = mask_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // Run neural network inference (batch size 1)
        let policies = {
            let mut pred = self.predictor.lock().expect("OnnxPredictor lock poisoned");
            let (policies, _values) = pred.predict(&obs, &mask_f32, 1);
            policies
        };

        // Argmax over masked policy
        let mut best_sem = 0usize;
        let mut best_prob = f32::NEG_INFINITY;
        for sem in 0..NUM_ACTIONS {
            if mask_bool[sem] && policies[sem] > best_prob {
                best_prob = policies[sem];
                best_sem = sem;
            }
        }

        // Map semantic action → deckgym action
        if let Some(&deckgym_idx) = action_map.get(&best_sem) {
            possible_actions[deckgym_idx].clone()
        } else {
            // Fallback: first legal action (should never happen)
            log::warn!(
                "OnnxPlayer: semantic action {} not found in action_map, using fallback",
                best_sem
            );
            possible_actions[0].clone()
        }
    }

    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }
}

impl std::fmt::Debug for OnnxPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OnnxPlayer")
    }
}

