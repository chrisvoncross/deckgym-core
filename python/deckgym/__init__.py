from .deckgym import (
    PyEnergyType as EnergyType,
    PyAttack as Attack,
    PyAbility as Ability,
    PyCard as Card,
    PyPlayedCard as PlayedCard,
    PyDeck as Deck,
    PyGame as Game,
    PyState as State,
    PyGameOutcome as GameOutcome,
    PySimulationResults as SimulationResults,
    PyRLEnv as RLEnv,
    PyMCTSConfig as MCTSConfig,
    PyMCTSEngine as MCTSEngine,
    py_simulate as simulate,
    get_player_types,
)

__all__ = [
    "EnergyType",
    "Attack",
    "Ability",
    "Card",
    "PlayedCard",
    "Deck",
    "Game",
    "State",
    "GameOutcome",
    "SimulationResults",
    "RLEnv",
    "MCTSConfig",
    "MCTSEngine",
    "simulate",
    "get_player_types",
]

# ONNX-dependent classes (only available when built with --features onnx)
try:
    from .deckgym import PyOnnxPredictor as OnnxPredictor
    __all__.append("OnnxPredictor")
except ImportError:
    pass

try:
    from .deckgym import PyCrossGameSelfPlay as CrossGameSelfPlay
    __all__.append("CrossGameSelfPlay")
except ImportError:
    pass
