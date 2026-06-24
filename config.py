"""config.py
=============

Centralised configuration for the AlphaZero-style chess engine.

Every tunable knob in the project lives here so that experiments are
reproducible and easy to run: change a value in this file and the whole
pipeline (self-play -> training -> evaluation) picks it up.

The configuration is split into three logical groups:

* ``NetworkConfig``  -- neural-network architecture.
* ``MCTSConfig``     -- Monte-Carlo-Tree-Search behaviour.
* ``TrainingConfig`` -- the self-play / optimisation loop.

They are bundled together in a single :class:`Config` dataclass.  Using
``dataclasses`` (instead of a bag of module-level globals) keeps the
configuration typed, documented and trivially serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os


# --------------------------------------------------------------------------- #
# Constants describing the encoding of the chess problem.
#
# These are NOT hyper-parameters -- they are fixed by the way we encode the
# board and the moves (see ``chess_game.py``).  They live here so that the
# network and the search can import a single source of truth.
# --------------------------------------------------------------------------- #

# Number of feature planes in the board tensor (see ``chess_game.encode_board``).
#   12  piece planes      (6 piece types x 2 colours)
#    4  castling rights    (us K-side / Q-side, them K-side / Q-side)
#    1  en-passant target square
#    1  half-move clock     (progress towards the 50-move rule)
NUM_INPUT_PLANES = 18

# The board is 8x8.
BOARD_SIZE = 8

# Size of the policy head.  We use the canonical AlphaZero move encoding:
#   64 from-squares  x  73 move-types  = 4672 possible moves.
# (56 "queen" moves + 8 knight moves + 9 under-promotions -- see chess_game.py)
NUM_ACTIONS = 64 * 73  # == 4672


# --------------------------------------------------------------------------- #
# Network architecture
# --------------------------------------------------------------------------- #
@dataclass
class NetworkConfig:
    """Hyper-parameters of the residual convolutional network (``model.py``)."""

    num_input_planes: int = NUM_INPUT_PLANES
    num_actions: int = NUM_ACTIONS

    #: Number of channels in the convolutional "trunk".
    num_channels: int = 64
    #: Number of residual blocks stacked in the trunk.  Deeper == stronger but
    #: slower.  4-6 is a good range for a CPU-friendly educational setup.
    num_residual_blocks: int = 4

    #: Width of the hidden layer in the value head.
    value_head_hidden: int = 64


# --------------------------------------------------------------------------- #
# Monte-Carlo Tree Search
# --------------------------------------------------------------------------- #
@dataclass
class MCTSConfig:
    """Hyper-parameters controlling the search (``mcts.py``)."""

    #: Number of simulations (network evaluations) per move.  More == stronger.
    num_simulations: int = 100

    #: Exploration constant in the PUCT formula.  Higher == more exploration of
    #: moves with high prior probability but few visits.
    c_puct: float = 1.5

    #: Dirichlet noise is mixed into the root priors during self-play to keep
    #: exploration alive.  ``alpha`` controls the shape (smaller -> more peaky),
    #: ``epsilon`` controls how much noise is mixed in.
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    #: Play-time assist: blend a simple material count into each leaf evaluation,
    #: ``value = (1 - w) * network_value + w * material``.  ``0.0`` is pure
    #: AlphaZero (used for *training*, so self-play stays "zero-knowledge").  A
    #: value like ``0.85`` gives a much stronger opponent *before* the network is
    #: fully trained -- it makes the search actually win material, capture hanging
    #: pieces and find basic mates.  Used for play / analysis, not self-play.
    material_weight: float = 0.0


# --------------------------------------------------------------------------- #
# Training / self-play loop
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """Hyper-parameters of the self-play + optimisation loop (``training.py``)."""

    # ---- outer loop -------------------------------------------------------- #
    #: Number of (self-play -> train) iterations to run.
    num_iterations: int = 20
    #: Number of self-play games generated per iteration.
    games_per_iteration: int = 10
    #: Gradient-update epochs over the replay buffer per iteration.
    epochs_per_iteration: int = 4

    # ---- self-play --------------------------------------------------------- #
    #: For the first ``temperature_moves`` plies of a game, moves are sampled in
    #: proportion to MCTS visit counts (temperature = 1) to encourage diverse
    #: openings.  Afterwards we play greedily (temperature -> 0).
    temperature_moves: int = 15
    #: Hard cap on game length so self-play cannot run forever in shuffly,
    #: drawish positions produced by a weak early network.
    max_moves: int = 200
    #: Number of worker processes used to generate self-play games in parallel.
    #: Self-play (not training) dominates runtime, and games are independent, so
    #: this is the single biggest speed-up available.  ``1`` runs sequentially in
    #: the main process (simplest, and what the tests use).  Workers always run
    #: on CPU regardless of the training device.
    num_self_play_workers: int = 1

    # ---- PGN logging ------------------------------------------------------- #
    #: If ``True``, every self-play game is appended to a PGN file per iteration
    #: (``<pgn_dir>/selfplay_iterNNN.pgn``) so games can be replayed/inspected in
    #: any chess GUI -- great for showcasing what the engine has learned.
    save_self_play_pgn: bool = False
    #: Directory for the PGN files written when ``save_self_play_pgn`` is set.
    pgn_dir: str = "pgn"

    # ---- periodic evaluation ----------------------------------------------- #
    #: Evaluate the current network against a baseline every ``eval_every``
    #: iterations and log an approximate Elo gain.  ``0`` disables evaluation
    #: (the default, so plain training runs stay fast).
    eval_every: int = 0
    #: Number of games played per evaluation match (split evenly across colours).
    eval_games: int = 10
    #: MCTS simulations to use *during evaluation* (kept low so eval is cheap).
    eval_simulations: int = 50

    # ---- optimisation ------------------------------------------------------ #
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularisation (AlphaZero uses this too)
    #: Relative weight of the value loss vs. the policy loss in the total loss.
    value_loss_weight: float = 1.0
    #: Gradient clipping threshold (None disables clipping).
    grad_clip: float | None = 1.0

    # ---- replay buffer ----------------------------------------------------- #
    #: Maximum number of (state, policy, value) examples kept in memory.  Old
    #: examples are discarded FIFO so the network trains on recent self-play.
    replay_buffer_size: int = 20_000

    # ---- bookkeeping ------------------------------------------------------- #
    #: ``"cpu"`` or ``"cuda"``.  ``"auto"`` picks CUDA when available.
    device: str = "auto"
    #: Directory where model checkpoints are written.
    checkpoint_dir: str = "checkpoints"
    #: Directory where text logs are written.
    log_dir: str = "logs"
    #: Save a checkpoint every ``checkpoint_every`` iterations (always saves the
    #: final one as ``best.pt``).
    checkpoint_every: int = 1
    #: Seed for reproducibility (set to ``None`` for nondeterministic runs).
    seed: int | None = 42


# --------------------------------------------------------------------------- #
# Top-level configuration object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Bundles the three sub-configurations into one object.

    Usage::

        from config import Config
        cfg = Config()                 # defaults
        cfg.mcts.num_simulations = 400 # tweak as needed
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolved_device(self) -> str:
        """Return the concrete torch device string.

        Resolves the ``"auto"`` setting to ``"cuda"`` when a GPU is available,
        otherwise ``"cpu"``.  Imported lazily so that simply importing the
        config does not force a (potentially slow) torch import.
        """
        if self.training.device != "auto":
            return self.training.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch always present in practice
            return "cpu"

    def ensure_directories(self) -> None:
        """Create the checkpoint/log/PGN directories if they do not yet exist."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        if self.training.save_self_play_pgn:
            os.makedirs(self.training.pgn_dir, exist_ok=True)


# A ready-to-use default instance for quick experimentation / imports.
DEFAULT_CONFIG = Config()
