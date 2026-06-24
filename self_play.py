"""self_play.py
===============

Generate training data by having the engine play games against *itself*.

This is the "reinforcement" in this reinforcement-learning system: the engine is
its own opponent and its own teacher.  No human games and no hand-crafted
heuristics are used -- the only ground truth is who eventually wins.

For every move of every self-play game we record a training example:

* ``state``         -- the canonical board tensor the network saw;
* ``policy_target`` -- the MCTS visit-count distribution over moves (a *better*
  policy than the raw network output, because search refined it);
* ``value_target``  -- filled in once the game ends: ``+1`` if the player who was
  to move at that position went on to win, ``-1`` if they lost, ``0`` for a draw.

Training the network to imitate the (state -> policy_target) mapping and to
predict (state -> value_target) is exactly what makes the next generation of the
network stronger, which in turn produces better self-play data: the virtuous
cycle at the core of AlphaZero.

Because self-play games are completely independent of one another, this module
also offers a **multiprocessing** path (:func:`generate_self_play_data_parallel`)
that fans the games out across CPU cores -- by far the biggest practical speed-up
for an AlphaZero loop, where self-play dominates the wall-clock time.
"""

from __future__ import annotations

import datetime as _dt
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import chess
import chess.pgn

from chess_game import ChessGame
from config import Config, NUM_ACTIONS
from mcts import MCTS, action_probabilities


@dataclass
class TrainingExample:
    """One (state, policy, value) tuple of supervision for the network.

    Attributes:
        state: canonical board tensor, shape ``(18, 8, 8)``.
        policy: full-length (``NUM_ACTIONS``) MCTS policy target that sums to 1.
        value: game outcome from the perspective of the player to move at this
            position, in ``{-1, 0, +1}`` (assigned after the game finishes).
    """

    state: np.ndarray
    policy: np.ndarray
    value: float


@dataclass
class GameResult:
    """Everything produced by a single self-play game.

    Attributes:
        examples: the per-move :class:`TrainingExample` list (the training data).
        result: the game result string in PGN convention (``"1-0"``, ``"0-1"``
            or ``"1/2-1/2"``).
        pgn: the full game in PGN text, replayable in any chess GUI.
    """

    examples: List[TrainingExample] = field(default_factory=list)
    result: str = "*"
    pgn: str = ""


def _policy_vector(game: ChessGame, moves, probs) -> np.ndarray:
    """Scatter a list of (move, probability) pairs into a full policy vector.

    The MCTS distribution only covers the legal moves; here we place each
    probability at its canonical action index so the target has the same layout
    as the network's policy head.

    Args:
        game: the position the moves belong to (provides the canonical encoding).
        moves: list of ``chess.Move``.
        probs: matching array of probabilities.

    Returns:
        A ``(NUM_ACTIONS,)`` float32 vector that sums to 1.
    """
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for move, prob in zip(moves, probs):
        policy[game.encode_move(move)] = prob
    return policy


def _board_to_pgn(board: chess.Board, round_no: Optional[int] = None) -> str:
    """Render a finished game's move stack as a PGN string.

    ``chess.pgn.Game.from_board`` walks the board's move history, so the board
    passed in must still hold its full stack (the self-play board does).
    """
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "RL-Chess-Engine self-play"
    game.headers["White"] = "AlphaZero-net"
    game.headers["Black"] = "AlphaZero-net"
    game.headers["Date"] = _dt.date.today().strftime("%Y.%m.%d")
    if round_no is not None:
        game.headers["Round"] = str(round_no)
    game.headers["Result"] = board.result(claim_draw=True)
    return str(game)


def play_game(
    network,
    config: Config,
    verbose: bool = False,
    collect_pgn: bool = False,
    round_no: Optional[int] = None,
) -> GameResult:
    """Play a single self-play game and return its :class:`GameResult`.

    Args:
        network: the current :class:`model.ChessNet`.
        config: the global :class:`config.Config`.
        verbose: if ``True``, print each move as it is played.
        collect_pgn: if ``True``, also render the game to PGN text in the result.
        round_no: optional round number written into the PGN headers.

    Returns:
        A :class:`GameResult` whose ``examples`` already have their ``value``
        back-filled from the final game outcome.
    """
    game = ChessGame()
    mcts = MCTS(network, config.mcts)

    # We stash the side-to-move alongside each example so we can sign the value
    # correctly once we know who won.
    examples: List[TrainingExample] = []
    players: List[bool] = []

    move_number = 0
    while not game.is_terminal() and move_number < config.training.max_moves:
        # Run MCTS from the current position (with root exploration noise).
        root = mcts.run(game, add_exploration_noise=True)

        # Early in the game we sample moves in proportion to visit counts to get
        # opening variety; later we play (near-)greedily for stronger play.
        temperature = 1.0 if move_number < config.training.temperature_moves else 0.0
        moves, probs = action_probabilities(root, temperature=temperature)

        # Record the training example.  The value is a placeholder for now.
        examples.append(
            TrainingExample(
                state=game.encode_state(),
                policy=_policy_vector(game, moves, probs),
                value=0.0,
            )
        )
        players.append(game.turn)

        # Sample the actual move to play from the (temperature-adjusted) policy.
        chosen = moves[np.random.choice(len(moves), p=probs)]
        if verbose:
            print(f"  move {move_number + 1}: {game.board.san(chosen)}")
        game.push(chosen)
        move_number += 1

    # ---- assign value targets from the final result -----------------------
    white_result = game.result_white()  # +1 / 0 / -1 from White's perspective
    for example, player in zip(examples, players):
        # Convert White's result into the perspective of the player to move.
        example.value = white_result if player == chess.WHITE else -white_result

    result_str = game.board.result(claim_draw=True)
    if verbose:
        print(f"  game over: {result_str} ({len(examples)} positions)")

    return GameResult(
        examples=examples,
        result=result_str,
        pgn=_board_to_pgn(game.board, round_no) if collect_pgn else "",
    )


def generate_self_play_data(
    network,
    config: Config,
    num_games: int,
    verbose: bool = False,
    collect_pgn: bool = False,
) -> List[GameResult]:
    """Play ``num_games`` self-play games sequentially in the current process.

    Args:
        network: the current network used by both sides.
        config: the global configuration.
        num_games: how many games to play.
        verbose: forwarded to :func:`play_game` (and prints a per-game header).
        collect_pgn: collect PGN text for each game.

    Returns:
        A list of :class:`GameResult`, one per game.
    """
    results: List[GameResult] = []
    for g in range(num_games):
        if verbose:
            print(f"[self-play] game {g + 1}/{num_games}")
        results.append(
            play_game(network, config, verbose=verbose,
                      collect_pgn=collect_pgn, round_no=g + 1)
        )
    return results


# --------------------------------------------------------------------------- #
# Parallel self-play
# --------------------------------------------------------------------------- #
# A self-play game only needs the network *weights*, and the resulting examples
# (numpy arrays) and PGN (text) are trivially picklable -- so games parallelise
# cleanly across processes with no shared state.
#
# Each worker rebuilds the network on the CPU from a passed-in ``state_dict`` and
# plays its share of the games.  We deliberately pin each worker to a single
# torch thread: spawning ``num_workers`` processes that each grab every core
# would oversubscribe the CPU and run *slower*.
# --------------------------------------------------------------------------- #
def _worker_play_games(args) -> List[GameResult]:
    """Top-level (picklable) worker entry point: play ``num_games`` games.

    Args:
        args: a tuple ``(state_dict, config, num_games, seed, collect_pgn,
            base_round)`` -- everything a worker needs, bundled for ``Pool.map``.

    Returns:
        The list of :class:`GameResult` produced by this worker.
    """
    import torch  # imported inside the worker (fresh interpreter under "spawn")
    from model import ChessNet

    state_dict, config, num_games, seed, collect_pgn, base_round = args

    torch.set_num_threads(1)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    network = ChessNet(config.network)
    network.load_state_dict(state_dict)
    network.eval()

    return [
        play_game(network, config, collect_pgn=collect_pgn, round_no=base_round + i)
        for i in range(num_games)
    ]


def _split_games(num_games: int, num_workers: int) -> List[int]:
    """Split ``num_games`` as evenly as possible into ``num_workers`` chunks."""
    base, extra = divmod(num_games, num_workers)
    return [base + (1 if i < extra else 0) for i in range(num_workers)]


def generate_self_play_data_parallel(
    network,
    config: Config,
    num_games: int,
    num_workers: int,
    collect_pgn: bool = False,
) -> List[GameResult]:
    """Generate self-play games across ``num_workers`` processes.

    Falls back to the sequential path when ``num_workers <= 1`` or only one game
    is requested, which keeps single-core and test environments simple.

    Args:
        network: the current network (its CPU weights are shipped to workers).
        config: the global configuration.
        num_games: total number of games to generate.
        num_workers: number of worker processes to spawn.
        collect_pgn: collect PGN text for each game.

    Returns:
        A combined list of :class:`GameResult` from all workers.
    """
    if num_workers <= 1 or num_games <= 1:
        return generate_self_play_data(network, config, num_games, collect_pgn=collect_pgn)

    # Move the weights to CPU once and share them with every worker.
    cpu_state = {k: v.detach().cpu() for k, v in network.state_dict().items()}

    chunks = _split_games(num_games, num_workers)
    base_seed = config.training.seed
    tasks = []
    base_round = 1
    for w, n in enumerate(chunks):
        if n == 0:
            continue
        seed = None if base_seed is None else base_seed + w
        tasks.append((cpu_state, config, n, seed, collect_pgn, base_round))
        base_round += n

    # ``spawn`` is the safe, cross-platform start method (and the only one on
    # Windows).  The pool must be created from a ``__main__``-guarded entry point.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        chunked_results = pool.map(_worker_play_games, tasks)

    # Flatten the per-worker lists back into a single list of games.
    return [game for worker_results in chunked_results for game in worker_results]
