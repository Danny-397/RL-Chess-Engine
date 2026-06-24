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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import chess

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


def play_game(network, config: Config, verbose: bool = False) -> List[TrainingExample]:
    """Play a single self-play game and return its list of training examples.

    Args:
        network: the current :class:`model.ChessNet`.
        config: the global :class:`config.Config`.
        verbose: if ``True``, print each move as it is played.

    Returns:
        A list of :class:`TrainingExample`, one per move, with ``value`` already
        back-filled from the final game result.
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

    if verbose:
        print(f"  game over: {game.board.result(claim_draw=True)} "
              f"({len(examples)} positions)")
    return examples


def generate_self_play_data(
    network, config: Config, num_games: int, verbose: bool = False
) -> List[TrainingExample]:
    """Play ``num_games`` self-play games and concatenate their examples.

    Args:
        network: the current network used by both sides.
        config: the global configuration.
        num_games: how many games to play.
        verbose: forwarded to :func:`play_game` (and prints a per-game header).

    Returns:
        A flat list of all training examples gathered across the games.
    """
    data: List[TrainingExample] = []
    for g in range(num_games):
        if verbose:
            print(f"[self-play] game {g + 1}/{num_games}")
        data.extend(play_game(network, config, verbose=verbose))
    return data
