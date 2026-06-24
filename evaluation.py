"""evaluation.py
================

Measure how strong the engine actually is.

Training loss going down is reassuring, but the only test that matters is *does
the engine win more games?*  This module pits two move-choosing **agents**
against each other over a match and summarises the outcome, including an
approximate **Elo** difference -- the standard way to express relative chess
strength as a single number.

Two agents are provided:

* :class:`NetworkAgent` -- plays the strongest move it can: greedy MCTS with no
  exploration noise (this is "competition mode", unlike noisy self-play).
* :class:`RandomAgent`  -- plays a uniformly random legal move.  A useful sanity
  baseline: a network that cannot reliably beat random has not learned anything.

The Elo conversion uses the logistic model that defines Elo: a player expected to
score ``S`` against an opponent is rated

    elo_difference = -400 * log10(1 / S - 1)

so scoring 50% is +0 Elo, ~64% is about +100 Elo, and so on.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol

import chess
import chess.pgn

from chess_game import ChessGame
from config import Config
from mcts import MCTS, action_probabilities


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class Agent(Protocol):
    """Anything that can choose a move for a position.

    Defining the agent interface as a ``Protocol`` keeps the match code fully
    decoupled from *how* a move is chosen -- a neural net, random play, or any
    future agent (e.g. a classical engine) all plug in the same way.
    """

    name: str

    def select_move(self, game: ChessGame) -> chess.Move:
        """Return the move this agent wants to play in ``game``."""
        ...


class RandomAgent:
    """Baseline agent that plays a uniformly random legal move."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def select_move(self, game: ChessGame) -> chess.Move:
        return random.choice(game.legal_moves())


class NetworkAgent:
    """Agent that plays the engine's strongest move (greedy MCTS, no noise).

    Args:
        network: the :class:`model.ChessNet` to play with.
        config: the global configuration (its ``mcts`` section is used).
        num_simulations: optional override for the number of MCTS simulations;
            defaults to ``config.mcts.num_simulations``.
        name: a human-readable label used in match reports.
    """

    def __init__(
        self,
        network,
        config: Config,
        num_simulations: Optional[int] = None,
        name: str = "network",
    ) -> None:
        self.network = network
        self.name = name
        # Build a (possibly simulation-overridden) copy of the MCTS config so we
        # never mutate the shared global configuration.
        mcts_cfg = config.mcts
        if num_simulations is not None:
            mcts_cfg = dataclasses.replace(mcts_cfg, num_simulations=num_simulations)
        self.mcts = MCTS(network, mcts_cfg)

    def select_move(self, game: ChessGame) -> chess.Move:
        root = self.mcts.run(game, add_exploration_noise=False)
        moves, probs = action_probabilities(root, temperature=0.0)  # greedy
        return moves[int(probs.argmax())]


# --------------------------------------------------------------------------- #
# Match results + Elo
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    """Summary of a match from the perspective of ``agent_a`` ("the candidate").

    Attributes:
        wins / draws / losses: game counts for ``agent_a``.
        score: ``(wins + 0.5 * draws) / games`` -- the expected-score in [0, 1].
        elo_difference: estimated Elo of ``agent_a`` minus ``agent_b``.
        pgns: optional list of PGN strings, one per game played.
    """

    wins: int
    draws: int
    losses: int
    pgns: List[str]

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo_difference(self) -> float:
        return elo_difference(self.score)

    def summary(self, a_name: str = "A", b_name: str = "B") -> str:
        """Return a one-line human-readable summary of the match."""
        return (
            f"{a_name} vs {b_name}: "
            f"+{self.wins} ={self.draws} -{self.losses} "
            f"(score {self.score:.1%}, Elo {self.elo_difference:+.0f})"
        )


def elo_difference(score: float) -> float:
    """Convert an expected match score in [0, 1] into an Elo difference.

    A perfect or zero score is mathematically infinite Elo, so we clamp the score
    away from the extremes to keep the result finite and reportable.

    Args:
        score: expected score (fraction of points won) in ``[0, 1]``.

    Returns:
        The estimated Elo advantage (positive means stronger than the opponent).
    """
    eps = 1e-4
    score = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


# --------------------------------------------------------------------------- #
# Playing matches
# --------------------------------------------------------------------------- #
def play_match_game(
    white: Agent, black: Agent, config: Config, collect_pgn: bool = False
) -> "tuple[float, str]":
    """Play one game between two agents and return ``(white_result, pgn)``.

    Args:
        white: the agent playing White.
        black: the agent playing Black.
        config: global configuration (``training.max_moves`` caps game length).
        collect_pgn: whether to also return the game's PGN text.

    Returns:
        ``(white_result, pgn)`` where ``white_result`` is ``+1`` / ``0`` / ``-1``
        from White's perspective and ``pgn`` is the game text (or ``""``).
    """
    game = ChessGame()
    moves_played = 0
    while not game.is_terminal() and moves_played < config.training.max_moves:
        agent = white if game.turn == chess.WHITE else black
        game.push(agent.select_move(game))
        moves_played += 1

    white_result = game.result_white()

    pgn = ""
    if collect_pgn:
        node_game = chess.pgn.Game.from_board(game.board)
        node_game.headers["Event"] = "RL-Chess-Engine evaluation"
        node_game.headers["White"] = white.name
        node_game.headers["Black"] = black.name
        node_game.headers["Result"] = game.board.result(claim_draw=True)
        pgn = str(node_game)

    return white_result, pgn


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    config: Config,
    num_games: int,
    collect_pgn: bool = False,
    verbose: bool = False,
) -> MatchResult:
    """Play a ``num_games`` match between two agents, alternating colours.

    Colours are swapped every game so that any first-move advantage is shared
    evenly -- otherwise the agent that always plays White would look artificially
    stronger.

    Args:
        agent_a: the "candidate" agent whose results are reported.
        agent_b: the opponent.
        config: global configuration.
        num_games: number of games to play.
        collect_pgn: collect PGN text for each game.
        verbose: print a line per game.

    Returns:
        A :class:`MatchResult` from ``agent_a``'s perspective.
    """
    wins = draws = losses = 0
    pgns: List[str] = []

    for g in range(num_games):
        a_is_white = (g % 2 == 0)
        white, black = (agent_a, agent_b) if a_is_white else (agent_b, agent_a)

        white_result, pgn = play_match_game(white, black, config, collect_pgn)
        if collect_pgn:
            pgns.append(pgn)

        # Translate the White-perspective result into agent_a's perspective.
        a_result = white_result if a_is_white else -white_result
        if a_result > 0:
            wins += 1
        elif a_result < 0:
            losses += 1
        else:
            draws += 1

        if verbose:
            outcome = {1: "win", 0: "draw", -1: "loss"}[int(round(a_result))]
            print(f"  game {g + 1}/{num_games}: {agent_a.name} ({'W' if a_is_white else 'B'}) -> {outcome}")

    return MatchResult(wins=wins, draws=draws, losses=losses, pgns=pgns)


def evaluate_against_random(network, config: Config, num_games: int,
                            num_simulations: Optional[int] = None) -> MatchResult:
    """Convenience helper: evaluate a network against the random baseline.

    Args:
        network: the network to evaluate.
        config: global configuration.
        num_games: number of games to play.
        num_simulations: optional MCTS simulation override for the network agent.

    Returns:
        The :class:`MatchResult` of network (candidate) vs. random.
    """
    candidate = NetworkAgent(network, config, num_simulations=num_simulations, name="network")
    baseline = RandomAgent()
    return play_match(candidate, baseline, config, num_games)
