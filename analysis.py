"""analysis.py
==============

Turn the engine's search into human-readable *advice* about a position.

The neural network + MCTS already compute everything needed to explain a
position; this module simply surfaces it:

* the engine's **recommended moves**, ranked by how much MCTS searched them
  (visit count is the engine's true preference -- a more reliable signal than the
  raw network prior), and
* an **evaluation** / **win probability** for the side to move, taken from the
  search-refined value at the root.

This is what powers the ``hint`` command in console play and the ``analyze`` CLI
mode -- and later the "recommended moves" panel in the web UI.  Keeping it in its
own module means every front-end shares exactly the same analysis logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chess

from chess_game import ChessGame
from config import Config
from mcts import MCTS


def value_to_win_probability(value: float) -> float:
    """Map a value in ``[-1, 1]`` to a win probability in ``[0, 1]``.

    The value head is trained on game outcomes in ``{-1, 0, +1}``, so a simple
    linear rescale ``(v + 1) / 2`` reads naturally as "expected score": ``+1`` ->
    100%, ``0`` -> 50% (drawish), ``-1`` -> 0%.
    """
    return (value + 1.0) / 2.0


@dataclass
class MoveSuggestion:
    """One recommended move and the engine's statistics about it.

    Attributes:
        move: the ``chess.Move``.
        san: the move in Standard Algebraic Notation (e.g. ``"Nf3"``).
        visits: how many MCTS simulations explored this move.
        visit_fraction: ``visits`` as a fraction of the root's total visits --
            i.e. how strongly the engine prefers this move.
        value: evaluation after the move, from the *mover's* perspective.
        win_probability: ``value`` expressed as a win probability in ``[0, 1]``.
    """

    move: chess.Move
    san: str
    visits: int
    visit_fraction: float
    value: float
    win_probability: float


@dataclass
class PositionAnalysis:
    """The engine's assessment of a single position.

    Attributes:
        value: search-refined evaluation for the side to move, in ``[-1, 1]``.
        win_probability: ``value`` as a win probability in ``[0, 1]``.
        suggestions: recommended moves, best first.
    """

    value: float
    win_probability: float
    suggestions: List[MoveSuggestion]

    def render(self, max_moves: int = 3) -> str:
        """Return a compact multi-line summary suitable for a console."""
        lines = [
            f"Engine eval: {self.value:+.2f}  "
            f"(win probability for side to move: {self.win_probability:.0%})",
            "Recommended moves:",
        ]
        for i, s in enumerate(self.suggestions[:max_moves], start=1):
            lines.append(
                f"  {i}. {s.san:<7} {s.visit_fraction:5.0%} of search, "
                f"eval {s.value:+.2f}"
            )
        return "\n".join(lines)


def analyze_position(
    network, game: ChessGame, config: Config, top_n: int = 5
) -> PositionAnalysis:
    """Run MCTS on ``game`` and report the engine's recommendations.

    This always searches in "competition mode" (no Dirichlet exploration noise)
    so the advice reflects the engine's genuine best judgement.

    Args:
        network: the :class:`model.ChessNet` to analyse with.
        game: the position to analyse (not modified).
        config: global configuration (its ``mcts`` section drives the search).
        top_n: maximum number of recommended moves to return.

    Returns:
        A :class:`PositionAnalysis` with the evaluation and ranked suggestions.
    """
    root = MCTS(network, config.mcts).run(game, add_exploration_noise=False)

    total_visits = sum(child.visit_count for child in root.children.values())
    total_visits = max(total_visits, 1)  # guard against divide-by-zero

    suggestions: List[MoveSuggestion] = []
    for move, child in root.children.items():
        # ``child.value()`` is from the child's (opponent's) perspective, so we
        # negate it to express the evaluation from the moving side's perspective.
        move_value = -child.value()
        suggestions.append(
            MoveSuggestion(
                move=move,
                san=game.board.san(move),
                visits=child.visit_count,
                visit_fraction=child.visit_count / total_visits,
                value=move_value,
                win_probability=value_to_win_probability(move_value),
            )
        )

    # Rank by how much the engine searched each move (its real preference).
    suggestions.sort(key=lambda s: s.visits, reverse=True)

    root_value = root.value()
    return PositionAnalysis(
        value=root_value,
        win_probability=value_to_win_probability(root_value),
        suggestions=suggestions[:top_n],
    )
