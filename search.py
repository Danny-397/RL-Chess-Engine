"""search.py
=============

A small, classical **alpha-beta** chess searcher used as the play-time opponent.

Why this exists alongside the neural network: the AlphaZero network in this repo
needs a lot of GPU self-play before it plays well, and an *untrained* network +
MCTS shuffles winning positions into draws.  This module is a dependency-light
(no PyTorch) negamax searcher that plays genuinely sound chess right now -- it
captures, avoids blunders, and *converts* won positions into checkmate -- so the
deployed demo is fun to play while the network trains.

It is deliberately classical and self-contained:

* **Negamax + alpha-beta pruning** with a fixed depth.
* **Quiescence search** on captures, so it doesn't blunder at the search horizon.
* **Move ordering** (captures first, most-valuable-victim first) for good pruning.
* An evaluation = **material** + an **endgame "mop-up"** term (drive the lone king
  to a corner, bring your king up, squeeze its escape squares) so it can actually
  deliver basic mates.
* **Draws score 0** (repetition / 50-move / stalemate / insufficient material), so
  a winning side refuses to repeat and instead makes progress toward mate, while
  mate scores are depth-adjusted so it prefers the *fastest* mate.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import chess

# Centipawn piece values (king handled via checkmate, not material).
_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
_MATE = 1_000_000          # score for delivering mate (depth-adjusted below)
_INF = 10_000_000


def _mopup_cp(board: chess.Board, strong: chess.Color) -> float:
    """Endgame bonus (centipawns) for ``strong`` cornering the losing king."""
    weak_king = board.king(not strong)
    strong_king = board.king(strong)
    if weak_king is None or strong_king is None:
        return 0.0
    wf, wr = chess.square_file(weak_king), chess.square_rank(weak_king)
    to_edge = max(3 - wf, wf - 4) + max(3 - wr, wr - 4)        # 0..6
    kings_close = 7 - chess.square_distance(strong_king, weak_king)  # 0..6
    escapes = 0
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[weak_king]):
        occ = board.piece_at(sq)
        if occ is not None and occ.color != strong:
            continue  # blocked by the losing king's own piece
        if not board.is_attacked_by(strong, sq):
            escapes += 1
    boxed_in = 8 - escapes                                     # 0..8
    return 14.0 * to_edge + 8.0 * kings_close + 12.0 * boxed_in


def evaluate(board: chess.Board) -> float:
    """Static evaluation in centipawns, from the side-to-move's perspective."""
    material = 0
    for piece_type, value in _VALUE.items():
        material += value * (
            len(board.pieces(piece_type, board.turn))
            - len(board.pieces(piece_type, not board.turn))
        )
    score = float(material)
    # Once clearly ahead, add the mop-up term so the win can be converted.
    if abs(material) >= 400:
        strong = board.turn if material > 0 else not board.turn
        mop = _mopup_cp(board, strong)
        score += mop if material > 0 else -mop
    return score


def _is_drawn(board: chess.Board) -> bool:
    """Cheap draw detection used inside the search (all score 0)."""
    return (
        board.is_insufficient_material()
        or board.halfmove_clock >= 100
        or board.is_repetition(3)
    )


def _ordered_moves(board: chess.Board) -> List[chess.Move]:
    """Legal moves with captures first (most valuable victim first)."""
    captures, quiets = [], []
    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            quiets.append(move)

    def victim_value(move: chess.Move) -> int:
        piece = board.piece_at(move.to_square)
        return _VALUE.get(piece.piece_type, 0) if piece else 0  # 0 for en passant

    captures.sort(key=victim_value, reverse=True)
    return captures + quiets


def _quiesce(board: chess.Board, alpha: float, beta: float) -> float:
    """Search only captures past the depth limit to avoid horizon blunders."""
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        board.push(move)
        score = -_quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
    """Negamax with alpha-beta pruning; positive == good for side to move."""
    if _is_drawn(board):
        return 0.0
    moves = _ordered_moves(board)
    if not moves:  # no legal move -> checkmate (we lost) or stalemate (draw)
        return -_MATE + ply if board.is_check() else 0.0
    if depth == 0:
        return _quiesce(board, alpha, beta)

    best = -_INF
    for move in moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # opponent won't allow this line
    return best


def search(board: chess.Board, depth: int = 4) -> Tuple[Optional[chess.Move], float, List[Tuple[chess.Move, float]]]:
    """Pick the best move for the side to move.

    Args:
        board: the position to search (not modified on return).
        depth: full-width search depth in plies (quiescence extends captures).

    Returns:
        ``(best_move, best_score_cp, ranked)`` where ``ranked`` is every root move
        paired with its score (centipawns, side-to-move perspective), best first.
    """
    best_move: Optional[chess.Move] = None
    best = -_INF
    ranked: List[Tuple[chess.Move, float]] = []

    # Each root move is searched with a *full* window so every returned score is
    # exact and the moves can be ranked/displayed correctly. (A narrowing
    # alpha-beta window at the root would still pick the best move, but it would
    # report mere bounds for the others -- making the analysis panel misleading.)
    # Pruning still happens inside each move's subtree.
    for move in _ordered_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -_INF, _INF, 1)
        board.pop()
        ranked.append((move, score))
        if score > best:
            best = score
            best_move = move

    ranked.sort(key=lambda ms: ms[1], reverse=True)
    return best_move, best, ranked


# --------------------------------------------------------------------------- #
# Human-readable analysis (shared by the console, the analyze CLI and the web UI)
# --------------------------------------------------------------------------- #
def cp_to_value(cp: float) -> float:
    """Map a centipawn score to a bounded value in [-1, 1] (mates saturate)."""
    if cp >= _MATE - 100_000:
        return 1.0
    if cp <= -_MATE + 100_000:
        return -1.0
    return math.tanh(cp / 400.0)


def cp_to_win_probability(cp: float) -> float:
    """Map a centipawn score to a win probability in [0, 1]."""
    return (cp_to_value(cp) + 1.0) / 2.0


def analyze(board: chess.Board, depth: int = 4, top_n: int = 3) -> Dict:
    """Search a position and return a JSON-friendly analysis dict.

    The shape matches what the web UI / console expect: an evaluation, a
    White-perspective win probability (for a stable eval bar), the best move, and
    the top ``top_n`` recommended moves.
    """
    if board.is_game_over(claim_draw=True):
        return {
            "game_over": True,
            "result": board.result(claim_draw=True),
            "suggestions": [],
            "white_win_probability": None,
        }

    best_move, best_cp, ranked = search(board, depth)
    stm_is_white = board.turn == chess.WHITE
    white_cp = best_cp if stm_is_white else -best_cp

    top = ranked[:top_n]
    suggestions: List[Dict] = [
        {
            "san": board.san(mv),
            "uci": mv.uci(),
            "value": cp_to_value(cp),
            "win_probability": cp_to_win_probability(cp),
        }
        for mv, cp in top
    ]
    # A "preference share" per move (softmax over scores) so the UI can show how
    # strongly the engine favours each candidate, analogous to MCTS visit shares.
    if top:
        hi = max(cp for _, cp in top)
        weights = [math.exp((cp - hi) / 120.0) for _, cp in top]
        total = sum(weights) or 1.0
        for s, w in zip(suggestions, weights):
            s["visit_fraction"] = w / total

    return {
        "game_over": False,
        "side_to_move": "white" if stm_is_white else "black",
        "value": cp_to_value(best_cp),
        "win_probability": cp_to_win_probability(best_cp),
        "white_win_probability": cp_to_win_probability(white_cp),
        "suggestions": suggestions,
        "move": {
            "uci": best_move.uci(),
            "san": board.san(best_move),
            "win_probability": cp_to_win_probability(best_cp),
        },
    }
