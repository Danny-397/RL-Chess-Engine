"""chess_game.py
================

Game logic and *encoding* for the chess engine.

This module is the bridge between the rules of chess and the tensors that the
neural network consumes/produces.  It relies on the excellent
`python-chess <https://python-chess.readthedocs.io>`_ library for the actual
rules (legal-move generation, check / checkmate / draw detection), which lets us
focus the intellectual effort on the *encoding* and the learning algorithm.

It is responsible for three things:

1. **A thin game wrapper** (:class:`ChessGame`) exposing exactly the operations
   the search and self-play code need: list legal moves, push a move, detect
   terminal states, and clone the position.

2. **Board -> tensor encoding** (:func:`encode_board`).  The network always sees
   the position *from the perspective of the side to move* ("canonical form"):
   when it is Black's turn we mirror the board so the network only ever has to
   reason about "the player at the bottom of the board to move".  This makes the
   network colour-agnostic and roughly halves what it has to learn.

3. **Move <-> index encoding** (:func:`move_to_index` / :func:`index_to_move`).
   We use the canonical AlphaZero action space of ``64 x 73 = 4672`` moves:

   ===========  =====  ===========================================================
   plane range  count  meaning
   ===========  =====  ===========================================================
   0  .. 55      56    "queen" moves: 8 compass directions x 7 distances
   56 .. 63       8    knight moves: the 8 L-shaped jumps
   64 .. 72       9    under-promotions: {knight, bishop, rook} x {left, fwd, right}
   ===========  =====  ===========================================================

   Queen-promotions are encoded as ordinary "queen" moves (direction + distance),
   so they do not need their own planes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import chess

from config import NUM_INPUT_PLANES, NUM_ACTIONS


# --------------------------------------------------------------------------- #
# Move-encoding lookup tables
# --------------------------------------------------------------------------- #
# The 8 "queen" directions as (delta_file, delta_rank), ordered
# N, NE, E, SE, S, SW, W, NW.  Any sliding move (queen/rook/bishop), king step
# or pawn push/capture is one of these directions at some distance in 1..7.
_QUEEN_DIRECTIONS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# The 8 knight jumps as (delta_file, delta_rank).
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Under-promotion target pieces (queen promotions use the "queen move" planes).
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Plane offsets within the 73 move-types for each move family.
_KNIGHT_PLANE_OFFSET = 56       # planes 56..63
_UNDERPROMO_PLANE_OFFSET = 64   # planes 64..72
_NUM_MOVE_TYPES = 73


def _sign(x: int) -> int:
    """Return -1, 0 or +1 -- the sign of ``x``."""
    return (x > 0) - (x < 0)


def move_to_index(move: chess.Move) -> int:
    """Encode a move (in *canonical / white-to-move* orientation) as an integer.

    The returned index is in ``[0, NUM_ACTIONS)`` and is the position of this
    move in the flat policy vector produced by the network.

    .. important::
       ``move`` must already be expressed in the canonical frame, i.e. as if the
       side to move were White moving "up" the board.  :class:`ChessGame` takes
       care of mirroring before calling this (see :meth:`ChessGame.encode_move`).

    Args:
        move: a ``chess.Move`` in canonical orientation.

    Returns:
        The integer action index ``from_square * 73 + move_type_plane``.
    """
    from_sq, to_sq = move.from_square, move.to_square
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    d_file, d_rank = to_file - from_file, to_rank - from_rank

    promo = move.promotion

    if promo in _UNDERPROMO_PIECES:
        # Under-promotion: 3 forward directions (capture-left, push, capture-right)
        # times 3 target pieces.  ``d_file`` is in {-1, 0, +1} for a promoting pawn.
        direction_index = {-1: 0, 0: 1, 1: 2}[d_file]
        piece_index = _UNDERPROMO_PIECES.index(promo)
        plane = _UNDERPROMO_PLANE_OFFSET + piece_index * 3 + direction_index
    elif (d_file, d_rank) in _KNIGHT_DELTAS:
        # Knight move.
        plane = _KNIGHT_PLANE_OFFSET + _KNIGHT_DELTAS.index((d_file, d_rank))
    else:
        # "Queen" move: a straight line in one of 8 directions.  This also covers
        # queen-promotions (a pawn stepping to the last rank) automatically.
        distance = max(abs(d_file), abs(d_rank))
        direction = (_sign(d_file), _sign(d_rank))
        direction_index = _QUEEN_DIRECTIONS.index(direction)
        plane = direction_index * 7 + (distance - 1)

    return from_sq * _NUM_MOVE_TYPES + plane


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Decode an action index back into a concrete ``chess.Move`` for ``board``.

    This is the inverse of :func:`move_to_index` and is provided mainly for
    completeness / debugging.  ``board`` must be in canonical orientation; the
    decoded move is validated against (and matched to) the board's legal moves so
    that promotions and en-passant are resolved correctly.

    Args:
        index: an action index in ``[0, NUM_ACTIONS)``.
        board: the canonical-orientation board the move applies to.

    Returns:
        The matching legal ``chess.Move``, or ``None`` if no legal move maps to
        this index in the given position.
    """
    from_sq = index // _NUM_MOVE_TYPES
    plane = index % _NUM_MOVE_TYPES
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)

    promotion = None
    if plane >= _UNDERPROMO_PLANE_OFFSET:
        # Under-promotion plane.
        p = plane - _UNDERPROMO_PLANE_OFFSET
        piece_index, direction_index = divmod(p, 3)
        promotion = _UNDERPROMO_PIECES[piece_index]
        d_file = {0: -1, 1: 0, 2: 1}[direction_index]
        d_rank = 1  # promotions always advance one rank in canonical frame
    elif plane >= _KNIGHT_PLANE_OFFSET:
        d_file, d_rank = _KNIGHT_DELTAS[plane - _KNIGHT_PLANE_OFFSET]
    else:
        direction_index, distance_minus_1 = divmod(plane, 7)
        d_file, d_rank = _QUEEN_DIRECTIONS[direction_index]
        distance = distance_minus_1 + 1
        d_file, d_rank = d_file * distance, d_rank * distance

    to_file, to_rank = from_file + d_file, from_rank + d_rank
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None
    to_sq = chess.square(to_file, to_rank)

    # Build candidate move(s) and match against the legal move list.  This is the
    # robust way to recover queen-promotions (encoded on the queen planes) and to
    # reject pseudo-moves that are not actually legal here.
    candidates = [chess.Move(from_sq, to_sq, promotion=promotion)]
    if promotion is None and to_rank in (0, 7):
        # A pawn reaching the last rank on a queen-plane is a queen promotion.
        candidates.append(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))

    for candidate in candidates:
        if candidate in board.legal_moves:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Board -> tensor encoding
# --------------------------------------------------------------------------- #
# Plane layout of the 18-channel board tensor (canonical / side-to-move view):
#   0..5    : our pieces      [pawn, knight, bishop, rook, queen, king]
#   6..11   : their pieces     (same order)
#   12      : our  king-side castling right
#   13      : our  queen-side castling right
#   14      : their king-side castling right
#   15      : their queen-side castling right
#   16      : en-passant target square
#   17      : half-move clock (normalised by 100, the 50-move-rule horizon)
_PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]


def _canonical_board(board: chess.Board) -> chess.Board:
    """Return ``board`` rotated so the side to move is always "White at the bottom".

    When it is Black's turn we use :meth:`chess.Board.mirror`, which swaps piece
    colours *and* flips the board vertically (and mirrors castling/en-passant
    state).  The result is a position with White to move that is strategically
    identical from the mover's point of view.
    """
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as an ``(NUM_INPUT_PLANES, 8, 8)`` float32 tensor.

    The encoding is *canonical*: it is always produced from the perspective of
    the side to move (see :func:`_canonical_board`).  The network therefore never
    needs to know which colour it is playing -- "us" is always planes 0..5.

    Args:
        board: any ``chess.Board`` position.

    Returns:
        A ``numpy`` array of shape ``(18, 8, 8)`` and dtype ``float32``.
    """
    cboard = _canonical_board(board)
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    # --- piece planes (0..11) ------------------------------------------------
    for square in chess.SQUARES:
        piece = cboard.piece_at(square)
        if piece is None:
            continue
        piece_idx = _PIECE_ORDER.index(piece.piece_type)
        # "us" == White in the canonical board; "them" == Black.
        plane = piece_idx if piece.color == chess.WHITE else piece_idx + 6
        rank, file = chess.square_rank(square), chess.square_file(square)
        planes[plane, rank, file] = 1.0

    # --- castling rights (12..15) -------------------------------------------
    if cboard.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if cboard.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if cboard.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if cboard.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    # --- en-passant target (16) ---------------------------------------------
    if cboard.ep_square is not None:
        rank, file = chess.square_rank(cboard.ep_square), chess.square_file(cboard.ep_square)
        planes[16, rank, file] = 1.0

    # --- half-move clock (17) -----------------------------------------------
    # Normalised so the plane is ~1.0 as we approach the 50-move (100-ply) rule.
    planes[17, :, :] = min(cboard.halfmove_clock, 100) / 100.0

    return planes


# --------------------------------------------------------------------------- #
# The game wrapper
# --------------------------------------------------------------------------- #
class ChessGame:
    """A thin, search-friendly wrapper around ``chess.Board``.

    The class deliberately exposes only what MCTS and self-play need.  Keeping
    this surface small makes the search code easy to read and means the
    underlying rules engine could in principle be swapped out.

    Attributes:
        board: the wrapped ``chess.Board`` (the single source of truth).
    """

    def __init__(self, board: Optional[chess.Board] = None) -> None:
        """Create a game.  With no argument, starts from the standard position."""
        self.board: chess.Board = board if board is not None else chess.Board()

    # ---- basic accessors --------------------------------------------------- #
    @property
    def turn(self) -> bool:
        """``chess.WHITE`` or ``chess.BLACK`` -- whose move it is."""
        return self.board.turn

    def legal_moves(self) -> List[chess.Move]:
        """Return the list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        """Apply ``move`` in place, advancing the position by one ply."""
        self.board.push(move)

    def clone(self) -> "ChessGame":
        """Return a deep copy of the game (used heavily by MCTS).

        ``stack=False`` skips copying the full move history, which we do not need
        during search and which makes cloning noticeably cheaper.
        """
        return ChessGame(self.board.copy(stack=False))

    # ---- terminal detection ------------------------------------------------ #
    def is_terminal(self) -> bool:
        """Return ``True`` if the game is over (checkmate, stalemate or draw)."""
        return self.board.is_game_over(claim_draw=True)

    def terminal_value(self) -> float:
        """Return the game result from the perspective of the *side to move*.

        Must only be called on a terminal position.  Returns:

        * ``-1.0`` if the side to move is checkmated (they lost), and
        * ``0.0`` for any drawn outcome (stalemate, insufficient material,
          repetition, 50-move rule, ...).

        A ``+1.0`` is never returned here because the side *to move* can never be
        the one delivering mate -- the win is observed by the opponent one ply
        earlier and propagated through search/self-play.
        """
        if self.board.is_checkmate():
            return -1.0
        return 0.0

    def result_white(self) -> float:
        """Return the final result from *White's* perspective: +1 / 0 / -1.

        Used by self-play to label every stored position once the game ends.
        """
        result = self.board.result(claim_draw=True)  # "1-0", "0-1" or "1/2-1/2"
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    # ---- encoding helpers (canonicalisation lives here) -------------------- #
    def encode_state(self) -> np.ndarray:
        """Return the canonical ``(18, 8, 8)`` tensor for the current position."""
        return encode_board(self.board)

    def encode_move(self, move: chess.Move) -> int:
        """Map a *real* legal move to its canonical policy index.

        If it is Black's turn we first mirror the move's squares vertically so it
        matches the mirrored (canonical) board the network sees.
        """
        if self.board.turn == chess.BLACK:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
        return move_to_index(move)

    # ---- misc -------------------------------------------------------------- #
    def push_san(self, san: str) -> chess.Move:
        """Apply a move given in Standard Algebraic Notation (e.g. ``"Nf3"``).

        Returns the parsed ``chess.Move``.  Raises ``ValueError`` on illegal or
        unparseable input -- handy for the human-vs-engine console.
        """
        move = self.board.parse_san(san)  # raises ValueError if illegal
        self.board.push(move)
        return move

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        """Pretty unicode board, convenient for console play and debugging."""
        return str(self.board)
