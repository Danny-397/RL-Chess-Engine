"""test_engine.py
=================

A small but meaningful test-suite.  These tests are deliberately readable: they
double as *documentation of the invariants* the system relies on, and they catch
the kind of subtle bugs (move-encoding round-trips, perspective sign errors) that
are otherwise extremely hard to notice in a self-learning system.

Run with::

    pytest -q          # or:  python -m pytest tests/
"""

from __future__ import annotations

import os
import sys

import numpy as np
import chess
import pytest

# Make the project root importable when running ``pytest`` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, NUM_ACTIONS, NUM_INPUT_PLANES
from chess_game import ChessGame, encode_board, move_to_index, index_to_move
from model import ChessNet
from mcts import MCTS, action_probabilities


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_board_tensor_shape_and_range():
    """The encoded start position has the right shape and is binary-ish."""
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor.shape == (NUM_INPUT_PLANES, 8, 8)
    assert tensor.dtype == np.float32
    # Start position: 32 pieces -> 32 ones across the 12 piece planes.
    assert tensor[0:12].sum() == 32


def test_move_index_in_range_for_all_legal_start_moves():
    """Every legal opening move maps to a valid, unique action index."""
    game = ChessGame()
    indices = [game.encode_move(m) for m in game.legal_moves()]
    assert all(0 <= i < NUM_ACTIONS for i in indices)
    assert len(set(indices)) == len(indices)  # no collisions


def test_move_encoding_roundtrip_white():
    """move_to_index -> index_to_move recovers the original move (White)."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move)
        recovered = index_to_move(idx, board)
        assert recovered == move


def test_move_encoding_roundtrip_promotions():
    """Queen- and under-promotions both round-trip correctly."""
    # White pawn on e7, ready to promote; bare kings elsewhere.
    board = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
    for move in board.legal_moves:
        if move.promotion is not None:
            idx = move_to_index(move)
            assert index_to_move(idx, board) == move


def test_canonical_encoding_is_color_symmetric():
    """A position and its colour-flipped mirror encode identically.

    This is the property that lets the network be colour-agnostic: the side to
    move always sees itself as "us" on planes 0..5.
    """
    board = chess.Board()
    board.push_san("e4")  # now Black to move
    black_view = encode_board(board)

    mirror = board.mirror()  # White to move, same structure
    white_view = encode_board(mirror)

    assert np.array_equal(black_view, white_view)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_network_output_shapes_and_value_range():
    """The network returns correctly shaped policy logits and a bounded value."""
    cfg = Config()
    net = ChessNet(cfg.network)
    policy, value = net.predict(ChessGame().encode_state())
    assert policy.shape == (NUM_ACTIONS,)
    assert -1.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# Terminal detection / values
# --------------------------------------------------------------------------- #
def test_terminal_value_checkmate():
    """Fool's mate: the side to move is checkmated, so terminal_value == -1."""
    game = ChessGame()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        game.push_san(san)
    assert game.is_terminal()
    assert game.terminal_value() == -1.0
    assert game.result_white() == -1.0  # Black delivered mate


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #
def test_mcts_returns_valid_distribution():
    """MCTS produces a normalised distribution over the legal moves."""
    cfg = Config()
    cfg.mcts.num_simulations = 16  # keep the test fast
    net = ChessNet(cfg.network)

    game = ChessGame()
    root = MCTS(net, cfg.mcts).run(game, add_exploration_noise=True)
    moves, probs = action_probabilities(root, temperature=1.0)

    assert len(moves) == len(list(game.legal_moves()))
    assert pytest.approx(probs.sum(), abs=1e-6) == 1.0
    # The most-visited move must be a legal move in this position.
    assert moves[int(np.argmax(probs))] in game.legal_moves()


def test_mcts_greedy_temperature_picks_single_move():
    """At temperature 0 all probability mass is on one move."""
    cfg = Config()
    cfg.mcts.num_simulations = 16
    net = ChessNet(cfg.network)
    root = MCTS(net, cfg.mcts).run(ChessGame(), add_exploration_noise=False)
    _, probs = action_probabilities(root, temperature=0.0)
    assert pytest.approx(probs.max(), abs=1e-9) == 1.0
    assert (probs > 0).sum() == 1
