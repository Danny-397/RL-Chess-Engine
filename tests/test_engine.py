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
from self_play import play_game, generate_self_play_data
from evaluation import (
    NetworkAgent, RandomAgent, play_match, elo_difference,
)
from analysis import analyze_position, value_to_win_probability


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


# --------------------------------------------------------------------------- #
# Self-play (data generation + PGN)
# --------------------------------------------------------------------------- #
def _fast_config() -> Config:
    """A tiny config so self-play/eval tests run in a couple of seconds."""
    cfg = Config()
    cfg.mcts.num_simulations = 8
    cfg.training.max_moves = 12
    cfg.training.temperature_moves = 4
    return cfg


def test_self_play_produces_consistent_examples():
    """A self-play game yields well-formed, value-labelled examples + PGN."""
    cfg = _fast_config()
    net = ChessNet(cfg.network)
    result = play_game(net, cfg, collect_pgn=True)

    assert len(result.examples) > 0
    for ex in result.examples:
        assert ex.state.shape == (NUM_INPUT_PLANES, 8, 8)
        assert ex.policy.shape == (NUM_ACTIONS,)
        assert pytest.approx(ex.policy.sum(), abs=1e-5) == 1.0
        assert ex.value in (-1.0, 0.0, 1.0)
    # The PGN should be non-empty and carry our self-play event tag.
    assert "RL-Chess-Engine self-play" in result.pgn


def test_generate_self_play_data_returns_one_result_per_game():
    cfg = _fast_config()
    net = ChessNet(cfg.network)
    results = generate_self_play_data(net, cfg, num_games=2)
    assert len(results) == 2


# --------------------------------------------------------------------------- #
# Evaluation + Elo
# --------------------------------------------------------------------------- #
def test_elo_difference_monotonic_and_symmetric():
    """Elo is 0 at 50%, positive above, negative below, and antisymmetric."""
    assert elo_difference(0.5) == pytest.approx(0.0, abs=1e-6)
    assert elo_difference(0.75) > 0
    assert elo_difference(0.25) < 0
    assert elo_difference(0.75) == pytest.approx(-elo_difference(0.25), abs=1e-6)


def test_play_match_tallies_all_games():
    """A match accounts for exactly the requested number of games."""
    cfg = _fast_config()
    net = ChessNet(cfg.network)
    candidate = NetworkAgent(net, cfg, num_simulations=8)
    result = play_match(candidate, RandomAgent(), cfg, num_games=4)
    assert result.games == 4
    assert result.wins + result.draws + result.losses == 4
    assert 0.0 <= result.score <= 1.0


# --------------------------------------------------------------------------- #
# Analysis (hints / recommended moves)
# --------------------------------------------------------------------------- #
def test_win_probability_mapping():
    """Value -> win-probability rescale hits the expected anchor points."""
    assert value_to_win_probability(1.0) == pytest.approx(1.0)
    assert value_to_win_probability(0.0) == pytest.approx(0.5)
    assert value_to_win_probability(-1.0) == pytest.approx(0.0)


def test_strip_bom_handles_both_encodings():
    """Console input parsing tolerates a leading byte-order-mark in either form."""
    from main import strip_bom

    assert strip_bom("﻿hint") == "hint"          # UTF-16 BOM char
    assert strip_bom("\xef\xbb\xbfhint") == "hint"     # UTF-8 BOM decoded bytes
    assert strip_bom("e4") == "e4"                      # untouched when absent


def test_analyze_position_recommends_legal_moves():
    """Analysis returns legal, visit-ranked suggestions and a valid win prob."""
    cfg = _fast_config()
    net = ChessNet(cfg.network)
    game = ChessGame()
    analysis = analyze_position(net, game, cfg, top_n=3)

    assert 0.0 <= analysis.win_probability <= 1.0
    assert 1 <= len(analysis.suggestions) <= 3
    legal = set(game.legal_moves())
    visits = [s.visits for s in analysis.suggestions]
    for s in analysis.suggestions:
        assert s.move in legal
        assert 0.0 <= s.win_probability <= 1.0
    # Suggestions must be sorted by visit count, strongest first.
    assert visits == sorted(visits, reverse=True)
