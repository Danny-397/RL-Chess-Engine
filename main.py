"""main.py
=========

Command-line entry point for the chess engine.

Two modes are provided:

* ``train`` -- run the full self-play + training loop (see :mod:`training`)::

      python main.py --mode train
      python main.py --mode train --iterations 5 --games 4 --simulations 50

* ``play``  -- play a game against a trained engine from the console, entering
  your moves in Standard Algebraic Notation (e.g. ``e4``, ``Nf3``, ``O-O``)::

      python main.py --mode play
      python main.py --mode play --checkpoint checkpoints/best.pt --color black

All hyper-parameters live in :mod:`config`; the most common ones are exposed as
CLI flags below so you can experiment without editing code.
"""

from __future__ import annotations

import argparse
import os

import chess

from config import Config
from chess_game import ChessGame
from model import ChessNet


# --------------------------------------------------------------------------- #
# Building a Config from CLI arguments
# --------------------------------------------------------------------------- #
def _build_config(args: argparse.Namespace) -> Config:
    """Create a :class:`Config` and apply any command-line overrides."""
    config = Config()
    if args.iterations is not None:
        config.training.num_iterations = args.iterations
    if args.games is not None:
        config.training.games_per_iteration = args.games
    if args.simulations is not None:
        config.mcts.num_simulations = args.simulations
    if args.device is not None:
        config.training.device = args.device
    return config


# --------------------------------------------------------------------------- #
# train mode
# --------------------------------------------------------------------------- #
def run_train(args: argparse.Namespace) -> None:
    """Launch the training loop."""
    # Imported lazily so that ``--help`` and ``play`` mode don't pay the cost of
    # importing the (heavier) training stack.
    from training import train

    config = _build_config(args)
    print("Starting training with configuration:")
    print(f"  iterations         = {config.training.num_iterations}")
    print(f"  games/iteration    = {config.training.games_per_iteration}")
    print(f"  MCTS simulations   = {config.mcts.num_simulations}")
    print(f"  device             = {config.resolved_device()}")
    train(config)


# --------------------------------------------------------------------------- #
# play mode
# --------------------------------------------------------------------------- #
def _load_engine(config: Config, checkpoint: str):
    """Load a network for play, falling back to an untrained one if needed."""
    from training import load_checkpoint  # lazy import (keeps startup snappy)

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loaded engine from {checkpoint}")
        return load_checkpoint(checkpoint, config)

    print(
        f"WARNING: checkpoint '{checkpoint}' not found -- playing against an "
        "UNTRAINED (random) network.  Run training first for a real opponent."
    )
    return ChessNet(config.network).to(config.resolved_device()).eval()


def _engine_move(network, game: ChessGame, config: Config):
    """Pick the engine's move: greedy MCTS with no root exploration noise."""
    from mcts import MCTS, action_probabilities

    mcts = MCTS(network, config.mcts)
    root = mcts.run(game, add_exploration_noise=False)
    moves, probs = action_probabilities(root, temperature=0.0)  # greedy
    return moves[int(probs.argmax())]


def run_play(args: argparse.Namespace) -> None:
    """Play a console game: human (SAN input) vs. the engine."""
    config = _build_config(args)
    network = _load_engine(config, args.checkpoint)

    human_is_white = args.color.lower() == "white"
    game = ChessGame()

    print("\nYou are playing as", "White" if human_is_white else "Black")
    print("Enter moves in algebraic notation (e.g. e4, Nf3, O-O).")
    print("Type 'quit' to resign, 'board' to redraw the position.\n")

    while not game.is_terminal():
        print(game)
        print()

        # ``chess.WHITE`` is ``True`` and ``chess.BLACK`` is ``False``, so the
        # boolean ``game.turn`` can be compared directly against ``human_is_white``.
        if game.turn == human_is_white:
            # ---- human's turn ----
            # ``﻿`` strips a stray byte-order-mark that some Windows
            # shells prepend when input is piped in.
            user_input = input("Your move: ").strip().lstrip("﻿")
            if user_input.lower() in {"quit", "resign", "exit"}:
                print("You resigned. Good game!")
                return
            if user_input.lower() == "board":
                continue
            try:
                game.push_san(user_input)
            except ValueError:
                print(f"  '{user_input}' is not a legal move here -- try again.\n")
                continue
        else:
            # ---- engine's turn ----
            print("Engine is thinking...")
            move = _engine_move(network, game, config)
            san = game.board.san(move)
            game.push(move)
            print(f"Engine plays: {san}\n")

    # ---- game over ----
    print(game)
    print("\nGame over:", game.board.result(claim_draw=True))
    if game.board.is_checkmate():
        # The side to move has been mated, so the *other* side won.
        winner = "Black" if game.turn == chess.WHITE else "White"
        print(f"Checkmate -- {winner} wins.")
    else:
        print("Draw.")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AlphaZero-style chess engine (train / play).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "play"], default="train",
        help="train a new engine via self-play, or play against a trained one.",
    )
    # training overrides
    parser.add_argument("--iterations", type=int, default=None,
                        help="number of self-play/train iterations (train mode).")
    parser.add_argument("--games", type=int, default=None,
                        help="self-play games per iteration (train mode).")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (both modes).")
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu', 'cuda', or 'auto'.")
    # play options
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="model checkpoint to load (play mode).")
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"],
                        help="the colour YOU play (play mode).")
    return parser


def main() -> None:
    """Parse arguments and dispatch to the requested mode."""
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_play(args)


if __name__ == "__main__":
    main()
