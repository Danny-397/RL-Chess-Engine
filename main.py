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
    if args.workers is not None:
        config.training.num_self_play_workers = args.workers
    if args.save_pgn:
        config.training.save_self_play_pgn = True
    if args.eval_every is not None:
        config.training.eval_every = args.eval_every
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
    print(f"  self-play workers  = {config.training.num_self_play_workers}")
    print(f"  save self-play PGN = {config.training.save_self_play_pgn}")
    print(f"  eval every         = {config.training.eval_every or 'off'}")
    print(f"  device             = {config.resolved_device()}")
    train(config)


# --------------------------------------------------------------------------- #
# play mode
# --------------------------------------------------------------------------- #
def strip_bom(text: str) -> str:
    """Remove a leading byte-order-mark from console input, if present.

    Some Windows shells prepend a BOM to the first line when input is piped in.
    It can arrive either as the Unicode char ``U+FEFF`` or, when the UTF-8 BOM
    bytes ``EF BB BF`` are decoded one-by-one, as the three characters
    ``\\xef\\xbb\\xbf``.  We handle both so commands like ``hint`` still parse.
    """
    for bom in ("﻿", "\xef\xbb\xbf"):
        if text.startswith(bom):
            return text[len(bom):]
    return text


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


def run_play(args: argparse.Namespace) -> None:
    """Play a console game: human (SAN input) vs. the engine."""
    config = _build_config(args)
    network = _load_engine(config, args.checkpoint)

    human_is_white = args.color.lower() == "white"
    game = ChessGame()

    print("\nYou are playing as", "White" if human_is_white else "Black")
    print("Enter moves in algebraic notation (e.g. e4, Nf3, O-O).")
    print("Commands: 'hint' = ask the engine for recommended moves,")
    print("          'eval' = show the engine's evaluation of the position,")
    print("          'board' = redraw, 'quit' = resign.\n")

    while not game.is_terminal():
        print(game)
        print()

        # ``chess.WHITE`` is ``True`` and ``chess.BLACK`` is ``False``, so the
        # boolean ``game.turn`` can be compared directly against ``human_is_white``.
        if game.turn == human_is_white:
            # ---- human's turn ----
            # ```` (byte-order-mark) is stripped because some Windows
            # shells prepend one to the first line when input is piped in.
            user_input = strip_bom(input("Your move: ")).strip()
            command = user_input.lower()
            if command in {"quit", "resign", "exit"}:
                print("You resigned. Good game!")
                return
            if command == "board":
                continue
            if command in {"hint", "eval", "analyze"}:
                # Ask the engine to assess *your* position and suggest moves.
                from analysis import analyze_position

                print("Analysing...")
                analysis = analyze_position(network, game, config)
                print(analysis.render() + "\n")
                continue
            try:
                game.push_san(user_input)
            except ValueError:
                print(f"  '{user_input}' is not a legal move here -- try again.\n")
                continue
        else:
            # ---- engine's turn ----
            from analysis import analyze_position

            print("Engine is thinking...")
            analysis = analyze_position(network, game, config)
            best = analysis.suggestions[0]  # most-searched move
            game.push(best.move)
            print(f"Engine plays: {best.san}  "
                  f"(its win estimate: {best.win_probability:.0%})\n")

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
# analyze mode
# --------------------------------------------------------------------------- #
def run_analyze(args: argparse.Namespace) -> None:
    """Print the engine's evaluation and recommended moves for one position.

    The position defaults to the standard opening but can be any legal position
    supplied as a FEN string via ``--fen``.
    """
    from analysis import analyze_position

    config = _build_config(args)
    network = _load_engine(config, args.checkpoint)

    board = chess.Board(args.fen) if args.fen else chess.Board()
    game = ChessGame(board)

    print(game)
    print(f"\n{'White' if game.turn == chess.WHITE else 'Black'} to move.\n")
    analysis = analyze_position(network, game, config, top_n=args.top_n)
    print(analysis.render(max_moves=args.top_n))


# --------------------------------------------------------------------------- #
# eval mode
# --------------------------------------------------------------------------- #
def run_eval(args: argparse.Namespace) -> None:
    """Play a match to measure engine strength and report an Elo estimate.

    The candidate (``--checkpoint``) plays ``--eval-games`` games against an
    opponent given by ``--opponent``: either ``"random"`` (the baseline) or the
    path to another checkpoint (head-to-head between two trained engines).
    """
    from evaluation import NetworkAgent, RandomAgent, play_match

    config = _build_config(args)
    candidate_net = _load_engine(config, args.checkpoint)
    candidate = NetworkAgent(candidate_net, config, name=os.path.basename(args.checkpoint))

    if args.opponent.lower() == "random":
        opponent = RandomAgent()
    else:
        opponent_net = _load_engine(config, args.opponent)
        opponent = NetworkAgent(opponent_net, config, name=os.path.basename(args.opponent))

    print(f"\nPlaying {args.eval_games} games: {candidate.name} vs {opponent.name} ...")
    result = play_match(
        candidate, opponent, config,
        num_games=args.eval_games,
        collect_pgn=bool(args.pgn_out),
        verbose=True,
    )
    print("\n" + result.summary(candidate.name, opponent.name))

    if args.pgn_out:
        with open(args.pgn_out, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(result.pgns) + "\n")
        print(f"Saved {len(result.pgns)} games to {args.pgn_out}")


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
        "--mode", choices=["train", "play", "eval", "analyze"], default="train",
        help="train, play against the engine, evaluate strength, or analyze a position.",
    )
    # training overrides
    parser.add_argument("--iterations", type=int, default=None,
                        help="number of self-play/train iterations (train mode).")
    parser.add_argument("--games", type=int, default=None,
                        help="self-play games per iteration (train mode).")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (all modes).")
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel self-play worker processes (train mode).")
    parser.add_argument("--save-pgn", action="store_true",
                        help="archive self-play games as PGN (train mode).")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="evaluate vs random every N iterations (train mode).")
    # play options
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="model checkpoint to load (play / eval mode).")
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"],
                        help="the colour YOU play (play mode).")
    # eval options
    parser.add_argument("--opponent", type=str, default="random",
                        help="'random' or a checkpoint path to play against (eval mode).")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="number of games to play (eval mode).")
    parser.add_argument("--pgn-out", type=str, default=None,
                        help="write evaluated games to this PGN file (eval mode).")
    # analyze options
    parser.add_argument("--fen", type=str, default=None,
                        help="FEN of the position to analyze (analyze mode).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="number of recommended moves to show (analyze mode).")
    return parser


def main() -> None:
    """Parse arguments and dispatch to the requested mode."""
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "analyze":
        run_analyze(args)
    else:
        run_play(args)


if __name__ == "__main__":
    main()
