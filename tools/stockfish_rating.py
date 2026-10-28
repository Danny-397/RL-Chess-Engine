"""Rate the classical engine against a strength-limited Stockfish.

"Elo vs a random-move player" says the engine learned *something*, but it is
not a number a chess player can place. This plays the site's classical
alpha-beta engine (search.py, the "Classical" opponent on rlchess.xyz) against
Stockfish with ``UCI_LimitStrength`` at a fixed ``UCI_Elo``, alternating colours,
and turns the score into a rating estimate with a 95% interval.

    python tools/stockfish_rating.py --stockfish path/to/stockfish.exe --depth 3 --games 40

Stockfish's UCI_Elo scale is calibrated against CCRL-style engine ratings at
blitz-like time controls, so the result is an engine rating on that scale, not
a FIDE or chess.com human rating. Results are written to
``experiment_out/stockfish_rating_d<depth>_e<elo>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import chess
import chess.engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import search  # noqa: E402

MAX_PLIES = 300


def elo_from_score(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / p - 1)


def play(sf: chess.engine.SimpleEngine, depth: int, ours_white: bool, movetime: float) -> float:
    """One game; returns our score (1, 0.5, 0)."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if (board.turn == chess.WHITE) == ours_white:
            move = search.search(board, depth)[0]
        else:
            move = sf.play(board, chess.engine.Limit(time=movetime)).move
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*":            # ply cap: score as a draw
        return 0.5
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == ours_white else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--depth", type=int, default=3, help="classical search depth (site: 2 easy, 3 normal, 4 hard)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--elo", type=int, default=1320, help="Stockfish UCI_Elo (1320 is its minimum)")
    ap.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
    args = ap.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1, "Hash": 16})
    scores, t0 = [], time.time()
    try:
        for g in range(args.games):
            tg = time.time()
            s = play(sf, args.depth, ours_white=(g % 2 == 0), movetime=args.movetime)
            scores.append(s)
            print(f"game {g + 1}/{args.games}: {s}  running {sum(scores)}/{len(scores)}"
                  f"  ({time.time() - tg:.0f}s)", flush=True)
    finally:
        sf.quit()

    n, total = len(scores), sum(scores)
    p = total / n
    # Standard error of the per-game score, then map the interval through the Elo curve.
    var = sum((s - p) ** 2 for s in scores) / max(n - 1, 1)
    se = math.sqrt(var / n)
    lo, hi = max(p - 1.96 * se, 0.0), min(p + 1.96 * se, 1.0)
    out = {
        "engine": f"classical alpha-beta, depth {args.depth}",
        "opponent": f"Stockfish UCI_Elo {args.elo}, {args.movetime}s/move",
        "games": n, "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
        "score": round(p, 3),
        "elo_diff": round(elo_from_score(p)), "elo_diff_95ci": [round(elo_from_score(lo)), round(elo_from_score(hi))],
        "rating_estimate": round(args.elo + elo_from_score(p)),
        "rating_95ci": [round(args.elo + elo_from_score(lo)), round(args.elo + elo_from_score(hi))],
        "seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "experiment_out"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "..", "experiment_out",
                        f"stockfish_rating_d{args.depth}_e{args.elo}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
