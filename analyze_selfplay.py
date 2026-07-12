"""analyze_selfplay.py
======================

Turn a training run's archived self-play games into the **draw-cycle evidence**
the technical report needs (see ``docs/technical-report.md`` §5 and §6).

``main.py train --save-pgn`` writes one PGN file per iteration to the ``pgn/``
directory (``selfplay_iterNNN.pgn``), each holding every self-play game of that
iteration.  A game's *final position* tells us exactly why it ended, so this
script needs no changes to the training loop: it replays each game and classifies
its termination.

For every iteration it reports how many games ended in each of:

* ``checkmate``            -- a decisive result (the engine *converted* a win)
* ``stalemate``
* ``insufficient_material``
* ``repetition``           -- threefold / fivefold
* ``fifty_move``           -- fifty- / seventyfive-move rule
* ``max_moves_cap``        -- hit the self-play length cap without a rules result

and the headline **non-decisive rate** = (games that were *not* a checkmate) /
games.  A run stuck in the *draw cycle* shows a non-decisive rate near 1.0
dominated by ``repetition`` / ``fifty_move`` / ``max_moves_cap``; a run that is
escaping it shows ``checkmate`` rising and that rate falling.

Usage::

    # analyse the default pgn/ directory, write a CSV and a chart
    python analyze_selfplay.py

    # compare two runs by giving each its own pgn dir + labelled outputs
    python analyze_selfplay.py --pgn-dir pgn_baseline --label baseline \
        --out-csv logs/term_baseline.csv --plot assets/term_baseline.png
    python analyze_selfplay.py --pgn-dir pgn_scaled   --label scaled \
        --out-csv logs/term_scaled.csv   --plot assets/term_scaled.png

The CSV columns are stable and plug straight into the report's §5.3 table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn


# The termination buckets, in the order they appear in the CSV / chart.  The
# first ("checkmate") is the only *decisive* outcome; the rest are the draw-cycle
# symptoms we want to watch shrink as compute scales.
BUCKETS = [
    "checkmate",
    "stalemate",
    "insufficient_material",
    "repetition",
    "fifty_move",
    "max_moves_cap",
    "other",
]

_ITER_RE = re.compile(r"selfplay_iter(\d+)\.pgn$")


def classify_termination(board: chess.Board) -> str:
    """Classify why a finished self-play game ended, from its final position.

    ``claim_draw=True`` so threefold-repetition and fifty-move *claims* count as
    draws (self-play stops on them).  A position that is not terminal under the
    rules means the game was cut off by the self-play length cap.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "max_moves_cap"

    T = chess.Termination
    mapping = {
        T.CHECKMATE: "checkmate",
        T.STALEMATE: "stalemate",
        T.INSUFFICIENT_MATERIAL: "insufficient_material",
        T.THREEFOLD_REPETITION: "repetition",
        T.FIVEFOLD_REPETITION: "repetition",
        T.FIFTY_MOVES: "fifty_move",
        T.SEVENTYFIVE_MOVES: "fifty_move",
    }
    return mapping.get(outcome.termination, "other")


def analyse_pgn_file(path: str) -> Dict[str, int]:
    """Tally termination buckets over every game in one PGN file."""
    counts: Dict[str, int] = {b: 0 for b in BUCKETS}
    counts["games"] = 0
    with open(path, encoding="utf-8") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.end().board()  # replay to the final position
            counts[classify_termination(board)] += 1
            counts["games"] += 1
    return counts


def iter_number(path: str) -> int:
    """Extract the iteration index from a ``selfplay_iterNNN.pgn`` filename."""
    m = _ITER_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0


def analyse_dir(pgn_dir: str) -> "OrderedDict[int, Dict[str, int]]":
    """Analyse every ``selfplay_iterNNN.pgn`` in ``pgn_dir``, ordered by iter."""
    files = sorted(glob.glob(os.path.join(pgn_dir, "selfplay_iter*.pgn")),
                   key=iter_number)
    if not files:
        raise SystemExit(
            f"No 'selfplay_iter*.pgn' files found in '{pgn_dir}'. "
            f"Did you run training with --save-pgn?"
        )
    per_iter: "OrderedDict[int, Dict[str, int]]" = OrderedDict()
    for path in files:
        per_iter[iter_number(path)] = analyse_pgn_file(path)
    return per_iter


def non_decisive_rate(row: Dict[str, int]) -> float:
    """Fraction of games that did NOT end in checkmate (the draw-cycle metric)."""
    games = row["games"]
    return 0.0 if games == 0 else (games - row["checkmate"]) / games


def write_csv(per_iter: "OrderedDict[int, Dict[str, int]]", out_csv: str,
              label: Optional[str]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    header = (["label", "iteration", "games", "non_decisive_rate"] + BUCKETS)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for it, row in per_iter.items():
            w.writerow(
                [label or "", it, row["games"], f"{non_decisive_rate(row):.4f}"]
                + [row[b] for b in BUCKETS]
            )
    print(f"Wrote {out_csv}  ({len(per_iter)} iterations)")


def print_table(per_iter: "OrderedDict[int, Dict[str, int]]", label: Optional[str]) -> None:
    tag = f" [{label}]" if label else ""
    print(f"\nSelf-play termination by iteration{tag}")
    print(f"{'iter':>4} {'games':>6} {'nondec%':>8} {'mate':>5} {'stale':>5} "
          f"{'insuf':>5} {'rep':>5} {'50mv':>5} {'cap':>5}")
    for it, r in per_iter.items():
        print(f"{it:>4} {r['games']:>6} {non_decisive_rate(r)*100:>7.1f}% "
              f"{r['checkmate']:>5} {r['stalemate']:>5} {r['insufficient_material']:>5} "
              f"{r['repetition']:>5} {r['fifty_move']:>5} {r['max_moves_cap']:>5}")
    # Whole-run summary line.
    tot = {b: sum(r[b] for r in per_iter.values()) for b in BUCKETS}
    tot["games"] = sum(r["games"] for r in per_iter.values())
    print(f"{'ALL':>4} {tot['games']:>6} {non_decisive_rate(tot)*100:>7.1f}% "
          f"{tot['checkmate']:>5} {tot['stalemate']:>5} {tot['insufficient_material']:>5} "
          f"{tot['repetition']:>5} {tot['fifty_move']:>5} {tot['max_moves_cap']:>5}")


def make_plot(per_iter: "OrderedDict[int, Dict[str, int]]", out_png: str,
              label: Optional[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(skipping plot: matplotlib unavailable: {exc})")
        return

    iters = list(per_iter.keys())
    nd_rate = [non_decisive_rate(per_iter[i]) * 100 for i in iters]
    mate = [per_iter[i]["checkmate"] for i in iters]
    games = [per_iter[i]["games"] for i in iters]
    mate_pct = [100 * m / g if g else 0 for m, g in zip(mate, games)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, nd_rate, marker="o", label="non-decisive rate (draw cycle)")
    ax.plot(iters, mate_pct, marker="s", label="checkmate rate (conversion)")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("% of self-play games")
    ax.set_ylim(0, 100)
    ax.set_title(f"Self-play outcomes{f' — {label}' if label else ''}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pgn-dir", default="pgn",
                   help="directory of selfplay_iterNNN.pgn files (default: pgn)")
    p.add_argument("--out-csv", default="logs/selfplay_termination.csv",
                   help="where to write the per-iteration CSV")
    p.add_argument("--plot", default=None,
                   help="optional path for a PNG chart (e.g. assets/termination.png)")
    p.add_argument("--label", default=None,
                   help="tag for this run (e.g. 'baseline' or 'scaled')")
    args = p.parse_args()

    per_iter = analyse_dir(args.pgn_dir)
    print_table(per_iter, args.label)
    write_csv(per_iter, args.out_csv, args.label)
    if args.plot:
        make_plot(per_iter, args.plot, args.label)


if __name__ == "__main__":
    main()
