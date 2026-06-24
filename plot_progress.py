"""plot_progress.py
==================

Turn a training run's log into a picture.

``training.py`` appends a line per iteration to ``logs/training.log`` (loss
breakdown) plus periodic evaluation lines (win rate / Elo vs. the random
baseline).  This script parses that log and renders two charts:

* **Training loss** (total / policy / value) versus iteration -- shows the
  network fitting the self-play data better over time.
* **Estimated Elo vs. random** versus iteration -- the bottom line: is the engine
  actually getting *stronger*?

Usage::

    python plot_progress.py                       # reads logs/training.log
    python plot_progress.py --log path/to.log --out assets/progress.png

The result is a single PNG, ideal for a README or a project write-up.  Parsing is
deliberately tolerant of both the older and newer log line formats.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Tuple

# Regexes for the two kinds of lines we care about.  ``.*?`` keeps them robust to
# extra fields that were added to the log format over time (e.g. ``games=``).
_ITER_RE = re.compile(
    r"\[iter\s+(\d+)/\d+\].*?loss=([\d.]+)\s*\(policy=([\d.]+),\s*value=([\d.]+)\)"
)
_EVAL_RE = re.compile(r"\[eval\].*?score\s+([\d.]+)%.*?Elo\s+([+-]?\d+)")


def parse_log(path: str):
    """Parse a training log into per-iteration series.

    Returns:
        A tuple ``(iters, total, policy, value, eval_iters, elo)`` of parallel
        lists.  Evaluation points are sparser than iterations, so they get their
        own x-axis (``eval_iters``); each eval line is attributed to the most
        recent iteration seen above it.
    """
    iters: List[int] = []
    total: List[float] = []
    policy: List[float] = []
    value: List[float] = []
    eval_iters: List[int] = []
    elo: List[float] = []

    last_iter: Optional[int] = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            # The log is append-only and may hold several runs.  Reset at each
            # "Training start" banner so we chart only the most recent run.
            if "Training start" in line:
                iters.clear(); total.clear(); policy.clear(); value.clear()
                eval_iters.clear(); elo.clear()
                last_iter = None
                continue
            m = _ITER_RE.search(line)
            if m:
                last_iter = int(m.group(1))
                iters.append(last_iter)
                total.append(float(m.group(2)))
                policy.append(float(m.group(3)))
                value.append(float(m.group(4)))
                continue
            e = _EVAL_RE.search(line)
            if e and last_iter is not None:
                eval_iters.append(last_iter)
                elo.append(float(e.group(2)))

    return iters, total, policy, value, eval_iters, elo


def make_plot(path: str, out: str) -> str:
    """Parse ``path`` and write the progress chart to ``out``. Returns ``out``."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend -- no display needed
    import matplotlib.pyplot as plt

    iters, total, policy, value, eval_iters, elo = parse_log(path)
    if not iters:
        raise SystemExit(f"No iteration lines found in {path!r} yet -- "
                         "let training run for at least one iteration.")

    has_elo = len(elo) > 0
    fig, axes = plt.subplots(1, 2 if has_elo else 1, figsize=(12 if has_elo else 6, 4.5))
    ax_loss = axes[0] if has_elo else axes

    ax_loss.plot(iters, total, label="total", color="#1f77b4", linewidth=2)
    ax_loss.plot(iters, policy, label="policy", color="#ff7f0e", linestyle="--")
    ax_loss.plot(iters, value, label="value", color="#2ca02c", linestyle=":")
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    if has_elo:
        ax_elo = axes[1]
        ax_elo.axhline(0, color="#999", linewidth=1)
        ax_elo.plot(eval_iters, elo, marker="o", color="#d62728", linewidth=2)
        ax_elo.set_title("Estimated Elo vs. random baseline")
        ax_elo.set_xlabel("iteration")
        ax_elo.set_ylabel("Elo")
        ax_elo.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  ({len(iters)} iterations"
          f"{', ' + str(len(elo)) + ' eval points' if has_elo else ''})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training progress from a log.")
    parser.add_argument("--log", default=os.path.join("logs", "training.log"),
                        help="path to the training log (default: logs/training.log)")
    parser.add_argument("--out", default=os.path.join("assets", "training_progress.png"),
                        help="output PNG path (default: assets/training_progress.png)")
    args = parser.parse_args()
    make_plot(args.log, args.out)


if __name__ == "__main__":
    main()
