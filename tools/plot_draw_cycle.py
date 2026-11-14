"""Draw the draw-cycle figure from the committed data, so it can be rebuilt.

Inputs (both committed):
  docs/data/gpu_run_2026-07-14.json   per-iteration self-play terminations and
                                      the periodic 50-game evaluations vs random
  docs/data/no_learning_control.json  untrained networks vs random (optional)

    python tools/plot_draw_cycle.py     # -> assets/draw_cycle.png (+ web copy)

Left panel: share of self-play games not ending in checkmate, per iteration.
Right panel: Elo vs random at each evaluation checkpoint with a 95% interval
(normal approximation on the per-game score, see
evaluation.MatchResult.score_interval), and the untrained-network controls as
horizontal reference bands.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation import MatchResult  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUN = os.path.join(ROOT, "docs", "data", "gpu_run_2026-07-14.json")
# Prefer the paired control: the seed-42 network every GPU arm started from.
START = os.path.join(ROOT, "docs", "data", "start_network_control.json")
CONTROL = os.path.join(ROOT, "docs", "data", "no_learning_control.json")
OUT = os.path.join(ROOT, "assets", "draw_cycle.png")
WEB_COPY = os.path.join(ROOT, "web", "static", "draw_cycle.png")

# The site's palette: walnut ground, maple text, three muted series colours.
BG, INK, INK2, RULE = "#1d140e", "#efe3cb", "#a8937155", "#3a2b1f"
COLOURS = {"baseline": "#d9c39a", "scaled": "#8fa9b8", "assisted": "#c9a368"}
LABELS = {"baseline": "baseline (100 games)", "scaled": "10x compute (1,000 games)",
          "assisted": "material assist w=0.30 (100 games)"}


def main() -> int:
    run = json.load(open(RUN, encoding="utf-8"))
    src = START if os.path.exists(START) else CONTROL if os.path.exists(CONTROL) else None
    control = json.load(open(src, encoding="utf-8")) if src else None
    who = "starting net" if src == START else "untrained nets"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": "#c4b08f", "xtick.color": "#8f7c60", "ytick.color": "#8f7c60",
        "axes.edgecolor": RULE, "axes.facecolor": BG, "figure.facecolor": BG,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

    for name, st in run["stages"].items():
        it = [t["iter"] for t in st["self_play_termination"]]
        games = [sum(v for k, v in t.items() if k != "iter") for t in st["self_play_termination"]]
        nondec = [100 * (1 - t["checkmate"] / g) for t, g in zip(st["self_play_termination"], games)]
        a.plot(it, nondec, color=COLOURS[name], lw=1.6, marker="o", ms=3, label=LABELS[name])

        xs, ys, lo, hi = [], [], [], []
        for c in st["eval_checkpoints"]:
            m = MatchResult(c["wins"], c["draws"], c["losses"], [])
            l, h = m.elo_interval()
            xs.append(c["iter"]); ys.append(m.elo_difference); lo.append(m.elo_difference - l); hi.append(h - m.elo_difference)
        off = {"baseline": -0.25, "scaled": 0.0, "assisted": 0.25}[name]
        b.errorbar([x + off for x in xs], ys, yerr=[lo, hi], color=COLOURS[name], lw=1.4, marker="o", ms=3.5,
                   capsize=2.5, elinewidth=0.9)

    a.set_title("Self-play games not ending in checkmate", loc="left", fontsize=10.5, color=INK, pad=10)
    a.set_xlabel("training iteration"); a.set_ylabel("%"); a.set_ylim(0, 103)
    a.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor="#c4b08f")

    if control:
        for arm in control["arms"]:
            lo_, hi_ = arm["elo_95ci"]
            w = arm["material_weight"]
            col = "#c9a368" if w > 0 else "#d9c39a"
            b.axhspan(lo_, hi_, color=col, alpha=0.10, lw=0)
            b.axhline(arm["elo"], color=col, lw=1.0, ls=":", label=f"{who}, w={w:.2f}")
        b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5,
                 labelcolor="#c4b08f", ncol=2, title="starting network: dotted line = estimate, band = 95% CI", title_fontsize=7.5)
    b.set_title("Elo vs random at each checkpoint (95% interval, 50 games)", loc="left", fontsize=10.5, color=INK, pad=10)
    b.set_xlabel("training iteration"); b.set_ylabel("Elo")

    fig.tight_layout(w_pad=3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    shutil.copyfile(OUT, WEB_COPY)
    print("wrote", OUT, "and", WEB_COPY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
