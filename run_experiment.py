"""run_experiment.py
====================

One-command driver for the **compute-scaling experiment** behind the technical
report (``docs/technical-report.md`` §5.3 and §6).

It runs the *identical* AlphaZero pipeline at two compute scales — a small
**baseline** and a larger **scaled** run on GPU — and compares two metrics over
training:

* the **non-decisive rate** of self-play games (the *draw cycle* — fraction of
  games that did NOT end in checkmate), and
* **Elo vs. a random baseline**.

If the scaled run's non-decisive rate falls and its Elo rises where the baseline
flatlines, the "plateau is a compute ceiling, not a bug" hypothesis is supported.

The script writes all artifacts under ``experiment_out/`` and a ready-to-paste
results file at ``docs/experiment-results.md``. It is **resumable**: each stage
drops a ``<stage>.result.json`` marker and is skipped on re-run, so a dropped
Colab session can just re-run the same command.

Usage
-----
    # validate the whole pipeline fast on CPU first (~1-2 min):
    python run_experiment.py --quick

    # the real run on a GPU box (Colab):
    python run_experiment.py --device cuda

    # scale up if you have the budget:
    python run_experiment.py --device cuda \
        --scaled-iters 40 --scaled-games 100 --scaled-sims 200

Everything is a knob; run ``python run_experiment.py --help`` for the full list.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

from config import Config
import analyze_selfplay as asp


OUT_DIR = "experiment_out"


# --------------------------------------------------------------------------- #
# One stage (baseline or scaled)
# --------------------------------------------------------------------------- #
def build_config(device: str, iters: int, games: int, sims: int, workers: int,
                 eval_every: int, eval_games: int, eval_sims: int) -> Config:
    """Assemble a Config for one stage. Everything not set here is the default,
    so the two stages differ ONLY in compute — a clean comparison."""
    cfg = Config()
    cfg.training.device = device
    cfg.training.num_iterations = iters
    cfg.training.games_per_iteration = games
    cfg.training.num_self_play_workers = workers
    cfg.training.save_self_play_pgn = True          # needed by analyze_selfplay
    cfg.training.eval_every = eval_every            # Elo curve in the log
    cfg.training.eval_games = eval_games
    cfg.training.eval_simulations = eval_sims
    cfg.mcts.num_simulations = sims
    return cfg


def _plot_progress(out_png: str) -> None:
    """Render loss + Elo curves from the (current) training log. Best-effort."""
    try:
        subprocess.run(
            [sys.executable, "plot_progress.py",
             "--log", os.path.join("logs", "training.log"), "--out", out_png],
            check=True, capture_output=True, text=True,
        )
        print(f"  wrote {out_png}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  (plot_progress skipped: {exc})")


def run_stage(name: str, cfg: Config, eval_games: int, eval_sims: int) -> Dict:
    """Train, evaluate, archive PGNs, and analyse one stage. Returns a metrics
    dict and caches it to ``experiment_out/<name>.result.json`` (resumable)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    marker = os.path.join(OUT_DIR, f"{name}.result.json")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            cached = json.load(fh)
        print(f"[{name}] already complete — skipping (delete {marker} to redo).")
        return cached

    # Import here so a plain --help / --quick dry parse doesn't load torch.
    from training import train
    from evaluation import evaluate_against_random

    device = cfg.resolved_device()
    print(f"\n=== STAGE: {name}  (device={device}, "
          f"{cfg.training.num_iterations} iters x {cfg.training.games_per_iteration} games, "
          f"{cfg.mcts.num_simulations} sims) ===")

    # Start each stage from a clean pgn/ dir so files don't mix between stages.
    if os.path.isdir(cfg.training.pgn_dir):
        shutil.rmtree(cfg.training.pgn_dir)

    t0 = time.time()
    model = train(cfg)
    train_secs = time.time() - t0

    print(f"[{name}] evaluating vs random ({eval_games} games)...")
    match = evaluate_against_random(model, cfg, num_games=eval_games,
                                    num_simulations=eval_sims)

    # Archive this stage's PGNs and training log, then analyse the games.
    pgn_dst = os.path.join(OUT_DIR, f"pgn_{name}")
    if os.path.isdir(pgn_dst):
        shutil.rmtree(pgn_dst)
    shutil.move(cfg.training.pgn_dir, pgn_dst)
    shutil.copy(os.path.join("logs", "training.log"),
                os.path.join(OUT_DIR, f"train_{name}.log"))

    per_iter = asp.analyse_dir(pgn_dst)
    asp.print_table(per_iter, name)
    asp.write_csv(per_iter, os.path.join(OUT_DIR, f"term_{name}.csv"), name)
    asp.make_plot(per_iter, os.path.join("assets", f"term_{name}.png"), name)
    _plot_progress(os.path.join("assets", f"progress_{name}.png"))

    last_it = max(per_iter)
    final_row = per_iter[last_it]
    metrics = {
        "name": name,
        "device": device,
        "iters": cfg.training.num_iterations,
        "games_per_iter": cfg.training.games_per_iteration,
        "simulations": cfg.mcts.num_simulations,
        "total_self_play_games": sum(r["games"] for r in per_iter.values()),
        "final_non_decisive_rate": round(asp.non_decisive_rate(final_row), 4),
        "final_checkmate_rate": round(
            final_row["checkmate"] / final_row["games"] if final_row["games"] else 0.0, 4),
        "eval_wins": match.wins,
        "eval_draws": match.draws,
        "eval_losses": match.losses,
        "eval_score": round(match.score, 4),
        "elo_vs_random": round(match.elo_difference, 1),
        "train_seconds": round(train_secs, 1),
    }
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[{name}] done in {train_secs/60:.1f} min — "
          f"non-decisive {metrics['final_non_decisive_rate']:.0%}, "
          f"Elo {metrics['elo_vs_random']:+.0f}")
    return metrics


# --------------------------------------------------------------------------- #
# Results write-up
# --------------------------------------------------------------------------- #
def write_results(baseline: Dict, scaled: Dict) -> str:
    """Write a ready-to-paste §5.3 results block and return its path."""
    nd_drop = baseline["final_non_decisive_rate"] - scaled["final_non_decisive_rate"]
    elo_gain = scaled["elo_vs_random"] - baseline["elo_vs_random"]
    if nd_drop > 0.01 and elo_gain > 0:
        verdict = ("**Hypothesis supported.** Scaling compute lowered the self-play "
                   "non-decisive rate and raised Elo vs. random, with the code "
                   "unchanged — consistent with a compute ceiling rather than an "
                   "algorithmic bug.")
    elif nd_drop > 0.01 or elo_gain > 0:
        verdict = ("**Hypothesis partially supported.** One of the two metrics moved "
                   "in the predicted direction; more compute is likely needed for a "
                   "clear reversal, but the direction is consistent with the "
                   "compute-ceiling explanation.")
    else:
        verdict = ("**Hypothesis not supported at this scale.** Neither metric moved "
                   "as predicted; report this honestly — either more compute is "
                   "required, or the plateau has an additional cause (e.g. missing "
                   "move-history planes, §8).")

    def row(m: Dict) -> str:
        return (f"| {m['name']} | {m['total_self_play_games']} | "
                f"{m['simulations']} | {m['final_non_decisive_rate']:.0%} | "
                f"{m['final_checkmate_rate']:.0%} | "
                f"{m['eval_wins']}/{m['eval_draws']}/{m['eval_losses']} | "
                f"{m['elo_vs_random']:+.0f} |")

    md = f"""# Experiment results — compute-scaling test (auto-generated)

Paste this table and verdict into `docs/technical-report.md` §5.3.

| Run | Self-play games | MCTS sims | Non-decisive rate | Checkmate rate | Eval W/D/L vs random | Elo vs random |
|---|---|---|---|---|---|---|
{row(baseline)}
{row(scaled)}

**Change (scaled − baseline):** non-decisive rate {(-nd_drop):+.1%}, Elo {elo_gain:+.0f}.

{verdict}

Charts: `assets/term_baseline.png`, `assets/term_scaled.png`,
`assets/progress_baseline.png`, `assets/progress_scaled.png`.
Raw per-iteration CSVs: `experiment_out/term_baseline.csv`, `experiment_out/term_scaled.csv`.
"""
    os.makedirs("docs", exist_ok=True)
    path = os.path.join("docs", "experiment-results.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'.")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel self-play workers (1 = sequential, safest on GPU).")
    p.add_argument("--quick", action="store_true",
                   help="tiny CPU run to validate the whole pipeline end-to-end.")
    # baseline knobs
    p.add_argument("--base-iters", type=int, default=10)
    p.add_argument("--base-games", type=int, default=10)
    p.add_argument("--base-sims", type=int, default=100)
    # scaled knobs
    p.add_argument("--scaled-iters", type=int, default=25)
    p.add_argument("--scaled-games", type=int, default=40)
    p.add_argument("--scaled-sims", type=int, default=160)
    # eval knobs
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--eval-sims", type=int, default=50)
    args = p.parse_args()

    if args.quick:
        args.device = "cpu"
        args.base_iters, args.base_games, args.base_sims = 2, 3, 10
        args.scaled_iters, args.scaled_games, args.scaled_sims = 3, 4, 12
        args.eval_games, args.eval_sims = 6, 10
        print(">>> QUICK MODE: tiny CPU run to validate the pipeline (results are "
              "not meaningful).")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # Warn early if CUDA was requested but is unavailable.
    if args.device in ("cuda", "auto") and not args.quick:
        try:
            import torch
            if not torch.cuda.is_available():
                print("WARNING: CUDA not available — this will run on CPU and be slow. "
                      "Stop (Ctrl-C) and fix the runtime if you meant to use a GPU.")
        except Exception:
            pass

    baseline_cfg = build_config(
        args.device, args.base_iters, args.base_games, args.base_sims,
        args.workers, eval_every=max(1, args.base_iters // 5),
        eval_games=args.eval_games, eval_sims=args.eval_sims)
    scaled_cfg = build_config(
        args.device, args.scaled_iters, args.scaled_games, args.scaled_sims,
        args.workers, eval_every=max(1, args.scaled_iters // 5),
        eval_games=args.eval_games, eval_sims=args.eval_sims)

    baseline = run_stage("baseline", baseline_cfg, args.eval_games, args.eval_sims)
    scaled = run_stage("scaled", scaled_cfg, args.eval_games, args.eval_sims)

    results_path = write_results(baseline, scaled)

    print("\n" + "=" * 68)
    print("EXPERIMENT COMPLETE")
    print("=" * 68)
    print(f"baseline: non-decisive {baseline['final_non_decisive_rate']:.0%}, "
          f"Elo {baseline['elo_vs_random']:+.0f}, "
          f"{baseline['total_self_play_games']} games")
    print(f"scaled:   non-decisive {scaled['final_non_decisive_rate']:.0%}, "
          f"Elo {scaled['elo_vs_random']:+.0f}, "
          f"{scaled['total_self_play_games']} games")
    print(f"\nResults written to {results_path}")
    print("Charts in assets/: term_baseline.png, term_scaled.png, "
          "progress_baseline.png, progress_scaled.png")


if __name__ == "__main__":
    main()
