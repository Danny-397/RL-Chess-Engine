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
                 eval_every: int, eval_games: int, eval_sims: int,
                 material_weight: float = 0.0) -> Config:
    """Assemble a Config for one stage. Everything not set here is the default,
    so stages differ ONLY in the variable under test (compute, or the self-play
    material assist) — a clean comparison.

    ``material_weight`` blends a material signal into the search's leaf value
    (``value = (1-w)*network_value + w*material``). It is ``0.0`` for the pure
    baseline/scaled runs; the *assisted* ablation sets it > 0 during self-play to
    test whether a richer self-play signal — at fixed compute — breaks the draw
    cycle by producing decisive games the value head can actually learn from.
    """
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
    cfg.mcts.material_weight = material_weight
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
        "material_weight": cfg.mcts.material_weight,
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
def _row(m: Dict) -> str:
    return (f"| {m['name']} | {m['total_self_play_games']} | "
            f"{m['simulations']} | {m.get('material_weight', 0.0):.2f} | "
            f"{m['final_non_decisive_rate']:.0%} | "
            f"{m['final_checkmate_rate']:.0%} | "
            f"{m['eval_wins']}/{m['eval_draws']}/{m['eval_losses']} | "
            f"{m['elo_vs_random']:+.0f} |")


def _verdict(ref: Dict, test: Dict, lever: str) -> str:
    """One honest verdict comparing a test run to the baseline on the two
    draw-cycle metrics."""
    nd_drop = ref["final_non_decisive_rate"] - test["final_non_decisive_rate"]
    elo_gain = test["elo_vs_random"] - ref["elo_vs_random"]
    if nd_drop > 0.01 and elo_gain > 0:
        return (f"**Supported.** {lever} lowered the non-decisive rate "
                f"({nd_drop:+.1%}) and raised Elo ({elo_gain:+.0f}).")
    if nd_drop > 0.01 or elo_gain > 0:
        return (f"**Partially supported.** Only one metric moved as predicted "
                f"(non-decisive {(-nd_drop):+.1%}, Elo {elo_gain:+.0f}).")
    return (f"**Not supported at this scale.** Neither metric moved as predicted "
            f"(non-decisive {(-nd_drop):+.1%}, Elo {elo_gain:+.0f}) — report honestly.")


def _combined_section(baseline: Dict, scaled: Dict, assisted: Dict,
                      combined: Dict | None, header: str) -> str:
    """Experiment C's block, or a note explaining that it wasn't run."""
    if combined is None:
        return (
            "\n## Experiment C — both levers together (not run)\n"
            "A and B each moved the metrics alone, so the open question is whether "
            "they *compound* or merely substitute. Run it with:\n"
            "```bash\npython run_experiment.py --device cuda --run-combined\n```\n"
        )

    # Does stacking the assist on top of scaled compute beat scaled compute alone?
    d_nd = scaled["final_non_decisive_rate"] - combined["final_non_decisive_rate"]
    d_elo = combined["elo_vs_random"] - scaled["elo_vs_random"]
    # ...and does it beat the assist alone?
    over_assisted = combined["elo_vs_random"] - assisted["elo_vs_random"]

    if d_elo > 0 and over_assisted > 0:
        verdict = (f"**The levers compound.** Combining beats scaled compute alone by "
                   f"{d_elo:+.0f} Elo and the fixed-compute assist alone by "
                   f"{over_assisted:+.0f} Elo, with non-decisive rate down a further "
                   f"{d_nd:.0%}. Neither lever was substituting for the other.")
    elif d_elo <= 0 and over_assisted <= 0:
        verdict = ("**The levers substitute.** Stacking them beats neither arm on its "
                   "own, so the assist and extra compute were fixing the same "
                   "bottleneck — a richer self-play signal — by different means.")
    else:
        better, worse = ("compute", "the assist") if d_elo <= 0 else ("the assist", "compute")
        verdict = (f"**Partial.** The combination improves on {worse} alone but not on "
                   f"{better} alone, so {better} is doing most of the work and the "
                   f"second lever adds little on top.")

    return f"""
## Experiment C — both levers together
Scaled compute *and* the self-play material assist. A and B each moved the
metrics on their own; this arm asks whether they compound or substitute.

{header}
{_row(scaled)}
{_row(assisted)}
{_row(combined)}

{verdict}
"""


def write_results(baseline: Dict, scaled: Dict, assisted: Dict,
                  combined: Dict | None = None) -> str:
    """Write a ready-to-paste results block and return its path.

    Experiment A (compute): baseline vs scaled — does *more compute* break the
    draw cycle?  Experiment B (signal, fixed compute): baseline vs assisted —
    does a richer self-play value signal break it *without* more compute?
    Reading the two together distinguishes the cause.

    Experiment C (both, optional): scaled compute *and* the material assist.
    A and B each moved the metrics on their own, which leaves the question they
    cannot answer — whether the two levers actually compound, or whether the
    assist just substitutes for compute and the two together buy nothing extra.
    C is the arm that settles it.
    """
    header = ("| Run | Self-play games | MCTS sims | Self-play material w | "
              "Non-decisive rate | Checkmate rate | Eval W/D/L | Elo vs random |\n"
              "|---|---|---|---|---|---|---|---|")

    ver_a = _verdict(baseline, scaled, "Scaling compute")
    ver_b = _verdict(baseline, assisted, "A self-play material assist (fixed compute)")

    # A small interpretation matrix so the causal reading is explicit.
    a_moved = (baseline["final_non_decisive_rate"] - scaled["final_non_decisive_rate"] > 0.01
               or scaled["elo_vs_random"] - baseline["elo_vs_random"] > 0)
    b_moved = (baseline["final_non_decisive_rate"] - assisted["final_non_decisive_rate"] > 0.01
               or assisted["elo_vs_random"] - baseline["elo_vs_random"] > 0)
    if a_moved and b_moved:
        reading = ("Both levers help — the plateau is driven by *both* limited compute "
                   "and an impoverished self-play signal (they compound).")
    elif b_moved and not a_moved:
        reading = ("The **signal**, not raw compute, dominates: enriching self-play "
                   "broke the cycle at fixed compute while extra compute alone did not.")
    elif a_moved and not b_moved:
        reading = ("**Compute** dominates: more self-play helped while the material "
                   "assist alone did not — consistent with the classic compute ceiling.")
    else:
        reading = ("Neither lever moved the metrics at this scale — the cause likely "
                   "lies elsewhere (e.g. encoding: missing move-history planes, §8).")

    md = f"""# Experiment results — causes of the draw cycle (auto-generated)

Two controlled experiments, identical code, isolating one variable each. Paste
the tables + verdicts into `docs/technical-report.md` §5.3.

## Experiment A — compute scaling
Does *more compute* break the draw cycle? (material assist = 0 in both)

{header}
{_row(baseline)}
{_row(scaled)}

{ver_a}

## Experiment B — self-play signal, at fixed compute
Does a richer self-play value signal break the draw cycle *without* more compute?
(same compute as baseline; only the self-play material weight differs)

{header}
{_row(baseline)}
{_row(assisted)}

{ver_b}

## Reading the two together
{reading}
{_combined_section(baseline, scaled, assisted, combined, header)}
Charts: `assets/term_baseline.png`, `assets/term_scaled.png`, `assets/term_assisted.png`,
and the matching `assets/progress_*.png`.
Raw CSVs: `experiment_out/term_baseline.csv`, `term_scaled.csv`, `term_assisted.csv`.
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
    # assisted-ablation knob (Experiment B): self-play material weight, fixed compute
    p.add_argument("--assist-weight", type=float, default=0.30,
                   help="self-play material blend for the fixed-compute ablation "
                        "(0 = off; the baseline always uses 0).")
    p.add_argument("--run-combined", action="store_true",
                   help="Also run Experiment C: scaled compute WITH the material "
                        "assist, to test whether the two levers compound. Costs "
                        "roughly as much as the scaled stage again.")
    p.add_argument("--skip-assisted", action="store_true",
                   help="run only Experiment A (compute scaling), skip the ablation.")
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
    # Experiment B: SAME compute as baseline, only the self-play material weight differs.
    assisted_cfg = build_config(
        args.device, args.base_iters, args.base_games, args.base_sims,
        args.workers, eval_every=max(1, args.base_iters // 5),
        eval_games=args.eval_games, eval_sims=args.eval_sims,
        material_weight=args.assist_weight)

    # Experiment C: scaled compute AND the assist — the only arm that can tell
    # whether the two levers compound or merely substitute for each other.
    combined_cfg = build_config(
        args.device, args.scaled_iters, args.scaled_games, args.scaled_sims,
        args.workers, eval_every=max(1, args.scaled_iters // 5),
        eval_games=args.eval_games, eval_sims=args.eval_sims,
        material_weight=args.assist_weight)

    baseline = run_stage("baseline", baseline_cfg, args.eval_games, args.eval_sims)
    scaled = run_stage("scaled", scaled_cfg, args.eval_games, args.eval_sims)
    if args.skip_assisted:
        assisted = dict(baseline)  # neutral placeholder so the write-up still renders
        assisted["name"] = "assisted (skipped)"
    else:
        assisted = run_stage("assisted", assisted_cfg, args.eval_games, args.eval_sims)

    combined = None
    if args.run_combined:
        combined = run_stage("combined", combined_cfg, args.eval_games, args.eval_sims)

    results_path = write_results(baseline, scaled, assisted, combined)

    print("\n" + "=" * 68)
    print("EXPERIMENT COMPLETE")
    print("=" * 68)
    for m in [x for x in (baseline, scaled, assisted, combined) if x is not None]:
        print(f"{m['name']:>20}: non-decisive {m['final_non_decisive_rate']:.0%}, "
              f"Elo {m['elo_vs_random']:+.0f}, "
              f"material_w {m.get('material_weight', 0.0):.2f}, "
              f"{m['total_self_play_games']} games")
    print(f"\nResults written to {results_path}")
    print("Charts in assets/: term_{baseline,scaled,assisted}.png + progress_*.png")


if __name__ == "__main__":
    main()
