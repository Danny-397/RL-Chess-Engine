"""Summarise the three-seed rerun: every trained network against its own starting network.

Reads what ``notebooks/seeds_rerun.ipynb`` leaves in ``experiment_out/``:

* ``seed{N}/{baseline,assisted}.result.json`` (from ``run_experiment.py --seed N``)
* ``seed1-legacy-draws/term_baseline.csv`` and ``seed{N}/term_*.csv``
* ``start_random.json`` / ``start_greedy.json`` (``tools/no_learning_control.py --seed-values 1,2,3``)
* ``value_probe.json`` (``tools/value_probe.py``)

and writes one machine-readable file, by default ``docs/data/seeds_rerun_2026-11.json``.

    python tools/summarise_seeds.py

"Change" is trained Elo minus starting-network Elo under the same evaluation, with a
95% interval from the two standard errors combined; "pooled" adds the games of all
three seeds before converting to Elo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import MatchResult  # noqa: E402

SEEDS = (1, 2, 3)
ARMS = ("baseline", "assisted")
EVALS = (  # label, result-file prefix, start opponent, start weight (None = the arm's own)
    ("random, own search", "eval", "random", None),
    ("random, w=0", "pure_eval", "random", 0.0),
    ("greedy, w=0", "greedy_eval", "greedy", 0.0),
)


def se(m: MatchResult) -> float:
    lo, hi = m.elo_interval()
    return (hi - lo) / (2 * 1.96)


def change(trained: MatchResult, start: MatchResult) -> dict:
    d = trained.elo_difference - start.elo_difference
    e = math.hypot(se(trained), se(start))
    return {"trained": round(trained.elo_difference), "start": round(start.elo_difference),
            "change": round(d), "ci": [round(d - 1.96 * e), round(d + 1.96 * e)],
            "trained_wdl": [trained.wins, trained.draws, trained.losses],
            "start_wdl": [start.wins, start.draws, start.losses]}


def add(a: MatchResult, b: MatchResult) -> MatchResult:
    return MatchResult(a.wins + b.wins, a.draws + b.draws, a.losses + b.losses, [])


def termination(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    games = sum(int(r["games"]) for r in rows)
    mates = sum(int(r["checkmate"]) for r in rows)
    rep = sum(int(r["repetition"]) for r in rows)
    late = rows[2:]  # iterations 3-10
    late_games = sum(int(r["games"]) for r in late)
    late_nd = sum(int(r["games"]) - int(r["checkmate"]) for r in late)
    return {"games": games, "checkmates": mates, "repetitions": rep,
            "non_decisive": round(1 - mates / games, 3),
            "non_decisive_iter3_10": round(late_nd / late_games, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="experiment_out")
    ap.add_argument("--out", default=os.path.join("docs", "data", "seeds_rerun_2026-11.json"))
    args = ap.parse_args()
    d = args.dir

    start = {o: json.load(open(os.path.join(d, f"start_{o}.json"))) for o in ("random", "greedy")}

    def start_cell(opp: str, w: float, seed: int) -> MatchResult:
        arm = next(a for a in start[opp]["arms"] if abs(a["material_weight"] - w) < 1e-9)
        c = next(p for p in arm["per_seed"] if p["seed"] == seed)
        return MatchResult(c["wins"], c["draws"], c["losses"], [])

    per_seed, pooled = [], []
    for arm in ARMS:
        for label, key, opp, sw in EVALS:
            tot_t = tot_s = MatchResult(0, 0, 0, [])
            for s in SEEDS:
                r = json.load(open(os.path.join(d, f"seed{s}", f"{arm}.result.json")))
                w = r["material_weight"] if sw is None else sw
                t = MatchResult(r[f"{key}_wins"], r[f"{key}_draws"], r[f"{key}_losses"], [])
                st = start_cell(opp, w, s)
                per_seed.append({"seed": s, "arm": arm, "eval": label, **change(t, st)})
                tot_t, tot_s = add(tot_t, t), add(tot_s, st)
            pooled.append({"arm": arm, "eval": label, **change(tot_t, tot_s)})

    term = {f"seed{s}/{a}": termination(os.path.join(d, f"seed{s}", f"term_{a}.csv"))
            for s in SEEDS for a in ARMS}
    term["seed1-legacy-draws/baseline"] = termination(
        os.path.join(d, "seed1-legacy-draws", "term_baseline.csv"))
    legacy = json.load(open(os.path.join(d, "seed1-legacy-draws", "baseline.result.json")))

    probe = json.load(open(os.path.join(d, "value_probe.json")))

    out = {
        "what": "three training seeds per arm (baseline, material-assisted w=0.30), corrected draw rule, "
                "each network measured against its own starting network",
        "hardware": "Kaggle T4, 2026-11-15/16 (training); starting-network controls on CPU",
        "training": "10 iterations x 10 self-play games x 100 sims per arm; final evals 50 games, 50 sims",
        "pooled": pooled,
        "per_seed": per_seed,
        "self_play_termination": term,
        "legacy_rule_seed1_baseline_eval": {k: legacy[k] for k in legacy if "eval" in k},
        "value_probe": probe,
        "start_controls": start,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    for x in pooled:
        print(f"POOLED {x['arm']:<9} {x['eval']:<19} trained {x['trained']:+5d}  start {x['start']:+5d}  "
              f"change {x['change']:+5d}  95% CI {x['ci']}")
    for x in per_seed:
        print(f"seed {x['seed']} {x['arm']:<9} {x['eval']:<19} trained {x['trained']:+5d}  "
              f"start {x['start']:+5d}  change {x['change']:+5d}  95% CI {x['ci']}")
    for k, v in term.items():
        print(f"{k:<28} {v}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
