"""No-learning control for Experiment B: what does the material blend buy an
*untrained* network?

Experiment B evaluated the assisted network with the same search it trained
with, so its search blended ``material_weight = 0.30`` of a material count into
every leaf value, while the baseline network was evaluated with no blend. Its
"Elo vs random" therefore measures network + heuristic, and the gap to the
baseline mixes two effects: what the network learned, and what the heuristic
does on its own.

This script measures the second effect directly. It plays freshly initialised
(never trained) networks against ``RandomAgent`` under the same evaluation
settings as Experiment B (50 MCTS simulations per move, 200-move cap, colours
alternated), once with ``material_weight = 0`` and once with ``0.30``. Several
initialisation seeds are pooled so the result does not hinge on one random
network.

    python tools/no_learning_control.py --games 50 --seeds 3 --workers 3

Writes ``docs/data/no_learning_control.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.set_num_threads(1)

from config import Config  # noqa: E402
from evaluation import MatchResult, evaluate_against_random  # noqa: E402
from model import ChessNet  # noqa: E402


def run_one(task: tuple[float, int, int, int]) -> dict:
    """One (weight, seed) cell: a fresh random-init network vs RandomAgent."""
    weight, seed, games, sims = task
    torch.set_num_threads(1)
    cfg = Config()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = ChessNet(cfg.network).eval()          # random init, no training
    t0 = time.time()
    m = evaluate_against_random(net, cfg, num_games=games,
                                num_simulations=sims, material_weight=weight)
    print(f"w={weight:.2f} seed={seed}: +{m.wins} ={m.draws} -{m.losses}", flush=True)
    return {"weight": weight, "seed": seed, "wins": m.wins, "draws": m.draws,
            "losses": m.losses, "seconds": round(time.time() - t0, 1)}


def summarise(weight: float, cells: list[dict]) -> dict:
    per_seed = [{k: c[k] for k in ("seed", "wins", "draws", "losses", "seconds")}
                for c in cells if c["weight"] == weight]
    total = MatchResult(sum(c["wins"] for c in per_seed),
                        sum(c["draws"] for c in per_seed),
                        sum(c["losses"] for c in per_seed), [])
    lo, hi = total.elo_interval()
    return {
        "material_weight": weight,
        "games": total.games,
        "wins": total.wins, "draws": total.draws, "losses": total.losses,
        "score": round(total.score, 4),
        "elo": round(total.elo_difference),
        "elo_95ci": [round(lo), round(hi)],
        "per_seed": per_seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=50, help="games per seed per arm")
    ap.add_argument("--seeds", type=int, default=3, help="number of init seeds (1..N)")
    ap.add_argument("--seed-values", default=None,
                    help="comma-separated init seeds instead, e.g. 42: the seed every "
                         "GPU arm trained from, so this rebuilds their shared starting network")
    ap.add_argument("--sims", type=int, default=50, help="MCTS sims per move (Exp. B eval used 50)")
    ap.add_argument("--weight", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join("docs", "data", "no_learning_control.json"))
    args = ap.parse_args()

    seeds = ([int(x) for x in args.seed_values.split(",")] if args.seed_values
             else list(range(1, args.seeds + 1)))
    t0 = time.time()
    tasks = [(w, s, args.games, args.sims) for w in (0.0, args.weight) for s in seeds]
    with multiprocessing.Pool(args.workers) as pool:
        cells = pool.map(run_one, tasks, chunksize=1)
    arms = [summarise(w, cells) for w in (0.0, args.weight)]
    out = {
        "what": "untrained (random-init) ChessNet + MCTS vs RandomAgent",
        "network": "4 residual blocks x 64 channels, random init, no training",
        "sims_per_move": args.sims,
        "max_moves": Config().training.max_moves,
        "seeds": seeds,
        "games_per_seed": args.games,
        "arms": arms,
        "seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for a in arms:
        print(f"w={a['material_weight']:.2f}: +{a['wins']} ={a['draws']} -{a['losses']} "
              f"score {a['score']:.3f}  Elo {a['elo']:+d}  95% CI {a['elo_95ci']}")
    print(f"wrote {args.out} ({out['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
