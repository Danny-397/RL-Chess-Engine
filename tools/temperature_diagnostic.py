"""Diagnostic: does the greedy switch at ply 15 drive the repetitions?

Plays self-play games with a trained network under three temperature schedules
(no material blend) and reports how they end. Run from the repository root:

    python tools/temperature_diagnostic.py experiment_out/seed1/baseline.pt 12 out.json

The 2026-11-16 run (seed-1 rerun baseline, 12 games each) is in
docs/data/temperature_diagnostic_2026-11-16.json.
"""
import json
import multiprocessing as mp
import random
import sys

sys.path.insert(0, '.')

import numpy as np
import torch

from config import Config
from self_play import play_game
from training import load_checkpoint

SCHEDULES = {'tau1_to_ply15 (default)': 15, 'tau1_to_ply60': 60, 'tau1_always': 10_000}


def one(args):
    ckpt, label, tmoves, seed = args
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = Config()
    cfg.mcts.material_weight = 0.0
    cfg.training.temperature_moves = tmoves
    net = load_checkpoint(ckpt, cfg, device='cpu').eval()
    r = play_game(net, cfg, collect_pgn=True)
    import io, chess, chess.pgn
    g = chess.pgn.read_game(io.StringIO(r.pgn)); b = g.end().board()
    term = ('mate' if b.is_checkmate() else 'threefold' if b.is_repetition(3) else
            'stalemate' if b.is_stalemate() else 'insufficient' if b.is_insufficient_material() else
            'fifty' if b.halfmove_clock >= 100 else 'cap' if len(r.examples) >= cfg.training.max_moves else 'other')
    return {'schedule': label, 'seed': seed, 'result': r.result, 'plies': len(r.examples), 'termination': term}


if __name__ == '__main__':
    ckpt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tasks = [(ckpt, lab, t, 1000 + i) for lab, t in SCHEDULES.items() for i in range(n)]
    with mp.Pool(6) as pool:
        res = pool.map(one, tasks, chunksize=1)
    json.dump(res, open(out, 'w'), indent=1)
    for lab in SCHEDULES:
        rs = [r for r in res if r['schedule'] == lab]
        dec = sum(r['result'] in ('1-0', '0-1') for r in rs)
        plies = sorted(r['plies'] for r in rs)
        print(f"{lab:<26} games {len(rs)}  decisive {dec}  median plies {plies[len(plies)//2]}  "
              f"terminations {dict(__import__('collections').Counter(r['termination'] for r in rs))}")
