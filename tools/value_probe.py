"""Does the value head know that being up a queen is good?

The draw-cycle explanation says that when every self-play game is drawn, the
value head learns that every position is even. That is a claim about the value
head, so this measures the value head directly instead of inferring it from a
falling value loss.

Positions: play ``--plies`` random legal moves from the start (seeded), keep
positions with roughly level material, then make two variants of each by
removing one queen: the side to move is then **up** a queen or **down** a
queen. Invalid positions (e.g. the side not to move left in check) are dropped.
Each network's value head scores all three, from the side to move's
perspective. A healthy head gives up > level > down; a collapsed head gives
the same number for all three.

    python tools/value_probe.py                         # untrained seed-42 net + checkpoints/best.pt
    python tools/value_probe.py --checkpoint path/a.pt --checkpoint path/b.pt

Writes ``docs/data/value_probe.json`` (or ``--out``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chess_game import ChessGame, _raw_material  # noqa: E402
from config import Config, NetworkConfig  # noqa: E402
from model import ChessNet  # noqa: E402


def make_positions(n: int, plies: tuple[int, int], seed: int) -> list[dict]:
    """Level positions plus 'up a queen' / 'down a queen' variants of each."""
    rng = random.Random(seed)
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        b = chess.Board()
        for _ in range(rng.randint(*plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if b.is_game_over() or abs(_raw_material(b)) > 1:
            continue
        us, them = b.turn, not b.turn
        our_q, their_q = list(b.pieces(chess.QUEEN, us)), list(b.pieces(chess.QUEEN, them))
        if not our_q or not their_q:
            continue
        up, down = b.copy(stack=False), b.copy(stack=False)
        up.remove_piece_at(their_q[0])
        down.remove_piece_at(our_q[0])
        if not (up.is_valid() and down.is_valid()):
            continue
        out.append({"level": b.fen(), "up": up.fen(), "down": down.fen()})
    return out


def load(path: str | None, seed: int = 42) -> tuple[str, ChessNet]:
    if path is None:
        torch.manual_seed(seed)
        return f"untrained (seed {seed})", ChessNet(Config().network).eval()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = next(v for v in ck.values() if isinstance(v, dict) and any("conv" in k or "value" in k for k in v))
    net = ChessNet(NetworkConfig(**ck["network_config"])) if "network_config" in ck else ChessNet()
    net.load_state_dict(state)
    it = ck.get("iteration")
    return path.replace(os.sep, "/") + (f" (iteration {it})" if it is not None else ""), net.eval()


def probe(net: ChessNet, positions: list[dict]) -> dict:
    res: dict[str, list[float]] = {"level": [], "up": [], "down": []}
    for p in positions:
        for k in res:
            res[k].append(float(net.predict(ChessGame(chess.Board(p[k])).encode_state())[1]))
    arr = {k: np.array(v) for k, v in res.items()}
    gap = arr["up"] - arr["down"]
    return {
        "mean_value": {k: round(float(v.mean()), 4) for k, v in arr.items()},
        "up_minus_down": round(float(gap.mean()), 4),
        "up_minus_down_95ci": [round(float(gap.mean() - 1.96 * gap.std(ddof=1) / np.sqrt(len(gap))), 4),
                               round(float(gap.mean() + 1.96 * gap.std(ddof=1) / np.sqrt(len(gap))), 4)],
        "share_up_above_down": round(float((gap > 0).mean()), 4),
        "value_spread_sd": round(float(np.concatenate(list(arr.values())).std()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="checkpoint(s) to probe; default: checkpoints/best.pt")
    ap.add_argument("--no-untrained", action="store_true", help="skip the untrained seed-42 reference")
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="seed for the position generator")
    ap.add_argument("--out", default=os.path.join("docs", "data", "value_probe.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    positions = make_positions(args.positions, (10, 40), args.seed)
    targets = ([] if args.no_untrained else [None]) + (args.checkpoint or [os.path.join("checkpoints", "best.pt")])
    results = []
    for t in targets:
        name, net = load(t)
        r = probe(net, positions)
        r["network"] = name
        results.append(r)
        mv = r["mean_value"]
        print(f"{name}: up {mv['up']:+.3f}  level {mv['level']:+.3f}  down {mv['down']:+.3f}  "
              f"gap {r['up_minus_down']:+.3f} {r['up_minus_down_95ci']}  up>down {r['share_up_above_down']:.0%}")
    out = {"what": "value head on level positions and the same positions with one queen removed "
                   "(side to move up / down a queen), side-to-move perspective",
           "positions": len(positions), "position_seed": args.seed, "results": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
