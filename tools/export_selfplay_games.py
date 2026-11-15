"""Record real self-play games for the site's draw-cycle replay.

Plays ``--games`` self-play games with a checkpoint under the conditions of the
draw-cycle baseline (``material_weight = 0``, the configured simulations, root
noise and temperature exactly as in training) and writes them, unedited, to
``web/static/selfplay_games.json`` with how each one ended. Nothing is
selected or trimmed: the site shows every recorded game and says which
network played it.

    python tools/export_selfplay_games.py --checkpoint checkpoints/best.pt --games 6

The Kaggle seeds notebook calls this on a GPU-trained checkpoint, so the
replay can be swapped for games from a properly trained network.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config  # noqa: E402
from self_play import play_game  # noqa: E402
from training import load_checkpoint  # noqa: E402


def termination(board: chess.Board, plies: int, cap: int) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_repetition(3):
        return "threefold repetition"
    if board.halfmove_clock >= 100:
        return "fifty-move rule"
    if plies >= cap:
        return "move cap"
    if board.can_claim_threefold_repetition():          # legacy rule only
        return "repetition available (claimed)"
    if board.can_claim_fifty_moves():
        return "fifty-move draw available (claimed)"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=os.path.join("checkpoints", "best.pt"))
    ap.add_argument("--label", default=None,
                    help="how the site names the network (default: from the checkpoint)")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=None, help="MCTS sims (default: config)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("web", "static", "selfplay_games.json"))
    args = ap.parse_args()

    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = Config()
    cfg.mcts.material_weight = 0.0                       # the draw-cycle baseline condition
    if args.sims:
        cfg.mcts.num_simulations = args.sims
    net = load_checkpoint(args.checkpoint, cfg, device="cpu")
    it = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("iteration")
    label = args.label or (f"checkpoints/best.pt, iteration {it} of a short CPU run" if it is not None
                           else os.path.basename(args.checkpoint))

    games = []
    for g in range(args.games):
        t0 = time.time()
        res = play_game(net, cfg, collect_pgn=True, round_no=g + 1)
        game = chess.pgn.read_game(__import__("io").StringIO(res.pgn))
        board = game.board()
        sans = []
        for mv in game.mainline_moves():
            sans.append(board.san(mv))
            board.push(mv)
        how = termination(board, len(sans), cfg.training.max_moves)
        games.append({"moves": sans, "plies": len(sans), "result": board.result(claim_draw=True),
                      "termination": how})
        print(f"game {g + 1}: {len(sans)} plies, {how} ({time.time() - t0:.0f} s)", flush=True)

    import chess_game
    out = {"network": label, "material_weight": 0.0,
           "draw_rule": "claimable (legacy)" if chess_game.END_ON_CLAIMABLE_DRAWS else "actual", "sims_per_move": cfg.mcts.num_simulations,
           "note": "every game recorded, none selected or edited", "games": games}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
