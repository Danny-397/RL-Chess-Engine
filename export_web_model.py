"""export_web_model.py
=======================

Export the trained network for **in-browser play** (the "AlphaZero net" opponent
on the website). It writes two files into ``web/static/``:

* ``model.onnx``      -- a tiny **value-only** ONNX model (the policy head is
  dropped, so the browser download stays small);
* ``model_meta.json`` -- "golden" self-check data: for a couple of reference
  positions, the encoding sum and the network's value output. The web page reads
  this to confirm its JavaScript board-encoding matches this Python encoding
  *for the exact model being shipped* -- so a freshly retrained model drops in
  without anyone having to hand-edit magic numbers in ``index.html``.

Run (after training, with a checkpoint in ``checkpoints/``)::

    python export_web_model.py                       # uses checkpoints/best.pt
    python export_web_model.py --checkpoint path.pt  # or an explicit checkpoint

Then commit ``web/static/model.onnx`` and ``web/static/model_meta.json`` and
redeploy.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from config import Config
from training import load_checkpoint
from chess_game import encode_board

# Reference positions for the JS<->Python encoding self-check. The second is
# *Black to move*, which exercises the colour-mirror path in the JS encoder.
GOLDEN_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 1",
]

OUT_ONNX = os.path.join("web", "static", "model.onnx")
OUT_META = os.path.join("web", "static", "model_meta.json")


class _ValueOnly(torch.nn.Module):
    """Wrap the net so ONNX exports ONLY the value head (small download)."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.net(x)
        return value


def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the value net for in-browser play.")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to export (default: checkpoints/best.pt, then "
        "checkpoints/example_checkpoint.pt).",
    )
    args = ap.parse_args()

    ckpt = args.checkpoint or _first_existing(
        os.path.join("checkpoints", "best.pt"),
        os.path.join("checkpoints", "example_checkpoint.pt"),
    )
    if not ckpt:
        raise SystemExit(
            "No checkpoint found. Train first (train_on_colab.ipynb) or pass "
            "--checkpoint PATH."
        )

    cfg = Config()
    net = load_checkpoint(ckpt, cfg, device="cpu")
    net.eval()
    value_only = _ValueOnly(net).eval()

    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    dummy = torch.zeros(1, cfg.network.num_input_planes, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        value_only,
        dummy,
        OUT_ONNX,
        input_names=["board"],
        output_names=["value"],
        dynamic_axes={"board": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript exporter (matches the value network the site expects)
    )
    print(f"exported {OUT_ONNX}  (from {ckpt})")

    # --- golden self-check data, computed from the exact ONNX we just wrote ----
    import onnxruntime as ort

    sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
    golden = []
    for fen in GOLDEN_FENS:
        import chess

        x = encode_board(chess.Board(fen))[None].astype(np.float32)
        v = float(np.asarray(sess.run(None, {"board": x})[0]).reshape(-1)[0])
        golden.append({"fen": fen, "value": round(v, 6), "sum": round(float(x.sum()), 3)})
        print(f"  golden  value={v:+.6f}  sum={float(x.sum()):.0f}  {fen}")

    with open(OUT_META, "w", encoding="utf-8") as fh:
        json.dump({"golden": golden}, fh, indent=2)
    print(f"wrote    {OUT_META}")


if __name__ == "__main__":
    main()
