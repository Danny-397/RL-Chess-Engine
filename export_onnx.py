"""export_onnx.py
=================

Export the trained PyTorch network to ONNX so it can run **in the browser**
(via onnxruntime-web) — this is what powers the "play the AlphaZero network"
option on the web UI, with no PyTorch backend required.

It also prints "golden" values (the network's value output for a few positions)
that the front-end uses as a self-check to confirm its JavaScript board encoding
matches this Python encoding exactly.

Run:  python export_onnx.py
Output: web/static/model.onnx
"""

import numpy as np
import torch
import chess

from config import Config
from training import load_checkpoint
from chess_game import encode_board

CKPT = "checkpoints/example_checkpoint.pt"
OUT = "web/static/model.onnx"

cfg = Config()
net = load_checkpoint(CKPT, cfg)
net.eval()

dummy = torch.zeros(1, cfg.network.num_input_planes, 8, 8, dtype=torch.float32)
torch.onnx.export(
    net, dummy, OUT,
    input_names=["board"], output_names=["policy", "value"],
    dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
    opset_version=17,
)
print(f"exported -> {OUT}")

# --- verify with onnxruntime + print golden values for the JS self-check ------
import onnxruntime as ort

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
fens = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # 1.e4 c5
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # italian-ish, black to... white to move
]
print("\nFEN, torch_value, onnx_value, encoding_sum:")
for f in fens:
    b = chess.Board(f)
    x = encode_board(b)[None].astype(np.float32)
    with torch.no_grad():
        _, v_t = net(torch.from_numpy(x))
    out = sess.run(None, {"board": x})
    v_onnx = float(np.asarray(out[1]).reshape(-1)[0])
    print(f"  {f}")
    print(f"    torch={float(v_t):+.6f}  onnx={v_onnx:+.6f}  encsum={float(x.sum()):.3f}")
