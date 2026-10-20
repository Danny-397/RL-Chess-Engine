"""test_training.py
===================

Tests for the parts of the pipeline that ``test_engine.py`` does not reach:
the loss, the replay buffer, checkpoint round-trips, the gradient step, and
the ONNX export that the public site actually loads.

The export test is the load-bearing one. ``rlchess.xyz`` does not run the
PyTorch model -- it downloads ``web/static/model.onnx`` and runs it in the
browser through onnxruntime-web. If the export silently diverges from the
network it was built from, every test here and in ``test_engine.py`` still
passes while the live site plays different, worse chess. So it is checked for
numerical parity rather than merely for "a file appeared".

Run with::

    pytest -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, NUM_ACTIONS, NUM_INPUT_PLANES
from model import ChessNet
from self_play import TrainingExample
from training import (
    ReplayBuffer,
    alphazero_loss,
    load_checkpoint,
    save_checkpoint,
    train_on_buffer,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _example(rng: np.random.Generator, value: float = 0.0) -> TrainingExample:
    """One syntactically valid training example with a normalised policy."""
    policy = rng.random(NUM_ACTIONS).astype(np.float32)
    policy /= policy.sum()
    return TrainingExample(
        state=rng.random((NUM_INPUT_PLANES, 8, 8)).astype(np.float32),
        policy=policy,
        value=value,
    )


def _batch(n: int, seed: int = 0) -> list[TrainingExample]:
    rng = np.random.default_rng(seed)
    return [_example(rng, value=float(rng.choice([-1.0, 0.0, 1.0]))) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_loss_components_sum_to_the_total():
    """total == policy + value_loss_weight * value, which is the whole contract."""
    torch.manual_seed(0)
    b, weight = 4, 1.5
    logits = torch.randn(b, NUM_ACTIONS)
    value_pred = torch.tanh(torch.randn(b))
    target = torch.softmax(torch.randn(b, NUM_ACTIONS), dim=1)
    value_target = torch.tensor([1.0, -1.0, 0.0, 1.0])

    total, p_loss, v_loss = alphazero_loss(
        logits, value_pred, target, value_target, weight
    )
    assert torch.isclose(total, p_loss + weight * v_loss, atol=1e-5)
    assert total.item() >= 0.0


def test_loss_is_lower_when_the_network_is_right():
    """A network predicting the targets must score better than one that isn't.

    Without this, a sign error in either head is invisible: the loss still
    decreases, just toward the wrong thing.
    """
    target = torch.softmax(torch.randn(6, NUM_ACTIONS), dim=1)
    value_target = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0, 1.0])

    confident = torch.log(target + 1e-9) * 6.0          # peaked on the target
    wrong = torch.log(target.flip(dims=[1]) + 1e-9) * 6.0

    good, _, _ = alphazero_loss(confident, value_target.clone(), target, value_target, 1.0)
    bad, _, _ = alphazero_loss(wrong, -value_target, target, value_target, 1.0)
    assert good.item() < bad.item()


def test_value_weight_scales_only_the_value_term():
    target = torch.softmax(torch.randn(3, NUM_ACTIONS), dim=1)
    value_target = torch.tensor([1.0, 0.0, -1.0])
    logits = torch.randn(3, NUM_ACTIONS)
    value_pred = torch.zeros(3)

    t1, p1, v1 = alphazero_loss(logits, value_pred, target, value_target, 1.0)
    t2, p2, v2 = alphazero_loss(logits, value_pred, target, value_target, 3.0)

    assert torch.isclose(p1, p2, atol=1e-6)      # policy term untouched
    assert torch.isclose(v1, v2, atol=1e-6)      # value term untouched
    assert t2.item() > t1.item()                 # only the blend changes


def test_loss_produces_gradients_for_every_parameter():
    """A head that never receives gradient would train silently to nothing."""
    cfg = Config()
    model = ChessNet(cfg.network)
    states, policies, values = _tensors(_batch(4, seed=1))

    logits, value_pred = model(states)
    total, _, _ = alphazero_loss(
        logits, value_pred, policies, values, cfg.training.value_loss_weight
    )
    total.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not missing, f"no gradient reached: {missing[:5]}"


def _tensors(batch):
    states = torch.from_numpy(np.stack([e.state for e in batch]))
    policies = torch.from_numpy(np.stack([e.policy for e in batch]))
    values = torch.tensor([e.value for e in batch], dtype=torch.float32)
    return states, policies, values


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def test_buffer_evicts_oldest_past_capacity():
    buf = ReplayBuffer(capacity=10)
    first = _batch(10, seed=2)
    buf.add(first)
    assert len(buf) == 10

    newer = _batch(4, seed=3)
    buf.add(newer)
    assert len(buf) == 10, "capacity must be a hard ceiling"

    # The four survivors from `newer` are present; the four oldest are gone.
    kept = {id(e) for e in buf.buffer}
    assert all(id(e) in kept for e in newer)
    assert not any(id(e) in kept for e in first[:4])


def test_sample_never_exceeds_what_is_stored():
    buf = ReplayBuffer(capacity=100)
    buf.add(_batch(3, seed=4))
    assert len(buf.sample(64)) == 3          # asking for more is not an error
    assert len(buf.sample(2)) == 2


def test_empty_buffer_samples_nothing():
    assert ReplayBuffer(capacity=5).sample(8) == []


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_preserves_the_network_exactly(tmp_path):
    """Weights must survive save/load bit-for-bit, or resuming silently restarts."""
    cfg = Config()
    torch.manual_seed(7)
    model = ChessNet(cfg.network)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, path, iteration=3)
    assert os.path.exists(path)

    restored = load_checkpoint(path, cfg, device="cpu")
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                  restored.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1.detach().cpu(), p2.detach().cpu()), f"{n1} changed"


def test_restored_network_gives_identical_outputs(tmp_path):
    cfg = Config()
    torch.manual_seed(8)
    model = ChessNet(cfg.network).eval()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, path)
    restored = load_checkpoint(path, cfg, device="cpu").eval()

    x = torch.from_numpy(_batch(2, seed=9)[0].state).unsqueeze(0)
    with torch.no_grad():
        p1, v1 = model(x)
        p2, v2 = restored(x)
    assert torch.allclose(p1, p2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


# --------------------------------------------------------------------------- #
# The gradient step
# --------------------------------------------------------------------------- #
def test_training_moves_the_weights_and_reports_finite_losses():
    cfg = Config()
    cfg.training.batch_size = 4
    cfg.training.epochs_per_iteration = 1

    model = ChessNet(cfg.network)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    buf = ReplayBuffer(capacity=32)
    buf.add(_batch(16, seed=10))

    total, policy, value = train_on_buffer(model, opt, buf, cfg, device="cpu")

    assert all(np.isfinite(x) for x in (total, policy, value))
    moved = any(not torch.equal(b, p.detach())
                for b, p in zip(before, model.parameters()))
    assert moved, "an optimiser step that changes nothing is not training"


def test_training_can_overfit_a_single_repeated_example():
    """The clearest signal that the loop actually learns: drive loss down."""
    cfg = Config()
    cfg.training.batch_size = 2
    cfg.training.epochs_per_iteration = 1

    torch.manual_seed(11)
    model = ChessNet(cfg.network)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    one = _batch(1, seed=12)[0]
    buf = ReplayBuffer(capacity=8)
    buf.add([one] * 4)

    first, _, _ = train_on_buffer(model, opt, buf, cfg, device="cpu")
    for _ in range(12):
        last, _, _ = train_on_buffer(model, opt, buf, cfg, device="cpu")
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


# --------------------------------------------------------------------------- #
# ONNX export -- what the public site actually runs
# --------------------------------------------------------------------------- #
# Skipped per-test, not per-module: onnxruntime is what the browser uses and
# what parity needs, so those tests must still run when the optional `onnx`
# package (graph checker only) is absent.
def _ort():
    return pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


def _require_exporter():
    """Skip unless this machine can actually produce an ONNX file.

    Checking that `onnx` imports is not enough: torch.onnx.export needs
    `onnxscript` for the torch.export path that became the default in 2.9, and
    a partially-installed `onnx` can import while still lacking the functions
    the legacy path calls. Both failures look like a broken toolchain, not a
    defect in this repo, so the only honest check is to try a trivial export
    and skip if it raises.
    """
    import tempfile

    class _Tiny(torch.nn.Module):
        def forward(self, x):
            return x * 2

    with tempfile.TemporaryDirectory() as d:
        try:
            torch.onnx.export(_Tiny(), torch.zeros(1, 2),
                              os.path.join(d, "probe.onnx"), opset_version=17)
        except Exception as exc:
            pytest.skip(f"no working ONNX exporter here ({type(exc).__name__}); "
                        f"pip install onnxscript")


def _check_graph(path: str) -> None:
    """Validate the graph if `onnx` is importable; it is optional tooling."""
    try:
        import onnx
    except Exception:
        return
    onnx.checker.check_model(onnx.load(path))


def test_value_only_wrapper_returns_just_the_value_head():
    from export_web_model import _ValueOnly

    cfg = Config()
    model = ChessNet(cfg.network).eval()
    wrapped = _ValueOnly(model).eval()

    x = torch.randn(1, NUM_INPUT_PLANES, 8, 8)
    with torch.no_grad():
        _, value = model(x)
        out = wrapped(x)
    assert torch.allclose(out, value, atol=1e-6)


def test_exported_onnx_matches_pytorch(tmp_path):
    """Numerical parity between the exported graph and the network it came from.

    The browser runs the ONNX file, so a mismatch here is a live-site bug that
    nothing else in this suite would catch.
    """
    _ort()
    _require_exporter()
    from export_web_model import _ValueOnly

    cfg = Config()
    torch.manual_seed(13)
    model = ChessNet(cfg.network).eval()
    wrapped = _ValueOnly(model).eval()

    path = str(tmp_path / "model.onnx")
    dummy = torch.randn(1, NUM_INPUT_PLANES, 8, 8)
    torch.onnx.export(
        wrapped, dummy, path,
        input_names=["board"], output_names=["value"],
        dynamic_axes={"board": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    _check_graph(path)

    sess = _ort().InferenceSession(path, providers=["CPUExecutionProvider"])
    for seed in (14, 15, 16):
        torch.manual_seed(seed)
        x = torch.randn(1, NUM_INPUT_PLANES, 8, 8)
        with torch.no_grad():
            expected = wrapped(x).numpy()
        got = sess.run(["value"], {"board": x.numpy()})[0]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_exported_onnx_accepts_a_batch(tmp_path):
    """dynamic_axes is declared, so more than one position must work."""
    _ort()
    _require_exporter()
    from export_web_model import _ValueOnly

    cfg = Config()
    model = ChessNet(cfg.network).eval()
    wrapped = _ValueOnly(model).eval()
    path = str(tmp_path / "model.onnx")
    torch.onnx.export(
        wrapped, torch.randn(1, NUM_INPUT_PLANES, 8, 8), path,
        input_names=["board"], output_names=["value"],
        dynamic_axes={"board": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    sess = _ort().InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(["value"], {"board": np.random.randn(5, NUM_INPUT_PLANES, 8, 8).astype(np.float32)})[0]
    assert out.shape[0] == 5
