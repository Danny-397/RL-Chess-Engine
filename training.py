"""training.py
==============

The optimisation half of the AlphaZero loop.

The overall algorithm is a cycle:

    +-----------------------------------------------------------------+
    |  1. self-play with the current network  -> training examples    |
    |  2. add them to a replay buffer                                  |
    |  3. take gradient steps to make the network predict the         |
    |     MCTS policies and game outcomes more accurately             |
    |  4. checkpoint, then repeat -- a stronger network produces       |
    |     stronger self-play, which produces better training data      |
    +-----------------------------------------------------------------+

The loss combines the two heads (see :func:`alphazero_loss`):

* **policy loss** -- cross-entropy between the MCTS visit distribution
  ``pi`` and the network's predicted move distribution;
* **value loss**  -- mean-squared error between the game outcome ``z`` and the
  network's predicted value ``v``;
* **L2 regularisation** -- applied via the optimiser's ``weight_decay``.

    loss = -sum( pi * log_softmax(policy_logits) )  +  c * (z - v)^2
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import ChessNet
from self_play import TrainingExample, generate_self_play_data


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def alphazero_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined AlphaZero loss.

    Args:
        policy_logits: network policy output, shape ``(B, num_actions)``.
        value_pred: network value output, shape ``(B,)``.
        policy_target: MCTS policy targets (each row sums to 1), ``(B, num_actions)``.
        value_target: game outcomes in ``[-1, 1]``, shape ``(B,)``.
        value_loss_weight: scalar weighting the value loss relative to policy.

    Returns:
        ``(total_loss, policy_loss, value_loss)`` -- the policy/value parts are
        returned separately purely for logging.
    """
    # Policy loss: cross-entropy between the soft target distribution pi and the
    # predicted distribution.  We use log-softmax for numerical stability and
    # take the negative dot product with the targets, averaged over the batch.
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Value loss: simple mean-squared error.
    value_loss = F.mse_loss(value_pred, value_target)

    total = policy_loss + value_loss_weight * value_loss
    return total, policy_loss, value_loss


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """A fixed-capacity FIFO buffer of training examples.

    Keeping a *window* of recent self-play (rather than only the latest batch)
    stabilises training: each gradient step sees a mix of positions from several
    recent network generations.  When full, the oldest examples are dropped.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[TrainingExample] = deque(maxlen=capacity)

    def add(self, examples: List[TrainingExample]) -> None:
        """Append a batch of examples (oldest are evicted past capacity)."""
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Return a uniformly random batch (without replacement when possible)."""
        size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, size)

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Checkpoint helpers (shared with main.py)
# --------------------------------------------------------------------------- #
def save_checkpoint(model: ChessNet, path: str, iteration: int = 0) -> None:
    """Serialise the model weights (and a little metadata) to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "network_config": model.config.__dict__,
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(path: str, config: Config, device: str | None = None) -> ChessNet:
    """Recreate a :class:`ChessNet` and load weights from ``path``.

    Args:
        path: checkpoint file written by :func:`save_checkpoint`.
        config: global config (its ``network`` section builds the architecture).
        device: device to map the weights onto; defaults to ``config.resolved_device()``.

    Returns:
        The model in ``eval`` mode, ready for inference / play.
    """
    device = device or config.resolved_device()
    checkpoint = torch.load(path, map_location=device)
    model = ChessNet(config.network).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# A single optimisation pass over a batch
# --------------------------------------------------------------------------- #
def _examples_to_tensors(
    batch: List[TrainingExample], device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of examples into (states, policies, values) tensors."""
    states = torch.from_numpy(np.stack([e.state for e in batch])).to(device)
    policies = torch.from_numpy(np.stack([e.policy for e in batch])).to(device)
    values = torch.tensor([e.value for e in batch], dtype=torch.float32, device=device)
    return states, policies, values


def train_on_buffer(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    config: Config,
    device: str,
) -> Tuple[float, float, float]:
    """Run ``epochs_per_iteration`` worth of gradient steps over the buffer.

    Returns:
        The mean ``(total, policy, value)`` losses over all steps, for logging.
    """
    model.train()
    tcfg = config.training

    # Roughly one "epoch" = enough minibatches to cover the buffer once.
    steps_per_epoch = max(1, len(buffer) // tcfg.batch_size)
    total_steps = tcfg.epochs_per_iteration * steps_per_epoch

    running = np.zeros(3, dtype=np.float64)
    for _ in range(total_steps):
        batch = buffer.sample(tcfg.batch_size)
        states, policy_target, value_target = _examples_to_tensors(batch, device)

        policy_logits, value_pred = model(states)
        loss, p_loss, v_loss = alphazero_loss(
            policy_logits, value_pred, policy_target, value_target,
            tcfg.value_loss_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        if tcfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        running += [loss.item(), p_loss.item(), v_loss.item()]

    running /= total_steps
    return float(running[0]), float(running[1]), float(running[2])


# --------------------------------------------------------------------------- #
# The full training loop
# --------------------------------------------------------------------------- #
def _set_seed(seed: int | None) -> None:
    """Seed python / numpy / torch RNGs for reproducible runs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _log(log_path: str, message: str) -> None:
    """Print ``message`` and append it to the run log file."""
    print(message)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def train(config: Config | None = None, model: ChessNet | None = None) -> ChessNet:
    """Run the complete self-play + training loop.

    Args:
        config: configuration to use (defaults to :class:`config.Config`).
        model: optionally continue training from an existing model; otherwise a
            freshly initialised network is created.

    Returns:
        The trained model.  Checkpoints are also written to
        ``config.training.checkpoint_dir`` along the way (and ``best.pt`` at the end).
    """
    config = config or Config()
    config.ensure_directories()
    _set_seed(config.training.seed)

    device = config.resolved_device()
    tcfg = config.training
    log_path = os.path.join(tcfg.log_dir, "training.log")

    model = (model or ChessNet(config.network)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
    )
    buffer = ReplayBuffer(tcfg.replay_buffer_size)

    _log(log_path, f"=== Training start: device={device}, "
                   f"{tcfg.num_iterations} iterations ===")

    for iteration in range(1, tcfg.num_iterations + 1):
        t0 = time.time()

        # 1. Self-play: generate fresh data with the current network.
        model.eval()
        examples = generate_self_play_data(
            model, config, num_games=tcfg.games_per_iteration
        )
        buffer.add(examples)

        # 2. Optimise: take gradient steps over the replay buffer.
        total, p_loss, v_loss = train_on_buffer(model, optimizer, buffer, config, device)

        dt = time.time() - t0
        _log(
            log_path,
            f"[iter {iteration:3d}/{tcfg.num_iterations}] "
            f"new_examples={len(examples):5d} buffer={len(buffer):6d} "
            f"loss={total:.4f} (policy={p_loss:.4f}, value={v_loss:.4f}) "
            f"time={dt:.1f}s",
        )

        # 3. Checkpoint periodically.
        if iteration % tcfg.checkpoint_every == 0:
            ckpt = os.path.join(tcfg.checkpoint_dir, f"checkpoint_iter{iteration:03d}.pt")
            save_checkpoint(model, ckpt, iteration=iteration)

    # Always save a final "best" checkpoint for easy loading in play mode.
    best_path = os.path.join(tcfg.checkpoint_dir, "best.pt")
    save_checkpoint(model, best_path, iteration=tcfg.num_iterations)
    _log(log_path, f"=== Training done. Final model saved to {best_path} ===")

    return model
