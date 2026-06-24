"""model.py
===========

The neural network at the heart of the engine.

Following AlphaZero, we use a single network with a shared convolutional
"trunk" and two output "heads":

* the **policy head** outputs a logit for every one of the 4672 possible moves
  (most of which are illegal in any given position -- the search masks those
  out), and
* the **value head** outputs a single scalar in ``[-1, 1]`` estimating the game
  outcome from the perspective of the side to move (``+1`` = we are winning,
  ``-1`` = we are losing, ``0`` = drawish).

Sharing a trunk between the two heads is a form of multi-task learning: the
features useful for choosing a move are largely the same as those useful for
judging a position, so learning them jointly is both faster and a stronger
regulariser.

Architecture (a small, CPU-friendly ResNet)::

    input (18 x 8 x 8)
      -> Conv 3x3 + BN + ReLU                      (the "stem")
      -> N x ResidualBlock                          (the "trunk")
      -> policy head: Conv 1x1 + BN + ReLU -> FC -> 4672 logits
      -> value  head: Conv 1x1 + BN + ReLU -> FC -> ReLU -> FC -> tanh
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NetworkConfig


class ResidualBlock(nn.Module):
    """A standard pre-activation-free residual block: ``out = ReLU(x + F(x))``.

    Residual ("skip") connections let gradients flow directly past each block,
    which is what makes it practical to train deep convolutional towers without
    the signal vanishing.  Each block keeps the spatial size (8x8) and channel
    count fixed, so blocks can be stacked freely.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual          # the skip connection
        return F.relu(out)


class ChessNet(nn.Module):
    """The dual-headed policy/value network.

    Args:
        config: a :class:`config.NetworkConfig` describing the architecture.
            Defaults to a fresh ``NetworkConfig`` so the network can be built
            with ``ChessNet()`` in tests/REPLs.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config.num_channels

        # --- stem: lift the 18 input planes up to ``c`` channels ------------
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.num_input_planes, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # --- trunk: a stack of residual blocks ------------------------------
        self.trunk = nn.Sequential(
            *[ResidualBlock(c) for _ in range(self.config.num_residual_blocks)]
        )

        # --- policy head ----------------------------------------------------
        # A 1x1 conv reduces channels, then a fully-connected layer maps the
        # flattened 8x8 feature map to one logit per possible move.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, self.config.num_actions)

        # --- value head -----------------------------------------------------
        # Squeeze to a single plane, then an MLP down to one scalar, squashed
        # into [-1, 1] by tanh.
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8, self.config.value_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_head_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: input batch of shape ``(B, 18, 8, 8)``.

        Returns:
            A tuple ``(policy_logits, value)`` where ``policy_logits`` has shape
            ``(B, num_actions)`` (raw, *un-normalised* logits) and ``value`` has
            shape ``(B,)`` with each entry in ``[-1, 1]``.
        """
        x = self.stem(x)
        x = self.trunk(x)

        # policy head
        p = self.policy_conv(x)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # value head
        v = self.value_conv(x)
        v = v.flatten(start_dim=1)
        value = self.value_fc(v).squeeze(-1)  # (B, 1) -> (B,)

        return policy_logits, value

    # ------------------------------------------------------------------ #
    # Convenience inference helper used by MCTS.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate a *single* encoded board for the search.

        This wraps the boilerplate (eval mode, no-grad, add/remove the batch
        dimension, move data to the network's device) so ``mcts.py`` stays clean.

        Args:
            state: a single board tensor of shape ``(18, 8, 8)`` from
                :func:`chess_game.encode_board`.

        Returns:
            ``(policy_logits, value)`` where ``policy_logits`` is a 1-D numpy
            array of length ``num_actions`` and ``value`` is a python float.
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, 18, 8, 8)
        policy_logits, value = self.forward(x)
        return policy_logits[0].cpu().numpy(), float(value.item())
