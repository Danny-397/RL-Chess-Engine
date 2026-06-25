"""mcts.py
=========

Monte-Carlo Tree Search (MCTS) guided by the neural network.

This is the algorithmic heart of an AlphaZero-style engine and the part a
reviewer should read most carefully.  The classic "Monte-Carlo" rollout (play
random moves to the end of the game to estimate a position's value) is replaced
by a single call to the neural network's **value head**.  Likewise, the network's
**policy head** provides a *prior* that focuses the search on promising moves.

Each call to :meth:`MCTS.run` performs ``num_simulations`` iterations; each
iteration walks the tree through four phases:

1. **Selection** -- starting at the root, repeatedly descend to the child that
   maximises the PUCT score until reaching a node that has not been expanded.
2. **Expansion** -- evaluate that leaf with the network to obtain a value and a
   prior distribution, and create its children (one per legal move).
3. **Evaluation** -- the leaf's value is the network's value estimate (or the
   exact game result if the leaf is terminal).
4. **Back-up** -- propagate that value up the path, flipping its sign at every
   ply because the players alternate (a position good for me is bad for my
   opponent).

The PUCT selection score balances *exploitation* (the running value estimate
``Q``) and *exploration* (the prior ``P`` damped by how often the child has
already been visited)::

    score(child) = Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess

from chess_game import ChessGame, heuristic_score
from config import MCTSConfig


class Node:
    """A single node in the search tree -- i.e. one board position.

    To keep memory modest we do **not** store the full board in every node.
    Instead each node remembers only the move that leads to it from its parent;
    the position is reconstructed on the fly by replaying moves from the root
    during selection.  The statistics below are what PUCT needs.

    Attributes:
        prior: ``P`` -- the network's prior probability for the move that leads
            to this node (set by the parent's expansion).
        visit_count: ``N`` -- how many simulations have passed through this node.
        value_sum: ``W`` -- the sum of (perspective-correct) values backed up
            through this node.  ``Q = W / N``.
        children: mapping from ``chess.Move`` to child :class:`Node`.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[chess.Move, "Node"] = {}

    def is_expanded(self) -> bool:
        """Return ``True`` once this node's children have been created."""
        return len(self.children) > 0

    def value(self) -> float:
        """Mean action value ``Q = W / N`` (0 for an unvisited node)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Neural-network-guided Monte-Carlo Tree Search.

    Args:
        network: a :class:`model.ChessNet` (anything exposing
            ``predict(state) -> (policy_logits, value)`` works, which keeps this
            module decoupled from the exact model class).
        config: an :class:`config.MCTSConfig` of search hyper-parameters.
    """

    def __init__(self, network, config: MCTSConfig) -> None:
        self.network = network
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, game: ChessGame, add_exploration_noise: bool = True) -> Node:
        """Build a search tree rooted at ``game`` and return its root.

        Args:
            game: the position to search from (not modified -- it is cloned).
            add_exploration_noise: if ``True`` (used during self-play) Dirichlet
                noise is mixed into the root priors so the search explores moves
                the network is currently under-rating.  Set ``False`` for the
                strongest possible play (e.g. evaluation / vs. a human).

        Returns:
            The expanded root :class:`Node`.  Call :func:`action_probabilities`
            (or read ``root.children``) to extract a move distribution.
        """
        root = Node(prior=0.0)

        # Expand the root immediately so it has children to explore / add noise to.
        self._expand(root, game)
        if add_exploration_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            scratch = game.clone()
            search_path: List[Node] = [node]

            # --- 1. Selection: descend until we hit an unexpanded node ------
            while node.is_expanded():
                move, node = self._select_child(node)
                scratch.push(move)
                search_path.append(node)

            # --- 2/3. Expansion + Evaluation -------------------------------
            if scratch.is_terminal():
                # Exact outcome from the perspective of the side to move at the leaf.
                value = scratch.terminal_value()
            else:
                value = self._expand(node, scratch)

            # --- 4. Back-up ------------------------------------------------
            self._backup(search_path, value)

        return root

    # ------------------------------------------------------------------ #
    # The four phases, factored into small private helpers.
    # ------------------------------------------------------------------ #
    def _select_child(self, node: Node) -> Tuple[chess.Move, Node]:
        """Return the ``(move, child)`` with the highest PUCT score."""
        # sqrt of the parent's visit count is shared across all children.
        sqrt_parent = math.sqrt(node.visit_count)

        best_score = -float("inf")
        best_move: Optional[chess.Move] = None
        best_child: Optional[Node] = None

        for move, child in node.children.items():
            # Exploitation term Q.  A child accumulates values from the *child's*
            # perspective; from the parent's point of view that is the opponent,
            # so we negate it.  Unvisited children get Q = 0 (neutral).
            q = -child.value() if child.visit_count > 0 else 0.0
            # Exploration term U.
            u = self.config.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_move, best_child = score, move, child

        assert best_move is not None and best_child is not None
        return best_move, best_child

    def _expand(self, node: Node, game: ChessGame) -> float:
        """Evaluate ``game`` with the network and create ``node``'s children.

        Args:
            node: the leaf node to expand (must currently have no children).
            game: the (canonicalisable) position the node represents.

        Returns:
            The network's value estimate for this position, from the perspective
            of the side to move -- ready to be backed up.
        """
        policy_logits, value = self.network.predict(game.encode_state())

        # Play-time assist: blend in a material heuristic so the search wins
        # material and captures hanging pieces even before the network is fully
        # trained.  Disabled (weight 0) during self-play to keep training pure.
        w = self.config.material_weight
        if w > 0.0:
            value = (1.0 - w) * value + w * heuristic_score(game.board)

        legal_moves = game.legal_moves()
        # Gather the priors for exactly the legal moves and softmax over *those*
        # only (illegal moves get zero probability -- "masking").
        indices = np.array([game.encode_move(m) for m in legal_moves], dtype=np.int64)
        legal_logits = policy_logits[indices]
        priors = _softmax(legal_logits)

        for move, prior in zip(legal_moves, priors):
            node.children[move] = Node(prior=float(prior))

        return value

    def _backup(self, search_path: List[Node], value: float) -> None:
        """Propagate ``value`` up ``search_path``, flipping sign each ply.

        ``value`` is from the perspective of the player to move at the leaf.  As
        we walk back towards the root the side to move alternates, so the sign
        flips at every step.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into the root priors to encourage exploration.

        ``P'(a) = (1 - eps) * P(a) + eps * noise(a)`` where ``noise`` is a sample
        from a symmetric Dirichlet distribution.  This is applied only at the
        root and only during self-play, exactly as in the AlphaZero paper.
        """
        moves = list(root.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(moves))
        eps = self.config.dirichlet_epsilon
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)


# --------------------------------------------------------------------------- #
# Helpers for turning a searched root into a move / training target.
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def action_probabilities(
    root: Node, temperature: float = 1.0
) -> Tuple[List[chess.Move], np.ndarray]:
    """Convert root visit counts into a move probability distribution.

    The visit count is MCTS's improved policy: better moves are searched more,
    so ``N(a)`` is a stronger signal than the raw network prior.  Temperature
    controls exploration:

    * ``temperature = 1`` -> probabilities proportional to visit counts
      (used for the opening plies of self-play, for variety);
    * ``temperature -> 0`` -> all probability on the most-visited move
      (greedy / strongest play).

    Args:
        root: an expanded root node returned by :meth:`MCTS.run`.
        temperature: the softmax-like temperature described above.

    Returns:
        ``(moves, probs)`` -- the list of moves and a matching numpy array of
        probabilities that sums to 1.
    """
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if temperature <= 1e-6:
        # Greedy: put all mass on the most-visited move (ties broken by argmax).
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
        return moves, probs

    # Apply temperature: N^(1/T), then normalise.
    scaled = visits ** (1.0 / temperature)
    total = scaled.sum()
    if total <= 0:  # pathological fallback (e.g. all zero visits): uniform.
        probs = np.ones_like(visits) / len(visits)
    else:
        probs = scaled / total
    return moves, probs
