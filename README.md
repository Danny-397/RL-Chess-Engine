# RL-Chess-Engine

A clean, fully-documented implementation of an **AlphaZero-style reinforcement
learning chess engine**, written from scratch in **Python + PyTorch**.

The engine learns to play chess **with zero human knowledge** beyond the rules:
it starts from random weights and improves purely by playing millions of moves
against itself. The same three ideas that powered DeepMind's AlphaZero are all
here, implemented in a way meant to be *read and understood*:

1. A **deep residual neural network** that looks at a position and outputs both a
   move-preference (**policy**) and an estimate of who is winning (**value**).
2. **Monte-Carlo Tree Search (MCTS)** that uses the network to look ahead and
   produce much stronger move choices than the raw network alone.
3. A **self-play training loop** that turns those searches into training data and
   feeds the improved network back into the search — a self-reinforcing cycle.

> This project was built to be a clear, correct, end-to-end demonstration of
> reinforcement learning and search, not to chase grandmaster strength. Every
> module is small, single-purpose and heavily commented.

---

## Why this is interesting

AlphaZero is famous for being conceptually simple but subtle to get *right*. This
repo deliberately surfaces the parts that are easy to get wrong, with tests to
prove they're correct:

- **Perspective handling.** The network always sees the board from the side to
  move's point of view (the board is mirrored when it's Black's turn), and value
  signs are flipped correctly as they propagate up the search tree. A single sign
  error here silently breaks learning — so there are tests for it.
- **The full AlphaZero move encoding.** Moves are encoded into the canonical
  `8 × 8 × 73 = 4672`-dimensional action space (queen moves, knight moves and
  under-promotions), with round-trip tests.
- **Search as policy improvement.** MCTS visit counts — not the raw network
  output — are used as the training target, which is precisely what makes the
  network get stronger over time.

---

## Architecture at a glance

```
                 +-------------------+
                 |   self_play.py    |  plays the engine against itself,
                 |  (data generator) |  records (state, MCTS policy, outcome)
                 +---------+---------+
                           |  training examples
                           v
   +-----------+     +-----------+     +-------------------+
   |  mcts.py  |<----|  model.py |     |   training.py     |
   |  (search) |     | (ResNet)  |---->| (loss + optimise) |
   +-----+-----+     +-----------+     +---------+---------+
         |  uses network priors + value           |  updated weights
         |                                         v
         |                                  checkpoints/*.pt
         v
   +-------------+
   | chess_game  |  rules, board<->tensor encoding, move<->index encoding
   | (.py)       |  (built on python-chess)
   +-------------+
```

| File             | Responsibility |
|------------------|----------------|
| `config.py`      | All hyper-parameters in one typed, documented place. |
| `chess_game.py`  | Rules wrapper + board→tensor and move↔index encodings. |
| `model.py`       | The dual-headed residual policy/value network. |
| `mcts.py`        | PUCT Monte-Carlo Tree Search guided by the network. |
| `self_play.py`   | Generates `(state, policy, value)` training data via self-play. |
| `training.py`    | Loss function, replay buffer, training loop, checkpointing. |
| `main.py`        | CLI entry point: `--mode train` / `--mode play`. |
| `tests/`         | Pytest suite covering encoding, model, search and terminals. |

---

## Installation

```bash
git clone https://github.com/Danny-397/RL-Chess-Engine.git
cd RL-Chess-Engine
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+. A GPU is optional — the defaults are tuned to run on a
laptop CPU.

---

## Quick start

### Play against the engine

A small, ready-to-use checkpoint ships in `checkpoints/example_checkpoint.pt` so
you can play immediately:

```bash
python main.py --mode play --checkpoint checkpoints/example_checkpoint.pt
```

You enter moves in standard algebraic notation (`e4`, `Nf3`, `O-O`, `exd5`,
`e8=Q`). Type `board` to redraw, `quit` to resign.

> The bundled checkpoint is trained only briefly (it's a *demonstration*, not a
> strong player). Train longer for a tougher opponent.

### Train your own engine

```bash
# A short demo run (minutes on a CPU):
python main.py --mode train --iterations 5 --games 4 --simulations 50

# A more serious run (uses the defaults in config.py):
python main.py --mode train
```

Checkpoints are written to `checkpoints/` and progress is logged to
`logs/training.log`. The final model is saved as `checkpoints/best.pt`, which is
the default the `play` mode looks for.

### Run the tests

```bash
pytest -q
```

---

## How it works (the 3-minute version)

**1. The network (`model.py`).** A small ResNet takes an `18 × 8 × 8` tensor of
the position and produces a policy (4672 logits, one per possible move) and a
scalar value in `[-1, 1]`. Sharing a trunk between the two heads is multi-task
learning: features useful for *choosing* a move also help *judging* a position.

**2. The search (`mcts.py`).** Pure neural-network move choice is short-sighted.
MCTS runs many simulations, each descending the game tree by the **PUCT** rule

```
score(move) = Q(move) + c_puct · P(move) · √N(parent) / (1 + N(move))
```

which balances exploiting moves with a high running value `Q` against exploring
moves the network rates highly (`P`) but hasn't tried much. Leaves are evaluated
by the network's value head (no random rollouts), and the value is backed up the
tree, flipping sign each ply because the players alternate.

**3. Self-play & training (`self_play.py`, `training.py`).** The engine plays
itself; for each position it stores the board, the MCTS visit distribution (a
*sharpened* policy), and — once the game ends — who won. The network is then
trained to match those policies (cross-entropy) and outcomes (MSE):

```
loss = −Σ π · log p(·)   +   c · (z − v)²   +   L2 regularisation
```

A stronger network produces stronger self-play, which produces better data, which
produces a stronger network. That loop is the whole idea.

---

## Extending it

The code is built to be tinkered with:

- **Different network?** Edit `model.py` (or swap in your own `nn.Module` that
  returns `(policy_logits, value)`). Adjust depth/width in `config.NetworkConfig`.
- **Stronger / faster search?** Change `num_simulations` and `c_puct` in
  `config.MCTSConfig`.
- **Training behaviour?** Everything (learning rate, batch size, games per
  iteration, replay buffer size, temperature schedule) lives in
  `config.TrainingConfig`.

---

## Limitations & honest notes

- This is an **educational** engine. With modest simulation counts and a small
  network it will not approach strong engines like Stockfish or Leela.
- Training is single-threaded and self-play (not the network) dominates runtime,
  which is the usual AlphaZero bottleneck. Parallelising self-play across
  processes is the highest-impact next step.
- For clarity the board encoding omits move-history planes (used by the original
  AlphaZero to detect repetitions); this is a documented simplification.

---

## References

- Silver et al., *“A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play”*, Science 2018 (AlphaZero).
- Silver et al., *“Mastering the game of Go without human knowledge”*, Nature 2017
  (AlphaGo Zero).
