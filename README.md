# RL-Chess-Engine

[![tests](https://github.com/Danny-397/RL-Chess-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Danny-397/RL-Chess-Engine/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
| `self_play.py`   | Generates `(state, policy, value)` training data via self-play (sequential **and** multiprocessing), with PGN export. |
| `training.py`    | Loss function, replay buffer, training loop, checkpointing, periodic evaluation. |
| `evaluation.py`  | Pluggable agents, match play, and an approximate **Elo** estimate. |
| `analysis.py`    | Engine move recommendations + win-probability (powers `hint`/`analyze`). |
| `main.py`        | CLI entry point: `--mode train` / `play` / `eval` / `analyze` / `serve`. |
| `web/`           | Optional FastAPI backend + chessboard.js front-end (browser play). |
| `tests/`         | Pytest suite covering encoding, model, search, self-play and evaluation. |

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

### Play in the browser (web UI)

The most demoable way to play: a drag-and-drop board with a live evaluation bar
and a "recommended moves" panel, served by a small FastAPI backend that wraps the
engine.

```bash
pip install fastapi "uvicorn[standard]"   # one-time, optional web extras
python main.py --mode serve               # then open http://127.0.0.1:8000
```

You play White by dragging pieces; the engine replies and shows its win estimate,
the **Hint** button asks the engine to recommend moves for your position, and the
eval bar tracks who's ahead. A banner announces checkmate/draw. The backend is
stateless (the browser sends the position as FEN), and it reuses the exact same
`analysis.py` logic as the console.

> **Why the bot plays sensibly even though the bundled net is barely trained:**
> for *play* (not training) the search blends in a simple material heuristic
> (`MCTSConfig.material_weight`, default `0.85` in play/serve, `0.0` in training).
> This makes it capture hanging pieces, avoid blunders and find basic mates, so
> it's a real opponent before a full GPU training run. Set
> `--material-weight 0` (CLI) or `RLCHESS_MATERIAL_WEIGHT=0` (server) to play the
> pure from-scratch network instead.

#### Deploy the web UI to Render

A [`render.yaml`](render.yaml) blueprint is included, so you can host the board
online:

1. Push this repo to GitHub (already done if you cloned it from there).
2. On [Render](https://render.com): **New + → Blueprint**, pick this repo, and
   apply. Render reads `render.yaml` and provisions the web service.
3. Open the URL Render gives you and play.

Or configure a **Web Service** manually with these settings:

| Setting | Value |
|---|---|
| Build command | `pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` |
| Start command | `uvicorn web.server:app --host 0.0.0.0 --port $PORT` |
| Env var | `RLCHESS_SIMULATIONS=60` (lower = faster responses) |

Two gotchas the blueprint already handles:

- **CPU-only PyTorch.** A plain `pip install torch` pulls a multi-gigabyte CUDA
  wheel that overflows Render's build; the `--index-url .../whl/cpu` keeps it small.
- **Port binding.** Render injects `$PORT`; the start command binds `0.0.0.0:$PORT`
  (the local default is `127.0.0.1:8000`).

> **Memory note:** PyTorch needs a fair bit of RAM. Render's *free* instance
> (512 MB) may OOM when the model loads — if so, upgrade to a larger instance, or
> lower `RLCHESS_SIMULATIONS`. The committed `example_checkpoint.pt` is served by
> default; set `RLCHESS_CHECKPOINT` to point at a stronger one.

#### Frontend on Vercel + backend on Render (split deploy)

> **Simplest option — you may not need Vercel at all.** The Render web service
> above already serves the board UI (at `/`) *and* the API from one place, so the
> Render URL is a complete, working site on its own. Only do the split below if you
> specifically want the frontend on Vercel's CDN (e.g. to match the TradeBot setup).

**Vercel cannot host the backend** — it runs Python only as serverless functions
with a 250 MB unzipped limit, and PyTorch alone is far larger. The standard
pattern is to host the **static board on Vercel** and keep the **PyTorch API on
Render**, with the page calling across to it:

1. Deploy the backend to Render (above). Note its URL, e.g.
   `https://rl-chess-engine.onrender.com`.
2. Deploy the frontend to Vercel: import this repo and **set the project's Root
   Directory to `web/static`** (Settings → Build & Deployment → Root Directory).
   That subfolder contains only `index.html`, so Vercel deploys it as a pure
   static site — no build, no Python.

   > ⚠️ If you deploy from the **repo root** instead, Vercel sees `requirements.txt`
   > and tries to build the Python backend, erroring with *"main.py does not define
   > a top-level app … add `[tool.vercel] entrypoint = web.server:app`"*. **Do not
   > add that entrypoint** — it would try to bundle PyTorch into a serverless
   > function and blow Vercel's 250 MB limit. Setting Root Directory to `web/static`
   > makes the error disappear by skipping the Python code entirely.
3. Tell the frontend where the backend is, either by:
   - opening it with `?api=https://rl-chess-engine.onrender.com`, or
   - editing `API_BASE` at the top of the `<script>` in
     [web/static/index.html](web/static/index.html).
4. Lock down CORS on the Render backend by setting
   `RLCHESS_ALLOW_ORIGINS=https://your-site.vercel.app` (it defaults to `*`).

The backend already sends the right CORS headers, so the cross-origin calls from
Vercel just work.

### Play against the engine (console)

A small, ready-to-use checkpoint ships in `checkpoints/example_checkpoint.pt` so
you can play immediately:

```bash
python main.py --mode play --checkpoint checkpoints/example_checkpoint.pt
```

You enter moves in standard algebraic notation (`e4`, `Nf3`, `O-O`, `exd5`,
`e8=Q`). During your turn you can also type:

- **`hint`** — the engine analyses *your* position and prints its top recommended
  moves (ranked by search effort) plus your win probability;
- **`eval`** — show the engine's evaluation of the current position;
- `board` to redraw, `quit` to resign.

The engine also reports its own win estimate each time it moves.

> The bundled checkpoint is trained only briefly (it's a *demonstration*, not a
> strong player). Train longer for a tougher opponent.

### Analyse any position

Get the engine's evaluation and recommended moves for a position without playing
a whole game — defaults to the opening, or pass any FEN:

```bash
python main.py --mode analyze --checkpoint checkpoints/best.pt
python main.py --mode analyze --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1" --top-n 5
```

Example output:

```
Engine eval: +0.18  (win probability for side to move: 59%)
Recommended moves:
  1. Nf3      62% of search, eval +0.18
  2. e4       21% of search, eval +0.12
  3. d4       11% of search, eval +0.09
```

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

#### Faster training: parallel self-play

Self-play (not the gradient updates) dominates AlphaZero's runtime, and games are
independent — so they fan out across CPU cores almost perfectly:

```bash
python main.py --mode train --workers 8        # 8 self-play worker processes
```

Workers each rebuild the network from the current weights and run on CPU (pinned
to one thread each to avoid oversubscription); their `(state, policy, value)`
examples are collected back in the main process for the gradient step.

#### Archiving games + tracking strength during training

```bash
# Save every self-play game to pgn/selfplay_iterNNN.pgn, and every 5 iterations
# play a quick match vs a random baseline and log the estimated Elo gain:
python main.py --mode train --save-pgn --eval-every 5
```

### Measure engine strength (Elo)

The `eval` mode plays a match and reports a win/draw/loss tally plus an
approximate Elo difference (colours are alternated so first-move advantage
cancels out):

```bash
# Trained engine vs. the random baseline (the basic "did it learn?" test):
python main.py --mode eval --checkpoint checkpoints/best.pt --opponent random --eval-games 40

# Head-to-head between two checkpoints (did training iteration N beat iteration M?):
python main.py --mode eval --checkpoint checkpoints/checkpoint_iter020.pt \
                           --opponent checkpoints/checkpoint_iter010.pt --eval-games 40

# Optionally dump the played games to a PGN file with --pgn-out games.pgn
```

Elo is derived from the expected score `S` by the standard logistic relation
`elo = -400 · log10(1/S − 1)`, so 50% → ±0, ~64% → ~+100, etc.

### Run the tests

```bash
pytest -q
```

The same suite runs automatically on every push via GitHub Actions (see the
`tests` badge above).

### Tracking training progress

Training appends a per-iteration loss breakdown to `logs/training.log`, plus a
periodic Elo estimate against the random baseline. Turn that log into charts:

```bash
python plot_progress.py            # -> assets/training_progress.png
```

This produces two plots — training loss over time, and estimated Elo vs. random
over time — which together answer the only questions that matter: *is the network
fitting the data, and is the engine actually getting stronger?*

### Training results (and an honest note on scale)

Below is a short **CPU** training run (~19 iterations, ~100 self-play games):

![training progress](assets/training_progress.png)

The **policy loss drops sharply** (4.7 → ~0.9): the network is clearly learning to
imitate the search. But two things are visible and worth being honest about:

1. The **value loss sits near zero**, and
2. the engine does **not yet beat the random baseline** (Elo ≈ 0).

Both have the same cause — a **draw cycle**. A weak engine's self-play games almost
always run to the move cap and end in *draws*, so the value target is almost always
`0`; the value head learns only to predict "even," which keeps the search weak, which
keeps games drawish. Escaping this requires **far more (and more decisive) self-play
games** than a single CPU can produce in reasonable time.

**To train a genuinely strong engine**, run on a GPU with more self-play. A ready-made
Colab notebook does exactly this on free, dedicated compute (so it won't crash from a
laptop's memory pressure or stall when the machine sleeps):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danny-397/RL-Chess-Engine/blob/main/notebooks/train_on_colab.ipynb)

It clones the repo, trains with deeper search and more games per iteration, plots the
learning curves, measures Elo vs. random, and lets you download the trained checkpoint to
drop in as `checkpoints/example_checkpoint.pt`. The learning machinery here is correct and
complete — the limiting factor is raw compute, not the algorithm.

---

## How it works (the 3-minute version)

> For a fuller treatment with the underlying math (PUCT, the loss function, and
> why the self-play loop converges), see [docs/how-it-works.md](docs/how-it-works.md).

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
- Self-play (not the network) dominates runtime, the usual AlphaZero bottleneck.
  Multiprocessing self-play (`--workers`) addresses this on a single machine; the
  next step up would be distributing self-play across multiple machines.
- For clarity the board encoding omits move-history planes (used by the original
  AlphaZero to detect repetitions); this is a documented simplification.

---

## References

- Silver et al., *“A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play”*, Science 2018 (AlphaZero).
- Silver et al., *“Mastering the game of Go without human knowledge”*, Nature 2017
  (AlphaGo Zero).
