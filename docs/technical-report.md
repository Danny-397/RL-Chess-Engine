# An AlphaZero-Style Chess Engine, Built From Scratch: Implementation, Results, and the "Draw Cycle"

> **Draft — technical report.** Sections 4–6 are drafted below from the actual
> implementation. Sections 1–3 and 7–10 are outlined as placeholders to be
> filled. Numbers marked `⟨FILL⟩` must be populated from a **real training run**
> (see §5 and the logging spec at the end) — do not cite aspirational figures.

---

## 1. Abstract  *(placeholder)*
One paragraph: a from-scratch AlphaZero implementation (network + MCTS + self-play);
the empirical finding that training plateaus into a self-reinforcing **draw cycle**;
and the compute-scaling experiment showing the plateau is a resource ceiling, not
an algorithmic error.

## 2. Introduction  *(placeholder)*
Motivation for building AlphaZero from first principles rather than using a library;
the appeal of zero-human-knowledge learning; what this report contributes.

## 3. Background  *(placeholder)*
The three pillars of AlphaZero — a policy/value network, PUCT MCTS, and a self-play
training loop — and how they interlock.

---

## 4. Method / Implementation

The system is ~3,200 lines of Python + PyTorch with no reinforcement-learning
libraries; `python-chess` is used only for the rules (legal-move generation and
terminal-state detection), so all intellectual effort sits in the encoding, the
network, the search, and the training loop.

### 4.1 Problem encoding

**Board → tensor** (`chess_game.encode_board`). Each position becomes an
`18 × 8 × 8` tensor:

| Planes | Count | Contents |
|---|---|---|
| Piece planes | 12 | 6 piece types × 2 colours |
| Castling rights | 4 | us K/Q-side, them K/Q-side |
| En-passant | 1 | target square, if any |
| Half-move clock | 1 | progress toward the 50-move rule |

Crucially, the board is encoded in **canonical (side-to-move) form**: when it is
Black's turn the board is mirrored, so the network only ever reasons about "the
player at the bottom, to move." This makes the network colour-agnostic and roughly
halves what it must learn.

**Move → index** (`move_to_index` / `index_to_move`). Moves use the canonical
AlphaZero action space of `64 × 73 = 4672`:

| Plane range | Count | Meaning |
|---|---|---|
| 0–55 | 56 | "queen" moves: 8 directions × 7 distances |
| 56–63 | 8 | knight moves (the 8 L-jumps) |
| 64–72 | 9 | under-promotions: {N,B,R} × {left, fwd, right} |

Queen-promotions reuse the "queen" move planes, so they need no dedicated encoding.
The mapping is bijective on legal moves and is verified by round-trip tests (§4.5).

### 4.2 Network architecture

A single small, CPU-friendly residual CNN (`model.py`) with a shared trunk and two
heads — a form of multi-task learning, since the features useful for choosing a move
largely overlap with those useful for judging a position:

```
input (18 × 8 × 8)
  → Conv 3×3 + BN + ReLU                         (stem)
  → N × ResidualBlock   (default N=4, 64 channels) (trunk)
  → policy head: Conv 1×1 + BN + ReLU → FC → 4672 logits
  → value  head: Conv 1×1 + BN + ReLU → FC → ReLU → FC → tanh  ∈ [-1, 1]
```

The value head's `tanh` output is interpreted as expected game outcome from the
side-to-move's perspective (`+1` winning, `0` drawish, `-1` losing). Residual
("skip") connections let gradients bypass each block so the tower trains without
vanishing signal.

### 4.3 Search: PUCT Monte-Carlo Tree Search

`mcts.py` is the algorithmic heart. The classic Monte-Carlo rollout is replaced by a
single call to the network's **value head**, and the **policy head** supplies a
prior that focuses the search. Each of `num_simulations` (default 100) iterations
walks the tree through four phases:

1. **Selection** — descend from the root by maximising the PUCT score until a
   not-yet-expanded node is reached.
2. **Expansion** — evaluate that leaf with the network to get a value and a prior
   distribution; create one child per legal move.
3. **Evaluation** — the leaf's value is the network estimate (or the exact result
   at a terminal node).
4. **Back-up** — propagate the value up the path, **flipping its sign at every ply**
   because the players alternate.

The selection score balances exploitation and exploration:

```
score(child) = Q(child) + c_puct · P(child) · √N(parent) / (1 + N(child))
```

During self-play, **Dirichlet noise** (`alpha = 0.3`, `epsilon = 0.25`) is mixed
into the root priors to keep exploration alive. To keep memory modest, nodes store
only the move that reaches them and reconstruct the position by replaying from the
root during selection.

### 4.4 Self-play and training targets

Self-play (`self_play.py`) is the "reinforcement": the engine is its own opponent
and its own teacher. For every move it records a training example:

- **state** — the canonical board tensor the network saw;
- **policy target** — the **MCTS visit-count distribution** (a *better* policy than
  the raw network output, because search refined it — this "search as policy
  improvement" is what makes the network get stronger);
- **value target** — filled in at game end: `+1 / 0 / −1` for the side to move.

For the first `temperature_moves = 15` plies, moves are sampled in proportion to
visit counts (diverse openings); thereafter play is greedy. A `max_moves = 200`
cap prevents weak early networks from shuffling forever.

**Loss** (`alphazero_loss`, `training.py`):

```
loss = −Σ πᵢ · log_softmax(policy_logits)ᵢ  +  c · (z − v)²
        └── cross-entropy to MCTS target ──┘     └─ MSE to outcome ─┘
```

The outer loop runs `num_iterations` rounds of *(generate self-play → train on the
replay buffer for `epochs_per_iteration` epochs)*. Self-play games are independent,
so a multiprocessing path fans them across CPU cores — the single biggest practical
speed-up, since self-play dominates wall-clock time.

### 4.5 Correctness: making the silent bugs loud

AlphaZero is conceptually simple but subtle to get *right*: the dangerous bugs don't
crash, they silently stop the network from learning. Two are called out and pinned
down with tests:

- **Perspective / colour symmetry** — the board must be mirrored for Black and value
  signs flipped consistently up the tree. Tested by
  `test_canonical_encoding_is_color_symmetric` and `test_material_score_perspective`.
- **Move-encoding integrity** — the 4672-space mapping must round-trip, including
  promotions. Tested by `test_move_encoding_roundtrip_white` /
  `_promotions` and `test_move_index_in_range_for_all_legal_start_moves`.

The full suite (25 tests, run in CI) also covers terminal values
(`test_terminal_value_checkmate`, `test_stalemate_is_a_draw`,
`test_insufficient_material_is_a_draw`), search behaviour
(`test_mcts_returns_valid_distribution`, `test_mcts_is_deterministic_without_noise`,
`test_search_finds_mate_in_one`), and the Elo math
(`test_elo_difference_monotonic_and_symmetric`).

### 4.6 A second, classical engine (and why)

An untrained network + MCTS shuffles winning positions into draws, which makes for a
poor playable demo. Rather than disguise that, the project ships a **second**,
dependency-light engine (`search.py`) as the play-time opponent and is explicit on
the site about which engine is doing what:

- **Negamax + alpha-beta** at fixed depth, with **quiescence search** on captures so
  it doesn't blunder at the horizon;
- **Move ordering** (captures first, most-valuable-victim) for pruning;
- Evaluation = **material** + an **endgame "mop-up"** term (drive the lone king to a
  corner, bring your own king up, squeeze escape squares) so it can actually deliver
  basic mates;
- Draws score `0` and mate scores are depth-adjusted, so a winning side refuses to
  repeat and prefers the *fastest* mate.

This is itself an engineering-judgment result (§7): use each tool where it is
genuinely strong, and tell the user the truth about what they are playing.

---

## 5. Experiments and Results

> **Integrity note.** The committed `logs/training.log` currently contains only
> short **smoke-test** runs (2 games; policy loss ≈ 7.3 → 5.9). All figures below
> must be regenerated from a real run using the logging spec at the end of this
> document. Report what actually happened.

### 5.1 Training dynamics
Plot policy loss, value loss, and total loss vs. iteration (`plot_progress.py`
already renders this to `assets/training_progress.png`). Report the starting and
ending policy loss over `⟨FILL: N⟩` iterations / `⟨FILL: G⟩` self-play games, and
comment on the value-loss behaviour (see §6 — a value head that collapses toward `0`
is the draw-cycle signature).

### 5.2 Playing strength
Using `evaluation.py`: play the greedy `NetworkAgent` against the `RandomAgent`
baseline over `⟨FILL: M⟩` games and report **win/draw/loss** and the implied **Elo
difference**. A network that cannot reliably beat random has not learned anything;
the honest early result here is roughly a 50% draw rate against random — the entry
point to §6.

### 5.3 Two experiments: *what* causes the draw cycle?

Rather than assert that the plateau is "just a compute limit," this section runs
**two controlled experiments** that isolate two competing causes. Both hold the
code fixed and vary exactly one thing; both are produced by a single driver
(`run_experiment.py`) and the numbers below are auto-written to
`docs/experiment-results.md` (paste them in).

The two headline metrics in both experiments are the **self-play non-decisive
rate** (fraction of games *not* won by checkmate — the draw cycle on a graph) and
**Elo vs. random**.

**Experiment A — compute scaling.** *Hypothesis: the plateau is a resource
ceiling.* Run the identical pipeline at a small baseline (~⟨FILL⟩ games) and a
larger scaled run (~⟨FILL⟩ games on GPU); the material assist is `0` in both. If
the hypothesis holds, the scaled run's non-decisive rate falls and Elo rises where
the baseline flatlines.

| Run | Self-play games | Non-decisive rate | Elo vs random |
|---|---|---|---|
| baseline | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |
| scaled | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |

**Experiment B — self-play signal at *fixed* compute.** *Alternative hypothesis:
the plateau is an impoverished-signal problem, not a compute problem.* The draw
cycle is ultimately a **value-learning** failure — if every self-play game draws,
the value target `z ≈ 0` everywhere and the value head can only learn "even" (§6).
This experiment tests whether enriching the self-play *search signal* — blending a
material term into leaf evaluation during self-play (`material_weight = w`), so the
search converts advantages and produces **decisive** games the value head can learn
from — breaks the cycle **without any extra compute**. Same compute as the
baseline; only the self-play material weight differs.

| Run | Self-play games | Material w | Non-decisive rate | Elo vs random |
|---|---|---|---|---|
| baseline | ⟨FILL⟩ | 0.00 | ⟨FILL⟩ | ⟨FILL⟩ |
| assisted | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ | ⟨FILL⟩ |

**Reading the two together (the point of the design).** Comparing which lever
moves the metrics *distinguishes the cause*, rather than assuming it:

| Compute helps (A)? | Signal helps (B)? | Interpretation |
|---|---|---|
| yes | no | classic **compute ceiling** |
| no | yes | an **impoverished self-play signal**, not raw compute |
| yes | yes | both compound |
| no | no | cause lies elsewhere — e.g. encoding (move-history planes, §8) |

This turns a *known* result ("AlphaZero needs compute") into an actual
*investigation* of a specific failure mode — the contribution here is the
diagnosis and the controlled test, not the (well-known) algorithm.

### 5.4 Ablations *(optional, cheap, strengthens the report)*
- **`material_weight`** blended into leaf evaluation at play time (0.0 = pure
  AlphaZero) vs. a value like 0.85 — effect on won-position conversion.
- **`num_simulations`** per move vs. strength.
- Classical engine **with vs. without piece-square tables** (the "weird openings"
  fix) — a clean demonstration that a little placed domain knowledge beats brute
  force.

---

## 6. The Draw Cycle

The central empirical finding, and the report's intellectual core.

**The observation.** The machinery worked — the network trained, loss fell, self-play
fed itself — and then strength plateaued. The engine could *win* material but not
*convert* it: up a queen, it would shuffle until the game petered out into a
repetition draw. Against a random-move opponent it drew roughly half its games.

**The mechanism.** This is a property of how AlphaZero learns, not a coding fault.
The value head is trained to predict the game outcome `z`. If nearly every self-play
game is a **draw**, then `z ≈ 0` for almost all training targets, so the only thing
the value head can learn is *"every position is even."* A value head that always
returns ≈0 cannot tell the search that being up a queen is *good*, so MCTS never
prioritises converting the advantage, so games stay drawn, so the next batch of
targets is again all-draws. A **self-reinforcing trap** — the *draw cycle*.

**The diagnosis.** Nothing was broken, which is what made it hard. The break came
from **logging the termination reason of every self-play game**: the histogram was
dominated by draws, which pointed at the value-target collapse rather than a bug.
The deeper lesson — and the hardest thing the project taught — is that *"my code is
wrong"* and *"my code is right but under-resourced"* look identical from the outside;
only instrumentation tells them apart. AlphaZero trained on tens of millions of
games; a laptop plays under a hundred.

**Why §5.3 is the right test.** Because the two explanations are externally
indistinguishable, the honest move is to *test* the resource hypothesis by scaling
compute while holding the code fixed — converting a plausible story into evidence.

**A likely secondary factor.** Real AlphaZero includes **move-history planes** in the
board encoding, which help the network reason about repetition; their absence here
plausibly deepens the draw cycle and is a natural future-work item (§8).

---

## 7. Engineering Decisions and Challenges  *(placeholder — content ready)*
Two-engine architecture and honest labeling (§4.6); deployment under a free-tier
constraint (a multi-GB PyTorch service was infeasible, so the site ships a tiny
PyTorch-free service and runs the trained network in-browser via **ONNX**, with a
JS↔Python board-encoding **self-check** before trusting it); and a memory/threading
fix (parallel self-play oversubscribed CPU/BLAS threads and exhausted memory —
capping thread counts and falling back to sequential self-play under pressure).

## 8. Limitations and Future Work
- **Move-history planes (the encoding hypothesis).** Real AlphaZero stacks the last
  *T* positions in its input, which helps the network reason about repetition — a
  plausible *third* cause of the draw cycle beyond §5.3's two. Adding them here is
  non-trivial because MCTS clones the board with `stack=False` and rebuilds
  positions by replaying from the root, so pre-root history is not consistently
  available inside the search. Doing it *correctly* requires threading recent
  history through the search so self-play and leaf evaluation see the *same*
  encoding — worth doing precisely because a botched version would reintroduce the
  silent encoding bugs §4.5 guards against. This is the natural next experiment: at
  fixed compute, does adding history planes lower the non-decisive rate?
- Rent GPU compute *before* drawing strength conclusions.
- Larger network / more residual blocks; stronger baselines than random.

## 9. Conclusion  *(placeholder)*
The three AlphaZero pieces fit together in code, not just diagrams; the subtle bugs
are silent and must be tested for; "did not reach grandmaster strength" and "the
project failed" are different statements; and the honest version of a project teaches
more than a polished fiction.

## 10. Reproducibility  *(placeholder)*
Link the repo, pin `requirements-train.txt`, list the exact config used for each run,
and give the commands (`main.py` train / evaluate) needed to reproduce every figure.

---

## Appendix A — Logging spec for the training run (so §5 writes itself)

Capture all of the following so the results section is plug-and-play. Most already
exist in `training.py` / `evaluation.py`; the **termination histogram** is the one
piece of new instrumentation and is the direct evidence for §6.

**Per training iteration** (already logged — keep):
- iteration index, games this iteration, new examples, buffer size
- `total`, `policy`, and `value` loss
- wall-clock time

**Per evaluation checkpoint** (set `eval_every > 0`; run at both compute scales):
- Elo vs. `RandomAgent`, and the raw **win / draw / loss** counts
- **self-play draw rate** for the iteration

**New: self-play game-termination histogram** — the draw-cycle evidence. For each
self-play game, record the reason it ended and aggregate per iteration:
- checkmate · stalemate · threefold repetition · fifty-move · insufficient material
  · **hit `max_moves` cap**
(A distribution dominated by repetition/50-move/max-moves is the draw cycle on a graph.)

**Per run, save once:**
- the full `Config` used (games, `num_simulations`, `num_residual_blocks`, LR, etc.)
- device (CPU vs. GPU), total wall-clock, and total games played
- the training-curve PNG (`plot_progress.py`) and the raw metrics as CSV/JSON

**The two headline series to plot for §5.3** (baseline vs. scaled, overlaid):
1. self-play **draw rate** vs. iteration
2. **Elo vs. random** vs. iteration

If the scaled curve bends where the baseline flatlines, the experiment is done —
even partially. Set a **hard GPU budget cap** before starting; stop when the trend
is legible, not when the engine is "good."

### How the metrics are produced (tooling already in the repo)

- **Loss curves + Elo vs. iteration** — `plot_progress.py` parses `logs/training.log`
  (loss lines + `[eval]` lines) into a PNG. Enable Elo lines by training with
  `--eval-every N`.
- **Draw cycle: non-decisive rate + termination histogram** — `analyze_selfplay.py`
  reads the per-iteration PGNs written by `--save-pgn`, replays each game to its
  final position, classifies the termination (checkmate / stalemate / insufficient
  / repetition / fifty-move / max-moves-cap), and writes a per-iteration CSV
  (+ optional chart). The headline series is the **non-decisive rate** (fraction of
  games *not* won by checkmate) — the draw cycle on a graph.

Concrete commands (note the CLI uses `--mode`, e.g. `--mode train`):

```bash
# a run that captures everything §5/§6 need
python main.py --mode train --iterations N --games G --simulations S \
    --eval-every K --save-pgn --device cuda
python analyze_selfplay.py --pgn-dir pgn --label scaled \
    --out-csv logs/term_scaled.csv --plot assets/term_scaled.png
python plot_progress.py --out assets/progress_scaled.png
```

Run the same two analysis commands against the baseline run's PGNs/log (with their
own `--label`/paths) and overlay the two `non_decisive_rate` columns for §5.3.
