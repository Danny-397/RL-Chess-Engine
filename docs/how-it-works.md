# How it works: the math behind the engine

This document explains the ideas that make the engine learn, aimed at a reader
who knows some programming and basic probability but hasn't studied
reinforcement learning. It connects each idea to the file that implements it.

---

## 1. The big idea: learning from self-play

The engine is given **only the rules of chess**. It has no opening book, no
hand-written evaluation of "a queen is worth nine pawns," nothing. It improves
through a loop:

1. **Play games against itself** using its current brain.
2. **Look at how those games turned out** and adjust the brain so it would make
   better decisions next time.
3. **Repeat.** Each better brain produces better games, which teach an even
   better brain.

This is the central loop of AlphaZero, and it is implemented across
[`self_play.py`](../self_play.py) (step 1) and [`training.py`](../training.py)
(step 2).

The subtle part is step 2: *what does "better" mean* when no human is labelling
the moves? The answer is the rest of this document.

---

## 2. The brain: one network, two outputs

The "brain" is a neural network (a residual CNN, in [`model.py`](../model.py)).
Given a board position $s$, it outputs two things:

$$f_\theta(s) = (\mathbf{p}, v)$$

- $\mathbf{p}$ — a **policy**: a probability for every possible move, i.e. "how
  promising does each move look at a glance?"
- $v \in [-1, 1]$ — a **value**: "who is winning?" ($+1$ = the side to move is
  certainly winning, $-1$ = certainly losing, $0$ = even).

$\theta$ is the set of network weights — the thing training adjusts.

A position is fed to the network as an $18\times 8\times 8$ stack of planes
(piece locations, castling rights, etc.), always shown **from the perspective of
the side to move** so the network never has to care whether it is White or Black
(see [`chess_game.py`](../chess_game.py)).

The network alone is a fast but shallow player: it reacts to a position without
thinking ahead. The next ingredient adds the thinking.

---

## 3. Thinking ahead: Monte-Carlo Tree Search (MCTS)

Strong chess requires *looking ahead*. MCTS (in [`mcts.py`](../mcts.py)) builds a
search tree of possible future positions, using the network to decide **where to
look** and **how good** the positions it reaches are.

Each move (edge) in the tree stores statistics:

- $N(s,a)$ — how many times we've tried move $a$ from position $s$,
- $W(s,a)$ — the total value found beneath it,
- $Q(s,a) = W(s,a)/N(s,a)$ — the **average** value found beneath it,
- $P(s,a)$ — the network's prior probability for $a$ (from $\mathbf{p}$).

One **simulation** does three things:

**Selection.** Starting at the root, repeatedly pick the move that maximises the
**PUCT** score until reaching a leaf (an unexplored position):

$$a^\* = \arg\max_a \Big[\, Q(s,a) + c_{\text{puct}}\, P(s,a)\, \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \,\Big]$$

The first term, $Q$, is **exploitation** — prefer moves that have looked good. The
second term is **exploration** — try moves the network likes ($P$ high) or that
we haven't visited much ($N$ low). The constant $c_{\text{puct}}$ sets the
balance. This single formula is why search and the network cooperate instead of
competing.

**Expansion & evaluation.** At the leaf, ask the network for $(\mathbf{p}, v)$.
The priors $\mathbf{p}$ initialise the children; the value $v$ estimates the leaf.
If the leaf is checkmate/stalemate we use the true game result instead.

**Backup.** Propagate $v$ back up the path, adding it to each $W$ and incrementing
each $N$ — **flipping its sign at every level**, because a position that's good
for me is exactly as bad for my opponent one move earlier. (A single missed sign
flip here silently breaks the engine — so there's a test for it.)

After hundreds of simulations, the move that was *visited most often*,
$N(s_\text{root}, a)$, is the search's considered best move — a far stronger
judgement than the network's raw prior $P$.

---

## 4. From search to training data

Here is the elegant trick. For each position in a self-play game we record:

- the position $s$,
- the **search policy** $\boldsymbol{\pi}$, where $\pi(a) \propto N(s_\text{root}, a)^{1/\tau}$
  — the normalised visit counts ($\tau$ is a temperature controlling
  exploration vs. greediness),
- a placeholder for the value.

When the game ends with result $z \in \{+1, 0, -1\}$, every recorded position is
labelled with $z$ **from that position's point of view** (so a position from the
eventual winner's turn gets $+1$). This happens in
[`self_play.py`](../self_play.py).

Why is $\boldsymbol{\pi}$ a good training target? Because MCTS used the network
*and then improved on it by searching*. So $\boldsymbol{\pi}$ is a strictly
better policy than the network's raw $\mathbf{p}$. Training the network to imitate
$\boldsymbol{\pi}$ pulls it toward the search's wisdom — this is **policy
improvement**, the engine of classical reinforcement learning, with MCTS playing
the role of the "improvement operator."

---

## 5. The loss function

Training (in [`training.py`](../training.py)) nudges $\theta$ so the network's
outputs match the data $(s, \boldsymbol{\pi}, z)$. The loss for one example is:

$$\mathcal{L}(\theta) = \underbrace{(v_\theta(s) - z)^2}_{\text{value: MSE}}
\;-\; \underbrace{\boldsymbol{\pi}^\top \log \mathbf{p}_\theta(s)}_{\text{policy: cross-entropy}}
\;+\; \underbrace{\lambda \lVert \theta \rVert^2}_{\text{weight decay}}$$

- The **value term** teaches the network to predict the eventual outcome — to
  *evaluate* positions.
- The **policy term** teaches it to predict the search's preferred moves — to
  *choose* moves. (Illegal moves are masked out before the cross-entropy.)
- **Weight decay** ($\lambda$) discourages over-large weights to reduce
  overfitting.

Minimising both terms at once is what makes a *single* network good at both
halves of the PUCT formula.

---

## 6. Why the loop converges to strong play

Putting it together, each iteration is one step of **generalised policy
iteration**:

- **Policy improvement:** MCTS turns the current network into a better policy
  $\boldsymbol{\pi}$ (by searching) and a better value estimate (by seeing real
  outcomes $z$).
- **Policy evaluation / projection:** training compresses that improvement back
  into the network's weights.

Because the improved network then drives the next round of self-play, the quality
of the *training data itself* rises every iteration. The system pulls itself up
by its own bootstraps — no external teacher required. Strength is measured
directly in [`evaluation.py`](../evaluation.py) by playing matches and converting
the score to Elo.

---

## 7. Correctness pitfalls (and how this repo guards them)

AlphaZero is conceptually simple but easy to get subtly wrong. The trickiest
parts, each covered by a test in [`tests/`](../tests/):

| Pitfall | Why it's dangerous | Guard |
|---|---|---|
| **Perspective/sign errors** | Value must flip every ply; a single missed flip teaches the net to help its opponent. | Color-symmetry + checkmate-sign tests. |
| **Move encoding** | The $8\times8\times73=4672$ action space (queen/knight moves + under-promotions) must round-trip exactly. | Encoding round-trip tests, incl. promotions. |
| **Search target** | Using raw priors instead of visit counts removes the improvement step. | Self-play uses $\boldsymbol{\pi}$ from $N$; distribution tests. |
| **Hidden randomness** | Non-reproducible search is impossible to debug. | Determinism test (no-noise MCTS is reproducible). |

---

### Further reading

- Silver et al., *"Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm"* (AlphaZero), 2017.
- Silver et al., *"Mastering the game of Go without human knowledge"*
  (AlphaGo Zero), 2017 — the source of the PUCT formulation used here.
