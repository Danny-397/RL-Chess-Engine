# Building this: a reflection

Notes on what I set out to do, what actually happened, and what I learned. I'm
keeping this honest — including the parts that didn't work — because those are the
parts I learned the most from.

## What I wanted

I wanted to understand AlphaZero by *building* it, not just reading about it. The
idea that a program could start from random moves, play only against itself, and
teach itself chess with **zero human strategy** felt like magic. I didn't want to
`pip install` someone's engine and watch it win — I wanted to write the neural
network, the search, and the self-play loop myself and see the magic happen (or
fail) with my own code.

## What actually happened

The machinery worked. The network trained, the loss dropped sharply, the search
ran, the self-play loop generated its own data and fed it back. Watching the
policy loss fall from 4.6 to ~1.0 — the network learning to imitate its own
search — is still the most satisfying graph I've ever produced, because I wrote
every piece of what made it move.

And then it plateaued. My engine could *win* material but couldn't *convert* it
into a checkmate. It would go up a queen and then shuffle pieces until the game
petered out into a draw by repetition. Against a bot playing random legal moves,
my "learning" engine drew about half its games. That is a humbling result to stare
at after weeks of work.

## The bug that wasn't a bug

The frustrating part was that nothing was *broken*. I went looking for a bug for
days. I added logging to record *why* every self-play game ended, and the pattern
jumped out: almost every game ended in a draw. That mattered because of how
AlphaZero learns — the network's "value head" is trained to predict who won, and
if every game is a draw, the only thing it can learn is "the position is even."
A value head that always says "even" can't tell the search that being up a queen
is *good*, so the search never tries to convert, so the games stay drawn. A
self-reinforcing trap. I named it the **draw cycle**.

The way out isn't a code fix — it's *scale*. AlphaZero played tens of millions of
games. My laptop played under a hundred. The algorithm was correct; the limiting
resource was compute I didn't have. Learning to tell the difference between "my
code is wrong" and "my code is right but under-resourced" was, weirdly, the most
important thing this project taught me. They look identical from the outside. Only
the logging told them apart.

## The decision I'm most proud of

I had a website to ship and a neural engine too weak to be a fair opponent. I could
have hidden that — quietly propped it up and called it "the AI." Instead I made a
deliberate call: build a *second* engine — a classical alpha-beta searcher with
material evaluation, piece-square tables, quiescence, and an endgame "mop-up" term
so it actually delivers checkmate — and be explicit on the site about which engine
is doing what. The trained network is still there to play (raw, or paired with a
small search), clearly labelled, honestly described as the still-learning project.

That felt, at first, like admitting failure. I've come to see it as the opposite:
it's engineering judgement — using each tool where it's actually strong, and
telling the user the truth about what they're playing. A demo that lies about
itself is worse than one that's honest about its limits.

## Other things that fought back

- **"Weird" openings.** My first classical engine only counted material, so every
  opening move scored zero and it played nonsense — knights to the rim, rooks
  shuffling. Adding piece-square tables (a table that says "knights belong near the
  centre") fixed it with no training at all. A good reminder that a little domain
  knowledge, well-placed, can beat a lot of brute force.
- **It wouldn't finish won games.** Material counting can't mate — a lone king is
  worth the same in the centre or the corner. I added an endgame term that rewards
  driving the enemy king to the edge and keeping my king close. Suddenly it could
  *finish*.
- **Deployment.** A PyTorch web service is gigabytes — too big for a free host. But
  because the classical engine and the board-encoding needed no PyTorch, I could
  ship the whole site as a tiny service, and even run the trained network in the
  browser itself (exported to ONNX) with a self-check that verifies my JavaScript
  board-encoding matches my Python one before trusting it. Constraints forced a
  cleaner architecture than I'd have designed if compute were free.
- **Crashes under memory pressure.** Parallel self-play spawned workers that
  oversubscribed the CPU and exhausted memory. Capping the math-library thread
  count and falling back to sequential self-play when memory is tight fixed it.

## What I'd do differently

Rent a GPU *first*, next time, before drawing conclusions about strength — I spent
a lot of energy debugging a "weakness" that was really just a compute ceiling. I'd
also add move-history planes to the board encoding (real AlphaZero uses them to
handle repetitions), which might have softened the draw cycle on its own.

## What I actually learned

- How the three pieces of AlphaZero — network, search, self-play — actually fit
  together, at the level of code, not just diagrams.
- That the subtle bugs (a value sign flipped in the wrong place, the board not
  mirrored for Black) are silent — they don't crash, they just quietly stop the
  thing from learning. That's why I wrote tests for exactly those.
- That "it didn't reach grandmaster strength" and "the project failed" are not the
  same statement. The learning worked and is visible; strength is a compute story.
- That the honest version of a project — the one that says what works, what
  doesn't, and why — is more interesting than a version that pretends everything
  went to plan. It certainly taught me more.
