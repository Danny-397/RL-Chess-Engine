"""web/server.py
================

A small FastAPI backend that puts the engine behind a web UI.

The server is intentionally **stateless**: the browser (using chess.js) owns the
game and sends the current position as a FEN string with every request.  The
server just answers two questions about a position:

* ``/analyze``     -- what does the engine think? (evaluation + recommended moves)
* ``/engine_move`` -- what move would the engine play here?

Both reuse :func:`analysis.analyze_position`, so the web UI, the console ``hint``
command and the ``analyze`` CLI all share exactly the same engine logic.

Run it with::

    python web/server.py
    # then open http://127.0.0.1:8000

or via the unified CLI::

    python main.py --mode serve
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Optional

# Allow ``python web/server.py`` to import the engine modules in the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from chess_game import ChessGame
from model import ChessNet
from analysis import analyze_position, value_to_win_probability


# --------------------------------------------------------------------------- #
# Engine setup (load the model once at startup)
# --------------------------------------------------------------------------- #
_CHECKPOINT = os.environ.get("RLCHESS_CHECKPOINT", "checkpoints/example_checkpoint.pt")
_SIMULATIONS = int(os.environ.get("RLCHESS_SIMULATIONS", "100"))

_config = Config()
_config.mcts.num_simulations = _SIMULATIONS


def _load_network() -> ChessNet:
    """Load the configured checkpoint, or fall back to an untrained network."""
    device = _config.resolved_device()
    if os.path.exists(_CHECKPOINT):
        from training import load_checkpoint

        print(f"[web] loaded engine from {_CHECKPOINT}")
        return load_checkpoint(_CHECKPOINT, _config, device)
    print(f"[web] WARNING: '{_CHECKPOINT}' not found -- using an untrained network.")
    return ChessNet(_config.network).to(device).eval()


_network = _load_network()

app = FastAPI(title="RL-Chess-Engine")

# Allow a browser front-end hosted elsewhere (e.g. on Vercel) to call this API.
# Defaults to "*"; set RLCHESS_ALLOW_ORIGINS to a comma-separated list of exact
# origins (e.g. "https://my-chess.vercel.app") to lock it down in production.
_ALLOW_ORIGINS = [o.strip() for o in os.environ.get("RLCHESS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class PositionRequest(BaseModel):
    """A position to reason about, plus an optional per-request search budget."""

    fen: str
    simulations: Optional[int] = None
    top_n: int = 3


def _analyse_fen(fen: str, simulations: Optional[int], top_n: int) -> dict:
    """Shared helper: analyse a FEN and return a JSON-friendly dict.

    Includes a White-perspective win probability so the front-end's evaluation
    bar stays stable (it does not flip every time the side to move changes).
    """
    board = chess.Board(fen)
    game = ChessGame(board)

    # Terminal positions have no move to recommend.
    if game.is_terminal():
        return {
            "game_over": True,
            "result": board.result(claim_draw=True),
            "suggestions": [],
            "white_win_probability": None,
        }

    cfg = _config
    if simulations is not None:
        cfg = dataclasses.replace(
            _config, mcts=dataclasses.replace(_config.mcts, num_simulations=simulations)
        )

    analysis = analyze_position(_network, game, cfg, top_n=top_n)

    # Convert the side-to-move value into a fixed White-perspective probability.
    white_value = analysis.value if board.turn == chess.WHITE else -analysis.value
    suggestions: List[dict] = [
        {
            "san": s.san,
            "uci": s.move.uci(),
            "visit_fraction": s.visit_fraction,
            "value": s.value,
            "win_probability": s.win_probability,
        }
        for s in analysis.suggestions
    ]
    return {
        "game_over": False,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "value": analysis.value,
        "win_probability": analysis.win_probability,
        "white_win_probability": value_to_win_probability(white_value),
        "suggestions": suggestions,
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    """Serve the single-page board UI."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/analyze")
def analyze(req: PositionRequest) -> dict:
    """Return the engine's evaluation and recommended moves for a position."""
    return _analyse_fen(req.fen, req.simulations, req.top_n)


@app.post("/engine_move")
def engine_move(req: PositionRequest) -> dict:
    """Return the move the engine would play (plus the supporting analysis)."""
    info = _analyse_fen(req.fen, req.simulations, req.top_n)
    if info["game_over"] or not info["suggestions"]:
        return info
    best = info["suggestions"][0]
    info["move"] = {"uci": best["uci"], "san": best["san"],
                    "win_probability": best["win_probability"]}
    return info


def main() -> None:
    """Launch the development server (``python web/server.py``)."""
    import uvicorn

    host = os.environ.get("RLCHESS_HOST", "127.0.0.1")
    port = int(os.environ.get("RLCHESS_PORT", "8000"))
    print(f"[web] serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
