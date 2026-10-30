"""web/server.py
================

A small, **dependency-light** FastAPI backend that puts the chess engine behind a
web UI.  It uses the classical alpha-beta searcher in :mod:`search` (no PyTorch),
so the deployed service is tiny, fast and fits a free cloud instance -- and it
plays genuinely sound chess (captures, avoids blunders, delivers mates).

The server is **stateless**: the browser (chess.js) owns the game and sends the
current position as a FEN with each request.  The server answers:

* ``/analyze``     -- evaluation + recommended moves for a position;
* ``/engine_move`` -- the move the engine would play here.

Run it with::

    python web/server.py        # then open http://127.0.0.1:8000
    # or:  python main.py --mode serve
"""

from __future__ import annotations

import mimetypes
import os
import sys
from typing import Optional

# Allow ``python web/server.py`` to import the engine modules in the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chess
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

import search

# Search depth (plies).  Higher = stronger but slower.  3 keeps the web snappy.
_DEPTH = int(os.environ.get("RLCHESS_DEPTH", "3"))

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

@app.middleware("http")
async def edge_cache(request: Request, call_next):
    """Let Vercel's edge cache the page and its static files.

    On Vercel every request otherwise wakes the Python function, and a cold one
    took ~1.7 s to first byte (Lighthouse, 2026-10-30). s-maxage is honoured by
    the edge only; browsers still revalidate (max-age=0), and each deployment
    starts with an empty edge cache, so a new build is never served stale.
    """
    response = await call_next(request)
    path = request.url.path
    if request.method == "GET" and response.status_code == 200 and (path == "/" or path.startswith("/static/")):
        response.headers["Cache-Control"] = "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800"
    return response


# Python's MIME table lacks .woff2 on some systems, which served the self-hosted
# fonts as application/octet-stream.
mimetypes.add_type("font/woff2", ".woff2")

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class PositionRequest(BaseModel):
    """A position to reason about, plus optional per-request knobs."""

    fen: str
    depth: Optional[int] = None
    top_n: int = 3
    # Accepted for backwards-compatibility with older clients; ignored.
    simulations: Optional[int] = None


def _analyse(req: PositionRequest) -> dict:
    """Run the searcher on the requested position and return a JSON dict."""
    board = chess.Board(req.fen)
    depth = req.depth if req.depth is not None else _DEPTH
    return search.analyze(board, depth=depth, top_n=req.top_n)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    """Serve the single-page board UI."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """A real page for browsers that hit a missing URL; JSON for everyone else."""
    if exc.status_code == 404 and "text/html" in request.headers.get("accept", ""):
        return FileResponse(os.path.join(_STATIC_DIR, "404.html"), status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/health")
def health() -> dict:
    """Lightweight readiness probe (also handy for pre-warming a cold instance)."""
    return {"status": "ok", "depth": _DEPTH}


@app.post("/analyze")
def analyze(req: PositionRequest) -> dict:
    """Return the engine's evaluation and recommended moves for a position."""
    return _analyse(req)


@app.post("/engine_move")
def engine_move(req: PositionRequest) -> dict:
    """Return the move the engine would play (plus the supporting analysis)."""
    return _analyse(req)  # analyze() already includes the chosen "move"


def main() -> None:
    """Launch the development server (``python web/server.py``)."""
    import uvicorn

    host = os.environ.get("RLCHESS_HOST", "127.0.0.1")
    port = int(os.environ.get("RLCHESS_PORT", "8000"))
    print(f"[web] serving on http://{host}:{port} (alpha-beta depth {_DEPTH})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
