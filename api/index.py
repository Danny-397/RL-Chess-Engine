"""Vercel serverless entrypoint.

Vercel's Python runtime serves an exported ASGI ``app``.  Our FastAPI app (the
board UI + the alpha-beta API) lives in :mod:`web.server`; we just re-export it
here and let ``vercel.json`` route every request to this function.

This works on Vercel because the play engine is the torch-free alpha-beta searcher
(``search.py``) -- only ``python-chess`` + ``fastapi`` -- which fits comfortably
inside a serverless function.
"""

import os
import sys

# Make the repo root importable from inside the /api function bundle.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.server import app  # noqa: E402  -- the ASGI app Vercel will serve

# Expose under the name Vercel looks for as well.
handler = app
