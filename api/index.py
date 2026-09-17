"""
Vercel serverless entrypoint.

Vercel's Python runtime (`@vercel/python`) detects a module-level ASGI `app`
object in a file under `api/` and wraps it as a serverless function
automatically -- this file just re-exports the real FastAPI app from
src/app/main.py, so there's a single source of truth for the application
itself (this file has no logic of its own).
"""

import os
import sys

# Mirrors tests/conftest.py's explicit sys.path insertion, so the `src.app...`
# absolute imports resolve the same way locally, in tests, and on Vercel,
# regardless of whatever working directory the platform actually invokes
# this from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.app.main import app  # noqa: E402

__all__ = ["app"]
