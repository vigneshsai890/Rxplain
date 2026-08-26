"""Vercel entrypoint.

Vercel's Python runtime imports ``app`` from this module and serves it as an ASGI
handler. There is no long-running process, so the application must start with no
database connection to open and no migration to run: ``vercel.json`` sets
``STORAGE_BACKEND=memory``, which serves RxNorm resolution from the committed seed
in process. See :mod:`app.adapters.rxnorm_local`.

Two limits are worth knowing before relying on this target.

**Function duration.** A cold transcription with
``meta/llama-3.2-11b-vision-instruct`` was measured at 46.7s, against a 60s
ceiling on Vercel's Hobby plan. It fits, with little margin. A warm call was 3.5s.
If ``NVIDIA_API_KEY`` is left unset the fixture replayer runs instead and every
request is fast, which is the safer configuration for a demo.

**Statelessness.** Rate-limit counters and the openFDA cache live in process, so
each function instance keeps its own. Limits are therefore per instance rather
than global; a shared Redis backend is required to make them global.

Cloud Run does not have either constraint and runs the committed Dockerfile
unchanged, so it remains the better target for anything beyond a demo.
"""

from __future__ import annotations

from app.main import app

__all__ = ["app"]
