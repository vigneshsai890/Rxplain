"""The shared rate limiter.

A module singleton, because two places need the same instance: the route module
decorates each endpoint, and the application factory registers the instance and
its error handler. Two limiters would mean the decorator counted against a bucket
nobody enforced.

Why the limits live in mutable module state
-------------------------------------------
slowapi's decorators are applied at import time, before any ``Settings`` object
exists, so a limit cannot be captured as a value. Passing callables solves the
ordering but raises a subtler problem: a callable that reads ``get_settings()``
consults the *environment*, which means a limit passed explicitly to
``create_app(Settings(rate_limit_guide=...))`` is silently ignored - the limiter
happily enforces a different number than the application was configured with.

So the callables read a holder that :func:`configure` populates from the settings
the application was actually built with. It is deliberately narrow: two strings,
written once during startup, and reset between tests.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import Settings

_DEFAULT_FALLBACK = "120/minute"
_GUIDE_FALLBACK = "10/minute"

_limits: dict[str, str] = {"default": _DEFAULT_FALLBACK, "guide": _GUIDE_FALLBACK}


def configure(settings: Settings) -> None:
    """Adopt the limits of the application being started."""
    _limits["default"] = settings.rate_limit_default
    _limits["guide"] = settings.rate_limit_guide
    limiter.enabled = settings.rate_limit_enabled


def reset() -> None:
    """Restore fallback limits and clear counters. For tests."""
    _limits["default"] = _DEFAULT_FALLBACK
    _limits["guide"] = _GUIDE_FALLBACK
    limiter.enabled = True
    limiter.reset()


def default_limit() -> str:
    """Return the limit applied to ordinary page requests."""
    return _limits["default"]


def guide_limit() -> str:
    """Return the tighter limit for the transcription endpoint.

    ``POST /guide`` can trigger a transcription lasting tens of seconds, so an
    unauthenticated caller could otherwise amplify cost with very little traffic.
    """
    return _limits["guide"]


limiter = Limiter(key_func=get_remote_address, default_limits=[default_limit])
