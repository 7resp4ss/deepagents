"""Langfuse helpers for Deep Agents instrumentation."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])
_LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


class _NoOpLangfuseContext:
    """Fallback Langfuse context that performs no actions."""

    def update_current_observation(self, **_: Any) -> None:
        return None

    def update_current_trace(self, **_: Any) -> None:
        return None

    def flush(self) -> None:
        return None


def _noop_observe(func: _F | None = None, **_: Any):
    def decorator(inner: _F) -> _F:
        return inner

    if func is not None and callable(func):
        return func
    return decorator


def _is_langfuse_configured() -> bool:
    return all(bool(os.environ.get(key)) for key in _LANGFUSE_KEYS)


def _safe_call(func: Callable[..., Any] | None, **kwargs: Any) -> None:
    if not callable(func):
        return None
    try:
        func(**kwargs)
    except Exception:
        return None


LANGFUSE_ENABLED = False
langfuse_context: Any = _NoOpLangfuseContext()
observe = _noop_observe

if _is_langfuse_configured():
    try:
        from langfuse import Langfuse as _Langfuse
        from langfuse import observe as _langfuse_observe
    except ImportError:
        pass
    else:
        client_kwargs: dict[str, Any] = {
            "public_key": os.environ.get("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.environ.get("LANGFUSE_SECRET_KEY"),
        }
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        if host:
            client_kwargs["base_url"] = host

        try:
            langfuse_context = _Langfuse(**client_kwargs)
        except Exception:
            langfuse_context = _NoOpLangfuseContext()
        else:
            observe = cast(Any, _langfuse_observe)
            LANGFUSE_ENABLED = True


def update_current_observation(**kwargs: Any) -> None:
    """Update the current Langfuse observation with additional metadata."""
    updater = getattr(langfuse_context, "update_current_observation", None)
    _safe_call(updater, **kwargs)


def update_current_trace(**kwargs: Any) -> None:
    """Update the current Langfuse trace with additional metadata."""
    updater = getattr(langfuse_context, "update_current_trace", None)
    _safe_call(updater, **kwargs)


def flush() -> None:
    """Flush pending Langfuse events if supported by the client."""
    flusher = getattr(langfuse_context, "flush", None)
    _safe_call(flusher)


__all__ = [
    "LANGFUSE_ENABLED",
    "langfuse_context",
    "observe",
    "update_current_observation",
    "update_current_trace",
    "flush",
]
