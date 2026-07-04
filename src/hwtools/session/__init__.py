"""Layer 5 (stateful) — the session: state that persists across tool calls.

Holds the pieces that must live across acquisitions within a session — the
:class:`~hwtools.session.store.CaptureStore` (frames addressed by handle) and the
single honest :func:`~hwtools.session.acquire.acquire` primitive — and the
:class:`~hwtools.session.scope_session.ScopeSession` facade that composes them with
the pure analysis lenses into the operations the agent performs. Everything below
this layer stays pure and instrument-agnostic; :mod:`hwtools.tools` is only the MCP
transport that exposes this session.
"""

from hwtools.session.acquire import AcquiredFrame, acquire
from hwtools.session.loop import (
    AutosetResult,
    LoopResult,
    SingleShotResult,
    autoset,
    capture_single,
    capture_until_usable,
)
from hwtools.session.scope_session import ScopeSession
from hwtools.session.store import CaptureInfo, CaptureStore

__all__ = [
    "AcquiredFrame",
    "AutosetResult",
    "CaptureInfo",
    "CaptureStore",
    "LoopResult",
    "ScopeSession",
    "SingleShotResult",
    "acquire",
    "autoset",
    "capture_single",
    "capture_until_usable",
]
