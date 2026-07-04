"""Layer 5 (stateful) — session glue over the pure analysis and thin drivers.

Holds the two things that must persist across tool calls within a session: the
:class:`~hwtools.session.store.CaptureStore` (frames addressed by handle) and the
single honest :func:`~hwtools.session.acquire.acquire` primitive. Everything below
this layer stays pure and instrument-agnostic; the MCP tool surface sits above it.
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
from hwtools.session.store import CaptureInfo, CaptureStore

__all__ = [
    "AcquiredFrame",
    "AutosetResult",
    "CaptureInfo",
    "CaptureStore",
    "LoopResult",
    "SingleShotResult",
    "acquire",
    "autoset",
    "capture_single",
    "capture_until_usable",
]
