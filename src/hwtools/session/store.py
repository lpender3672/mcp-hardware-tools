"""Server-side store of acquired frames, addressed by handle.

The missing piece the expansion goals demand: a captured frame becomes an
*addressable artifact* that every downstream lens (judge, decode, describe,
classify) computes over by id — never re-acquiring. This is non-negotiable for a
one-shot: the event happens once, so the single stored frame must serve every
question asked of it.

Management guarantees (see ``docs/signal-infrastructure-plan.md``):

* **IDs are monotonic and never reused** — ``cap-7`` is that frame forever, or it
  is gone. A stale reference can only ever *fail*, never resolve to a different
  frame.
* **Eviction is loud** — :meth:`get` on an evicted id raises, naming the live
  captures; it never returns ``None`` or the wrong frame.
* **Bounded by bytes, not count** — a deep frame is ~96 MB/channel, so a count cap
  is meaningless; eviction is driven by a resident-byte budget.
* **Two tiers** — *ephemeral* working frames self-reclaim (a fresh :meth:`put`
  drops the previous ephemeral, so a convergence loop never accumulates); *kept*
  frames survive until an explicit :meth:`drop`, and are never silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, SweepMode

#: Default resident-byte budget for the store (~256 MB). Deep frames are large.
DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class CaptureInfo:
    """Provenance of one stored frame — what it is, how big, kept or not."""

    capture_id: str
    bytes: int
    n_samples: int
    channels: tuple[ChannelId, ...]
    triggered: bool
    kept: bool = False
    keep_reason: str | None = None
    label: str | None = None
    sweep: SweepMode | None = None


@dataclass
class _Entry:
    capture: Capture
    info: CaptureInfo


def _capture_bytes(capture: Capture) -> int:
    return int(sum(wf.samples.nbytes for wf in capture.waveforms.values()))


def _capture_samples(capture: Capture) -> int:
    return max((wf.n for wf in capture.waveforms.values()), default=0)


class CaptureStore:
    """A bounded, handle-addressed store of immutable capture frames."""

    def __init__(self, *, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self._budget = budget_bytes
        self._entries: dict[str, _Entry] = {}  # insertion-ordered (oldest first)
        self._counter = 0

    # -- writing --------------------------------------------------------------

    def put(
        self,
        capture: Capture,
        *,
        keep: bool = False,
        label: str | None = None,
        keep_reason: str | None = None,
        sweep: SweepMode | None = None,
    ) -> str:
        """Store a frame and return its fresh, never-reused handle (``cap-N``).

        A fresh put drops the previous *ephemeral* working frame (loops don't
        accumulate). Kept frames survive. Raises if a single frame exceeds the
        budget, or if only kept frames remain and there is no room — never silently
        drops a kept frame.
        """
        nbytes = _capture_bytes(capture)
        if nbytes > self._budget:
            raise ValueError(
                f"capture is {nbytes} bytes, over the {self._budget}-byte store budget; "
                f"reduce the memory depth or raise the budget"
            )
        # A new acquisition supersedes the previous working frame.
        for cid in [c for c, e in self._entries.items() if not e.info.kept]:
            del self._entries[cid]
        # Only kept frames remain now; if they leave no room, refuse loudly rather
        # than silently evict something the agent deliberately kept.
        if self.resident_bytes() + nbytes > self._budget:
            raise MemoryError(
                f"storing {nbytes} bytes would exceed the {self._budget}-byte budget; "
                f"kept frames occupy {self.resident_bytes()} bytes — drop one explicitly "
                f"(drop()/clear()) before capturing"
            )

        self._counter += 1
        capture_id = f"cap-{self._counter}"
        info = CaptureInfo(
            capture_id=capture_id,
            bytes=nbytes,
            n_samples=_capture_samples(capture),
            channels=tuple(capture.waveforms),
            triggered=capture.triggered,
            kept=keep,
            keep_reason=keep_reason if keep else None,
            label=label,
            sweep=sweep,
        )
        self._entries[capture_id] = _Entry(capture=capture, info=info)
        return capture_id

    def keep(
        self, capture_id: str, *, label: str | None = None, reason: str | None = None
    ) -> None:
        """Promote a frame to *kept*, exempting it from ephemeral auto-eviction."""
        entry = self._resolve(capture_id)
        entry.info = replace(
            entry.info,
            kept=True,
            keep_reason=reason or entry.info.keep_reason,
            label=label if label is not None else entry.info.label,
        )

    def drop(self, capture_id: str) -> None:
        """Explicitly reclaim one frame (resolves ``latest``/labels to its real id)."""
        entry = self._resolve(capture_id)
        del self._entries[entry.info.capture_id]

    def clear_ephemeral(self) -> None:
        """Drop every non-kept frame."""
        for cid in [c for c, e in self._entries.items() if not e.info.kept]:
            del self._entries[cid]

    def clear(self) -> None:
        """Drop every frame, kept or not."""
        self._entries.clear()

    # -- reading --------------------------------------------------------------

    def get(self, capture_id: str) -> Capture:
        """Resolve a handle (``cap-N``, a label, or ``latest``) to its frame.

        Raises :class:`KeyError` naming the live captures if the handle is unknown
        or evicted — never returns ``None`` or a different frame.
        """
        return self._resolve(capture_id).capture

    def info(self, capture_id: str) -> CaptureInfo:
        """Provenance for one frame."""
        return self._resolve(capture_id).info

    def list(self) -> list[CaptureInfo]:
        """Every resident frame's provenance, oldest first — the audit view."""
        return [entry.info for entry in self._entries.values()]

    def latest_id(self) -> str | None:
        """The most recent frame's id, or ``None`` if the store is empty."""
        if not self._entries:
            return None
        return next(reversed(self._entries))

    def resident_bytes(self) -> int:
        return int(sum(e.info.bytes for e in self._entries.values()))

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals ------------------------------------------------------------

    def _resolve(self, ref: str) -> _Entry:
        if not self._entries:
            raise KeyError(f"no captures in the store (requested {ref!r})")
        if ref == "latest":
            return next(reversed(self._entries.values()))
        if ref in self._entries:
            return self._entries[ref]
        # Fall back to a label match, most recent first.
        for cid in reversed(self._entries):
            if self._entries[cid].info.label == ref:
                return self._entries[cid]
        live = ", ".join(self._entries) or "(none)"
        raise KeyError(f"unknown or evicted capture {ref!r}; live captures: {live}")
