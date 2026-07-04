"""The decision-ready view of one acquisition — per channel and overall.

Produced by :func:`hwtools.analysis.judge.judge_capture` from a
:class:`~hwtools.model.capture.Capture`: it folds each channel's measurements
(vpp / midline / frequency) together with the clip/fill judgement into a single
payload the agent reasons over *without re-capturing to look*. Each channel echoes
the :class:`~hwtools.model.channel.ChannelConfig` it was taken with, so the agent
computes the next setup statelessly (it sees "taken at 1 V/div, vpp 6 V, clipping"
and sizes the next scale directly).

This is the lean *loop* view. Dense characterisation (histograms, spectra, the
joint time-vs-value grid) lives in :mod:`hwtools.analysis.describe`, not here — so
this type stays small and does not rot into a god-object as new consumers arrive.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.timebase import TimebaseConfig


class ChannelReading(BaseModel):
    """One channel's decision-ready numbers: what it was set to, what it measured."""

    model_config = ConfigDict(frozen=True)

    config: ChannelConfig = Field(description="The vertical config this was TAKEN with.")
    vpp: float = Field(description="Peak-to-peak volts (unreliable if clipping).")
    midline: float = Field(description="(vmax+vmin)/2 — the offset & trigger-level target.")
    mean: float = Field(description="DC level; mean != midline implies DC offset / asym duty.")
    frequency: float | None = Field(
        default=None, description="Dominant tone (Hz), or None for flat/noise — keep timebase."
    )
    clipping: bool = Field(default=False, description="A meaningful fraction sits at the rails.")
    clipped_fraction: float = Field(
        default=0.0, description="Fraction at the rails; 1% grazing vs 40% flat-top."
    )
    fill_fraction: float = Field(default=0.0, description="Vertical span used, 0..~1.")


class AcquireResult(BaseModel):
    """Everything the agent needs to judge one acquisition and decide the next move.

    ``capture_id`` ties this back to the stored frame (set by the session layer;
    ``None`` for a pure in-memory judgement). ``timebase`` echoes the horizontal
    context so the agent's decision stays stateless.
    """

    model_config = ConfigDict(frozen=True)

    triggered: bool
    channels: dict[ChannelId, ChannelReading] = Field(default_factory=dict)
    bandwidth_ok: bool = True
    timebase: TimebaseConfig | None = None
    capture_id: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Triggered and no channel clipped — the loop's stop condition."""
        return self.triggered and not any(c.clipping for c in self.channels.values())

    def with_capture_id(self, capture_id: str) -> AcquireResult:
        """Return a copy stamped with its stored-frame handle (session layer use)."""
        return self.model_copy(update={"capture_id": capture_id})
