"""Decision-ready judgement of a capture, and the adjustment it implies.

These two types are the vocabulary of the self-correcting loop: ``judge`` (M5)
maps a :class:`~hwtools.model.capture.Capture` to a :class:`CaptureQuality`, and
``adjust`` maps that quality plus the current setup to an :class:`Adjustment` the
loop applies before re-capturing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig


class CaptureQuality(BaseModel):
    """Whether a capture is usable, and if not, why not."""

    model_config = ConfigDict(frozen=True)

    triggered: bool
    clipping: dict[ChannelId, bool] = Field(default_factory=dict)
    fill_fraction: dict[ChannelId, float] = Field(
        default_factory=dict, description="Vertical span used, 0..~1, per channel."
    )
    bandwidth_ok: bool = True
    notes: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A capture is usable when it triggered and no channel is clipped."""
        return self.triggered and not any(self.clipping.values())


class Adjustment(BaseModel):
    """A proposed change to the setup: replacement configs plus a rationale.

    Only the fields that need to change are populated; the loop overlays them on
    the current configuration. An empty adjustment means "nothing to change".
    """

    model_config = ConfigDict(frozen=True)

    channels: dict[ChannelId, ChannelConfig] = Field(default_factory=dict)
    timebase: TimebaseConfig | None = None
    trigger: TriggerConfig | None = None
    reason: str = ""

    def is_empty(self) -> bool:
        return not self.channels and self.timebase is None and self.trigger is None
