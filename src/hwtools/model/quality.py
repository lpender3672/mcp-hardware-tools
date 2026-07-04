"""The adjustment a judgement implies.

``adjust`` (:mod:`hwtools.analysis.adjust`) maps an
:class:`~hwtools.model.reading.AcquireResult` plus the current setup to an
:class:`Adjustment` the loop overlays before re-capturing. The judgement type
itself lives in :mod:`hwtools.model.reading`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig


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
