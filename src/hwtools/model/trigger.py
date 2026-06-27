"""Trigger configuration.

A trigger is modelled as a discriminated union on ``kind`` so new trigger types
(pulse, slope, pattern, protocol) slot in behind the same ``TriggerConfig``
without the loop logic above changing. Today only the edge trigger exists.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.ids import ChannelId, Slope, SweepMode, TriggerCoupling


class EdgeTrigger(BaseModel):
    """Fire on a rising/falling/either edge of a channel crossing ``level_v``."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["edge"] = "edge"
    source: ChannelId
    level_v: float = Field(description="Trigger threshold voltage.")
    slope: Slope = Slope.RISING


# The discriminated union of every trigger type. Add ``| PulseTrigger`` etc. here.
AnyTrigger = Annotated[EdgeTrigger, Field(discriminator="kind")]


class TriggerConfig(BaseModel):
    """A trigger plus how the sweep re-arms and how the trigger path is coupled."""

    model_config = ConfigDict(frozen=True)

    trigger: AnyTrigger
    sweep: SweepMode = SweepMode.AUTO
    coupling: TriggerCoupling = TriggerCoupling.DC

    @property
    def source(self) -> ChannelId:
        """Convenience accessor for the triggering channel."""
        return self.trigger.source
