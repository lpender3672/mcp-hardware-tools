"""The result of one acquisition: per-channel waveforms plus instrument state."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.waveform import Waveform


class Capture(BaseModel):
    """Everything one acquisition produced, ready to judge or decode.

    Keyed by channel so the judge can reason per-channel. ``trigger_status``
    carries whether the frame was actually triggered — decision-ready state the
    self-correcting loop needs without re-reading the instrument.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    waveforms: dict[ChannelId, Waveform]
    trigger_status: TriggerStatus
    sample_rate_hz: float = Field(gt=0)
    memory_depth: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_keys(self) -> Capture:
        for ch, wf in self.waveforms.items():
            if wf.channel is not ch:
                raise ValueError(f"waveform under key {ch!r} has channel {wf.channel!r}")
        return self

    @property
    def channels(self) -> list[ChannelId]:
        return list(self.waveforms)

    @property
    def triggered(self) -> bool:
        """Whether a frame was actually captured (vs still hunting)."""
        return self.trigger_status in (TriggerStatus.TRIGGERED, TriggerStatus.AUTO)
