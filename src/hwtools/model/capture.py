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
        """Whether a frame was actually captured (vs still hunting).

        WAIT/RUN mean the scope is still hunting the trigger — no frame. Everything
        else means a frame is present: TRIGGERED / AUTO (free-run), and STOP — a
        completed SINGLE acquisition latches to STOP, and a deep read stops the
        scope to read frozen memory, so STOP is the normal state of a captured frame.
        """
        return self.trigger_status not in (TriggerStatus.WAIT, TriggerStatus.RUN)
