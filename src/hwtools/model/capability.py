"""What a given instrument can do — so tools and the loop adapt across vendors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.ids import ChannelId


class ScopeCapabilities(BaseModel):
    """Static description of a scope model's limits and feature set.

    The loop reads these to stay within range (don't request a sample rate the
    instrument can't reach) and to convert volts/seconds-per-division into full
    spans (``vertical_divisions`` / ``horizontal_divisions``) without hard-coding
    any one vendor's grid.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    n_channels: int = Field(gt=0)
    max_sample_rate_hz: float = Field(gt=0)
    analog_bandwidth_hz: float = Field(gt=0)
    vertical_divisions: int = Field(default=8, gt=0)
    horizontal_divisions: int = Field(default=12, gt=0)
    memory_depths: tuple[int, ...] = ()
    trigger_kinds: tuple[str, ...] = ("edge",)
    decoder_kinds: tuple[str, ...] = ()

    @property
    def channels(self) -> tuple[ChannelId, ...]:
        return tuple(ChannelId(i) for i in range(1, self.n_channels + 1))

    def has_channel(self, channel: ChannelId) -> bool:
        return int(channel) <= self.n_channels
