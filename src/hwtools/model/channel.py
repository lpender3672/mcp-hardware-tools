"""Vertical (per-channel) configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.ids import ChannelId, Coupling


class ChannelConfig(BaseModel):
    """How a single analog channel is set up: coupling, scale, offset, probe.

    Scale is volts-per-division (the front-panel knob), not full-scale; the full
    vertical window depends on the instrument's division count, so
    :meth:`full_scale_v` takes that as a parameter rather than hard-coding it.
    """

    model_config = ConfigDict(frozen=True)

    channel: ChannelId
    coupling: Coupling = Coupling.DC
    scale_v_per_div: float = Field(gt=0, description="Vertical sensitivity, volts/division.")
    offset_v: float = Field(default=0.0, description="Vertical offset of the channel reference.")
    probe_ratio: float = Field(default=10.0, gt=0, description="Probe attenuation ratio, e.g. 10x.")
    bandwidth_limit: bool = Field(default=False, description="Engage the input bandwidth limit.")
    invert: bool = False
    enabled: bool = True

    def full_scale_v(self, divisions: int) -> float:
        """Total displayable vertical span for a scope with ``divisions`` divisions."""
        if divisions <= 0:
            raise ValueError("divisions must be positive")
        return self.scale_v_per_div * divisions
