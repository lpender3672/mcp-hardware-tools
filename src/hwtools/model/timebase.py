"""Horizontal (timebase) configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.model.ids import TimebaseMode


class TimebaseConfig(BaseModel):
    """Horizontal scale/offset and acquisition mode.

    Scale is seconds-per-division; the captured time span depends on the
    instrument's horizontal division count, so :meth:`full_span_s` takes it as a
    parameter rather than hard-coding it.
    """

    model_config = ConfigDict(frozen=True)

    scale_s_per_div: float = Field(gt=0, description="Horizontal scale, seconds/division.")
    offset_s: float = Field(default=0.0, description="Horizontal (trigger) offset.")
    mode: TimebaseMode = TimebaseMode.MAIN

    def full_span_s(self, divisions: int) -> float:
        """Total captured time window for a scope with ``divisions`` divisions."""
        if divisions <= 0:
            raise ValueError("divisions must be positive")
        return self.scale_s_per_div * divisions
