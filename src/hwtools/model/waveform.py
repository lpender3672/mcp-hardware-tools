"""Captured sample data.

A :class:`Waveform` is one channel's analog samples in volts on a uniform time
grid. A :class:`DigitalTrace` is the same idea after thresholding to logic
levels — the bridge from analog capture into the protocol decoders.

Both hold numpy arrays. Pydantic stores them via ``arbitrary_types_allowed`` and
serializes a compact *summary* (never the raw array) at the MCP boundary: dumping
a million samples as JSON helps no one, and the agent reasons over the summary
plus targeted measurements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from hwtools.model.ids import ChannelId

#: Fraction of samples at a rail above which a waveform counts as clipped. A few
#: rail-grazing samples (logic-edge overshoot, a lone transient) stay below this;
#: real flat-topping piles a sustained population against the rail.
CLIP_FRACTION_THRESHOLD = 0.01


def _as_1d_float(value: Any) -> npt.NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"samples must be 1-D, got {arr.ndim}-D")
    return arr


def _as_1d_bool(value: Any) -> npt.NDArray[np.bool_]:
    arr = np.asarray(value, dtype=np.bool_)
    if arr.ndim != 1:
        raise ValueError(f"levels must be 1-D, got {arr.ndim}-D")
    return arr


class Waveform(BaseModel):
    """One channel's analog samples (volts) on a uniform time grid."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    channel: ChannelId
    samples: npt.NDArray[np.float64] = Field(description="Sample voltages, volts.")
    t0_s: float = Field(description="Timestamp of the first sample.")
    dt_s: float = Field(gt=0, description="Sample interval, seconds.")
    units: str = "V"
    saturation: tuple[float, float] | None = Field(
        default=None,
        description="(low, high) volts at which the digitiser saturates, if known. "
        "A sample at a rail means the true signal was clipped there.",
    )

    def fraction_beyond_rails(
        self, low: float, high: float, *, margin_frac: float = 0.005
    ) -> float:
        """Fraction of samples sitting within ``margin_frac`` of either rail.

        Clipping is flat-topping — a *population* of samples piled against a rail —
        not a single excursion. This is the percentile that distinguishes real
        clipping from a few rail-grazing samples (e.g. logic-edge overshoot).
        """
        if self.n == 0:
            return 0.0
        margin = (high - low) * margin_frac
        at_rail = (self.samples >= high - margin) | (self.samples <= low + margin)
        return float(np.count_nonzero(at_rail)) / self.n

    @property
    def clipped_fraction(self) -> float:
        """Fraction of samples at the known digitiser saturation rails (0 if unknown)."""
        if self.saturation is None:
            return 0.0
        return self.fraction_beyond_rails(*self.saturation)

    @property
    def is_clipped(self) -> bool:
        """Whether a *meaningful fraction* of samples reach the saturation rails.

        A handful of rail-grazing samples (transient overshoot) does not count —
        only a sustained population at the rail (see :data:`CLIP_FRACTION_THRESHOLD`).
        """
        return self.clipped_fraction >= CLIP_FRACTION_THRESHOLD

    @field_validator("samples", mode="before")
    @classmethod
    def _coerce_samples(cls, value: Any) -> npt.NDArray[np.float64]:
        return _as_1d_float(value)

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.dt_s

    @property
    def duration_s(self) -> float:
        return self.n * self.dt_s

    @property
    def vmin(self) -> float:
        return float(self.samples.min()) if self.n else float("nan")

    @property
    def vmax(self) -> float:
        return float(self.samples.max()) if self.n else float("nan")

    @property
    def vpp(self) -> float:
        return self.vmax - self.vmin

    def time_axis(self) -> npt.NDArray[np.float64]:
        """Absolute sample times: ``t0 + i*dt``."""
        return self.t0_s + np.arange(self.n, dtype=np.float64) * self.dt_s

    @field_serializer("samples")
    def _summarize_samples(self, samples: npt.NDArray[np.float64]) -> dict[str, Any]:
        """Serialize a compact summary instead of the raw array."""
        return {
            "n": int(samples.size),
            "vmin": self.vmin,
            "vmax": self.vmax,
            "sample_rate_hz": self.sample_rate_hz,
        }


class DigitalTrace(BaseModel):
    """One channel thresholded to logic levels on a uniform time grid."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    channel: ChannelId
    levels: npt.NDArray[np.bool_] = Field(description="Logic levels, True = high.")
    t0_s: float
    dt_s: float = Field(gt=0)

    @field_validator("levels", mode="before")
    @classmethod
    def _coerce_levels(cls, value: Any) -> npt.NDArray[np.bool_]:
        return _as_1d_bool(value)

    @property
    def n(self) -> int:
        return int(self.levels.size)

    @property
    def duration_s(self) -> float:
        return self.n * self.dt_s

    def time_axis(self) -> npt.NDArray[np.float64]:
        return self.t0_s + np.arange(self.n, dtype=np.float64) * self.dt_s

    @field_serializer("levels")
    def _summarize_levels(self, levels: npt.NDArray[np.bool_]) -> dict[str, Any]:
        return {"n": int(levels.size), "transitions": int(np.count_nonzero(np.diff(levels)))}
