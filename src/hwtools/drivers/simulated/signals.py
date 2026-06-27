"""Synthetic analog signals for the simulated scope.

These describe the *true* signal a simulated DUT puts on a channel — the ground
truth the scope then has to capture and the self-correcting loop has to make
sense of. Kept deliberately simple (sine / square / DC); the interesting
behaviour (clipping, triggering, fill) is modelled by the scope rendering these
through a vertical/timebase/trigger configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class SimSignal(Protocol):
    """A true signal: volts as a function of time, plus its voltage extent."""

    def sample(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    @property
    def vrange(self) -> tuple[float, float]:
        """(min, max) true volts — used for triggering and autoscale."""
        ...


@dataclass(frozen=True)
class Sine:
    """A sine wave."""

    amplitude_v: float
    frequency_hz: float
    offset_v: float = 0.0
    phase_rad: float = 0.0

    def sample(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.offset_v + self.amplitude_v * np.sin(
            2.0 * np.pi * self.frequency_hz * t + self.phase_rad
        )

    @property
    def vrange(self) -> tuple[float, float]:
        return (self.offset_v - self.amplitude_v, self.offset_v + self.amplitude_v)


@dataclass(frozen=True)
class Square:
    """A square wave swinging +/- amplitude about offset."""

    amplitude_v: float
    frequency_hz: float
    offset_v: float = 0.0
    duty: float = 0.5

    def sample(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phase = (t * self.frequency_hz) % 1.0
        return self.offset_v + np.where(phase < self.duty, self.amplitude_v, -self.amplitude_v)

    @property
    def vrange(self) -> tuple[float, float]:
        return (self.offset_v - self.amplitude_v, self.offset_v + self.amplitude_v)


@dataclass(frozen=True)
class Dc:
    """A constant voltage."""

    value_v: float

    def sample(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(t.shape, self.value_v, dtype=np.float64)

    @property
    def vrange(self) -> tuple[float, float]:
        return (self.value_v, self.value_v)
