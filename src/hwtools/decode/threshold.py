"""Analog → digital: thresholding and edge finding.

The bridge from a :class:`~hwtools.model.waveform.Waveform` (volts) into the
logic-level :class:`~hwtools.model.waveform.DigitalTrace` the protocol decoders
consume. A Schmitt-trigger style hysteresis band suppresses chatter on slow or
noisy edges. Pure functions, no hardware awareness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hwtools.model.waveform import DigitalTrace, Waveform


@dataclass(frozen=True, slots=True)
class Edge:
    """A logic transition: where, when, and which direction."""

    index: int
    time_s: float
    rising: bool


def threshold(wf: Waveform, level_v: float, *, hysteresis_v: float = 0.0) -> DigitalTrace:
    """Convert a waveform to logic levels about ``level_v``.

    With ``hysteresis_v`` > 0 the line must rise above ``level + h/2`` to read
    high and fall below ``level - h/2`` to read low, holding state in between.
    """
    if hysteresis_v < 0:
        raise ValueError("hysteresis_v must be non-negative")

    samples = wf.samples
    high = level_v + hysteresis_v / 2.0
    low = level_v - hysteresis_v / 2.0
    levels = np.empty(samples.size, dtype=np.bool_)

    state = bool(samples[0] >= level_v) if samples.size else False
    for i in range(samples.size):
        v = samples[i]
        if state and v < low:
            state = False
        elif not state and v > high:
            state = True
        levels[i] = state

    return DigitalTrace(channel=wf.channel, levels=levels, t0_s=wf.t0_s, dt_s=wf.dt_s)


def find_edges(trace: DigitalTrace) -> list[Edge]:
    """Return every transition in ``trace``, in time order."""
    levels: npt.NDArray[np.bool_] = trace.levels
    changes = np.nonzero(np.diff(levels.astype(np.int8)))[0] + 1
    return [
        Edge(
            index=int(i),
            time_s=trace.t0_s + int(i) * trace.dt_s,
            rising=bool(levels[int(i)]),
        )
        for i in changes
    ]
