"""Pure measurements over a captured waveform.

No instrument awareness: these operate on a :class:`~hwtools.model.waveform.Waveform`
and back the judgement that drives the self-correcting loop.
"""

from __future__ import annotations

import numpy as np

from hwtools.decode.threshold import find_edges, threshold
from hwtools.model.waveform import Waveform


def mean(wf: Waveform) -> float:
    return float(wf.samples.mean()) if wf.n else float("nan")


def rms(wf: Waveform) -> float:
    return float(np.sqrt(np.mean(np.square(wf.samples)))) if wf.n else float("nan")


def frequency(wf: Waveform) -> float | None:
    """Estimate the fundamental frequency from rising-edge spacing.

    Thresholds at the midpoint with hysteresis (Schmitt) and measures
    rising-edge-to-rising-edge periods. Using the midpoint (not the mean) plus
    hysteresis is robust to overshoot ringing and asymmetric duty cycles, which
    fool a mean-crossing estimate. Returns ``None`` for a flat/DC trace or too
    few cycles to be meaningful.
    """
    if wf.n < 4 or wf.vpp <= 0:
        return None
    midpoint = (wf.vmax + wf.vmin) / 2.0
    trace = threshold(wf, level_v=midpoint, hysteresis_v=wf.vpp * 0.2)
    rising = [edge.time_s for edge in find_edges(trace) if edge.rising]
    if len(rising) < 2:
        return None
    period = float(np.median(np.diff(rising)))
    return 1.0 / period if period > 0 else None


def samples_per_period(wf: Waveform) -> float | None:
    """How many samples cover one period of the dominant tone (None if flat)."""
    freq = frequency(wf)
    if freq is None or freq <= 0:
        return None
    return (1.0 / freq) / wf.dt_s
