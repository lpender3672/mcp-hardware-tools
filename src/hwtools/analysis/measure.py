"""Pure measurements over a captured waveform.

No instrument awareness: these operate on a :class:`~hwtools.model.waveform.Waveform`
and back the judgement that drives the self-correcting loop.
"""

from __future__ import annotations

import numpy as np

from hwtools.model.waveform import Waveform


def mean(wf: Waveform) -> float:
    return float(wf.samples.mean()) if wf.n else float("nan")


def rms(wf: Waveform) -> float:
    return float(np.sqrt(np.mean(np.square(wf.samples)))) if wf.n else float("nan")


def frequency(wf: Waveform) -> float | None:
    """Estimate the fundamental frequency via mean-level crossings.

    Returns ``None`` for a flat/DC trace or too few crossings to be meaningful.
    """
    if wf.n < 4:
        return None
    midline = float(wf.samples.mean())
    above = wf.samples >= midline
    crossings = np.flatnonzero(np.diff(above.astype(np.int8)) != 0)
    if crossings.size < 2:
        return None
    half_periods = np.diff(crossings).astype(np.float64) * wf.dt_s
    period = 2.0 * float(np.median(half_periods))
    return 1.0 / period if period > 0 else None


def samples_per_period(wf: Waveform) -> float | None:
    """How many samples cover one period of the dominant tone (None if flat)."""
    freq = frequency(wf)
    if freq is None or freq <= 0:
        return None
    return (1.0 / freq) / wf.dt_s
