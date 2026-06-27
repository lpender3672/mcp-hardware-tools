"""Pure measurements over a captured waveform.

No instrument awareness: these operate on a :class:`~hwtools.model.waveform.Waveform`
and back the judgement that drives the self-correcting loop.
"""

from __future__ import annotations

import numpy as np

from hwtools.analysis import spectrum
from hwtools.model.waveform import Waveform


def mean(wf: Waveform) -> float:
    return float(wf.samples.mean()) if wf.n else float("nan")


def rms(wf: Waveform) -> float:
    return float(np.sqrt(np.mean(np.square(wf.samples)))) if wf.n else float("nan")


def frequency(wf: Waveform) -> float | None:
    """Dominant frequency of the waveform, or None for a flat/noisy trace.

    A thin consumer of the spectral foundation (:func:`hwtools.analysis.spectrum.
    peak_frequency`): a windowed FFT peak with prominence gating and sub-bin
    interpolation — robust to noise, overshoot ringing, and asymmetric duty.
    """
    return spectrum.peak_frequency(wf)


def samples_per_period(wf: Waveform) -> float | None:
    """How many samples cover one period of the dominant tone (None if flat)."""
    freq = frequency(wf)
    if freq is None or freq <= 0:
        return None
    return (1.0 / freq) / wf.dt_s
