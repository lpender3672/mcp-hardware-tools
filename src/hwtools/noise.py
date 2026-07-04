"""Synthetic noise sources — white / pink / brown, band-limitable, seeded.

The output-side companion to :mod:`hwtools.synth` (analytic shapes). Two consumers:

* the **software noise-robustness suite** (Thread A): full-length coloured noise
  added to a captured trace -- arbitrary length, good statistics, sweepable
  sigma/SNR;
* **arb calibration** (HIL): a short realisation rendered into a generator's
  arbitrary slot via :func:`noise_arbitrary`. That path is inherently *frozen and
  periodic* (the 2048-point buffer repeats at the replay rate, so its spectrum is a
  comb, not a continuum) and band-limited to the buffer's harmonics -- good only for
  anchoring the sim scope's noise model against the real front-end, not as a true
  noise source.

Colour is imposed by FIR-shaping white noise (``scipy.signal.firwin2`` designs a
filter whose magnitude response is ``f**alpha``, applied zero-phase with
``filtfilt``); the result is validated by the log-log Welch PSD slope: white 0,
pink -3, brown -6 dB/octave (i.e. 0 / -10 / -20 dB/decade). Randomness comes from
NumPy's PCG64 (:func:`numpy.random.default_rng`), seeded so every realisation is
reproducible.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import signal as sps

if TYPE_CHECKING:
    from hwtools.model.siggen import ArbitraryWaveform

NDArrayF = npt.NDArray[np.float64]


class NoiseColor(StrEnum):
    """A noise power-spectral shape."""

    WHITE = "WHITE"  # flat PSD
    PINK = "PINK"  # PSD proportional to 1/f    (-3 dB/oct)
    BROWN = "BROWN"  # PSD proportional to 1/f^2  (-6 dB/oct)


#: Amplitude-response exponent for each colour: ``|H(f)| ∝ f**alpha`` applied to
#: white noise, so output power ∝ ``f**(2*alpha)``.
_AMPLITUDE_EXPONENT: dict[NoiseColor, float] = {
    NoiseColor.WHITE: 0.0,
    NoiseColor.PINK: -0.5,
    NoiseColor.BROWN: -1.0,
}

#: Expected Welch PSD slope (dB per decade) — the validation target for each colour.
PSD_SLOPE_DB_PER_DECADE: dict[NoiseColor, float] = {
    NoiseColor.WHITE: 0.0,
    NoiseColor.PINK: -10.0,
    NoiseColor.BROWN: -20.0,
}


def white_noise(n: int, *, seed: int, sigma: float = 1.0) -> NDArrayF:
    """``n`` samples of Gaussian white noise, standard deviation ``sigma``."""
    if n <= 0:
        raise ValueError("n must be positive")
    return np.random.default_rng(seed).standard_normal(n) * sigma


def colored_noise(
    n: int,
    color: NoiseColor,
    *,
    seed: int,
    sigma: float = 1.0,
    numtaps: int | None = None,
) -> NDArrayF:
    """``n`` samples of ``color`` noise, scaled to standard deviation ``sigma``.

    White noise is FIR-shaped to the target spectral slope; the output is then
    renormalised so ``sigma`` is the requested RMS regardless of the colouring gain.
    """
    x = white_noise(n, seed=seed)
    alpha = _AMPLITUDE_EXPONENT[color]
    if alpha != 0.0:
        # filtfilt runs the filter forward and back, squaring its magnitude
        # response, so design for half the target exponent to land on f**alpha.
        taps = _color_fir(alpha / 2.0, numtaps if numtaps is not None else _default_numtaps(n))
        x = sps.filtfilt(taps, [1.0], x)
    std = float(x.std())
    return x / std * sigma if std > 0 else x


def band_limited(
    x: npt.ArrayLike,
    fs: float,
    *,
    low_hz: float | None = None,
    high_hz: float | None = None,
    order: int = 4,
) -> NDArrayF:
    """Zero-phase Butterworth band-limit of ``x`` (sample rate ``fs``).

    Pass ``low_hz`` and ``high_hz`` for a band-pass, one for a high-/low-pass, or
    neither to return the input unchanged.
    """
    signal = np.asarray(x, dtype=np.float64)
    nyquist = 0.5 * fs
    if low_hz is not None and high_hz is not None:
        band = [low_hz / nyquist, high_hz / nyquist]
        sos = sps.butter(order, band, btype="bandpass", output="sos")
    elif high_hz is not None:
        sos = sps.butter(order, high_hz / nyquist, btype="lowpass", output="sos")
    elif low_hz is not None:
        sos = sps.butter(order, low_hz / nyquist, btype="highpass", output="sos")
    else:
        return signal
    return np.asarray(sps.sosfiltfilt(sos, signal), dtype=np.float64)


def noise_arbitrary(points: int, color: NoiseColor, *, seed: int) -> ArbitraryWaveform:
    """A one-period noise realisation rendered as a normalised arbitrary waveform.

    Frozen and periodic by construction (see the module docstring) — for HIL
    calibration, not as a general noise source.
    """
    from hwtools.model.siggen import ArbitraryWaveform

    return ArbitraryWaveform.normalised(colored_noise(points, color, seed=seed))


def band_limited_noise_arbitrary(
    points: int, *, low_cycles: int, high_cycles: int, seed: int
) -> ArbitraryWaveform:
    """Band-limited noise in an arb slot: energy only in a chosen frequency window.

    The band is given in **cycles per period** (harmonic index) — a buffer of
    ``points`` samples is one period, so bin ``k`` is ``k`` cycles/period and the
    band spans ``[low_cycles, high_cycles]``. When the slot is replayed at ``f_hz``
    the passband lands at ``[low_cycles * f_hz, high_cycles * f_hz]`` Hz, so sweeping
    the band (or the replay rate) sweeps the noise window across the spectrum.

    Uses FFT bin-masking rather than an IIR filter: on a single-period buffer that is
    exactly circular, so the result stays perfectly periodic (no wrap discontinuity
    leaking energy out of band).
    """
    from hwtools.model.siggen import ArbitraryWaveform

    if not 0 <= low_cycles <= high_cycles <= points // 2:
        raise ValueError(f"require 0 <= low_cycles <= high_cycles <= {points // 2}")
    spectrum = np.fft.rfft(white_noise(points, seed=seed))
    k = np.arange(spectrum.size)
    spectrum[(k < low_cycles) | (k > high_cycles)] = 0.0
    return ArbitraryWaveform.normalised(np.fft.irfft(spectrum, n=points))


def _default_numtaps(n: int) -> int:
    """An odd FIR length that shapes well without an overlong transient for size ``n``."""
    return max(15, min(511, (n // 4) | 1))


def _color_fir(alpha: float, numtaps: int) -> NDArrayF:
    """FIR taps whose magnitude response is ``f**alpha`` (DC held finite)."""
    freqs = np.linspace(0.0, 1.0, 512)
    with np.errstate(divide="ignore"):
        gains = np.where(freqs > 0, freqs**alpha, 0.0)
    gains[0] = gains[1]  # a finite DC gain in place of the f=0 singularity
    gains = gains / gains.max()
    return np.asarray(sps.firwin2(numtaps, freqs, gains), dtype=np.float64)
