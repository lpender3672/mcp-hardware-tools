"""Frequency-domain foundation.

The principled basis for every spectral measurement — frequency estimation today,
and the analog/noise characterisation the README envisions (system ID from a
flat-noise stimulus, frequency response) tomorrow. Built on ``scipy.signal`` so
the window functions and the amplitude/PSD normalisation are correct by
construction rather than by hand.

Key facts this API makes explicit:

* **Resolution** is ``fs / N`` — set by the record length. You buy resolution with
  more samples (deeper captures), not zero-padding.
* **Window choice** trades main-lobe width (resolution) against side-lobe level
  (leakage): ``boxcar`` (sharpest, leakiest), ``hann`` (good general), ``flattop``
  (amplitude-accurate, wide), ``blackmanharris`` (low leakage). It is a parameter.
* **Scaling** differs for amplitude vs power: a sinusoid's amplitude comes from the
  ``'spectrum'`` scaling; a noise floor in V²/Hz from the ``'density'`` scaling
  (NENBW-normalised). scipy handles both; we expose both.

Welch averaging (segment + average, trading resolution for variance) is the
natural next addition for noise PSD; the single-segment periodogram here is its
``nperseg = N`` special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy import signal as sps

from hwtools.model.waveform import Waveform

#: Default analysis window — a sensible general-purpose resolution/leakage balance.
DEFAULT_WINDOW = "hann"
#: A peak must exceed this multiple of the median spectrum to count as a real tone.
DEFAULT_PEAK_PROMINENCE = 8.0


class SpectrumKind(StrEnum):
    AMPLITUDE_V = "amplitude_v"  # sinusoid amplitude, volts
    PSD_V2_PER_HZ = "psd_v2_per_hz"  # power spectral density, V^2/Hz


@dataclass(frozen=True)
class Spectrum:
    """A one-sided spectrum: frequencies and their values."""

    frequencies_hz: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    kind: SpectrumKind

    def peak(self) -> tuple[float, float] | None:
        """The (frequency, value) of the largest non-DC bin, or None if flat."""
        if self.values.size < 2:
            return None
        values = self.values.copy()
        values[0] = 0.0  # ignore the DC bin
        index = int(np.argmax(values))
        if index == 0:
            return None
        return float(self.frequencies_hz[index]), float(values[index])


def amplitude_spectrum(wf: Waveform, *, window: str = DEFAULT_WINDOW) -> Spectrum:
    """Single-sided amplitude spectrum (volts): a tone reads its own amplitude."""
    freqs, power = sps.periodogram(
        wf.samples, fs=wf.sample_rate_hz, window=window, scaling="spectrum", detrend="constant"
    )
    # 'spectrum' power is A^2/2 for a single-sided sinusoid; amplitude = sqrt(2*P).
    amplitude = np.sqrt(2.0 * np.asarray(power, dtype=np.float64))
    return Spectrum(np.asarray(freqs, dtype=np.float64), amplitude, SpectrumKind.AMPLITUDE_V)


def power_spectral_density(wf: Waveform, *, window: str = DEFAULT_WINDOW) -> Spectrum:
    """Power spectral density (V^2/Hz): the right basis for noise-floor work."""
    freqs, psd = sps.periodogram(
        wf.samples, fs=wf.sample_rate_hz, window=window, scaling="density", detrend="constant"
    )
    return Spectrum(
        np.asarray(freqs, dtype=np.float64),
        np.asarray(psd, dtype=np.float64),
        SpectrumKind.PSD_V2_PER_HZ,
    )


def welch_psd(
    wf: Waveform, *, nperseg: int | None = None, window: str = DEFAULT_WINDOW
) -> Spectrum:
    """Welch-averaged power spectral density (V^2/Hz): lower-variance noise floor.

    Splits the record into overlapping ``nperseg``-length segments and averages
    their periodograms — trading frequency resolution for a smoother estimate, the
    right basis for noise / system-ID work. ``nperseg`` defaults to a segment that
    balances resolution and averaging; the single-segment periodogram is the
    ``nperseg = N`` special case.
    """
    n = wf.n
    seg = nperseg if nperseg is not None else max(8, min(n, 256))
    seg = min(seg, n)
    freqs, psd = sps.welch(
        wf.samples, fs=wf.sample_rate_hz, window=window, nperseg=seg, scaling="density"
    )
    return Spectrum(
        np.asarray(freqs, dtype=np.float64),
        np.asarray(psd, dtype=np.float64),
        SpectrumKind.PSD_V2_PER_HZ,
    )


def spectral_flatness(wf: Waveform, *, window: str = DEFAULT_WINDOW) -> float:
    """Wiener entropy of the spectrum: ~0 for a pure tone, ~1 for white noise.

    The ratio of the geometric to the arithmetic mean of the (non-DC) power
    spectrum. A single dominant tone concentrates power in one bin (low flatness);
    broadband noise spreads it evenly (high flatness). One cheap number that
    separates "a signal" from "a noise source" — central to triage and system-ID.
    """
    if wf.n < 8:
        return 0.0
    psd = power_spectral_density(wf, window=window).values[1:]  # drop the DC bin
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    geo_mean = float(np.exp(np.mean(np.log(psd))))
    arith_mean = float(np.mean(psd))
    return geo_mean / arith_mean if arith_mean > 0 else 0.0


def peak_frequency(
    wf: Waveform,
    *,
    window: str = DEFAULT_WINDOW,
    min_prominence: float = DEFAULT_PEAK_PROMINENCE,
) -> float | None:
    """Dominant tone frequency, or None when no bin stands above the noise floor.

    Locates the strongest non-DC bin of the amplitude spectrum, requires it to
    exceed ``min_prominence`` times the median (noise floor), and refines it with
    a parabolic fit for sub-bin accuracy.
    """
    if wf.n < 8 or wf.vpp <= 0:
        return None
    spectrum = amplitude_spectrum(wf, window=window)
    values = spectrum.values.copy()
    if values.size < 3:
        return None
    values[0] = 0.0

    index = int(np.argmax(values))
    if index == 0:
        return None
    noise_floor = float(np.median(values))
    if noise_floor <= 0 or values[index] < min_prominence * noise_floor:
        return None

    offset = _parabolic_offset(values, index)
    bin_hz = float(spectrum.frequencies_hz[1] - spectrum.frequencies_hz[0])
    return float(spectrum.frequencies_hz[index]) + offset * bin_hz


def _parabolic_offset(values: npt.NDArray[np.float64], index: int) -> float:
    """Sub-bin peak offset in [-0.5, 0.5] from a 3-point parabolic fit."""
    if not 0 < index < values.size - 1:
        return 0.0
    left, mid, right = values[index - 1], values[index], values[index + 1]
    denom = left - 2.0 * mid + right
    if denom == 0:
        return 0.0
    return float(0.5 * (left - right) / denom)
