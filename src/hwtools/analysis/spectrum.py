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


def complex_tone(wf: Waveform, frequency_hz: float, *, window: str = DEFAULT_WINDOW) -> complex:
    """The complex amplitude ``A * exp(1j*phase)`` of a *known* frequency in ``wf``.

    Every other function here finds a frequency by searching a bin grid, which is
    right when the frequency is unknown but throws away phase (``amplitude_spectrum``)
    or only recovers it to within a bin's leakage skirt. When the frequency is
    externally known instead — set on a generator, not discovered — direct
    quadrature demodulation (a single-frequency DTFT, the lock-in-amplifier
    technique) is both simpler and exact for it: ``coeff = 2 * <x(t) * e^-j*w*t>``
    recovers ``A*exp(1j*phase)`` for ``x(t) = A*cos(w*t + phase)`` to the extent the
    record spans many cycles (the cross term at ``2*frequency_hz`` averages toward
    zero). The phase reference is ``wf.time_axis()``, i.e. the capture's own
    ``t0_s`` — meaningless in isolation (it depends on when the scope happened to
    trigger), but directly comparable between two channels from the *same*
    multi-channel acquisition, which share that origin exactly. That is what makes
    a phase-sensitive measurement (e.g. impedance from a drive/response pair)
    possible without a dedicated phase-reference instrument.

    A window is applied even though nothing can leak *into* an exactly-known
    frequency the way it leaks into a bin: broadband noise at other frequencies
    still leaks in less through a tapered record than a hard-edged rectangle, and
    the coherent-gain normalisation (dividing by ``sum(window)`` rather than ``N``)
    keeps the amplitude scaling exact regardless.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if frequency_hz >= wf.sample_rate_hz / 2.0:
        raise ValueError(
            f"frequency_hz ({frequency_hz:g} Hz) must be below Nyquist "
            f"({wf.sample_rate_hz / 2.0:g} Hz) or the estimate aliases"
        )
    n = wf.n
    if n < 2:
        raise ValueError("need at least 2 samples")
    w = sps.get_window(window, n)
    phasor = np.exp(-1j * 2.0 * np.pi * frequency_hz * wf.time_axis())
    return complex(2.0 * np.sum(wf.samples * w * phasor) / np.sum(w))


def harmonic_amplitudes(
    wf: Waveform,
    fundamental_hz: float,
    n_harmonics: int,
    *,
    window: str = DEFAULT_WINDOW,
    search_frac: float = 0.25,
) -> list[float]:
    """Amplitude (volts) at each of the first ``n_harmonics`` multiples of the
    fundamental — index 0 is the fundamental itself, index 1 the 2nd harmonic, etc.

    For each harmonic ``k`` the *peak* amplitude within ``+/- search_frac*fundamental``
    of ``k*fundamental`` is taken, so a small mistuning or a leakage skirt doesn't
    make it read zero. ``search_frac`` must be < 0.5 so the search band never reaches
    a neighbouring harmonic. Capture many periods (fine bins) for a clean read.
    """
    if fundamental_hz <= 0:
        raise ValueError("fundamental_hz must be positive")
    if not 0.0 < search_frac < 0.5:
        raise ValueError("search_frac must be in (0, 0.5)")
    spec = amplitude_spectrum(wf, window=window)
    freqs, amps = spec.frequencies_hz, spec.values
    half = search_frac * fundamental_hz
    out: list[float] = []
    for k in range(1, n_harmonics + 1):
        center = k * fundamental_hz
        band = (freqs >= center - half) & (freqs <= center + half)
        out.append(float(amps[band].max()) if bool(band.any()) else 0.0)
    return out


def thd(
    wf: Waveform,
    fundamental_hz: float,
    *,
    n_harmonics: int = 10,
    window: str = DEFAULT_WINDOW,
) -> float:
    """Total harmonic distortion: ``sqrt(Σ h_k², k≥2) / h_1`` over ``n_harmonics``.

    ~0 for a pure sine, ~0.48 for an ideal square, ~0.12 for a triangle. Raises if
    there is no energy at the fundamental.
    """
    amps = harmonic_amplitudes(wf, fundamental_hz, n_harmonics, window=window)
    fundamental = amps[0]
    if fundamental <= 0:
        raise ValueError("no energy at the fundamental; cannot compute THD")
    rest = np.asarray(amps[1:], dtype=np.float64)
    return float(np.sqrt(np.sum(rest**2)) / fundamental)


def psd_slope_db_per_decade(
    wf: Waveform,
    *,
    f_lo: float,
    f_hi: float,
    nperseg: int | None = None,
    window: str = DEFAULT_WINDOW,
) -> float:
    """Least-squares slope of the log-log Welch PSD over ``[f_lo, f_hi]``, dB/decade.

    The spectral-colour metric for noise sources: white ~0, pink ~-10, brown ~-20
    dB/decade. Choose the band to exclude DC and the top of the spectrum (filter
    roll-off / the Nyquist edge), where the estimate is unreliable.
    """
    spec = welch_psd(wf, nperseg=nperseg, window=window)
    freqs, psd = spec.frequencies_hz, spec.values
    mask = (freqs >= f_lo) & (freqs <= f_hi) & (psd > 0)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError(f"need >=2 positive PSD bins in [{f_lo}, {f_hi}] Hz; got fewer")
    slope, _ = np.polyfit(np.log10(freqs[mask]), 10.0 * np.log10(psd[mask]), 1)
    return float(slope)


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


def peak_frequency_and_prominence(
    wf: Waveform,
    *,
    window: str = DEFAULT_WINDOW,
    min_prominence: float = DEFAULT_PEAK_PROMINENCE,
) -> tuple[float | None, float]:
    """The dominant tone's (sub-bin) frequency and its prominence over the noise floor.

    Locates the strongest non-DC bin of the amplitude spectrum, measures its
    prominence (peak / spectral median), and — when that clears ``min_prominence`` —
    refines the frequency with a parabolic fit. Returns ``(None, prominence)`` when no
    bin stands out, so a caller can still read how tonal the spectrum is. Computing
    both here lets ``describe`` characterise a tone with a single periodogram.
    """
    if wf.n < 8 or wf.vpp <= 0:
        return None, 0.0
    spectrum = amplitude_spectrum(wf, window=window)
    values = spectrum.values.copy()
    if values.size < 3:
        return None, 0.0
    values[0] = 0.0

    index = int(np.argmax(values))
    if index == 0:
        return None, 0.0
    median = float(np.median(values))
    prominence = float(values[index] / median) if median > 0 else 0.0
    if median <= 0 or prominence < min_prominence:
        return None, prominence

    offset = _parabolic_offset(values, index)
    bin_hz = float(spectrum.frequencies_hz[1] - spectrum.frequencies_hz[0])
    freq = float(spectrum.frequencies_hz[index]) + offset * bin_hz
    return freq, prominence


def peak_frequency(
    wf: Waveform,
    *,
    window: str = DEFAULT_WINDOW,
    min_prominence: float = DEFAULT_PEAK_PROMINENCE,
) -> float | None:
    """Dominant tone frequency, or None when no bin stands above the noise floor."""
    return peak_frequency_and_prominence(wf, window=window, min_prominence=min_prominence)[0]


def _parabolic_offset(values: npt.NDArray[np.float64], index: int) -> float:
    """Sub-bin peak offset in [-0.5, 0.5] from a 3-point parabolic fit."""
    if not 0 < index < values.size - 1:
        return 0.0
    left, mid, right = values[index - 1], values[index], values[index + 1]
    denom = left - 2.0 * mid + right
    if denom == 0:
        return 0.0
    return float(0.5 * (left - right) / denom)
