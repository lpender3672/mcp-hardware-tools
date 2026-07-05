"""Spectral foundation: amplitude scaling, PSD, and peak frequency."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.analysis import spectrum
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import Waveform


def _tone(amp_v: float, freq_hz: float, *, fs: float = 1e6, n: int = 8192) -> Waveform:
    t = np.arange(n) / fs
    samples = 1.0 + amp_v * np.sin(2 * np.pi * freq_hz * t)  # +1 V DC offset
    return Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=1.0 / fs)


def test_amplitude_spectrum_reads_tone_amplitude() -> None:
    # flattop is amplitude-accurate even when the tone is off-bin.
    spec = spectrum.amplitude_spectrum(_tone(2.0, 12_345.0), window="flattop")
    peak = spec.peak()
    assert peak is not None
    _, amplitude = peak
    assert amplitude == pytest.approx(2.0, rel=0.02)  # recovers the 2 V amplitude


def test_peak_frequency_is_accurate() -> None:
    assert spectrum.peak_frequency(_tone(1.0, 12_345.0)) == pytest.approx(12_345.0, rel=1e-3)


def test_peak_frequency_none_for_noise() -> None:
    rng = np.random.default_rng(0)
    noise = Waveform(
        channel=ChannelId.CH1, samples=rng.normal(0, 1, 8192), t0_s=0.0, dt_s=1e-6
    )
    assert spectrum.peak_frequency(noise) is None


def test_peak_frequency_none_for_dc() -> None:
    dc = Waveform(channel=ChannelId.CH1, samples=np.full(8192, 2.5), t0_s=0.0, dt_s=1e-6)
    assert spectrum.peak_frequency(dc) is None


def test_psd_has_density_units_and_finds_the_tone() -> None:
    spec = spectrum.power_spectral_density(_tone(1.0, 10_000.0))
    assert spec.kind is spectrum.SpectrumKind.PSD_V2_PER_HZ
    peak = spec.peak()
    assert peak is not None
    assert peak[0] == pytest.approx(10_000.0, rel=1e-2)


# -- harmonic content ---------------------------------------------------------

_F0 = 1_000.0


def _periodic(kind: str, *, fs: float = 1e6, periods: int = 100) -> Waveform:
    """One coherent capture (integer periods) of a unit-amplitude shape at _F0."""
    from scipy import signal as sps

    n = round(fs / _F0 * periods)
    phase = 2 * np.pi * _F0 * np.arange(n) / fs
    if kind == "sine":
        samples = np.sin(phase)
    elif kind == "square":
        samples = np.sign(np.sin(phase))
    elif kind == "triangle":
        samples = sps.sawtooth(phase, width=0.5)
    else:  # sawtooth (all harmonics, 1/n)
        samples = sps.sawtooth(phase)
    return Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=1.0 / fs)


def test_harmonic_amplitudes_square_is_odd_one_over_n() -> None:
    amps = spectrum.harmonic_amplitudes(_periodic("square"), _F0, 5)
    h1, h2, h3, h4, h5 = amps
    assert h3 / h1 == pytest.approx(1 / 3, rel=0.03)  # 3rd ~ -9.5 dB
    assert h5 / h1 == pytest.approx(1 / 5, rel=0.03)  # 5th ~ -14 dB
    assert h2 / h1 < 0.01  # even harmonics suppressed
    assert h4 / h1 < 0.01


def test_harmonic_amplitudes_triangle_is_odd_one_over_n_squared() -> None:
    amps = spectrum.harmonic_amplitudes(_periodic("triangle"), _F0, 5)
    h1, _h2, h3, _h4, h5 = amps
    assert h3 / h1 == pytest.approx(1 / 9, rel=0.05)  # 3rd ~ -19 dB
    assert h5 / h1 == pytest.approx(1 / 25, rel=0.08)  # 5th ~ -28 dB


def test_thd_matches_textbook_values() -> None:
    assert spectrum.thd(_periodic("sine"), _F0) < 0.01  # pure tone
    # The square's 1/n series converges slowly, so sum enough harmonics to approach
    # the textbook 48.3 %; the default 10 would truncate at ~43 %.
    assert spectrum.thd(_periodic("square"), _F0, n_harmonics=50) == pytest.approx(0.483, abs=0.02)
    assert spectrum.thd(_periodic("triangle"), _F0) == pytest.approx(0.121, abs=0.02)


def test_harmonic_amplitudes_rejects_bad_args() -> None:
    wf = _periodic("sine")
    with pytest.raises(ValueError, match="fundamental_hz"):
        spectrum.harmonic_amplitudes(wf, 0.0, 3)
    with pytest.raises(ValueError, match="search_frac"):
        spectrum.harmonic_amplitudes(wf, _F0, 3, search_frac=0.6)
