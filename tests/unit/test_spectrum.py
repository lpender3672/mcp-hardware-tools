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
