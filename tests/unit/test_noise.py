"""Synthetic noise: spectral slope, reproducibility, band-limiting, arb rendering."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.analysis.spectrum import psd_slope_db_per_decade, welch_psd
from hwtools.model.ids import ChannelId
from hwtools.model.siggen import ArbitraryWaveform
from hwtools.model.waveform import Waveform
from hwtools.noise import (
    PSD_SLOPE_DB_PER_DECADE,
    NoiseColor,
    band_limited,
    band_limited_noise_arbitrary,
    colored_noise,
    noise_arbitrary,
    white_noise,
)

_N = 1 << 16
_FS = 1.0


def _as_waveform(x: np.ndarray, fs: float = _FS) -> Waveform:
    return Waveform(channel=ChannelId.CH1, samples=x, t0_s=0.0, dt_s=1.0 / fs)


@pytest.mark.parametrize("color", list(NoiseColor))
def test_colored_noise_matches_target_slope(color: NoiseColor) -> None:
    wf = _as_waveform(colored_noise(_N, color, seed=7))
    slope = psd_slope_db_per_decade(wf, f_lo=_FS * 0.01, f_hi=_FS * 0.4, nperseg=4096)
    assert slope == pytest.approx(PSD_SLOPE_DB_PER_DECADE[color], abs=2.0)


def test_white_noise_is_reproducible_and_seed_dependent() -> None:
    assert np.array_equal(white_noise(1000, seed=3), white_noise(1000, seed=3))
    assert not np.array_equal(white_noise(1000, seed=3), white_noise(1000, seed=4))


def test_sigma_scales_the_rms() -> None:
    x = colored_noise(_N, NoiseColor.PINK, seed=1, sigma=2.5)
    assert x.std() == pytest.approx(2.5, rel=0.05)


def test_band_limited_lowpass_attenuates_above_cutoff() -> None:
    x = white_noise(_N, seed=1)
    y = band_limited(x, fs=1000.0, high_hz=100.0)
    spec = welch_psd(_as_waveform(y, fs=1000.0), nperseg=4096)
    freqs, psd = spec.frequencies_hz, spec.values
    passband = psd[freqs < 80.0].mean()
    stopband = psd[freqs > 200.0].mean()
    assert stopband < passband * 1e-3  # well attenuated beyond the cutoff


def test_noise_arbitrary_is_normalised_full_scale() -> None:
    wave = noise_arbitrary(2048, NoiseColor.WHITE, seed=5)
    assert isinstance(wave, ArbitraryWaveform)
    assert wave.n == 2048
    peak = max(abs(s) for s in wave.samples)
    assert peak == pytest.approx(1.0)  # scaled so the peak sits at full scale


def test_normalised_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        ArbitraryWaveform.normalised(np.zeros(16))


def test_band_limited_noise_confines_energy_to_the_band() -> None:
    lo, hi = 100, 200
    wave = band_limited_noise_arbitrary(2048, low_cycles=lo, high_cycles=hi, seed=3)
    mag = np.abs(np.fft.rfft(np.array(wave.samples)))
    k = np.arange(mag.size)
    in_band = mag[(k >= lo) & (k <= hi)]
    out_of_band = mag[(k < lo) | (k > hi)]
    # FFT masking is exact: essentially no energy leaks outside the chosen harmonics.
    assert out_of_band.max() < in_band.mean() * 1e-6


def test_band_limited_noise_rejects_invalid_band() -> None:
    with pytest.raises(ValueError, match="low_cycles"):
        band_limited_noise_arbitrary(2048, low_cycles=200, high_cycles=100, seed=1)
    with pytest.raises(ValueError, match="low_cycles"):
        band_limited_noise_arbitrary(2048, low_cycles=0, high_cycles=2000, seed=1)  # > Nyquist
