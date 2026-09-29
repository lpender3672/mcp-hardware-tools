"""The dense characterisation lens: feature vector separates signal shapes, and
the opt-in views (percentiles, value histogram, time bins, PSD, joint grid)
carry the right dense data."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hwtools.analysis.describe import describe
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import Waveform

CH1 = ChannelId.CH1
_DT = 1e-6
_N = 20_000


def _wf(samples: np.ndarray) -> Waveform:
    return Waveform(channel=CH1, samples=samples, t0_s=0.0, dt_s=_DT)


def _t() -> np.ndarray:
    return np.arange(_N) * _DT


def _sine(freq: float = 1_000.0, amp: float = 1.0, offset: float = 0.0) -> Waveform:
    return _wf(offset + amp * np.sin(2 * np.pi * freq * _t()))


def _square(freq: float = 1_000.0, amp: float = 1.0, duty: float = 0.5) -> Waveform:
    phase = (_t() * freq) % 1.0
    return _wf(np.where(phase < duty, amp, -amp))


def _noise(sigma: float = 1.0) -> Waveform:
    rng = np.random.default_rng(0)
    return _wf(rng.normal(0.0, sigma, size=_N))


def _dc(value: float = 2.0) -> Waveform:
    return _wf(np.full(_N, value))


# -- feature vector: shape discrimination -------------------------------------


def test_crest_factor_matches_theory() -> None:
    # crest = peak / rms has a closed form for these ideal (noise-free) shapes:
    # square 1, sine sqrt(2), triangle sqrt(3). Gaussian-noise crest is statistical.
    assert describe(_square()).crest_factor == pytest.approx(1.0, abs=0.02)
    assert describe(_sine()).crest_factor == pytest.approx(math.sqrt(2), abs=0.02)
    assert describe(_noise()).crest_factor > 2.5


def test_spectral_flatness_separates_tone_from_noise() -> None:
    assert describe(_sine()).spectral_flatness < 0.01  # a pure tone -> flatness ~ 0
    assert describe(_noise()).spectral_flatness > 0.3  # broadband noise spreads it


def test_square_reads_two_levels_high_edge_rate_and_duty() -> None:
    c = describe(_square(freq=1_000.0, duty=0.5))
    assert c.n_levels == 2
    assert c.duty is not None and abs(c.duty - 0.5) < 0.05
    # ~1000 rising + 1000 falling per second = ~2000 transitions/s.
    assert 1500 < c.edge_rate_hz < 2500


def test_square_duty_tracks_command() -> None:
    c = describe(_square(freq=1_000.0, duty=0.25))
    assert c.duty is not None and abs(c.duty - 0.25) < 0.05


def test_dc_is_one_level_flat_aperiodic() -> None:
    c = describe(_dc(2.0))
    assert c.n_levels == 1
    assert c.vpp == 0.0
    assert c.peak_hz is None
    assert abs(c.dc_level - 2.0) < 1e-6
    assert c.duty is None


def test_sine_is_periodic_with_a_tone() -> None:
    c = describe(_sine(freq=1_000.0))
    assert c.peak_hz is not None and abs(c.peak_hz - 1_000.0) < 20
    assert c.peak_prominence > 5
    assert c.periodicity > 0.7
    assert c.period_s is not None and abs(c.period_s - 1e-3) < 1e-4


def test_noise_is_aperiodic_and_flat() -> None:
    c = describe(_noise())
    assert c.periodicity < 0.5
    assert c.peak_prominence < 5  # no single bin dominates


def test_dc_offset_shows_in_dc_level_not_midline_confusion() -> None:
    c = describe(_sine(amp=1.0, offset=2.0))
    assert abs(c.dc_level - 2.0) < 0.05  # mean tracks the DC offset


# -- opt-in dense views -------------------------------------------------------


def test_default_gives_percentiles_no_heavy_views() -> None:
    c = describe(_sine())
    assert c.percentiles is not None and "50" in c.percentiles
    assert c.value_histogram is None
    assert c.time_bins is None
    assert c.psd is None
    assert c.joint is None


def test_value_histogram_opt_in() -> None:
    c = describe(_square(), value_bins=16)
    assert c.value_histogram is not None
    assert len(c.value_histogram.counts) == 16
    assert len(c.value_histogram.edges) == 17
    assert sum(c.value_histogram.counts) == _N


def test_time_bins_opt_in() -> None:
    c = describe(_sine(), time_bins=10)
    assert c.time_bins is not None and len(c.time_bins) == 10
    assert "50" in c.time_bins[0].percentiles


def test_psd_opt_in_down_binned() -> None:
    c = describe(_sine(), psd=True, freq_bins=32)
    assert c.psd is not None
    assert len(c.psd.freqs_hz) == len(c.psd.power) <= 32


def test_joint_grid_opt_in() -> None:
    c = describe(_square(), joint=True, joint_bins=24)
    assert c.joint is not None
    assert len(c.joint.counts) == 24  # t bins
    assert len(c.joint.counts[0]) == 24  # v bins
    total = sum(sum(row) for row in c.joint.counts)
    assert total == _N  # every sample landed in a cell
