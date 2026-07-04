"""Analytic arbitrary-waveform synthesis: shape, range, crest factor."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.model.siggen import ArbitraryWaveform, WaveShape
from hwtools.synth import analytic_arbitrary


@pytest.mark.parametrize(
    ("shape", "crest"),
    [(WaveShape.SINE, 1.414), (WaveShape.TRIANGLE, 1.732), (WaveShape.SQUARE, 1.0)],
)
def test_shapes_have_the_expected_crest_factor(shape: WaveShape, crest: float) -> None:
    wave = analytic_arbitrary(shape, points=2048)
    assert isinstance(wave, ArbitraryWaveform)
    s = np.array(wave.samples)
    assert len(s) == 2048
    assert s.min() >= -1.0 and s.max() <= 1.0
    assert 0.5 * (s.max() - s.min()) / s.std() == pytest.approx(crest, abs=1e-2)


def test_square_duty_controls_high_fraction() -> None:
    s = np.array(analytic_arbitrary(WaveShape.SQUARE, points=1000, duty=0.25).samples)
    assert np.mean(s > 0) == pytest.approx(0.25, abs=1e-3)


def test_unsupported_shape_raises() -> None:
    with pytest.raises(ValueError, match="no analytic synthesiser"):
        analytic_arbitrary(WaveShape.NOISE, points=8)


def test_non_positive_points_raises() -> None:
    with pytest.raises(ValueError, match="points"):
        analytic_arbitrary(WaveShape.SINE, points=0)
