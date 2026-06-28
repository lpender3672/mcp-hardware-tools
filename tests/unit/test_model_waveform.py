"""Waveform / DigitalTrace coercion, derived metrics, and compact serialization."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hwtools.model import ChannelId, DigitalTrace, Waveform


def _ramp() -> Waveform:
    # 0,1,2,3 volts at 1 MHz starting at t=0.
    return Waveform(channel=ChannelId.CH1, samples=[0.0, 1.0, 2.0, 3.0], t0_s=0.0, dt_s=1e-6)


def test_waveform_coerces_list_to_float_array() -> None:
    wf = _ramp()
    assert isinstance(wf.samples, np.ndarray)
    assert wf.samples.dtype == np.float64
    assert wf.n == 4


def test_waveform_derived_metrics() -> None:
    wf = _ramp()
    assert wf.sample_rate_hz == pytest.approx(1e6)
    assert wf.duration_s == pytest.approx(4e-6)
    assert wf.vmin == pytest.approx(0.0)
    assert wf.vmax == pytest.approx(3.0)
    assert wf.vpp == pytest.approx(3.0)


def test_waveform_time_axis() -> None:
    wf = _ramp()
    np.testing.assert_allclose(wf.time_axis(), [0.0, 1e-6, 2e-6, 3e-6])


def test_waveform_rejects_2d_samples() -> None:
    with pytest.raises(ValidationError):
        Waveform(channel=ChannelId.CH1, samples=[[1.0, 2.0]], t0_s=0.0, dt_s=1e-6)


def test_waveform_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValidationError):
        Waveform(channel=ChannelId.CH1, samples=[1.0], t0_s=0.0, dt_s=0.0)


def test_waveform_serializes_summary_not_raw_samples() -> None:
    dumped = _ramp().model_dump()
    assert dumped["samples"] == {
        "n": 4,
        "vmin": 0.0,
        "vmax": 3.0,
        "sample_rate_hz": pytest.approx(1e6),
    }


def test_clipping_is_a_population_not_a_single_sample() -> None:
    # 1000 samples mid-range with TWO at the rail = 0.2% — a transient overshoot,
    # not clipping. The old "any sample at the rail" test would have flagged it.
    rails = (-5.0, 5.0)
    samples = np.zeros(1000)
    samples[0] = 5.0  # a lone rail-grazing spike
    samples[1] = 5.0
    wf = Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=1e-6, saturation=rails)
    assert wf.clipped_fraction == pytest.approx(0.002)
    assert wf.is_clipped is False


def test_clipping_flags_a_sustained_rail_population() -> None:
    # A flat-topped signal: 30% of samples pinned at the high rail -> clipped.
    rails = (-5.0, 5.0)
    samples = np.concatenate([np.full(300, 5.0), np.zeros(700)])
    wf = Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=1e-6, saturation=rails)
    assert wf.clipped_fraction == pytest.approx(0.3)
    assert wf.is_clipped is True


def test_clipped_fraction_zero_without_known_rails() -> None:
    wf = Waveform(channel=ChannelId.CH1, samples=[0.0, 1.0, 2.0], t0_s=0.0, dt_s=1e-6)
    assert wf.clipped_fraction == 0.0
    assert wf.is_clipped is False


def test_digital_trace_counts_transitions() -> None:
    trace = DigitalTrace(
        channel=ChannelId.CH1, levels=[0, 1, 1, 0, 1], t0_s=0.0, dt_s=1e-6
    )
    assert trace.levels.dtype == np.bool_
    assert trace.n == 5
    assert trace.model_dump()["levels"] == {"n": 5, "transitions": 3}
