"""Analog → digital thresholding and edge finding."""

from __future__ import annotations

import numpy as np

from hwtools.decode.threshold import find_edges, threshold
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import Waveform


def _wf(samples: list[float], dt_s: float = 1e-6) -> Waveform:
    return Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=dt_s)


def test_threshold_simple_step() -> None:
    trace = threshold(_wf([0.0, 0.0, 3.3, 3.3]), level_v=1.65)
    np.testing.assert_array_equal(trace.levels, [False, False, True, True])


def test_hysteresis_suppresses_chatter_around_level() -> None:
    # Wobble across the bare level but never past the hysteresis band -> no edges.
    samples = [0.0, 1.7, 1.6, 1.7, 1.6, 0.0]
    noisy = threshold(_wf(samples), level_v=1.65, hysteresis_v=1.0)  # band [1.15, 2.15]
    # Starts low (0.0 < 1.65) and never exceeds 2.15, so it stays low throughout.
    np.testing.assert_array_equal(noisy.levels, [False] * 6)


def test_find_edges_reports_direction_and_time() -> None:
    trace = threshold(_wf([0.0, 3.3, 3.3, 0.0], dt_s=1e-3), level_v=1.65)
    edges = find_edges(trace)
    assert [(e.index, e.rising) for e in edges] == [(1, True), (3, False)]
    assert edges[0].time_s == 1e-3
