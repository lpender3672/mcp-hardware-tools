"""Measurements and capture judgement."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.analysis import measure
from hwtools.analysis.judge import judge_capture
from hwtools.drivers.simulated import DEFAULT_CAPABILITIES, SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.model.waveform import Waveform


def _sine_wf(freq_hz: float, dt_s: float, n: int = 2000) -> Waveform:
    t = np.arange(n) * dt_s
    samples = np.sin(2 * np.pi * freq_hz * t)
    return Waveform(channel=ChannelId.CH1, samples=samples, t0_s=0.0, dt_s=dt_s)


def test_frequency_estimate() -> None:
    wf = _sine_wf(1_000.0, dt_s=1e-6)
    assert measure.frequency(wf) == pytest.approx(1_000.0, rel=0.05)


def test_frequency_none_for_dc() -> None:
    wf = Waveform(channel=ChannelId.CH1, samples=np.full(100, 2.0), t0_s=0.0, dt_s=1e-6)
    assert measure.frequency(wf) is None


def _scope() -> SimulatedScope:
    sig = {ChannelId.CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)}
    scope = SimulatedScope(sig)
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=200e-6))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=0.0, slope=Slope.RISING),
            sweep=SweepMode.NORMAL,
        )
    )
    return scope


def _judge(scope: SimulatedScope) -> object:
    cap = scope.capture([ChannelId.CH1])
    return judge_capture(cap, scope._channels, DEFAULT_CAPABILITIES)


def test_judge_flags_clipping() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.1))  # clips
    quality = _judge(scope)
    assert quality.clipping[ChannelId.CH1] is True  # type: ignore[attr-defined]
    assert quality.usable is False  # type: ignore[attr-defined]


def test_judge_good_capture_is_usable() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.3))
    quality = _judge(scope)
    assert quality.triggered is True  # type: ignore[attr-defined]
    assert quality.clipping[ChannelId.CH1] is False  # type: ignore[attr-defined]
    assert quality.usable is True  # type: ignore[attr-defined]


def test_judge_low_fill_noted_but_usable() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=5.0))  # tiny
    quality = _judge(scope)
    assert quality.fill_fraction[ChannelId.CH1] < 0.1  # type: ignore[attr-defined]
    assert any("fills only" in n for n in quality.notes)  # type: ignore[attr-defined]


def test_judge_untriggered_is_unusable() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.3))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=9.0),  # out of range
            sweep=SweepMode.NORMAL,
        )
    )
    quality = _judge(scope)
    assert quality.triggered is False  # type: ignore[attr-defined]
    assert quality.usable is False  # type: ignore[attr-defined]
