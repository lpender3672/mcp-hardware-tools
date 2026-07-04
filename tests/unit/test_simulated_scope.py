"""SimulatedScope rendering behaviour: window, clipping, fill, triggering."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.drivers.simulated import Dc, SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig


def _scope(**kw: object) -> SimulatedScope:
    sig = {ChannelId.CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)}
    return SimulatedScope(sig, **kw)  # type: ignore[arg-type]


def test_capture_window_matches_timebase() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5))
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3))  # *12 div = 12 ms
    cap = scope.capture([ChannelId.CH1])
    wf = cap.waveforms[ChannelId.CH1]
    assert wf.duration_s == pytest.approx(12e-3, rel=1e-3)
    assert np.all(np.isfinite(wf.samples))


def test_small_scale_clips_signal() -> None:
    scope = _scope()
    # The digitiser captures ~+/-5.1 div (overscan), so at 0.1 V/div a +/-1 V sine
    # clips at +/-0.51 V -- matching the real DS1000Z (divergence #1).
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.1))
    wf = scope.capture([ChannelId.CH1]).waveforms[ChannelId.CH1]
    assert wf.vmax == pytest.approx(0.51, abs=1e-3)
    assert wf.vmin == pytest.approx(-0.51, abs=1e-3)


def test_large_scale_does_not_clip() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=1.0))
    wf = scope.capture([ChannelId.CH1]).waveforms[ChannelId.CH1]
    assert wf.vmax == pytest.approx(1.0, abs=2e-2)
    assert wf.vmin == pytest.approx(-1.0, abs=2e-2)


def test_trigger_in_range_triggers() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=0.0, slope=Slope.RISING),
            sweep=SweepMode.NORMAL,
        )
    )
    cap = scope.capture([ChannelId.CH1])
    assert cap.trigger_status is TriggerStatus.TRIGGERED
    assert cap.triggered is True


def test_trigger_out_of_range_misses_in_normal_sweep() -> None:
    scope = _scope()
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=5.0),  # above the +/-1 V sine
            sweep=SweepMode.NORMAL,
        )
    )
    cap = scope.capture([ChannelId.CH1])
    assert cap.trigger_status is TriggerStatus.WAIT
    assert cap.triggered is False


def test_autoscale_fills_without_clipping() -> None:
    scope = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=2.5, frequency_hz=500.0)})
    scope.autoscale()
    wf = scope.capture([ChannelId.CH1]).waveforms[ChannelId.CH1]
    # filled a good chunk of the 8 divisions but not clipped
    assert wf.vpp == pytest.approx(5.0, rel=0.05)


def test_dc_signal_is_flat() -> None:
    scope = SimulatedScope({ChannelId.CH1: Dc(1.5)})
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=1.0))
    wf = scope.capture([ChannelId.CH1]).waveforms[ChannelId.CH1]
    assert wf.vpp == pytest.approx(0.0, abs=1e-9)
