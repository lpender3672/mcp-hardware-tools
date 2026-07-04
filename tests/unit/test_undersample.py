"""Undersampling: judge flags it and the loop speeds the timebase up.

This path was previously dead even in the sim (its signals were always well
sampled). A square avoids the sine-aliasing-to-DC trap, so 2 samples/period is
both genuinely undersampled and still measurable.
"""

from __future__ import annotations

from hwtools.analysis.adjust import suggest_adjustment
from hwtools.analysis.judge import judge_capture
from hwtools.drivers.simulated import DEFAULT_CAPABILITIES, SimulatedScope, Square
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session.loop import capture_until_usable

CH1 = ChannelId.CH1
# 0.05 s/div -> ~2 samples per 1 kHz period.
_SLOW = TimebaseConfig(scale_s_per_div=0.05)
_TRIGGER = TriggerConfig(
    trigger=EdgeTrigger(source=CH1, level_v=0.0, slope=Slope.RISING),
    sweep=SweepMode.NORMAL,
)


def _scope() -> SimulatedScope:
    scope = SimulatedScope({CH1: Square(amplitude_v=1.0, frequency_hz=1_000.0)})
    scope.configure_channel(ChannelConfig(channel=CH1, scale_v_per_div=0.3))
    scope.configure_timebase(_SLOW)
    scope.configure_trigger(_TRIGGER)
    return scope


def test_judge_flags_undersampling() -> None:
    scope = _scope()
    cap = scope.capture([CH1])
    result = judge_capture(cap, scope._channels, DEFAULT_CAPABILITIES)
    assert result.bandwidth_ok is False
    assert any("undersampled" in note for note in result.notes)


def test_adjust_speeds_up_undersampled_timebase() -> None:
    scope = _scope()
    cap = scope.capture([CH1])
    result = judge_capture(cap, scope._channels, DEFAULT_CAPABILITIES)
    adjustment = suggest_adjustment(
        result, cap, scope._channels, _SLOW, _TRIGGER, DEFAULT_CAPABILITIES
    )
    assert adjustment.timebase is not None
    assert adjustment.timebase.scale_s_per_div < _SLOW.scale_s_per_div


def test_loop_resolves_undersampling() -> None:
    scope = _scope()
    result = capture_until_usable(
        scope,
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.3)},
        timebase=_SLOW,
        trigger=_TRIGGER,
    )
    assert result.converged
    assert any("undersampled" in a.reason for a in result.adjustments)
