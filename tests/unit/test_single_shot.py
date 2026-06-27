"""Single-shot acquisition on the simulated scope."""

from __future__ import annotations

from hwtools.analysis.loop import capture_single
from hwtools.drivers.simulated import SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig

CH1 = ChannelId.CH1


def _trigger(level: float) -> TriggerConfig:
    return TriggerConfig(
        trigger=EdgeTrigger(source=CH1, level_v=level, slope=Slope.RISING),
        sweep=SweepMode.SINGLE,
    )


def test_single_shot_captures_when_trigger_in_range() -> None:
    scope = SimulatedScope({CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)})
    result = capture_single(
        scope,
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.3)},
        timebase=TimebaseConfig(scale_s_per_div=200e-6),
        trigger=_trigger(level=0.0),  # in range
    )
    assert result.triggered
    assert result.capture is not None
    assert result.quality is not None
    assert result.quality.usable


def test_single_shot_times_out_when_trigger_out_of_range() -> None:
    scope = SimulatedScope({CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)})
    result = capture_single(
        scope,
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.3)},
        timebase=TimebaseConfig(scale_s_per_div=200e-6),
        trigger=_trigger(level=9.0),  # never fires
        poll_timeout_s=0.1,
    )
    assert result.triggered is False
    assert result.capture is None
