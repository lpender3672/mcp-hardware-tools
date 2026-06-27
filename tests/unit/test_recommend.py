"""Single-stage setup recommendation and autoset on the simulated scope."""

from __future__ import annotations

import pytest

from hwtools.analysis.loop import autoset
from hwtools.analysis.recommend import recommend_setup
from hwtools.drivers.simulated import DEFAULT_CAPABILITIES, SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig

CH1 = ChannelId.CH1


def _trigger(level: float = 0.0) -> TriggerConfig:
    return TriggerConfig(
        trigger=EdgeTrigger(source=CH1, level_v=level, slope=Slope.RISING),
        sweep=SweepMode.NORMAL,
    )


def test_recommend_sizes_scale_offset_timebase_trigger_in_one_shot() -> None:
    # Offset sine: vrange [-1, 3], midline 1, vpp 4, 1 kHz.
    scope = SimulatedScope({CH1: Sine(amplitude_v=2.0, frequency_hz=1_000.0, offset_v=1.0)})
    wide = {CH1: ChannelConfig(channel=CH1, scale_v_per_div=5.0)}
    scope.configure_channel(wide[CH1])
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3))
    measurement = scope.capture([CH1])

    setup = recommend_setup(
        measurement,
        channels=wide,
        timebase=TimebaseConfig(scale_s_per_div=1e-3),
        trigger=_trigger(),
        capabilities=DEFAULT_CAPABILITIES,
    )

    ch = setup.channels[CH1]
    assert ch.scale_v_per_div == pytest.approx(4.0 / (0.6 * 8), rel=0.1)  # ~60% fill
    assert ch.offset_v == pytest.approx(-1.0, abs=0.05)  # centre the +1 V offset
    # 4 periods of 1 kHz across 12 divisions -> ~0.333 ms/div
    assert setup.timebase.scale_s_per_div == pytest.approx(4e-3 / 12, rel=0.1)
    assert setup.trigger.trigger.level_v == pytest.approx(1.0, abs=0.05)  # source midline


def test_autoset_from_clipped_reaches_usable_in_one_recommendation() -> None:
    scope = SimulatedScope({CH1: Sine(amplitude_v=3.0, frequency_hz=1_000.0)})
    # Start clipped and badly triggered; autoset should not care.
    result = autoset(
        scope,
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.1)},
        timebase=TimebaseConfig(scale_s_per_div=2e-3),
        trigger=_trigger(level=9.0),
    )

    assert result.converged
    assert result.quality.usable
    assert result.widen_steps == 0  # 5 V/div was wide enough to measure
    assert not result.quality.clipping[CH1]
    assert 0.4 < result.quality.fill_fraction[CH1] < 0.8  # well filled, not clipped


def test_autoset_widens_for_a_giant_signal() -> None:
    scope = SimulatedScope({CH1: Sine(amplitude_v=40.0, frequency_hz=1_000.0)})  # 80 Vpp
    result = autoset(
        scope,
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.1)},
        timebase=TimebaseConfig(scale_s_per_div=2e-3),
        trigger=_trigger(),
    )
    assert result.widen_steps >= 1  # 5 V/div clipped an 80 Vpp signal; had to widen
    assert result.converged
