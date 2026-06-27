"""The self-correcting loop converges on the simulated scope."""

from __future__ import annotations

from hwtools.analysis.loop import LoopResult, capture_until_usable
from hwtools.drivers.simulated import SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig


def _run(
    scope: SimulatedScope, *, scale: float, level: float, max_iterations: int = 10
) -> LoopResult:
    channels = {ChannelId.CH1: ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=scale)}
    timebase = TimebaseConfig(scale_s_per_div=200e-6)
    trigger = TriggerConfig(
        trigger=EdgeTrigger(source=ChannelId.CH1, level_v=level, slope=Slope.RISING),
        sweep=SweepMode.NORMAL,
    )
    return capture_until_usable(
        scope, channels=channels, timebase=timebase, trigger=trigger, max_iterations=max_iterations
    )


def test_converges_from_clipped_and_untriggered() -> None:
    scope = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=3.0, frequency_hz=1_000.0)})
    # scale 0.1 V/div -> 0.8 V span vs 6 Vpp signal (clips); level 9 V out of range.
    result = _run(scope, scale=0.1, level=9.0)

    assert result.converged
    assert result.quality.usable
    assert result.quality.triggered
    assert not any(result.quality.clipping.values())
    assert result.iterations >= 1
    assert result.adjustments  # it had to adjust


def test_loop_exhausts_budget_without_converging() -> None:
    # L3: a clipped start needs several grows; one iteration isn't enough.
    scope = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=3.0, frequency_hz=1_000.0)})
    result = _run(scope, scale=0.1, level=9.0, max_iterations=1)
    assert not result.converged
    assert result.iterations == 1


def test_already_usable_converges_immediately() -> None:
    scope = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)})
    result = _run(scope, scale=0.3, level=0.0)

    assert result.converged
    assert result.iterations == 0
    assert result.adjustments == []


def test_zooms_in_on_tiny_signal() -> None:
    scope = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=0.05, frequency_hz=1_000.0)})
    # 2 V/div -> 16 V span, 0.1 Vpp signal fills <1%. Triggers at 0 V (in range).
    result = _run(scope, scale=2.0, level=0.0)

    assert result.converged
    final_scale = result.adjustments[-1].channels[ChannelId.CH1].scale_v_per_div
    assert final_scale < 2.0  # zoomed in
    assert result.quality.fill_fraction[ChannelId.CH1] > 0.3
