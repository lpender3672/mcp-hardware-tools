"""The single acquire primitive over the simulated scope: honest one-shot vs
free-run, store integration, and the decision-ready result it returns."""

from __future__ import annotations

from hwtools.drivers.simulated import SimulatedScope, Sine
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session.acquire import acquire
from hwtools.session.store import CaptureStore

CH1 = ChannelId.CH1


def _scope() -> SimulatedScope:
    return SimulatedScope({CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)})


def _channels(scale: float = 0.3) -> dict[ChannelId, ChannelConfig]:
    return {CH1: ChannelConfig(channel=CH1, scale_v_per_div=scale)}


def _trigger(level: float) -> TriggerConfig:
    return TriggerConfig(
        trigger=EdgeTrigger(source=CH1, level_v=level, slope=Slope.RISING),
        sweep=SweepMode.SINGLE,
    )


def _tb() -> TimebaseConfig:
    return TimebaseConfig(scale_s_per_div=200e-6)


def test_single_shot_triggers_stores_kept_and_stamps_id() -> None:
    scope = _scope()
    store = CaptureStore()
    frame = acquire(
        scope, store, channels=_channels(), timebase=_tb(), trigger=_trigger(0.0),
        sweep=SweepMode.SINGLE,
    )
    assert frame.result.triggered
    assert frame.capture is not None
    assert frame.capture_id == "cap-1"
    assert frame.result.capture_id == "cap-1"  # stamped onto the agent-facing result
    # A one-shot is auto-kept (unrepeatable) and provenance records why.
    info = store.info("cap-1")
    assert info.kept and info.keep_reason == "one-shot"


def test_single_shot_miss_returns_untriggered_and_stores_nothing() -> None:
    scope = _scope()
    store = CaptureStore()
    frame = acquire(
        scope, store, channels=_channels(), timebase=_tb(), trigger=_trigger(9.0),
        sweep=SweepMode.SINGLE,  # level out of range → never fires
    )
    assert frame.result.triggered is False
    assert frame.capture is None
    assert frame.capture_id is None
    assert len(store) == 0  # never fabricates or stores a missed frame


def test_free_run_always_yields_an_ephemeral_frame() -> None:
    scope = _scope()
    store = CaptureStore()
    frame = acquire(
        scope, store, channels=_channels(), timebase=_tb(), trigger=_trigger(9.0),
        sweep=SweepMode.AUTO,  # out-of-range trigger, but AUTO free-runs anyway
    )
    assert frame.capture is not None  # AUTO always produces a frame
    assert frame.result.triggered  # AUTO counts as triggered
    assert frame.capture_id is not None
    assert store.info(frame.capture_id).kept is False  # ephemeral


def test_free_run_self_reclaims_previous_frame() -> None:
    scope = _scope()
    store = CaptureStore()
    for _ in range(5):
        acquire(
            scope, store, channels=_channels(), timebase=_tb(), trigger=_trigger(0.0),
            sweep=SweepMode.AUTO,
        )
    assert len(store) == 1  # ephemerals never accumulate


def test_result_carries_measurements_and_config_echo() -> None:
    scope = _scope()
    store = CaptureStore()
    frame = acquire(
        scope, store, channels=_channels(scale=0.3), timebase=_tb(),
        trigger=_trigger(0.0), sweep=SweepMode.SINGLE,
    )
    reading = frame.result.channels[CH1]
    assert reading.config.scale_v_per_div == 0.3  # echoes what it was taken with
    assert reading.vpp > 0
    assert reading.frequency is not None
    assert frame.result.timebase == _tb()
