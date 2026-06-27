"""Validation and derived behaviour for the config value objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hwtools.model import (
    AcqType,
    AcquireConfig,
    ChannelConfig,
    ChannelId,
    Coupling,
    EdgeTrigger,
    Slope,
    SweepMode,
    TimebaseConfig,
    TriggerConfig,
    TriggerCoupling,
)

# -- ChannelConfig -----------------------------------------------------------


def test_channel_defaults_and_full_scale() -> None:
    cfg = ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5)
    assert cfg.coupling is Coupling.DC
    assert cfg.probe_ratio == 10.0
    assert cfg.enabled is True
    assert cfg.full_scale_v(divisions=8) == pytest.approx(4.0)


def test_channel_is_frozen() -> None:
    cfg = ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5)
    with pytest.raises(ValidationError):
        cfg.scale_v_per_div = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_channel_rejects_nonpositive_scale(bad: float) -> None:
    with pytest.raises(ValidationError):
        ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=bad)


def test_full_scale_rejects_nonpositive_divisions() -> None:
    cfg = ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=0.5)
    with pytest.raises(ValueError):
        cfg.full_scale_v(divisions=0)


# -- TimebaseConfig ----------------------------------------------------------


def test_timebase_full_span() -> None:
    tb = TimebaseConfig(scale_s_per_div=1e-3)
    assert tb.full_span_s(divisions=12) == pytest.approx(12e-3)


def test_timebase_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValidationError):
        TimebaseConfig(scale_s_per_div=0.0)


# -- AcquireConfig -----------------------------------------------------------


def test_acquire_normal_ignores_averages() -> None:
    cfg = AcquireConfig(type=AcqType.NORMAL, averages=3)
    assert cfg.type is AcqType.NORMAL


@pytest.mark.parametrize("n", [2, 4, 16, 1024])
def test_acquire_average_accepts_power_of_two(n: int) -> None:
    cfg = AcquireConfig(type=AcqType.AVERAGE, averages=n)
    assert cfg.averages == n


@pytest.mark.parametrize("n", [1, 3, 6, 2048])
def test_acquire_average_rejects_invalid_counts(n: int) -> None:
    with pytest.raises(ValidationError):
        AcquireConfig(type=AcqType.AVERAGE, averages=n)


def test_acquire_memory_depth_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AcquireConfig(memory_depth=0)


# -- Trigger (discriminated union) -------------------------------------------


def test_trigger_config_defaults() -> None:
    cfg = TriggerConfig(trigger=EdgeTrigger(source=ChannelId.CH1, level_v=1.5))
    assert cfg.sweep is SweepMode.AUTO
    assert cfg.coupling is TriggerCoupling.DC
    assert cfg.source is ChannelId.CH1
    assert cfg.trigger.slope is Slope.RISING


def test_trigger_round_trips_through_discriminator() -> None:
    cfg = TriggerConfig(
        trigger=EdgeTrigger(source=ChannelId.CH2, level_v=-0.2, slope=Slope.FALLING),
        sweep=SweepMode.NORMAL,
    )
    restored = TriggerConfig.model_validate(cfg.model_dump())
    assert isinstance(restored.trigger, EdgeTrigger)
    assert restored.trigger.kind == "edge"
    assert restored.trigger.level_v == pytest.approx(-0.2)
    assert restored.sweep is SweepMode.NORMAL
