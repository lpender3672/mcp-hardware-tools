"""M0 sanity: the shared enums are stable, ordered, and string/int valued."""

from __future__ import annotations

from hwtools.model.ids import AcqType, ChannelId, Coupling, Slope, SweepMode


def test_channel_ids_are_one_indexed() -> None:
    assert [c.value for c in ChannelId] == [1, 2, 3, 4]
    assert int(ChannelId.CH1) == 1


def test_coupling_members() -> None:
    assert {c.value for c in Coupling} == {"AC", "DC", "GND"}


def test_slope_members() -> None:
    assert {s.value for s in Slope} == {"RISING", "FALLING", "EITHER"}


def test_sweep_and_acq_round_trip_by_value() -> None:
    assert SweepMode("AUTO") is SweepMode.AUTO
    assert AcqType("HIGH_RES") is AcqType.HIGH_RES
