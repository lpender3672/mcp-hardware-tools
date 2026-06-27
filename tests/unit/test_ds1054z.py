"""DS1054Z driver: SCPI serialization and waveform scaling, via FakeTransport."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import (
    AcqType,
    ChannelId,
    Coupling,
    Slope,
    SweepMode,
    TriggerStatus,
)
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.transport.fake import FakeTransport


def _scope(
    queries: dict[str, str] | None = None, blocks: dict[str, bytes] | None = None
) -> tuple[DS1054Z, FakeTransport]:
    fake = FakeTransport(queries=queries, blocks=blocks)
    return DS1054Z(fake), fake


def test_idn_and_lifecycle() -> None:
    scope, fake = _scope(queries={"*IDN?": "RIGOL,DS1054Z,DS1ZA,00.04.03"})
    with scope:
        assert fake.opened is True
        assert scope.idn().startswith("RIGOL,DS1054Z")
    assert fake.opened is False


def test_configure_channel_emits_expected_scpi() -> None:
    scope, fake = _scope()
    scope.configure_channel(
        ChannelConfig(
            channel=ChannelId.CH1,
            coupling=Coupling.DC,
            scale_v_per_div=0.5,
            offset_v=-1.0,
            probe_ratio=10.0,
        )
    )
    assert fake.log == [
        ":CHANnel1:DISPlay ON",
        ":CHANnel1:PROBe 10",
        ":CHANnel1:COUPling DC",
        ":CHANnel1:SCALe 0.5",
        ":CHANnel1:OFFSet -1",
        ":CHANnel1:BWLimit OFF",
        ":CHANnel1:INVert OFF",
    ]


def test_configure_timebase_emits_mode_scale_offset() -> None:
    scope, fake = _scope()
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3, offset_s=0.0))
    assert fake.log == [
        ":TIMebase:MODE MAIN",
        ":TIMebase:MAIN:SCALe 0.001",
        ":TIMebase:MAIN:OFFSet 0",
    ]


def test_configure_trigger_maps_enums() -> None:
    scope, fake = _scope()
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH2, level_v=1.2, slope=Slope.FALLING),
            sweep=SweepMode.SINGLE,
        )
    )
    assert ":TRIGger:EDGe:SOURce CHANnel2" in fake.log
    assert ":TRIGger:EDGe:SLOPe NEGative" in fake.log
    assert ":TRIGger:EDGe:LEVel 1.2" in fake.log
    assert ":TRIGger:SWEep SINGle" in fake.log


def test_configure_acquire_average_includes_count() -> None:
    scope, fake = _scope()
    scope.configure_acquire(AcquireConfig(type=AcqType.AVERAGE, averages=16, memory_depth=12000))
    assert ":ACQuire:TYPE AVERages" in fake.log
    assert ":ACQuire:AVERages 16" in fake.log
    assert ":ACQuire:MDEPth 12000" in fake.log


def test_acquire_auto_memory_depth() -> None:
    scope, fake = _scope()
    scope.configure_acquire(AcquireConfig(type=AcqType.NORMAL))
    assert ":ACQuire:MDEPth AUTO" in fake.log
    assert not any(cmd.startswith(":ACQuire:AVERages") for cmd in fake.log)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("TD", TriggerStatus.TRIGGERED), ("STOP", TriggerStatus.STOP), ("auto", TriggerStatus.AUTO)],
)
def test_trigger_status_parsing(reply: str, expected: TriggerStatus) -> None:
    scope, _ = _scope(queries={":TRIGger:STATus?": reply})
    assert scope.trigger_status() is expected


def test_capture_scales_samples_from_preamble() -> None:
    # preamble: format,type,points,count,xinc,xorig,xref,yinc,yorig,yref
    preamble = "0,0,4,1,1.000000e-06,-2.000000e-06,0,4.000000e-02,125,0"
    payload = bytes([125, 150, 100, 200])  # codes -> volts via (code - 125 - 0) * 0.04
    scope, fake = _scope(
        queries={
            ":ACQuire:SRATe?": "1.000000e+09",
            ":WAVeform:PREamble?": preamble,
            ":TRIGger:STATus?": "STOP",
        },
        blocks={":WAVeform:DATA?": payload},
    )

    cap = scope.capture([ChannelId.CH1])

    assert cap.sample_rate_hz == pytest.approx(1e9)
    assert cap.trigger_status is TriggerStatus.STOP
    wf = cap.waveforms[ChannelId.CH1]
    assert wf.dt_s == pytest.approx(1e-6)
    assert wf.t0_s == pytest.approx(-2e-6)
    np.testing.assert_allclose(wf.samples, [0.0, 1.0, -1.0, 3.0])
    # the right SCPI preceded the block read
    assert ":WAVeform:SOURce CHANnel1" in fake.log
    assert ":WAVeform:FORMat BYTE" in fake.log
