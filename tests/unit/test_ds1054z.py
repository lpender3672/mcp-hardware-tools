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
        ":CHANnel1:SCALe 0.5",  # already a 1-2-5 step; sent unchanged
        ":CHANnel1:OFFSet -1",
        ":CHANnel1:BWLimit OFF",
        ":CHANnel1:INVert OFF",
    ]


def test_non_1_2_5_scale_is_snapped_to_the_grid() -> None:
    # recommend/adjust produce off-grid scales (1.25 V/div, or x2 growth to 0.4/0.8).
    # The scope's scale is 1-2-5 constrained (VERNier only fine-tunes around a coarse
    # step), so an off-grid value conflicts -> "Parameter limited!". Snap to the grid.
    scope, fake = _scope()
    scope.configure_channel(
        ChannelConfig(channel=ChannelId.CH2, scale_v_per_div=1.25, probe_ratio=10.0)
    )
    assert ":CHANnel2:SCALe 1" in fake.log  # 1.25 -> nearest 1-2-5 step (1 V/div)
    assert not any("VERNier" in cmd for cmd in fake.log)


def test_channel_scale_and_offset_clamped_to_documented_range() -> None:
    scope, fake = _scope()
    # scale 200 V/div is over the 10X max (100 V); offset 50 V exceeds the ±20 V
    # limit at 1 V/div, 10X (scale < 5 V/div). Both must be clamped, not clamped by
    # the scope with a "Parameter limited!" beep.
    scope.configure_channel(
        ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=200.0, probe_ratio=10.0)
    )
    assert ":CHANnel1:SCALe 100" in fake.log
    fake.log.clear()
    scope.configure_channel(
        ChannelConfig(
            channel=ChannelId.CH1, scale_v_per_div=1.0, offset_v=50.0, probe_ratio=10.0
        )
    )
    assert ":CHANnel1:OFFSet 20" in fake.log


def test_trigger_level_clamped_to_source_screen_range() -> None:
    # At 0.1 V/div a first-guess 9 V level must clamp to just inside +/-5 div
    # (4.9 div = 0.49 V) rather than be clamped by the scope ("Parameter limited!").
    # Setting it *at* the +/-5 div edge is the "at limit" case that beeps.
    scope, fake = _scope()
    scope.configure_channel(
        ChannelConfig(channel=ChannelId.CH2, scale_v_per_div=0.1, probe_ratio=10.0)
    )
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH2, level_v=9.0), sweep=SweepMode.SINGLE
        )
    )
    assert ":TRIGger:EDGe:LEVel 0.49" in fake.log


def test_trigger_level_unclamped_for_unconfigured_source() -> None:
    # Without a known source scale the driver cannot bound it; send as-is.
    scope, fake = _scope()
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH2, level_v=1.2), sweep=SweepMode.SINGLE
        )
    )
    assert ":TRIGger:EDGe:LEVel 1.2" in fake.log


def test_timebase_snapped_to_1_2_5_step() -> None:
    scope, fake = _scope()
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=3.33e-4))  # 333 us
    assert ":TIMebase:MAIN:SCALe 0.0005" in fake.log  # snapped to 500 us
    fake.log.clear()
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3))  # already 1-2-5
    assert ":TIMebase:MAIN:SCALe 0.001" in fake.log


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


def test_run_control_commands() -> None:
    scope, fake = _scope(queries={":TRIGger:STATus?": "STOP"})
    scope.run()
    scope.stop()
    scope.single()
    scope.force_trigger()
    assert scope.trigger_status() is TriggerStatus.STOP
    assert fake.log == [":RUN", ":STOP", ":SINGle", ":TFORce", ":TRIGger:STATus?"]


def test_trigger_status_rejects_unknown_reply() -> None:
    scope, _ = _scope(queries={":TRIGger:STATus?": "BOGUS"})
    with pytest.raises(ValueError, match="unexpected trigger status"):
        scope.trigger_status()


def test_autoscale_emits_scpi() -> None:
    scope, fake = _scope()
    scope.autoscale()
    assert fake.log == [":AUToscale"]


def _display(states: dict[int, str]) -> dict[str, str]:
    """Scripted :CHANnel<n>:DISPlay? replies for the four analog channels."""
    return {f":CHANnel{n}:DISPlay?": states.get(n, "0") for n in (1, 2, 3, 4)}


def test_configure_acquire_average_includes_count() -> None:
    # one channel on -> 12000 is a legal record length
    scope, fake = _scope(queries=_display({1: "1"}))
    scope.configure_acquire(AcquireConfig(type=AcqType.AVERAGE, averages=16, memory_depth=12000))
    assert ":ACQuire:TYPE AVERages" in fake.log
    assert ":ACQuire:AVERages 16" in fake.log
    assert ":ACQuire:MDEPth 12000" in fake.log


def test_acquire_auto_memory_depth() -> None:
    scope, fake = _scope()
    scope.configure_acquire(AcquireConfig(type=AcqType.NORMAL))
    assert ":ACQuire:MDEPth AUTO" in fake.log
    assert not any(cmd.startswith(":ACQuire:AVERages") for cmd in fake.log)


def test_memory_depth_validated_against_enabled_channel_count() -> None:
    # two channels on -> 120000 is NOT legal (the 2-channel set is 6k/60k/600k/...).
    # The driver must refuse it rather than send a command that beeps/desyncs the scope.
    scope, _ = _scope(queries=_display({1: "1", 2: "1"}))
    with pytest.raises(ValueError, match="not a legal record length with 2 analog"):
        scope.configure_acquire(AcquireConfig(memory_depth=120_000))


def test_memory_depth_legal_for_two_channels_is_sent() -> None:
    scope, fake = _scope(queries=_display({1: "1", 2: "1"}))
    scope.configure_acquire(AcquireConfig(memory_depth=60_000))  # legal for 2 channels
    assert ":ACQuire:MDEPth 60000" in fake.log


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("TD", TriggerStatus.TRIGGERED), ("STOP", TriggerStatus.STOP), ("auto", TriggerStatus.AUTO)],
)
def test_trigger_status_parsing(reply: str, expected: TriggerStatus) -> None:
    scope, _ = _scope(queries={":TRIGger:STATus?": reply})
    assert scope.trigger_status() is expected


class _PagingTransport(FakeTransport):
    """A transport that serves ``:WAVeform:DATA?`` from a backing buffer, sliced
    by the most recent ``:WAVeform:STARt``/``:STOP`` window — so the driver's deep
    paging loop can be exercised end to end."""

    def __init__(self, memory: bytes, queries: dict[str, str]) -> None:
        super().__init__(queries=queries)
        self._memory = memory
        self._start = 1
        self._stop = len(memory)

    def write(self, command: str) -> None:
        super().write(command)
        if command.startswith(":WAVeform:STARt "):
            self._start = int(command.rsplit(" ", 1)[1])
        elif command.startswith(":WAVeform:STOP "):
            self._stop = int(command.rsplit(" ", 1)[1])

    def query_block(self, command: str) -> bytes:
        self.log.append(command)
        return self._memory[self._start - 1 : self._stop]


def test_deep_capture_pages_full_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # 7 points read in 3-point chunks: 3 + 3 + 1 — the loop must stitch them back
    # into the full memory, not just the first screen-sized block.
    monkeypatch.setattr(DS1054Z, "_RAW_CHUNK", 3)
    memory = bytes([125, 126, 127, 128, 129, 130, 131])
    preamble = f"0,0,{len(memory)},1,1.000000e-06,0,0,1.000000e+00,125,0"
    fake = _PagingTransport(
        memory,
        queries={
            ":ACQuire:SRATe?": "1.000000e+09",
            ":WAVeform:PREamble?": preamble,
            ":TRIGger:STATus?": "STOP",
        },
    )
    scope = DS1054Z(fake)

    cap = scope.capture([ChannelId.CH1])

    wf = cap.waveforms[ChannelId.CH1]
    assert wf.samples.size == len(memory)
    np.testing.assert_allclose(wf.samples, np.arange(7, dtype=np.float64))
    # deep read stops the scope first and pages in RAW mode
    assert ":STOP" in fake.log
    assert ":WAVeform:MODE RAW" in fake.log
    assert fake.log.count(":WAVeform:DATA?") == 3  # 3 chunks for 7 points @ chunk=3


def test_deep_capture_throws_when_no_deep_record() -> None:
    # RAW points == 0 means a free-running/untriggered acquisition with no deep
    # memory. The tool must surface that, not silently fall back to the screen trace.
    preamble = "0,0,0,1,1.000000e-06,0,0,1.000000e+00,125,0"
    fake = _PagingTransport(
        b"",
        queries={
            ":ACQuire:SRATe?": "1.000000e+09",
            ":WAVeform:PREamble?": preamble,
            ":TRIGger:STATus?": "STOP",
        },
    )
    scope = DS1054Z(fake)
    with pytest.raises(RuntimeError, match="no deep acquisition in memory"):
        scope.capture([ChannelId.CH1])


def test_deep_capture_rejects_short_memory_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DS1054Z, "_RAW_CHUNK", 100)
    # preamble promises 8 points but memory only holds 4 -> must not pass silently.
    preamble = "0,0,8,1,1.000000e-06,0,0,1.000000e+00,125,0"
    fake = _PagingTransport(
        bytes([125, 126, 127, 128]),
        queries={
            ":ACQuire:SRATe?": "1.000000e+09",
            ":WAVeform:PREamble?": preamble,
            ":TRIGger:STATus?": "STOP",
        },
    )
    scope = DS1054Z(fake)
    with pytest.raises(OSError, match="short-read"):
        scope.capture([ChannelId.CH1])


def test_screen_capture_uses_normal_mode() -> None:
    preamble = "0,0,4,1,1.000000e-06,-2.000000e-06,0,4.000000e-02,125,0"
    scope, fake = _scope(
        queries={
            ":ACQuire:SRATe?": "1.000000e+09",
            ":WAVeform:PREamble?": preamble,
            ":TRIGger:STATus?": "STOP",
        },
        blocks={":WAVeform:DATA?": bytes([125, 150, 100, 200])},
    )

    cap = scope.capture([ChannelId.CH1], deep=False)

    assert ":WAVeform:MODE NORMal" in fake.log
    assert ":WAVeform:MODE RAW" not in fake.log
    assert ":STOP" not in fake.log  # screen read leaves the run state alone
    # The NORMal read pins STARt to 1 (a stale STARt > 1200 from a prior RAW read
    # otherwise makes :WAV:DATA? return a single point). It must NOT set :WAV:STOP,
    # which would chirp "Stop point changed!" on every screen read.
    assert ":WAVeform:STARt 1" in fake.log
    assert not any(cmd.startswith(":WAVeform:STOP") for cmd in fake.log)
    np.testing.assert_allclose(cap.waveforms[ChannelId.CH1].samples, [0.0, 1.0, -1.0, 3.0])


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
