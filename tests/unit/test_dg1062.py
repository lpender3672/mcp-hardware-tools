"""DG1062Z driver: SCPI serialization and readback parsing, via FakeTransport.

No hardware. The behavioural contract (round-trips, over-range, arb) lives in
``tests/contract/test_signal_generator_contract.py``; this file pins the exact wire
strings against ``docs/DG1000Z_ProgrammingGuide_EN.pdf`` — the command *order* the
guide's dependency rules require, full-precision number formatting, and the
arbitrary-to-built-in transition that a naive ``:APPLy`` write path gets wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from hwtools.drivers.rigol import dg1062 as dg1062_mod
from hwtools.drivers.rigol.dg1062 import DG1062
from hwtools.model.siggen import (
    ArbitraryWaveform,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
    arb_repetition_hz,
)
from hwtools.transport.fake import FakeTransport

_NO_ERROR = {":SYSTem:ERRor?": '0,"No error"'}
# Both channels already in Vpp — the normal case, in which connect() writes nothing.
_UNITS_VPP = {":SOUR1:VOLTage:UNIT?": "VPP", ":SOUR2:VOLTage:UNIT?": "VPP"}


def _gen(
    queries: dict[str, str] | None = None, blocks: dict[str, bytes] | None = None
) -> tuple[DG1062, FakeTransport]:
    fake = FakeTransport(queries={**_NO_ERROR, **(queries or {})}, blocks=blocks)
    # output_settle_s=0: no output stage to restart behind a fake transport, and the
    # driver's settles would otherwise add half a second to every configure here.
    return DG1062(fake, output_settle_s=0.0), fake


def _sine(**kw: object) -> SignalGeneratorConfig:
    base: dict[str, object] = {
        "channel": SigGenChannel.CH1,
        "frequency_hz": 1_000.0,
        "amplitude_vpp": 2.0,
    }
    base.update(kw)
    return SignalGeneratorConfig(**base)


class _ErrorQueueTransport(FakeTransport):
    """FakeTransport replaying a scripted :SYSTem:ERRor? queue (one reply per read)."""

    def __init__(self, error_replies: list[str]) -> None:
        super().__init__()
        self._errors = error_replies

    def query(self, command: str) -> str:
        if command == ":SYSTem:ERRor?":
            self.log.append(command)
            return self._errors.pop(0)
        return super().query(command)


def test_idn_lifecycle_and_connect_leaves_correct_units_alone() -> None:
    gen, fake = _gen(
        queries={"*IDN?": "Rigol Technologies,DG1062Z,DG1ZA,00.01.03", **_UNITS_VPP}
    )
    with gen:
        assert fake.opened is True
        assert "DG1062Z" in gen.idn()
    assert fake.opened is False
    # Clears stale errors, then *checks* each channel's amplitude unit rather than
    # writing it: any :SOUR2: command drags the front panel to CH2, so an unconditional
    # write makes the display flip channels on every connect for no reason.
    assert fake.log[:3] == ["*CLS", ":SOUR1:VOLTage:UNIT?", ":SOUR2:VOLTage:UNIT?"]
    assert not any("UNIT VPP" in cmd for cmd in fake.log)


def test_connect_corrects_a_channel_left_in_the_wrong_unit() -> None:
    # The front panel can leave a channel in Vrms or dBm, and :VOLTage would then speak
    # units the model does not (guide 2-185) — so a wrong unit *is* rewritten.
    gen, fake = _gen(
        queries={":SOUR1:VOLTage:UNIT?": "VRMS", ":SOUR2:VOLTage:UNIT?": "VPP"}
    )
    gen.connect()
    assert ":SOUR1:VOLTage:UNIT VPP" in fake.log
    assert ":SOUR2:VOLTage:UNIT VPP" not in fake.log  # already correct, left untouched


# -- the write path: discrete commands, in the guide's dependency order -------------


def test_configure_sine_emits_discrete_commands_in_dependency_order() -> None:
    gen, fake = _gen()
    gen.configure_channel(_sine(waveform=WaveShape.SINE, offset_v=0.0))
    assert fake.log == [
        # Output down first: with it live, a channel playing the volatile arbitrary
        # cannot be talked onto a built-in waveform at all (scope-measured).
        ":OUTP1:STATe OFF",
        # Escape sample-rate arb mode, where :FREQuency is ignored (guide 2-100).
        ":SOUR1:FUNCtion:ARBitrary:MODE FREQ",
        # Load first: amplitude's range is limited by the impedance.
        ":OUTP1:LOAD INFinity",
        # Type next: the frequency carry-over clamp fires here...
        ":SOUR1:FUNCtion SINusoid",
        # ...and these explicit writes overwrite whatever it clamped to.
        ":SOUR1:FREQuency 1000.0",
        ":SOUR1:VOLTage 2.0",
        ":SOUR1:VOLTage:OFFSet 0.0",
        ":OUTP1:STATe ON",
        ":SYSTem:ERRor?",
    ]


def test_every_written_shape_name_reads_back_to_the_same_shape() -> None:
    """The forward and reverse shape maps are hand-maintained and must not drift.

    ``_SHAPE_NAME`` is what gets written and also seeds ``capabilities.waveforms``;
    ``_NAME_SHAPE`` decodes ``:FUNCtion?``. Nothing else couples them, so a shape added
    to one and forgotten in the other would advertise a waveform the driver could set
    but never read back — and only on hardware.
    """
    for shape, name in dg1062_mod._SHAPE_NAME.items():
        decoded = dg1062_mod._NAME_SHAPE.get(name.upper())
        assert decoded is shape, f"{shape} writes {name!r}, which reads back as {decoded}"
    # And the reverse map must not claim shapes the driver never advertises.
    assert set(dg1062_mod._NAME_SHAPE.values()) == set(dg1062_mod._SHAPE_NAME)


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture the driver's ``time.sleep`` durations instead of actually waiting."""
    slept: list[float] = []
    monkeypatch.setattr("hwtools.drivers.rigol.dg1062.time.sleep", slept.append)
    return slept


def test_configure_restarts_the_output_stage_with_a_settle_each_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output cycle and its two settles are load-bearing, not defensive padding.

    Scope-measured on a DG1062Z, both invisible to read-back: with the output left on, a
    channel playing the volatile arbitrary cannot be moved onto a built-in waveform by
    *any* command sequence; and an enable that overtakes the parameter writes comes up on
    the previous waveform (0/4 correct at a 0 ms gap, 4/4 at 50 ms and above). Asserted
    on ordering here; the analog proof is in cross/test_dg1062_scope.
    """
    slept = _record_sleeps(monkeypatch)
    fake = FakeTransport(queries=_NO_ERROR)
    DG1062(fake, output_settle_s=0.25).configure_channel(_sine())
    assert slept == [0.25, 0.25], "configure must settle after the off and before the on"
    # ...and the wait must land after the last parameter write, before the enable.
    assert fake.log.index(":SOUR1:VOLTage:OFFSet 0.0") < fake.log.index(":OUTP1:STATe ON")


def test_disabling_a_channel_skips_the_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Switching off cannot strand a waveform, so it must not pay the delay.
    slept = _record_sleeps(monkeypatch)
    fake = FakeTransport(queries=_NO_ERROR)
    DG1062(fake, output_settle_s=0.25).configure_channel(_sine(enabled=False))
    assert slept == []
    assert ":OUTP1:STATe OFF" in fake.log


def test_configure_never_writes_phase_so_it_survives_set_phase_deg() -> None:
    """:APPLy would reset the start phase to its 0-degree default (guide Ch1, "[]").

    The discrete path names only what it sets, so a configure after set_phase_deg
    leaves the CH1/CH2 phase relationship alone. This is the regression that made the
    two methods fight.
    """
    gen, fake = _gen()
    gen.set_phase_deg(90.0)
    fake.log.clear()
    gen.configure_channel(_sine(channel=SigGenChannel.CH2))
    assert not any("PHAS" in cmd for cmd in fake.log)
    assert not any("APPL" in cmd for cmd in fake.log)


def test_swapping_from_arbitrary_back_to_a_builtin_leaves_sample_rate_mode() -> None:
    """The headline failure: an upload leaves the channel in SRATe mode, where
    ":FUNCtion:ARBitrary:MODE" (2-100) says the frequency "cannot be set". Unless the
    driver switches back to FREQ mode, the new :FREQuency is ignored and the stale
    arb-era value is what the type-change clamp acts on."""
    gen, fake = _gen()
    gen.upload_arbitrary(SigGenChannel.CH1, ArbitraryWaveform(samples=np.linspace(-1, 1, 16)))
    fake.log.clear()
    gen.configure_channel(_sine(waveform=WaveShape.TRIANGLE, frequency_hz=500.0))
    mode = fake.log.index(":SOUR1:FUNCtion:ARBitrary:MODE FREQ")
    assert mode < fake.log.index(":SOUR1:FUNCtion RAMP") < fake.log.index(":SOUR1:FREQuency 500.0")


def test_numbers_keep_full_precision_on_the_wire() -> None:
    """``%g`` would round to 6 significant digits and shift the request silently — the
    instrument stores the truncated value and echoes it back, so no read-back catches
    it. 1234567.5 Hz must not become 1.23457e+06."""
    gen, fake = _gen()
    gen.configure_channel(
        _sine(
            waveform=WaveShape.SQUARE,
            frequency_hz=1_234_567.5,
            amplitude_vpp=2.0000005,
            offset_v=-0.0312505,
            duty_pct=99.9995,
        )
    )
    assert ":SOUR1:FREQuency 1234567.5" in fake.log
    assert ":SOUR1:VOLTage 2.0000005" in fake.log
    assert ":SOUR1:VOLTage:OFFSet -0.0312505" in fake.log
    assert ":SOUR1:FUNCtion:SQUare:DCYCle 99.9995" in fake.log
    assert not any("e+0" in cmd for cmd in fake.log)


def test_dc_and_noise_skip_the_parameters_they_do_not_have() -> None:
    gen, fake = _gen()
    # DC: ":APPLy:DC" notes frequency and amplitude "are not applicable to the DC
    # function"; only the offset means anything.
    gen.configure_channel(_sine(waveform=WaveShape.DC, amplitude_vpp=0.0, offset_v=2.5))
    assert ":SOUR1:FUNCtion DC" in fake.log
    assert ":SOUR1:VOLTage:OFFSet 2.5" in fake.log
    assert not any(cmd.startswith(":SOUR1:FREQuency") for cmd in fake.log)
    assert not any(cmd.startswith(":SOUR1:VOLTage ") for cmd in fake.log)
    fake.log.clear()
    gen.configure_channel(_sine(waveform=WaveShape.NOISE, amplitude_vpp=1.0))
    assert ":SOUR1:VOLTage 1.0" in fake.log  # noise has a level...
    assert not any(cmd.startswith(":SOUR1:FREQuency") for cmd in fake.log)  # ...but no frequency


def test_finite_load_is_sent_as_an_integer() -> None:
    gen, fake = _gen()
    gen.configure_channel(_sine(channel=SigGenChannel.CH2, output_load_ohms=50.0))
    assert ":OUTP2:LOAD 50" in fake.log  # not "50.0": <ohms> is an Integer parameter


def test_non_integer_load_is_refused() -> None:
    # Ch1: an Integer parameter given a decimal means "errors will occur".
    gen, _ = _gen()
    with pytest.raises(ValueError, match="integer output load"):
        gen.configure_channel(_sine(output_load_ohms=75.5))


def test_out_of_range_load_raises() -> None:
    gen, _ = _gen()
    for bad in (0.5, 20_000.0):
        with pytest.raises(ValueError, match="output load"):
            gen.configure_channel(_sine(output_load_ohms=bad))


# -- duty routing, per shape -------------------------------------------------------


def test_duty_routes_per_shape() -> None:
    gen, fake = _gen()
    gen.configure_channel(_sine(waveform=WaveShape.SQUARE, duty_pct=30.0))
    assert ":SOUR1:FUNCtion:SQUare:DCYCle 30.0" in fake.log
    fake.log.clear()
    gen.configure_channel(_sine(waveform=WaveShape.PULSE, duty_pct=25.0))
    assert ":SOUR1:FUNCtion:PULSe:DCYCle 25.0" in fake.log
    fake.log.clear()
    # A triangle is a ramp whose symmetry carries the shape, and duty_pct is that
    # knob — not a hard-coded 50, which would silently drop the caller's request.
    gen.configure_channel(_sine(waveform=WaveShape.TRIANGLE, duty_pct=75.0))
    assert ":SOUR1:FUNCtion RAMP" in fake.log
    assert ":SOUR1:FUNCtion:RAMP:SYMMetry 75.0" in fake.log


def test_duty_on_a_shape_without_one_is_refused_not_dropped() -> None:
    gen, _ = _gen()
    with pytest.raises(ValueError, match="no duty-cycle or symmetry control"):
        gen.configure_channel(_sine(waveform=WaveShape.SINE, duty_pct=30.0))


def test_pulse_duty_outside_the_documented_span_is_refused() -> None:
    gen, _ = _gen()
    for bad in (0.0, 100.0):  # guide 2-101: 0.001 % to 99.999 %
        with pytest.raises(ValueError, match="pulse duty cycle"):
            gen.configure_channel(_sine(waveform=WaveShape.PULSE, duty_pct=bad))


# -- limits ------------------------------------------------------------------------


def test_per_shape_frequency_ceilings_come_from_table_2_1() -> None:
    gen, _ = _gen()
    for shape, bad_hz, limit in [
        (WaveShape.TRIANGLE, 2e6, "1e+06"),  # ramp: 1 Hz to 1 MHz
        (WaveShape.SQUARE, 30e6, "2.5e+07"),  # square: 1 Hz to 25 MHz
        (WaveShape.SINC, 25e6, "2e+07"),  # arbitrary: 1 Hz to 20 MHz
    ]:
        with pytest.raises(ValueError, match=f"limited to {limit.replace('+', chr(92) + '+')}"):
            gen.configure_channel(_sine(waveform=shape, frequency_hz=bad_hz))


def test_below_minimum_frequency_and_amplitude_are_refused() -> None:
    """The instrument would clamp both to its floor and say nothing (guide 2-96,
    2-180), so the driver refuses instead of letting the divergence through."""
    gen, _ = _gen()
    with pytest.raises(ValueError, match="below the 1 Hz floor"):
        gen.configure_channel(_sine(frequency_hz=0.5))
    with pytest.raises(ValueError, match=r"below the 0\.002 Vpp floor"):
        gen.configure_channel(_sine(amplitude_vpp=0.001))


def test_over_range_frequency_and_unsupported_shape_raise() -> None:
    gen, _ = _gen()
    with pytest.raises(ValueError, match="exceeds"):
        gen.configure_channel(_sine(frequency_hz=120e6))
    with pytest.raises(ValueError, match="does not support"):
        gen.configure_channel(_sine(waveform=WaveShape.CMOS))


def test_enable_output_is_per_channel() -> None:
    gen, fake = _gen()
    gen.enable_output(SigGenChannel.CH2, True)
    gen.enable_output(SigGenChannel.CH1, False)
    assert fake.log == [":OUTP2:STATe ON", ":OUTP1:STATe OFF"]


def test_set_phase_offsets_ch2_and_aligns() -> None:
    gen, fake = _gen()
    gen.set_phase_deg(90.0)
    assert fake.log == [":SOUR1:PHASe 0.0", ":SOUR2:PHASe 90.0", ":SOUR1:PHASe:SYNChronize"]
    with pytest.raises(ValueError, match="within 0"):
        gen.set_phase_deg(361.0)


# -- readback ----------------------------------------------------------------------


def test_read_channel_parses_scientific_notation() -> None:
    gen, _ = _gen(
        queries={
            ":SOUR1:FUNCtion?": "SIN",
            ":SOUR1:FREQuency?": "1.000000E+03",
            ":SOUR1:VOLTage?": "2.000000E+00",
            ":SOUR1:VOLTage:OFFSet?": "0.000000E+00",
            ":OUTP1:STATe?": "ON",
            ":OUTP1:LOAD?": "9.900000E+37",  # high-Z
        }
    )
    cfg = gen.read_channel(SigGenChannel.CH1)
    assert cfg.waveform is WaveShape.SINE
    assert cfg.frequency_hz == pytest.approx(1_000.0)
    assert cfg.amplitude_vpp == pytest.approx(2.0)
    assert cfg.enabled is True
    assert math.isinf(cfg.output_load_ohms)


def test_read_channel_reports_finite_load_and_queried_duty() -> None:
    gen, _ = _gen(
        queries={
            ":SOUR2:FUNCtion?": "SQU",
            ":SOUR2:FREQuency?": "1.000000E+03",
            ":SOUR2:VOLTage?": "1.000000E+00",
            ":SOUR2:VOLTage:OFFSet?": "0.000000E+00",
            ":SOUR2:FUNCtion:SQUare:DCYCle?": "4.500000E+01",
            ":OUTP2:STATe?": "OFF",
            ":OUTP2:LOAD?": "5.000000E+01",
        }
    )
    cfg = gen.read_channel(SigGenChannel.CH2)
    assert cfg.waveform is WaveShape.SQUARE
    assert cfg.duty_pct == pytest.approx(45.0)
    assert cfg.enabled is False
    assert cfg.output_load_ohms == pytest.approx(50.0)


def test_read_channel_queries_ramp_symmetry_for_a_triangle() -> None:
    gen, fake = _gen(
        queries={
            ":SOUR1:FUNCtion?": "RAMP",
            ":SOUR1:FREQuency?": "1.000000E+03",
            ":SOUR1:VOLTage?": "1.000000E+00",
            ":SOUR1:VOLTage:OFFSet?": "0.000000E+00",
            ":SOUR1:FUNCtion:RAMP:SYMMetry?": "2.000000E+01",
            ":OUTP1:STATe?": "ON",
            ":OUTP1:LOAD?": "9.900000E+37",
        }
    )
    cfg = gen.read_channel(SigGenChannel.CH1)
    assert cfg.waveform is WaveShape.TRIANGLE
    assert cfg.duty_pct == pytest.approx(20.0)  # the real symmetry, not an assumed 50
    assert ":SOUR1:FUNCtion:RAMP:SYMMetry?" in fake.log


def test_read_channel_on_a_volatile_arb_points_at_read_arbitrary() -> None:
    # ":APPLy?" collapses arbitrary waveforms to "USER" (2-72) and :FUNCtion? reports
    # the same for the volatile buffer: an output mode, not a WaveShape.
    gen, _ = _gen(queries={":SOUR1:FUNCtion?": "USER"})
    with pytest.raises(RuntimeError, match="read_arbitrary"):
        gen.read_channel(SigGenChannel.CH1)


def test_read_channel_on_an_unmapped_builtin_is_loud() -> None:
    gen, _ = _gen(queries={":SOUR1:FUNCtion?": "CARDIAC"})
    with pytest.raises(RuntimeError, match="does not map"):
        gen.read_channel(SigGenChannel.CH1)


@pytest.mark.parametrize("name", ["DC", "NOIS"])
def test_shapes_without_a_frequency_report_zero_and_are_not_queried(name: str) -> None:
    """DC and noise have no frequency, so read_channel reports 0.0 and never asks.

    The :FREQuency register still holds *something* for these shapes — whatever the last
    frequency-bearing waveform left — and echoing that, or substituting a plausible
    stand-in, would misreport what the instrument is doing. No scripted reply for
    :FREQuency? is provided, so a query would raise rather than pass silently.
    """
    gen, fake = _gen(
        queries={
            ":SOUR1:FUNCtion?": name,
            ":SOUR1:VOLTage?": "0.000000E+00",
            ":SOUR1:VOLTage:OFFSet?": "2.500000E+00",
            ":OUTP1:STATe?": "ON",
            ":OUTP1:LOAD?": "9.900000E+37",
        }
    )
    cfg = gen.read_channel(SigGenChannel.CH1)
    assert cfg.frequency_hz == 0.0
    assert cfg.offset_v == pytest.approx(2.5)
    assert not any("FREQuency" in cmd for cmd in fake.log)


def test_a_frequency_bearing_shape_cannot_be_configured_at_zero_hz() -> None:
    # The model's ``ge=0`` exists only so DC/noise can say "no frequency"; it must not
    # become a licence for a 0 Hz sine.
    with pytest.raises(ValidationError, match="needs a frequency above 0"):
        _sine(frequency_hz=0.0)


@pytest.mark.parametrize("field", ["frequency_hz", "amplitude_vpp", "offset_v", "duty_pct"])
def test_nan_is_refused_before_it_can_reach_the_wire(field: str) -> None:
    """A NaN passes every bounds comparison (``nan > limit`` is False), so without an
    explicit guard it would be formatted onto the wire as the literal "nan".
    ``offset_v``, which carries no range of its own, was exactly that hole."""
    with pytest.raises(ValidationError):
        _sine(**{field: float("nan")})


# -- arbitrary waveforms -----------------------------------------------------------


def test_upload_arbitrary_states_mode_and_rate_before_downloading() -> None:
    gen, fake = _gen()
    samples = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.75]
    gen.upload_arbitrary(SigGenChannel.CH1, ArbitraryWaveform(samples=samples))
    assert fake.log[:3] == [
        # SRATe *before* the download, or a <=8k buffer is interpolated to 8192 points
        # (guide 2-174) and the read-back would not match.
        ":SOUR1:FUNCtion:ARBitrary:MODE SRATe",
        # The clock is stated, not inherited from whatever the front panel left.
        ":SOUR1:FUNCtion:ARBitrary:SRATe 20000000.0",
        ":SOUR1:DATA:DAC VOLATILE,",
    ]
    cmd, payload = fake.written_blocks[0]
    assert cmd == ":SOUR1:DATA:DAC VOLATILE,"
    # 14-bit codes (0..16383) as little-endian uint16.
    assert payload == np.rint((np.array(samples) + 1.0) / 2.0 * 16383).astype("<u2").tobytes()
    assert fake.log[-1] == ":SYSTem:ERRor?"


def test_upload_arbitrary_honours_an_explicit_sample_rate() -> None:
    gen, fake = _gen()
    wave = ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 1024))
    gen.upload_arbitrary(SigGenChannel.CH2, wave, sample_rate_hz=1e6)
    assert ":SOUR2:FUNCtion:ARBitrary:SRATe 1000000.0" in fake.log
    # The buffer is one period, so it repeats at rate / points.
    assert arb_repetition_hz(1e6, wave.n) == pytest.approx(976.5625)


def test_upload_arbitrary_rejects_an_out_of_range_rate() -> None:
    gen, _ = _gen()
    wave = ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 16))
    with pytest.raises(ValueError, match="Sa/s"):
        gen.upload_arbitrary(SigGenChannel.CH1, wave, sample_rate_hz=100e6)


def test_read_arbitrary_decodes_uint16_codes() -> None:
    codes = np.array([0, 8192, 16383], dtype="<u2")
    gen, _ = _gen(
        queries={":SOUR1:DATA:LOAD? VOLATILE": "1"},
        blocks={":SOUR1:DATA:LOAD? 1": codes.tobytes()},
    )
    wave = gen.read_arbitrary(SigGenChannel.CH1)
    assert wave.n == 3
    assert wave.samples[0] == pytest.approx(-1.0)
    assert wave.samples[2] == pytest.approx(1.0)


def test_read_arbitrary_accepts_a_scientific_notation_package_count() -> None:
    # Nearly every numeric query on this instrument answers in sci-notation; the count
    # must not be the one place that explodes.
    codes = np.array([0, 16383], dtype="<u2")
    gen, _ = _gen(
        queries={":SOUR1:DATA:LOAD? VOLATILE": "1.000000E+00"},
        blocks={":SOUR1:DATA:LOAD? 1": codes.tobytes()},
    )
    assert gen.read_arbitrary(SigGenChannel.CH1).n == 2


def test_read_arbitrary_concatenates_multiple_packages() -> None:
    first = np.array([0, 4096], dtype="<u2").tobytes()
    second = np.array([8192, 16383], dtype="<u2").tobytes()
    gen, _ = _gen(
        queries={":SOUR1:DATA:LOAD? VOLATILE": "2"},
        blocks={":SOUR1:DATA:LOAD? 1": first, ":SOUR1:DATA:LOAD? 2": second},
    )
    assert gen.read_arbitrary(SigGenChannel.CH1).n == 4


def test_arbitrary_upload_read_round_trips_through_fake() -> None:
    gen, fake = _gen()
    samples = np.linspace(-1.0, 1.0, 16).tolist()
    gen.upload_arbitrary(SigGenChannel.CH1, ArbitraryWaveform(samples=samples))
    _, payload = fake.written_blocks[0]
    gen2, _ = _gen(
        queries={":SOUR1:DATA:LOAD? VOLATILE": "1"},
        blocks={":SOUR1:DATA:LOAD? 1": payload},  # feed the written block back as the read
    )
    read = gen2.read_arbitrary(SigGenChannel.CH1)
    assert read.samples == pytest.approx(samples, abs=2 / 16383)  # within one 14-bit LSB


def test_arbitrary_rejects_a_slot() -> None:
    gen, _ = _gen()
    wave = ArbitraryWaveform(samples=[-1.0, 0.0, 1.0, 0.5, -0.5, 0.25, -0.25, 0.75])
    with pytest.raises(ValueError, match="slot=None"):
        gen.upload_arbitrary(SigGenChannel.CH1, wave, slot=1)
    with pytest.raises(ValueError, match="no numbered arbitrary slots"):
        gen.read_arbitrary(SigGenChannel.CH1, slot=1)


def test_read_arbitrary_rejects_odd_byte_count() -> None:
    gen, _ = _gen(
        queries={":SOUR1:DATA:LOAD? VOLATILE": "1"},
        blocks={":SOUR1:DATA:LOAD? 1": b"\x00\x01\x02"},  # 3 bytes
    )
    with pytest.raises(OSError, match="odd"):
        gen.read_arbitrary(SigGenChannel.CH1)


def test_configure_raises_and_drains_all_queued_errors() -> None:
    # A bad batch can queue several SCPI errors; configure surfaces them all, reading
    # until the terminating 0,"No error" rather than reporting only the first.
    fake = _ErrorQueueTransport(
        ['-222,"Data out of range"', '-113,"Undefined header"', '0,"No error"']
    )
    gen = DG1062(fake, output_settle_s=0.0)
    with pytest.raises(RuntimeError, match=r"-222.*-113"):
        gen.configure_channel(_sine())
