"""JDS6600 driver: command framing and reply parsing, no hardware.

The wire encodings asserted here (centi-Hz frequency, mV amplitude, biased offset,
tenths-of-a-percent duty, the ``:ok`` write acknowledgement) were all confirmed on
the bench — these tests pin them so a manual-only "fix" can't silently regress them.
"""

from __future__ import annotations

import pytest

from hwtools.drivers.joyit.jds6600 import JDS6600
from hwtools.model.siggen import (
    ArbitraryWaveform,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
)

_ARB_POINTS = 2048


class FakeSerial:
    """Records written commands and serves canned reply lines (JDS6600 framing)."""

    def __init__(self, replies: list[str]) -> None:
        self.written: list[bytes] = []
        self._replies = [r.encode() + b"\r\n" for r in replies]

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def readline(self) -> bytes:
        return self._replies.pop(0) if self._replies else b""

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def commands(self) -> list[str]:
        return [w.decode().strip() for w in self.written]


def _ok(n: int) -> list[str]:
    return [":ok"] * n


def test_idn_reads_model_and_serial() -> None:
    fake = FakeSerial([":r00=15.", ":r01=2166125956."])
    gen = JDS6600.from_serial(fake)
    with gen:
        assert gen.idn() == "JDS6600-15 SN:2166125956"
    assert fake.commands == [":r00=0.", ":r01=0."]


def test_configure_channel_formats_all_commands() -> None:
    # 6 writes (waveform, freq, amplitude, offset, duty) + enable's read-modify-write.
    fake = FakeSerial([*_ok(5), ":r20=0,0.", ":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH1,
            waveform=WaveShape.SINE,
            frequency_hz=1_000.0,
            amplitude_vpp=2.0,
            offset_v=0.0,
            duty_pct=50.0,
            enabled=True,
        )
    )
    assert fake.commands == [
        ":w21=0.",  # SINE
        ":w23=100000,0.",  # 1 kHz -> 100000 centi-Hz, unit code 0
        ":w25=2000.",  # 2.0 Vpp -> 2000 mV
        ":w27=1000.",  # 0 V bias -> code 1000
        ":w29=500.",  # 50.0 % -> 500
        ":r20=0.",  # read output pair before enabling
        ":w20=1,0.",  # enable CH1, leave CH2 as-was
    ]


def test_configure_channel_selects_arb_slot() -> None:
    # arb_slot picks waveform code 100 + slot instead of a built-in.
    fake = FakeSerial([*_ok(5), ":r20=0,0.", ":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH1, frequency_hz=1_000.0, amplitude_vpp=4.0, arb_slot=7
        )
    )
    assert fake.commands[0] == ":w21=107."  # arb slot 7


def test_read_channel_reports_arb_slot() -> None:
    fake = FakeSerial(
        [":r21=107.", ":r23=100000,0.", ":r25=4000.", ":r27=1000.", ":r29=500.", ":r20=1,0."]
    )
    gen = JDS6600.from_serial(fake)
    gen.connect()
    cfg = gen.read_channel(SigGenChannel.CH1)
    assert cfg.arb_slot == 7  # code 107 -> slot 7


def test_configure_channel_2_uses_even_function_codes() -> None:
    fake = FakeSerial([*_ok(5), ":r20=1,0.", ":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH2,
            waveform=WaveShape.SQUARE,
            frequency_hz=250_000.0,
            amplitude_vpp=5.0,
        )
    )
    assert fake.commands == [
        ":w22=1.",  # SQUARE on CH2
        ":w24=25000000,0.",  # 250 kHz
        ":w26=5000.",  # 5.0 Vpp
        ":w28=1000.",
        ":w30=500.",
        ":r20=0.",
        ":w20=1,1.",  # enable CH2, preserve CH1 (was on)
    ]


def test_negative_offset_encodes_below_bias_zero() -> None:
    fake = FakeSerial([*_ok(5), ":r20=0,0.", ":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH1,
            frequency_hz=1_000.0,
            amplitude_vpp=1.0,
            offset_v=-2.5,
        )
    )
    assert ":w27=750." in fake.commands  # -2.5 V -> 1000 + (-250)


def test_write_rejection_raises() -> None:
    fake = FakeSerial([":err"])  # anything but ":ok" is a rejection
    gen = JDS6600.from_serial(fake)
    gen.connect()
    with pytest.raises(RuntimeError, match="rejected"):
        gen.set_phase_deg(90.0)


def test_enable_output_read_modify_writes_pair() -> None:
    fake = FakeSerial([":r20=1,1.", ":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.enable_output(SigGenChannel.CH2, False)
    assert fake.commands == [":r20=0.", ":w20=1,0."]


def test_read_channel_decodes_all_fields() -> None:
    fake = FakeSerial(
        [
            ":r21=0.",  # waveform SINE
            ":r23=100000000,0.",  # 1 MHz
            ":r25=5000.",  # 5.0 Vpp
            ":r27=1000.",  # 0 V
            ":r29=500.",  # 50 %
            ":r20=1,0.",  # output pair (for enabled)
        ]
    )
    gen = JDS6600.from_serial(fake)
    gen.connect()
    cfg = gen.read_channel(SigGenChannel.CH1)
    assert cfg.waveform is WaveShape.SINE
    assert cfg.frequency_hz == 1_000_000.0
    assert cfg.amplitude_vpp == 5.0
    assert cfg.offset_v == 0.0
    assert cfg.duty_pct == 50.0
    assert cfg.enabled is True


def test_frequency_over_limit_raises_before_touching_wire() -> None:
    fake = FakeSerial([])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    with pytest.raises(ValueError, match="exceeds"):
        gen.configure_channel(
            SignalGeneratorConfig(
                channel=SigGenChannel.CH1, frequency_hz=20e6, amplitude_vpp=1.0
            )
        )
    assert fake.commands == []  # nothing sent — validated first


def test_amplitude_over_limit_raises() -> None:
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError, match="amplitude"):
        gen.configure_channel(
            SignalGeneratorConfig(
                channel=SigGenChannel.CH1, frequency_hz=1_000.0, amplitude_vpp=25.0
            )
        )


def test_offset_over_limit_raises() -> None:
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError, match="offset"):
        gen.configure_channel(
            SignalGeneratorConfig(
                channel=SigGenChannel.CH1,
                frequency_hz=1_000.0,
                amplitude_vpp=1.0,
                offset_v=12.0,
            )
        )


def test_set_phase_encodes_tenths_of_degree() -> None:
    fake = FakeSerial([":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    gen.set_phase_deg(90.0)
    assert fake.commands == [":w31=900."]


def test_set_phase_out_of_range_raises() -> None:
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError):
        gen.set_phase_deg(400.0)


# -- arbitrary-waveform codec (the `a`/`b` path) ------------------------------


def _pad(head: list[float]) -> tuple[float, ...]:
    """A full 2048-point buffer: ``head`` then zeros."""
    return tuple(head + [0.0] * (_ARB_POINTS - len(head)))


def test_upload_arbitrary_encodes_12bit_csv() -> None:
    fake = FakeSerial([":ok"])
    gen = JDS6600.from_serial(fake)
    gen.connect()
    wave = ArbitraryWaveform(samples=_pad([-1.0, 0.0, 1.0]))
    gen.upload_arbitrary(SigGenChannel.CH1, wave, slot=5)
    cmd = fake.commands[0]
    assert cmd.startswith(":a05=0,2048,4095,")  # -1 -> 0, 0 -> mid, +1 -> full scale
    assert cmd.endswith(".")
    values = cmd.split("=", 1)[1].rstrip(".").split(",")
    assert len(values) == _ARB_POINTS


def test_read_arbitrary_decodes_to_unit_range() -> None:
    codes = [0, 2048, 4095] + [2048] * (_ARB_POINTS - 3)
    reply = ":b07=" + ",".join(str(c) for c in codes) + "."
    gen = JDS6600.from_serial(FakeSerial([reply]))
    gen.connect()
    wave = gen.read_arbitrary(SigGenChannel.CH1, slot=7)
    assert wave.n == _ARB_POINTS
    assert wave.samples[0] == pytest.approx(-1.0, abs=1e-3)
    assert wave.samples[1] == pytest.approx(0.0, abs=1e-3)
    assert wave.samples[2] == pytest.approx(1.0, abs=1e-3)


def test_upload_read_round_trips_through_the_codec() -> None:
    ramp = _pad([-1.0 + 2.0 * i / 15 for i in range(16)])
    upload = FakeSerial([":ok"])
    gen = JDS6600.from_serial(upload)
    gen.connect()
    gen.upload_arbitrary(SigGenChannel.CH1, ArbitraryWaveform(samples=ramp), slot=3)
    # Feed the exact codes it wrote back as a `b` reply and decode them.
    written = upload.commands[0].split("=", 1)[1].rstrip(".")
    gen2 = JDS6600.from_serial(FakeSerial([f":b03={written}."]))
    gen2.connect()
    read = gen2.read_arbitrary(SigGenChannel.CH1, slot=3)
    assert read.samples == pytest.approx(ramp, abs=2 / 4095)  # within one 12-bit LSB


def test_upload_wrong_point_count_raises() -> None:
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError, match="2048"):
        gen.upload_arbitrary(SigGenChannel.CH1, ArbitraryWaveform(samples=(0.0, 0.5, -0.5)), slot=1)


def test_arbitrary_slot_out_of_range_raises() -> None:
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError, match="slot"):
        gen.read_arbitrary(SigGenChannel.CH1, slot=61)
    with pytest.raises(ValueError, match="slot"):
        gen.read_arbitrary(SigGenChannel.CH1, slot=0)


def test_arbitrary_requires_a_slot() -> None:
    # SLOT-addressed: omitting the slot is a loud error, not an implicit default.
    gen = JDS6600.from_serial(FakeSerial([]))
    gen.connect()
    with pytest.raises(ValueError, match="global slot"):
        gen.read_arbitrary(SigGenChannel.CH1)


def test_arbitrary_waveform_rejects_out_of_unit_range() -> None:
    with pytest.raises(ValueError, match="normalised"):
        ArbitraryWaveform(samples=(0.0, 1.5))
