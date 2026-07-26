"""HIL: every JDS6600 driver pathway exercised directly on the generator.

The unit suite pins the wire framing against a fake serial line; this module runs
each *driver* branch against the real device: identity, every per-channel encode
(waveform, centi-Hz frequency, mV amplitude, biased offset, duty), the shared
output-enable read-modify-write, phase, and the on-metal write-then-read-back that
proves the device stores exactly what the driver computed.

These are *self-consistency* checks — the generator's only observable is what you
wrote. The generator's actual output is validated one instrument over in
``tests/hardware/cross/test_siggen_scope_first_light.py``.

    uv run pytest -m hardware -s

Restores CH1 to a 1 kHz 2 Vpp sine on exit so the bench is left in a known state.
"""

from __future__ import annotations

import pytest

from hwtools.drivers.joyit import JDS6600
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.synth import analytic_arbitrary

CH1, CH2 = SigGenChannel.CH1, SigGenChannel.CH2


@pytest.mark.hardware
def test_idn_reports_model_and_serial(generator: JDS6600) -> None:
    idn = generator.idn()
    assert idn.startswith("JDS6600-")
    assert "SN:" in idn


@pytest.mark.hardware
def test_capabilities_match_the_bench_device(generator: JDS6600) -> None:
    caps = generator.capabilities
    assert caps.n_channels == 2
    assert caps.max_frequency_hz == pytest.approx(15e6)
    assert caps.max_amplitude_vpp == pytest.approx(20.0)


@pytest.mark.hardware
@pytest.mark.parametrize(
    "waveform", [WaveShape.SINE, WaveShape.SQUARE, WaveShape.TRIANGLE, WaveShape.NOISE]
)
def test_waveform_round_trips(generator: JDS6600, waveform: WaveShape) -> None:
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=CH1, waveform=waveform, frequency_hz=1_000.0, amplitude_vpp=1.0
        )
    )
    assert generator.read_channel(CH1).waveform is waveform


@pytest.mark.hardware
@pytest.mark.parametrize("frequency_hz", [10.0, 1_000.0, 100_000.0, 5_000_000.0])
def test_frequency_saved_verbatim(generator: JDS6600, frequency_hz: float) -> None:
    """Centi-Hz encode/decode round-trips exactly on grid across the device's range."""
    generator.configure_channel(
        SignalGeneratorConfig(channel=CH1, frequency_hz=frequency_hz, amplitude_vpp=1.0)
    )
    assert generator.read_channel(CH1).frequency_hz == pytest.approx(frequency_hz, rel=1e-6)


@pytest.mark.hardware
def test_amplitude_offset_duty_saved_verbatim(generator: JDS6600) -> None:
    config = SignalGeneratorConfig(
        channel=CH1,
        waveform=WaveShape.SQUARE,
        frequency_hz=1_000.0,
        amplitude_vpp=3.3,
        offset_v=-1.5,
        duty_pct=25.0,
    )
    generator.configure_channel(config)
    read = generator.read_channel(CH1)
    assert read.amplitude_vpp == pytest.approx(3.3, abs=1e-3)
    assert read.offset_v == pytest.approx(-1.5, abs=1e-2)
    assert read.duty_pct == pytest.approx(25.0, abs=0.1)


@pytest.mark.hardware
def test_output_enable_is_per_channel(generator: JDS6600) -> None:
    """Enabling one channel must not disturb the other (shared fn-20 register)."""
    for ch in (CH1, CH2):
        generator.configure_channel(
            SignalGeneratorConfig(
                channel=ch, frequency_hz=1_000.0, amplitude_vpp=1.0, enabled=False
            )
        )
    generator.enable_output(CH1, True)
    generator.enable_output(CH2, False)
    assert generator.output_enabled(CH1) is True
    assert generator.output_enabled(CH2) is False
    generator.enable_output(CH2, True)
    assert generator.output_enabled(CH1) is True  # CH1 undisturbed
    assert generator.output_enabled(CH2) is True


@pytest.mark.hardware
def test_over_range_frequency_raises_on_metal(generator: JDS6600) -> None:
    """The driver rejects an out-of-range request rather than letting the device
    silently clamp it (the device clamps 99 MHz -> 15 MHz; the driver must not)."""
    with pytest.raises(ValueError, match="exceeds"):
        generator.configure_channel(
            SignalGeneratorConfig(channel=CH1, frequency_hz=99e6, amplitude_vpp=1.0)
        )


@pytest.mark.hardware
def test_phase_round_trips_via_ch2(generator: JDS6600) -> None:
    """set_phase_deg reaches the device cleanly (fn 31); the link stays healthy."""
    generator.set_phase_deg(90.0)
    assert generator.idn().startswith("JDS6600-")  # link still in sync after the write


@pytest.mark.hardware
def test_arbitrary_upload_read_back_on_metal(generator: JDS6600) -> None:
    """The `a`/`b` codec round-trips a real 2048-point buffer through the device to
    within one 12-bit LSB — the driver-layer proof of the upload path."""
    caps = generator.capabilities
    assert caps.arb_length is not None
    slot = caps.arb_slots  # highest slot, to avoid clobbering low front-panel presets
    n = caps.arb_length.representative_length()
    wave = analytic_arbitrary(WaveShape.TRIANGLE, points=n)
    generator.upload_arbitrary(slot, wave)
    read = generator.read_arbitrary(slot)
    assert read.n == n
    assert read.samples == pytest.approx(wave.samples, abs=2 / (caps.arb_code_levels - 1))


@pytest.mark.hardware
def test_arb_slot_selection_reads_back(generator: JDS6600) -> None:
    """Selecting an arb slot round-trips through the waveform code (100 + slot)."""
    caps = generator.capabilities
    assert caps.arb_length is not None
    slot = caps.arb_slots
    n = caps.arb_length.representative_length()
    generator.upload_arbitrary(slot, analytic_arbitrary(WaveShape.SINE, points=n))
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=CH1, frequency_hz=1_000.0, amplitude_vpp=2.0, arb_slot=slot
        )
    )
    assert generator.read_channel(CH1).arb_slot == slot
