"""Behavioural contract every SignalGenerator implementation must satisfy.

Parametrised over the simulated generator (always runs) and the real JDS6600 (only
under ``-m hardware``), so the same expectations hold the sim and the driver to the
same behaviour. Assertions are structural, and the values chosen sit exactly on the
JDS6600's quantisation grid (centi-Hz / mV) so round-trips are exact on real metal.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hwtools.drivers.joyit.jds6600 import JDS6600
from hwtools.drivers.simulated import SimulatedSignalGenerator
from hwtools.drivers.simulated.signal_generator import DEFAULT_CAPABILITIES
from hwtools.interfaces.signal_generator import SignalGenerator
from hwtools.model.siggen import (
    ArbitraryWaveform,
    ArbLength,
    PlaybackMode,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
)

# A simulated *variable-length, deep-memory* generator — DG1000Z-shaped: 8..16384
# points per upload, 14-bit codes, burst-capable. Running the whole contract against
# this alongside the fixed-length JDS6600 profile proves the abstraction spans both
# device families in CI, before any Rigol hardware is on the bench.
_VARIABLE_ARB_PROFILE = DEFAULT_CAPABILITIES.model_copy(
    update={
        "model_name": "SimulatedVariableAWG",
        "max_frequency_hz": 60e6,
        "playback_modes": frozenset(
            {PlaybackMode.CONTINUOUS, PlaybackMode.BURST, PlaybackMode.SWEEP}
        ),
        "arb_length": ArbLength(min_points=8, max_points=16384),
        "arb_code_levels": 16384,
    }
)


@pytest.fixture(
    params=[
        pytest.param("sim", id="simulated"),
        pytest.param("sim_awg", id="simulated-variable-awg"),
        pytest.param("real", id="jds6600", marks=pytest.mark.hardware),
    ]
)
def generator(request: pytest.FixtureRequest) -> Iterator[SignalGenerator]:
    instrument: SignalGenerator
    if request.param == "sim":
        instrument = SimulatedSignalGenerator()
    elif request.param == "sim_awg":
        instrument = SimulatedSignalGenerator(capabilities=_VARIABLE_ARB_PROFILE)
    else:
        port = os.environ.get("HWTOOLS_JDS6600_PORT", "COM7")
        instrument = JDS6600(port)
    instrument.connect()
    try:
        yield instrument
    finally:
        instrument.disconnect()


def test_capabilities_are_sane(generator: SignalGenerator) -> None:
    caps = generator.capabilities
    assert caps.n_channels >= 1
    assert caps.max_frequency_hz > 0
    assert caps.max_amplitude_vpp > 0
    assert SigGenChannel.CH1 in caps.channels


def test_idn_is_nonempty(generator: SignalGenerator) -> None:
    assert generator.idn().strip()


def test_configure_then_read_round_trips(generator: SignalGenerator) -> None:
    config = SignalGeneratorConfig(
        channel=SigGenChannel.CH1,
        waveform=WaveShape.SINE,
        frequency_hz=1_000.0,
        amplitude_vpp=2.0,
        offset_v=0.0,
        duty_pct=50.0,
        enabled=True,
    )
    generator.configure_channel(config)
    read = generator.read_channel(SigGenChannel.CH1)

    assert read.waveform is WaveShape.SINE
    assert read.frequency_hz == pytest.approx(1_000.0, rel=1e-3)
    assert read.amplitude_vpp == pytest.approx(2.0, rel=1e-3)
    assert read.offset_v == pytest.approx(0.0, abs=1e-3)
    assert read.duty_pct == pytest.approx(50.0, rel=1e-3)


def test_enable_output_toggles_readback(generator: SignalGenerator) -> None:
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH1, frequency_hz=1_000.0, amplitude_vpp=1.0
        )
    )
    generator.enable_output(SigGenChannel.CH1, True)
    assert generator.output_enabled(SigGenChannel.CH1) is True
    generator.enable_output(SigGenChannel.CH1, False)
    assert generator.output_enabled(SigGenChannel.CH1) is False


def test_over_range_frequency_raises(generator: SignalGenerator) -> None:
    over = generator.capabilities.max_frequency_hz * 2.0
    with pytest.raises(ValueError):
        generator.configure_channel(
            SignalGeneratorConfig(
                channel=SigGenChannel.CH1, frequency_hz=over, amplitude_vpp=1.0
            )
        )


def test_phase_in_range_is_accepted(generator: SignalGenerator) -> None:
    generator.set_phase_deg(0.0)
    generator.set_phase_deg(90.0)
    with pytest.raises(ValueError):
        generator.set_phase_deg(361.0)


def test_arbitrary_upload_read_round_trips(generator: SignalGenerator) -> None:
    caps = generator.capabilities
    if not caps.supports_arbitrary():
        pytest.skip("no arbitrary-waveform support")
    assert caps.arb_length is not None  # narrowed by supports_arbitrary()
    n = caps.arb_length.representative_length()
    ramp = tuple(-1.0 + 2.0 * i / (n - 1) for i in range(n))  # one-period sawtooth
    slot = caps.arb_slots  # highest slot, to avoid clobbering low front-panel presets
    generator.upload_arbitrary(slot, ArbitraryWaveform(samples=ramp))
    read = generator.read_arbitrary(slot)
    assert read.n == n
    # Round-trip is exact on the sim, ±1 LSB on real 12-bit hardware.
    tol = 2.0 / (caps.arb_code_levels - 1) if caps.arb_code_levels else 1e-9
    assert read.samples == pytest.approx(ramp, abs=tol)
