"""DG1062Z driver pathway & limit tests on real hardware (signal generator only).

Complements the shared behavioural contract by pushing the instrument to its
documented edges and *deliberately* provoking rejections from both sides of the
driver:

* **client-side** — ``check_within`` / driver validation refuses an out-of-range
  request before any I/O (a fast ``ValueError``);
* **instrument-side** — a command the driver forwards but the DG1062Z rejects lands
  in its SCPI error queue and ``_check_error`` turns it into a loud ``RuntimeError``.

It also covers the two state-transition traps the programming guide documents and a
naive ``:APPLy`` write path falls into — leaving sample-rate arbitrary mode, and
preserving the channel phase across a reconfigure. Those are asserted here through
read-back; the *analog* proof is one instrument over, in ``cross/test_dg1062_scope``.

Needs only the DG1062Z on the LAN (no scope, no particular output wiring), so these
carry no ``cfg_*`` rig marker — just ``hardware``.

    uv run pytest -m hardware tests/hardware/drivers/rigol/
"""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.drivers.rigol import DG1062
from hwtools.model.siggen import (
    ArbitraryWaveform,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
)

CH1 = SigGenChannel.CH1


def _sine(**kw: object) -> SignalGeneratorConfig:
    base: dict[str, object] = {"channel": CH1, "frequency_hz": 1_000.0, "amplitude_vpp": 1.0}
    base.update(kw)
    return SignalGeneratorConfig(**base)


# -- limit acceptance: push to the valid edge, confirm it takes and reads back ----


@pytest.mark.hardware
def test_max_sine_frequency_is_accepted(dg1062: DG1062) -> None:
    dg1062.configure_channel(_sine(waveform=WaveShape.SINE, frequency_hz=60e6))
    assert dg1062.read_channel(CH1).frequency_hz == pytest.approx(60e6, rel=1e-4)


@pytest.mark.hardware
def test_square_at_its_frequency_ceiling_is_accepted(dg1062: DG1062) -> None:
    dg1062.configure_channel(_sine(waveform=WaveShape.SQUARE, frequency_hz=25e6))
    read = dg1062.read_channel(CH1)
    assert read.waveform is WaveShape.SQUARE
    assert read.frequency_hz == pytest.approx(25e6, rel=1e-4)


@pytest.mark.hardware
def test_documented_floors_are_accepted(dg1062: DG1062) -> None:
    """1 Hz (Table 2-1) and 2 mVpp (:VOLTage, 2-180) — the exact documented minima."""
    dg1062.configure_channel(_sine(frequency_hz=1.0, amplitude_vpp=0.002))
    read = dg1062.read_channel(CH1)
    assert read.frequency_hz == pytest.approx(1.0, rel=1e-3)
    assert read.amplitude_vpp == pytest.approx(0.002, abs=5e-4)


@pytest.mark.hardware
def test_fine_frequency_resolution_survives_the_wire(dg1062: DG1062) -> None:
    """A frequency needing more than 6 significant digits must arrive intact.

    ``%g`` formatting would send 1.23457e+06 for this and the instrument would store
    1234570 Hz — then echo that back, so only a tolerance tighter than the truncation
    catches it.
    """
    dg1062.configure_channel(_sine(frequency_hz=1_234_567.5))
    assert dg1062.read_channel(CH1).frequency_hz == pytest.approx(1_234_567.5, abs=0.5)


# A deep buffer that round-trips reliably through the *readable* VOLATILE download.
# NB: the guide/spec rate the arb at 16384 points, but bench-testing shows the SCPI
# DATA:DAC readback buffer populates only up to ~16000 (a 16383/16384 download silently
# no-ops) — 16384 is the front-panel/output ceiling, not the readable-download one. So
# the readback round-trip is validated at a safely-below-the-cap depth.
_DEEP_ARB_POINTS = 8192


@pytest.mark.hardware
def test_deep_arbitrary_round_trips(dg1062: DG1062) -> None:
    """A deep buffer uploads via binary DATA:DAC (sample-rate mode) and reads back exactly."""
    caps = dg1062.capabilities
    assert caps.arb_length is not None
    ramp = np.linspace(-1.0, 1.0, _DEEP_ARB_POINTS)
    dg1062.upload_arbitrary(CH1, ArbitraryWaveform(samples=ramp))
    read = dg1062.read_arbitrary(CH1)
    assert read.n == _DEEP_ARB_POINTS
    assert read.samples == pytest.approx(ramp, abs=4 / (caps.arb_code_levels - 1))  # ~2 LSB


# -- state transitions the guide documents as hazards ------------------------------


@pytest.mark.hardware
@pytest.mark.parametrize("shape", [WaveShape.SINE, WaveShape.TRIANGLE, WaveShape.SQUARE])
def test_swapping_from_arbitrary_back_to_a_builtin_takes_effect(
    dg1062: DG1062, shape: WaveShape
) -> None:
    """The reported failure, on metal: configure a built-in waveform straight after an
    arbitrary upload and the new frequency must actually be in force.

    While a channel sits in sample-rate arbitrary mode ":FUNCtion:ARBitrary:MODE"
    (2-100) says the frequency "cannot be set", so a driver that does not leave that
    mode has its :FREQuency ignored — and the stale arb-era value is then what
    :FREQuency[:FIXed]'s type-change clamp acts on. TRIANGLE is the sharp case: its
    1 MHz ceiling is well under the arb's 20 MHz, so a carried-over frequency gets
    pinned to a limit instead of taking the requested value.
    """
    dg1062.upload_arbitrary(CH1, ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 4096)))
    dg1062.configure_channel(_sine(waveform=shape, frequency_hz=5_000.0, amplitude_vpp=2.0))
    read = dg1062.read_channel(CH1)
    assert read.waveform is shape
    assert read.frequency_hz == pytest.approx(5_000.0, rel=1e-3)
    assert read.amplitude_vpp == pytest.approx(2.0, rel=1e-2)


@pytest.mark.hardware
def test_arbitrary_upload_after_a_builtin_keeps_the_configured_level(dg1062: DG1062) -> None:
    """An upload must not reset the amplitude the preceding configure established.

    ":APPLy:ARBitrary" would: its omitted <amplitude>/<offset> revert to the 5 Vpp /
    0 VDC defaults (Ch1, Square Brackets). The driver uses discrete
    :FUNCtion:ARBitrary:MODE / :SRATe writes instead, which name nothing else.
    """
    dg1062.configure_channel(_sine(amplitude_vpp=3.0, offset_v=0.0))
    dg1062.upload_arbitrary(CH1, ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 1024)))
    # read_channel refuses a volatile arb (it is an output mode, not a WaveShape), so
    # the level is checked at the SCPI level.
    assert float(dg1062._t.query(":SOUR1:VOLTage?")) == pytest.approx(3.0, rel=1e-2)


@pytest.mark.hardware
def test_channel_phase_survives_a_reconfigure(dg1062: DG1062) -> None:
    """set_phase_deg then configure_channel must not silently zero the phase.

    Every ":APPLy:*" verb ends in an optional <phase> that defaults to 0 deg, and an
    omitted parameter "is set to its default" (Ch1) — so an :APPLy-based configure
    destroys the phase relationship. The discrete write path never names phase.
    """
    dg1062.set_phase_deg(90.0)
    dg1062.configure_channel(
        SignalGeneratorConfig(
            channel=SigGenChannel.CH2, waveform=WaveShape.SINE,
            frequency_hz=1_000.0, amplitude_vpp=1.0,
        )
    )
    assert float(dg1062._t.query(":SOUR2:PHASe?")) == pytest.approx(90.0, abs=0.1)


@pytest.mark.hardware
def test_explicit_sample_rate_sets_the_playback_clock(dg1062: DG1062) -> None:
    dg1062.upload_arbitrary(
        CH1, ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 2048)), sample_rate_hz=2e6
    )
    assert float(dg1062._t.query(":SOUR1:FUNCtion:ARBitrary:SRATe?")) == pytest.approx(
        2e6, rel=1e-4
    )


# -- client-side rejections: the driver refuses before touching the instrument ----


@pytest.mark.hardware
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"frequency_hz": 120e6}, "exceeds"),  # over the 60 MHz model ceiling
        ({"waveform": WaveShape.TRIANGLE, "frequency_hz": 2e6}, "limited to"),  # ramp > 1 MHz
        ({"waveform": WaveShape.SQUARE, "frequency_hz": 30e6}, "limited to"),  # square > 25 MHz
        ({"waveform": WaveShape.SINC, "frequency_hz": 25e6}, "limited to"),  # arb > 20 MHz
        ({"frequency_hz": 0.5}, "below the"),  # under the 1 Hz floor (Table 2-1)
        ({"amplitude_vpp": 0.001}, "below the"),  # under the 2 mVpp floor
        ({"amplitude_vpp": 25.0}, "amplitude"),  # over 20 Vpp
        ({"offset_v": 12.0}, "offset"),  # over +/-10 V
        ({"waveform": WaveShape.CMOS}, "does not support"),  # no DG1062 mapping
        ({"output_load_ohms": 0.5}, "output load"),  # below 1 ohm
        ({"output_load_ohms": 75.5}, "integer output load"),  # <ohms> is an Integer
        ({"duty_pct": 30.0}, "no duty-cycle"),  # a sine has no duty control
        ({"waveform": WaveShape.PULSE, "duty_pct": 100.0}, "pulse duty"),  # > 99.999 %
    ],
)
def test_out_of_range_requests_are_rejected(
    dg1062: DG1062, kw: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        dg1062.configure_channel(_sine(**kw))


@pytest.mark.hardware
def test_over_length_arbitrary_is_rejected(dg1062: DG1062) -> None:
    caps = dg1062.capabilities
    assert caps.arb_length is not None
    too_long = np.linspace(-1.0, 1.0, caps.arb_length.max_points + 2)
    with pytest.raises(ValueError, match="arbitrary"):
        dg1062.upload_arbitrary(CH1, ArbitraryWaveform(samples=too_long))


@pytest.mark.hardware
def test_out_of_range_sample_rate_is_rejected(dg1062: DG1062) -> None:
    wave = ArbitraryWaveform(samples=np.linspace(-1.0, 1.0, 1024))
    with pytest.raises(ValueError, match="Sa/s"):  # ceiling is 60 MSa/s (2-101)
        dg1062.upload_arbitrary(CH1, wave, sample_rate_hz=100e6)


# -- instrument-side errors: the SCPI error queue, caught by _check_error ----------


@pytest.mark.hardware
def test_error_check_catches_a_real_scpi_error(dg1062: DG1062) -> None:
    """A malformed parameter the driver forwards is queued by the instrument and
    surfaced loudly — the _check_error path proven against real metal."""
    dg1062._t.write(":SOUR1:FUNCtion NOSUCHSHAPE")  # invalid discrete parameter
    with pytest.raises(RuntimeError, match="error"):
        dg1062._check_error()


@pytest.mark.hardware
def test_amplitude_beyond_the_high_frequency_limit_is_surfaced(dg1062: DG1062) -> None:
    """20 Vpp at 60 MHz is past the instrument's high-frequency amplitude derating.

    The driver's per-parameter checks pass (each value is within the static caps), so
    it forwards the request; the DG1062Z must not silently deliver 20 Vpp. It either
    queues an error (caught as RuntimeError) or clamps — either way the divergence is
    visible, never a silent wrong amplitude."""
    config = _sine(waveform=WaveShape.SINE, frequency_hz=60e6, amplitude_vpp=20.0)
    try:
        dg1062.configure_channel(config)
    except RuntimeError:
        return  # instrument rejected it; _check_error surfaced the queued error
    assert dg1062.read_channel(CH1).amplitude_vpp < 20.0  # otherwise it must have clamped
