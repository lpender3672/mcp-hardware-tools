"""Cross-instrument HIL: the JDS6600 drives, the DS1054Z measures.

The single place that says what runs against this generator — the same shared suite the
DG1062Z runs, at this instrument's :data:`PROFILE`, plus what is specific to it.

This is the *strong* validation a generator's own read-backs cannot give: its only
observable is what you wrote (a tautology), so the actual output is ground-truthed here
by an independent instrument digitising it. The spectral and harmonic checks lean on that
the other way round too — with a trustworthy source, they validate
``hwtools.analysis.spectrum`` and ``hwtools.noise`` against real captures.

Wiring: JDS6600 CH1 -> scope CH1 via BNC coax (**probe ratio 1x**, not a 10x probe).

    uv run pytest tests/hardware/cross/test_jds6600_scope.py -m cfg_siggen_1ch

Leaves CH1 on a benign 1 kHz 2 Vpp sine.
"""

from __future__ import annotations

import pytest

from hwtools.drivers.joyit import JDS6600
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.noise import NoiseColor
from tests.hardware.cross import _siggen_suite as suite
from tests.hardware.cross._measure import measure

GEN_CH = SigGenChannel.CH1

# Frequency is where this instrument excels — a centi-Hz DDS tracks to well under 1 % —
# so that one is tightened. Everything else keeps the shared defaults, which were set
# from this generator's own measured behaviour: a 12-bit DAC, a fixed 2048-point
# wavetable, and bench-observed edge ringing on replayed squares (crest ~1.16 against an
# ideal 1.0, with Vpp overshooting a 4.0 request to ~4.5).
PROFILE = suite.SiggenProfile(frequency_rel=0.02)

pytestmark = [pytest.mark.hardware, pytest.mark.cfg_siggen_1ch]


# -- built-in waveforms ------------------------------------------------------------


def test_sine_level_and_frequency(generator: JDS6600, live_scope: DS1054Z) -> None:
    suite.check_sine_level_and_frequency(generator, live_scope, PROFILE)


@pytest.mark.parametrize("frequency_hz", [1_000.0, 50_000.0])
def test_frequency_tracks_the_request(
    generator: JDS6600, live_scope: DS1054Z, frequency_hz: float
) -> None:
    suite.check_frequency_tracks(generator, live_scope, PROFILE, frequency_hz=frequency_hz)


def test_dc_offset_reaches_the_output(generator: JDS6600, live_scope: DS1054Z) -> None:
    suite.check_dc_offset(generator, live_scope, PROFILE)


@pytest.mark.parametrize("shape", suite.ARB_SHAPES)
def test_builtin_shape_reaches_the_output(
    generator: JDS6600, live_scope: DS1054Z, shape: WaveShape
) -> None:
    suite.check_builtin_shape(generator, live_scope, PROFILE, shape=shape)


def test_dc_shape_sets_and_reads_back(generator: JDS6600) -> None:
    suite.check_dc_selects(generator)


# -- arbitrary waveforms -----------------------------------------------------------


@pytest.mark.parametrize("shape", suite.ARB_SHAPES)
def test_uploaded_arbitrary_replays_with_expected_shape(
    generator: JDS6600, live_scope: DS1054Z, shape: WaveShape
) -> None:
    suite.check_arb_replays_shape(generator, live_scope, PROFILE, shape=shape)


@pytest.mark.parametrize("color", list(NoiseColor))
def test_arb_noise_replays_with_the_intended_spectral_slope(
    generator: JDS6600, live_scope: DS1054Z, color: NoiseColor
) -> None:
    suite.check_arb_noise_slope(generator, live_scope, PROFILE, color=color)


def test_white_arb_noise_reads_as_broadband_noise(
    generator: JDS6600, live_scope: DS1054Z
) -> None:
    suite.check_white_arb_noise_is_broadband(generator, live_scope, PROFILE)


@pytest.mark.parametrize(("low_cycles", "high_cycles"), [(50, 100), (150, 250), (300, 400)])
def test_band_limited_noise_lands_in_its_window(
    generator: JDS6600, live_scope: DS1054Z, low_cycles: int, high_cycles: int
) -> None:
    suite.check_band_limited_noise(
        generator, live_scope, PROFILE, low_cycles=low_cycles, high_cycles=high_cycles
    )


# -- spectral analysis validated against this source -------------------------------


@pytest.mark.parametrize("f0", [100.0, 1_000.0, 10_000.0, 100_000.0])
def test_peak_frequency_tracks_across_decades(
    generator: JDS6600, live_scope: DS1054Z, f0: float
) -> None:
    suite.check_spectral_peak_tracks(generator, live_scope, PROFILE, frequency_hz=f0)


@pytest.mark.parametrize("amplitude_vpp", PROFILE.amplitude_sweep_vpp)
def test_amplitude_spectrum_reads_true_vpp(
    generator: JDS6600, live_scope: DS1054Z, amplitude_vpp: float
) -> None:
    suite.check_amplitude_spectrum_reads_true_vpp(
        generator, live_scope, PROFILE, amplitude_vpp=amplitude_vpp
    )


def test_sub_bin_interpolation_beats_nearest_bin(
    generator: JDS6600, live_scope: DS1054Z
) -> None:
    suite.check_sub_bin_interpolation_beats_nearest_bin(generator, live_scope, PROFILE)


@pytest.mark.parametrize("f0", suite.HARMONIC_FREQS)
@pytest.mark.parametrize("shape", suite.AC_SHAPES, ids=lambda s: s.value)
def test_shape_spectral_fingerprint(
    generator: JDS6600, live_scope: DS1054Z, shape: WaveShape, f0: float
) -> None:
    suite.check_shape_spectral_fingerprint(generator, live_scope, PROFILE, shape=shape, f0=f0)


# -- JDS6600-specific behaviour ----------------------------------------------------


@pytest.mark.parametrize("declared_load_ohms", [float("inf"), 50.0])
def test_amplitude_is_true_vpp_regardless_of_declared_load(
    generator: JDS6600, live_scope: DS1054Z, declared_load_ohms: float
) -> None:
    """The JDS6600 is a fixed-level source: ``output_load_ohms`` changes nothing.

    The exact opposite of the DG1000Z, where declaring 50 Ω into a high-Z scope doubles
    the delivered voltage. Pinning it here is what justifies the model treating the field
    as advisory for this instrument rather than applying it: 4 Vpp requested is 4 Vpp
    delivered whichever load is declared, so a driver that started "helpfully"
    compensating would break this.
    """
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=10_000.0,
            amplitude_vpp=4.0, output_load_ohms=declared_load_ohms,
        )
    )
    assert measure(live_scope, expected_hz=10_000.0, span_vpp=4.0).vpp == pytest.approx(
        4.0, rel=PROFILE.vpp_rel
    )
