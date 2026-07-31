"""Cross-instrument HIL: the DG1062Z drives, the DS1054Z measures.

The single place that says what runs against this generator. Two halves:

* **the generic suite** — thin wrappers over ``_siggen_suite``, written purely against
  :class:`SignalGenerator` / :class:`Oscilloscope`. Shape fidelity, spectral accuracy,
  harmonic fingerprints and noise replay are all shared bodies; only :data:`PROFILE`
  differs per instrument;
* **DG1000Z-specific behaviour** — written out here, because it has no counterpart on
  another generator.

Read-back cannot substitute for any of this: the generator's only observable is its own
registers, so a driver that reports the right settings while emitting the wrong signal
passes every self-consistency check. An independent instrument digitising the output is
the only thing that catches it.

Bench: DG1062Z (192.168.64.49, VISA) CH1 -> scope (192.168.64.7) CH1 via BNC coax
(**probe ratio 1x**, not a 10x probe).

    uv run pytest tests/hardware/cross/test_dg1062_scope.py -m cfg_siggen_1ch

The ``dg1062`` fixture disables both outputs and leaves CH1 benign afterwards.
"""

from __future__ import annotations

import math

import pytest

from hwtools.analysis.spectrum import harmonic_amplitudes
from hwtools.drivers.rigol import DG1062
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.synth import analytic_arbitrary
from tests.hardware.cross import _siggen_suite as suite
from tests.hardware.cross._measure import capture, measure

GEN_CH = SigGenChannel.CH1

# Only frequency is tightened: a 14-bit DAC and 60 MHz earn it there. Every other axis
# matched the shared default when measured, so it is left inherited rather than restated.
PROFILE = suite.SiggenProfile(frequency_rel=0.02)

# KNOWN FAILING, deliberately: the six SINC / FULL_WAVE / LORENZ fingerprint cases. Left
# red as a visible record that these three shapes diverge between the two generators,
# rather than hidden behind an xfail. Measured here at 1 kHz and 100 kHz, agreeing to
# ~0.5 % between the two:
#
#   SINC       crest 3.167 / 3.162 (shared band 1.85-2.35); h2..h6 all within 1 % of the
#              fundamental, so the flat-band harmonic model still holds — Rigol's sinc is
#              simply narrower than the JDS6600's. A (2.95, 3.40) band would pass.
#   FULL_WAVE  crest 1.657 (in band); fails only on peak_mult, because peak/f0 = 1.00 not
#              2.00. Rigol's ABSSINE takes the set frequency as the *rectified* repetition
#              rate where the JDS6600 takes it as the underlying sine. The Fourier series
#              is identical either way: measured 0.200/0.086/0.048/0.030/0.021 against a
#              theoretical 3/(4n^2-1) = 0.2000/0.0857/0.0476/0.0303/0.0210.
#   LORENZ     crest 1.740 (in band); fails the "steep" envelope on h2 = 0.510 vs < 0.45.
#              h2..h6 = 0.510/0.281/0.145/0.080/0.041 — monotone, and below 1/n from the
#              3rd up, but a gentler roll-off than the JDS6600's.
#
# SINC and LORENZ were always *empirical* bands (a sinc's bandwidth and a Lorentzian's
# width are vendor waveform-table choices, not constants), so per-instrument values there
# would be routine. FULL_WAVE is different in kind: WaveShape.FULL_WAVE at 1 kHz currently
# produces physically different signals on the two instruments, which is an abstraction
# leak belonging in a driver, not in test tolerances.

pytestmark = [pytest.mark.hardware, pytest.mark.cfg_siggen_1ch]


# -- built-in waveforms ------------------------------------------------------------


def test_sine_level_and_frequency(dg1062: DG1062, live_scope: DS1054Z) -> None:
    suite.check_sine_level_and_frequency(dg1062, live_scope, PROFILE)


@pytest.mark.parametrize("frequency_hz", [1_000.0, 50_000.0])
def test_frequency_tracks_the_request(
    dg1062: DG1062, live_scope: DS1054Z, frequency_hz: float
) -> None:
    suite.check_frequency_tracks(dg1062, live_scope, PROFILE, frequency_hz=frequency_hz)


def test_dc_offset_reaches_the_output(dg1062: DG1062, live_scope: DS1054Z) -> None:
    suite.check_dc_offset(dg1062, live_scope, PROFILE)


@pytest.mark.parametrize("shape", suite.ARB_SHAPES)
def test_builtin_shape_reaches_the_output(
    dg1062: DG1062, live_scope: DS1054Z, shape: WaveShape
) -> None:
    suite.check_builtin_shape(dg1062, live_scope, PROFILE, shape=shape)


def test_dc_shape_sets_and_reads_back(dg1062: DG1062) -> None:
    suite.check_dc_selects(dg1062)


# -- arbitrary waveforms -----------------------------------------------------------


@pytest.mark.parametrize("shape", suite.ARB_SHAPES)
def test_uploaded_arbitrary_replays_with_expected_shape(
    dg1062: DG1062, live_scope: DS1054Z, shape: WaveShape
) -> None:
    suite.check_arb_replays_shape(dg1062, live_scope, PROFILE, shape=shape)


# NB: the arbitrary-noise checks (check_arb_noise_slope / check_white_arb_noise_is_broadband
# / check_band_limited_noise) are deliberately NOT run here. Uploading a synthesised noise
# buffer is a workaround for a generator with no decent noise source of its own — the
# result is frozen and periodic (a comb at the replay rate, not a continuum) and band-
# limited to the buffer's harmonics, as hwtools.noise's own docstring says. The DG1062Z
# has a native 60 MHz noise source (Table 2-1), which is continuous and needs no
# reconstruction filter, so it is tested directly below instead. Those arbitrary-noise
# checks remain in the shared suite and still run against the JDS6600, where they earn
# their keep.
def test_builtin_noise_is_broadband(dg1062: DG1062, live_scope: DS1054Z) -> None:
    suite.check_builtin_noise_is_broadband(dg1062, live_scope, PROFILE)


# -- spectral analysis validated against this source -------------------------------


@pytest.mark.parametrize("f0", [100.0, 1_000.0, 10_000.0, 100_000.0])
def test_peak_frequency_tracks_across_decades(
    dg1062: DG1062, live_scope: DS1054Z, f0: float
) -> None:
    suite.check_spectral_peak_tracks(dg1062, live_scope, PROFILE, frequency_hz=f0)


@pytest.mark.parametrize("amplitude_vpp", PROFILE.amplitude_sweep_vpp)
def test_amplitude_spectrum_reads_true_vpp(
    dg1062: DG1062, live_scope: DS1054Z, amplitude_vpp: float
) -> None:
    suite.check_amplitude_spectrum_reads_true_vpp(
        dg1062, live_scope, PROFILE, amplitude_vpp=amplitude_vpp
    )


def test_sub_bin_interpolation_beats_nearest_bin(
    dg1062: DG1062, live_scope: DS1054Z
) -> None:
    suite.check_sub_bin_interpolation_beats_nearest_bin(dg1062, live_scope, PROFILE)


@pytest.mark.parametrize("f0", suite.HARMONIC_FREQS)
@pytest.mark.parametrize("shape", suite.AC_SHAPES, ids=lambda s: s.value)
def test_shape_spectral_fingerprint(
    dg1062: DG1062, live_scope: DS1054Z, shape: WaveShape, f0: float
) -> None:
    suite.check_shape_spectral_fingerprint(dg1062, live_scope, PROFILE, shape=shape, f0=f0)


# -- DG1000Z-specific behaviour ----------------------------------------------------


def test_fifty_ohm_load_setting_doubles_into_highz(
    dg1062: DG1062, live_scope: DS1054Z
) -> None:
    """Declaring 50 Ω while driving the high-Z scope makes a 2 Vpp request measure ~4 Vpp.

    The DG1000Z scales the delivered level to its configured load, so mis-declaring it
    doubles the real voltage — the exact trap ``output_load_ohms`` exists to prevent, and
    the reason the model carries the field at all. The JDS6600 has no analogue: it does
    not compensate, so the same test there asserts the *opposite* (nothing changes).
    """
    dg1062.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=10_000.0,
            amplitude_vpp=2.0, output_load_ohms=50.0,
        )
    )
    assert measure(live_scope, expected_hz=10_000.0, span_vpp=4.0).vpp == pytest.approx(
        4.0, rel=0.15
    )


def test_builtin_waveform_after_an_arbitrary_actually_comes_out(
    dg1062: DG1062, live_scope: DS1054Z
) -> None:
    """Swap arbitrary -> sine and the scope must see the sine. The regression that matters.

    Read-back cannot make this assertion. With no settle between the parameter writes and
    the output enable, the DG1062Z comes up on the *previous* waveform and keeps emitting
    it — measured 0/4 correct at a 0 ms gap versus 4/4 at 50 ms and above — while
    ``read_channel`` cheerfully reports the new settings. Frequency, level and shape all
    differ between the two states, so a channel stuck in sample-rate arbitrary mode
    cannot coincidentally satisfy one of them.
    """
    caps = dg1062.capabilities
    assert caps.arb_length is not None
    n = caps.arb_length.representative_length()

    # Establish the premise rather than assuming it: a triangle arb really is on the wire.
    dg1062.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=1_000.0, amplitude_vpp=4.0
        )
    )
    dg1062.upload_arbitrary(
        GEN_CH, analytic_arbitrary(WaveShape.TRIANGLE, points=n), sample_rate_hz=5_000.0 * n
    )
    arb = measure(live_scope, expected_hz=5_000.0, span_vpp=4.0)
    assert arb.peak_hz == pytest.approx(5_000.0, rel=0.05)
    assert arb.vpp == pytest.approx(4.0, rel=0.15)
    assert arb.crest_factor == pytest.approx(math.sqrt(3), abs=0.2)

    dg1062.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=20_000.0, amplitude_vpp=2.0
        )
    )
    f = measure(live_scope, expected_hz=20_000.0, span_vpp=2.0)
    assert f.peak_hz == pytest.approx(20_000.0, rel=0.05)  # not the arb's 5 kHz
    assert f.vpp == pytest.approx(2.0, rel=0.15)  # not the arb's 4 Vpp
    assert f.crest_factor == pytest.approx(math.sqrt(2), abs=0.15)  # a sine, not a triangle


def test_ramp_symmetry_shapes_the_output(dg1062: DG1062, live_scope: DS1054Z) -> None:
    """``duty_pct`` drives ``:FUNCtion:RAMP:SYMMetry``, so it must change the shape.

    A symmetric triangle and a sawtooth share a crest factor of sqrt(3) and both spend
    half their period above the midline, so neither level nor duty separates them.
    Harmonic content does: a symmetric triangle is odd-harmonic only, suppressing its 2nd
    deeply, whereas a sawtooth carries every harmonic at roughly 1/n. Guards the
    ``duty_pct`` -> symmetry wiring that a hard-coded 50 would silently drop.
    """
    fundamental_hz = 10_000.0

    def second_harmonic_ratio(duty_pct: float) -> float:
        dg1062.configure_channel(
            SignalGeneratorConfig(
                channel=GEN_CH, waveform=WaveShape.TRIANGLE, frequency_hz=fundamental_hz,
                amplitude_vpp=4.0, duty_pct=duty_pct,
            )
        )
        wf = capture(live_scope, expected_hz=fundamental_hz, span_vpp=4.0)
        h1, h2 = harmonic_amplitudes(wf, fundamental_hz, 2)
        return h2 / h1

    symmetric = second_harmonic_ratio(50.0)
    sawtooth = second_harmonic_ratio(99.0)
    assert symmetric < 0.05  # even harmonics cancel in a symmetric triangle
    assert sawtooth > 0.2  # a sawtooth's 2nd sits near 1/2
    assert sawtooth > symmetric * 4
