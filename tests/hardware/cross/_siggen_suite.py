"""Generic, measured checks that hold for *any* signal generator.

Both sides of a cross-instrument test are interfaces — :class:`SignalGenerator` and
:class:`Oscilloscope` — so the checks themselves need not know which instruments are on
the bench. They live here as plain functions; the ``@pytest.mark`` / ``def test_*``
entry points live only in ``test_<instrument>_scope.py``, so for any given generator
there is exactly one file that says what runs against it and with what numbers.

Three things are kept strictly apart:

* **theory** — the Fourier harmonic models, ideal crest factors and noise slopes are
  mathematics, identical for every instrument, and live here;
* **capability** — arbitrary addressing, buffer depth, playback clock and the supported
  waveform set are read from :class:`~hwtools.model.siggen.SigGenCapabilities`, so a
  check an instrument cannot satisfy skips itself naming what it wanted;
* **fidelity** — how closely a *particular* generator is expected to hit theory is a
  :class:`SiggenProfile`, declared once per instrument by its test file.

That last split is what makes the same body serve a 12-bit DDS and a 14-bit AWG without
either being held to the other's standard.

This module imports pytest for ``approx`` and ``skip`` — the failure messages are worth
far more than purity on a bench, where a bad number is the whole diagnosis — but it
deliberately defines no tests and is never collected.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from hwtools.analysis import spectrum
from hwtools.analysis.describe import describe
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.interfaces.signal_generator import SignalGenerator
from hwtools.model.siggen import (
    ArbAddressing,
    ArbitraryWaveform,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
)
from hwtools.model.waveform import Waveform
from hwtools.noise import (
    PSD_SLOPE_DB_PER_DECADE,
    NoiseColor,
    band_limited_noise_arbitrary,
    noise_arbitrary,
)
from hwtools.synth import analytic_arbitrary
from tests.hardware.cross._measure import capture, measure

GEN_CH = SigGenChannel.CH1
_TWO_PI = 2.0 * math.pi

# Theoretical crest factor (peak / AC-rms) per shape — amplitude-invariant, so it
# discriminates *shape* without caring about level. Square 1, sine sqrt(2), triangle
# sqrt(3): well separated and correctly ordered, which is what makes it a usable gate.
IDEAL_CREST: dict[WaveShape, float] = {
    WaveShape.SQUARE: 1.0,
    WaveShape.SINE: math.sqrt(2.0),
    WaveShape.TRIANGLE: math.sqrt(3.0),
}
# The shapes `analytic_arbitrary` can synthesise, and so the ones an arb replay can use.
ARB_SHAPES = (WaveShape.SQUARE, WaveShape.SINE, WaveShape.TRIANGLE)


class SiggenProfile(BaseModel):
    """One instrument's measured-fidelity budget and measurement windows.

    More than tolerances: it also carries the sweeps and analysis bands that depend on
    the source rather than on the code under test. Defaults describe the loosest
    instrument on the bench, so a new generator can be brought up with
    ``SiggenProfile()`` and tightened from what it actually achieves, rather than
    guessed at up front.
    """

    model_config = ConfigDict(frozen=True)

    # -- time-domain fidelity --
    frequency_rel: float = Field(default=0.05, gt=0, description="Relative frequency error.")
    vpp_rel: float = Field(default=0.1, gt=0, description="Relative amplitude error.")
    dc_abs: float = Field(default=0.2, gt=0, description="Absolute offset error, volts.")
    crest_abs: float = Field(default=0.15, gt=0, description="Crest error, built-in shapes.")
    # A replayed arbitrary is reconstructed through the DAC's output filter, which rings
    # on fast edges — a square overshoots in level and crest far more than a built-in
    # square does. Hence a separate, wider band for that one case.
    arb_crest_abs: float = Field(default=0.1, gt=0, description="Crest error, replayed arb.")
    arb_square_crest_abs: float = Field(
        default=0.3, gt=0, description="Crest error, replayed arb square (edge ringing)."
    )
    periodicity_min: float = Field(
        default=0.8, gt=0, le=1.0, description="Minimum autocorrelation peak strength."
    )

    # -- spectral accuracy (validates hwtools.analysis.spectrum) --
    spectral_frequency_rel: float = Field(
        default=5e-3, gt=0, description="Peak-frequency error on a deep coherent capture."
    )
    spectral_amplitude_rel: float = Field(
        default=0.06, gt=0, description="Flattop amplitude-spectrum error vs Vpp/2."
    )
    # Amplitudes to sweep the amplitude-spectrum check over. Instrument-specific: the
    # top of the range is the generator's own ceiling.
    amplitude_sweep_vpp: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    # Crest tolerance where a shape's crest factor has a closed form. Tighter than
    # `crest_abs` because these run on a deep, near-coherent capture.
    crest_theory_abs: float = Field(default=0.08, gt=0)

    # -- arbitrary noise replay (validates hwtools.noise end to end) --
    noise_replay_hz: float = Field(
        default=1_000.0, gt=0, description="Buffer repetition rate for noise replay."
    )
    noise_slope_band_hz: tuple[float, float] = Field(
        default=(5e3, 5e5),
        description="Mid-band for the PSD slope fit: above the replay rate, below the "
        "front-end roll-off.",
    )
    noise_slope_abs: float = Field(
        default=5.0, gt=0, description="PSD slope error, dB/decade."
    )
    noise_crest_min: float = Field(default=2.5, gt=0)
    noise_flatness_min: float = Field(default=0.08, gt=0)
    band_rejection_db_min: float = Field(default=15.0, gt=0)

    def arb_crest_for(self, shape: WaveShape) -> float:
        return self.arb_square_crest_abs if shape is WaveShape.SQUARE else self.arb_crest_abs


# -- theoretical harmonic models (|h_n / h_1| from the Fourier series) --------------
# Mathematics, not measurements: identical for every generator. What differs per
# instrument is only how closely it is expected to hit these, i.e. the profile.


def _pure(n: int) -> float:
    return 1.0 if n == 1 else 0.0


def _square(n: int) -> float:
    return 1.0 / n if n % 2 == 1 else 0.0  # odd harmonics, 1/n


def _triangle(n: int) -> float:
    return 1.0 / (n * n) if n % 2 == 1 else 0.0  # odd harmonics, 1/n^2


def _sawtooth(n: int) -> float:
    return 1.0 / n  # every harmonic, 1/n


def _half_wave(n: int) -> float:
    if n == 1:
        return 1.0
    return (4.0 / math.pi) / (n * n - 1) if n % 2 == 0 else 0.0  # even only


def full_wave_coeff(m: int) -> float:
    """Full-wave rectified sine: harmonic m at 3/(4m^2-1) of the fundamental.

    Public because an instrument override may need to reuse it with a different
    ``fundamental_mult`` — vendors disagree on whether the set frequency means the
    rectified repetition rate or the underlying sine.
    """
    return 3.0 / (4 * m * m - 1)


def _multitone(n: int) -> float:
    return 1.0 if n <= 5 else 0.0


@dataclass(frozen=True)
class Fourier:
    """A closed-form harmonic model. ``coeff(n)`` is the theoretical |h_n/h_1|.

    Present harmonics (coeff >= floor) must match within ``rel_tol``; absent ones
    (coeff < floor) must measure below ``floor``. ``fundamental_mult`` analyses the
    harmonics of that multiple of the set frequency (full-wave's fundamental is 2x).
    """

    coeff: Callable[[int], float]
    fundamental_mult: int = 1
    rel_tol: float = 0.25
    floor: float = 0.03
    n_check: int = 5


@dataclass(frozen=True)
class Envelope:
    """A harmonic pattern without a fixed closed form (parameter-dependent shapes).

    ``h2_max``/``h3_max`` bound the ``"steep"`` kind. They are defaults, not theory: how
    steeply a vendor's built-in Lorenz actually rolls off is a property of that vendor's
    waveform table, so an instrument whose version decays more gently overrides them
    rather than the check being loosened for everyone.
    """

    kind: str  # "exp" | "steep" | "flat_band"
    h2_max: float = 0.45  # 2nd below the sawtooth's 1/2
    h3_max: float = 0.20  # 3rd well below 1/3


HarmonicModel = Fourier | Envelope | None  # None -> broadband, no harmonic check

_SQRT2 = math.sqrt(2)  # sine crest
_SQRT3 = math.sqrt(3)  # triangle crest


@dataclass(frozen=True)
class ShapeSpec:
    """How one waveform shape should look, spectrally and in crest factor.

    ``crest`` as a bare float is theoretical (peak/rms); a ``(lo, hi)`` tuple is an
    empirical band, for shapes with no closed form or where a DDS bandlimit softens the
    peak below theory (the sawtooths). ``trigger_v`` is a property of where the shape
    sits relative to ground — the logic-level and rectified shapes never cross 0 V, so a
    0 V edge trigger would wait forever.
    """

    trigger_v: float
    crest: float | tuple[float, float]
    harmonics: HarmonicModel
    peak_mult: float | None = 1.0  # expected peak_hz / f0, or None to skip


SPECS: dict[WaveShape, ShapeSpec] = {
    WaveShape.SINE: ShapeSpec(0.0, _SQRT2, Fourier(_pure)),
    WaveShape.PARTIAL_SINE: ShapeSpec(0.0, (1.35, 1.55), Fourier(_pure)),
    WaveShape.SQUARE: ShapeSpec(0.0, 1.0, Fourier(_square)),
    WaveShape.PULSE: ShapeSpec(0.0, 1.0, Fourier(_square)),
    WaveShape.CMOS: ShapeSpec(1.0, 1.0, Fourier(_square)),
    WaveShape.TRIANGLE: ShapeSpec(0.0, _SQRT3, Fourier(_triangle)),
    WaveShape.POS_STEP: ShapeSpec(0.0, (1.45, 1.80), Fourier(_sawtooth)),
    WaveShape.NEG_STEP: ShapeSpec(0.0, (1.45, 1.80), Fourier(_sawtooth)),
    WaveShape.HALF_WAVE: ShapeSpec(1.0, (1.20, 1.55), Fourier(_half_wave)),
    WaveShape.FULL_WAVE: ShapeSpec(
        1.0, (1.50, 1.95), Fourier(full_wave_coeff, fundamental_mult=2, n_check=4), peak_mult=2.0
    ),
    WaveShape.MULTI_TONE: ShapeSpec(
        0.0, (2.25, 2.85), Fourier(_multitone, n_check=6, floor=0.05), peak_mult=None
    ),
    WaveShape.EXP_RISE: ShapeSpec(0.0, (1.90, 2.40), Envelope("exp")),
    WaveShape.EXP_DECAY: ShapeSpec(0.0, (1.90, 2.40), Envelope("exp")),
    WaveShape.SINC: ShapeSpec(0.0, (1.85, 2.35), Envelope("flat_band"), peak_mult=None),
    WaveShape.LORENZ: ShapeSpec(0.0, (1.40, 1.75), Envelope("steep")),
    WaveShape.NOISE: ShapeSpec(0.0, (1.50, 3.50), None, peak_mult=None),
}
AC_SHAPES = tuple(SPECS)  # every shape with a spectrum (DC is checked separately)

# Per fundamental, a cycle count whose 12-division window spans many whole periods, so
# the capture is near-coherent and leakage stays small. Deep memory on top of that keeps
# the bin width fine.
HARMONIC_CYCLES: dict[float, float] = {1_000.0: 120.0, 100_000.0: 240.0}
HARMONIC_FREQS = tuple(HARMONIC_CYCLES)
HARMONIC_MEMORY = 1_200_000


def _require_shape(generator: SignalGenerator, shape: WaveShape) -> None:
    waveforms = generator.capabilities.waveforms
    if waveforms and shape not in waveforms:
        pytest.skip(f"{generator.capabilities.model_name} has no {shape} waveform")


def _require_arbitrary(generator: SignalGenerator) -> int:
    """Skip unless the instrument has arbitrary support; return a usable point count."""
    caps = generator.capabilities
    if not caps.supports_arbitrary():
        pytest.skip(f"{caps.model_name} has no arbitrary-waveform support")
    assert caps.arb_length is not None  # narrowed by supports_arbitrary()
    return caps.arb_length.representative_length()


def _play_arbitrary(
    generator: SignalGenerator,
    wave: ArbitraryWaveform,
    *,
    frequency_hz: float,
    amplitude_vpp: float,
) -> None:
    """Upload ``wave`` and get it repeating at ``frequency_hz``, whichever family it is.

    The two arbitrary families reach the same repetition rate by different routes:

    * ``SLOT`` — a DDS wavetable played at the channel's configured frequency, so the
      frequency comes from the channel config;
    * ``VOLATILE`` — clocked out point by point, so the frequency is the sample rate
      divided by the point count, and the level comes from a prior channel config.
    """
    caps = generator.capabilities
    if caps.arb_addressing is ArbAddressing.SLOT:
        slot = caps.arb_slots  # the highest slot, sparing low front-panel presets
        generator.upload_arbitrary(GEN_CH, wave, slot=slot)
        generator.configure_channel(
            SignalGeneratorConfig(
                channel=GEN_CH, frequency_hz=frequency_hz,
                amplitude_vpp=amplitude_vpp, arb_slot=slot,
            )
        )
    else:
        generator.configure_channel(
            SignalGeneratorConfig(
                channel=GEN_CH, waveform=WaveShape.SINE,
                frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp,
            )
        )
        rate_hz = frequency_hz * wave.n if caps.arb_sample_rate is not None else None
        generator.upload_arbitrary(GEN_CH, wave, sample_rate_hz=rate_hz)


def _set_sine(
    generator: SignalGenerator, *, frequency_hz: float, amplitude_vpp: float
) -> None:
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE,
            frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp,
        )
    )


# -- built-in waveforms ------------------------------------------------------------


def check_sine_level_and_frequency(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    frequency_hz: float = 10_000.0,
    amplitude_vpp: float = 4.0,
) -> None:
    """A sine arrives at the requested frequency and level, and looks like a sine.

    ``output_load_ohms`` is left at its high-Z default, matching the scope's 1 MΩ input
    — the one amplitude case unambiguous for every generator, whether or not it
    compensates for load.
    """
    _set_sine(generator, frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp)
    f = measure(scope, expected_hz=frequency_hz, span_vpp=amplitude_vpp)
    assert f.peak_hz == pytest.approx(frequency_hz, rel=prof.frequency_rel)
    assert f.vpp == pytest.approx(amplitude_vpp, rel=prof.vpp_rel)
    assert f.crest_factor == pytest.approx(IDEAL_CREST[WaveShape.SINE], abs=prof.crest_abs)
    assert f.periodicity > prof.periodicity_min


def check_frequency_tracks(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    frequency_hz: float,
    amplitude_vpp: float = 4.0,
) -> None:
    """The measured tone follows the requested frequency."""
    _set_sine(generator, frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp)
    f = measure(scope, expected_hz=frequency_hz, span_vpp=amplitude_vpp)
    assert f.peak_hz == pytest.approx(frequency_hz, rel=prof.frequency_rel)


def check_dc_offset(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    offset_v: float = 1.0,
    frequency_hz: float = 10_000.0,
    amplitude_vpp: float = 2.0,
) -> None:
    """A DC bias shows up as the captured mean level."""
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=frequency_hz,
            amplitude_vpp=amplitude_vpp, offset_v=offset_v,
        )
    )
    # Span and trigger both allow for the bias: a 0 V trigger level would otherwise sit
    # on, or outside, the shifted waveform's lower rail.
    f = measure(
        scope, expected_hz=frequency_hz,
        span_vpp=amplitude_vpp + 2.0 * abs(offset_v), trigger_level_v=offset_v,
    )
    assert f.dc_level == pytest.approx(offset_v, abs=prof.dc_abs)


def check_builtin_shape(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    shape: WaveShape,
    frequency_hz: float = 1_000.0,
    amplitude_vpp: float = 4.0,
) -> None:
    """A built-in shape reaches the output with its characteristic crest factor."""
    _require_shape(generator, shape)
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=shape,
            frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp,
        )
    )
    f = measure(scope, expected_hz=frequency_hz, span_vpp=amplitude_vpp)
    assert f.peak_hz == pytest.approx(frequency_hz, rel=prof.frequency_rel)
    assert f.periodicity > prof.periodicity_min
    assert f.crest_factor == pytest.approx(IDEAL_CREST[shape], abs=prof.crest_abs)


def check_dc_selects(generator: SignalGenerator) -> None:
    """DC has no edge to trigger on and no spectrum; verify the driver selects it.

    Generator-only — no scope involved — but it belongs with the shape coverage.
    """
    _require_shape(generator, WaveShape.DC)
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.DC, frequency_hz=1_000.0, amplitude_vpp=1.0
        )
    )
    assert generator.read_channel(GEN_CH).waveform is WaveShape.DC


# -- arbitrary waveforms -----------------------------------------------------------


def check_arb_replays_shape(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    shape: WaveShape,
    frequency_hz: float = 1_000.0,
    amplitude_vpp: float = 4.0,
) -> None:
    """An uploaded analytic waveform replays at the requested rate and shape.

    The crest-factor gate is what proves the upload -> replay chain reproduces *shape*
    and not merely amplitude.
    """
    n = _require_arbitrary(generator)
    _play_arbitrary(
        generator, analytic_arbitrary(shape, points=n),
        frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp,
    )
    f = measure(scope, expected_hz=frequency_hz, span_vpp=amplitude_vpp)
    assert f.peak_hz == pytest.approx(frequency_hz, rel=prof.frequency_rel)
    assert f.periodicity > prof.periodicity_min
    assert f.crest_factor == pytest.approx(IDEAL_CREST[shape], abs=prof.arb_crest_for(shape))


def _capture_noise_wideband(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    wave: ArbitraryWaveform,
) -> Waveform:
    """Play a noise buffer and capture it with enough bandwidth to see it, not alias it.

    The buffer replays at ``n * noise_replay_hz`` samples per second — megahertz of
    content — so the screen-rate trace would alias it beyond recognition. A short window
    with deep memory gives ~10 MSa/s instead.
    """
    _play_arbitrary(
        generator, wave, frequency_hz=prof.noise_replay_hz, amplitude_vpp=4.0
    )
    # 1.2 periods of the replay rate is a ~1.2 ms window; 12k samples over that is
    # ~10 MSa/s.
    return capture(
        scope, expected_hz=prof.noise_replay_hz, span_vpp=4.0, cycles=1.2, memory_depth=12_000
    )


def check_arb_noise_slope(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    color: NoiseColor,
    seed: int = 11,
) -> None:
    """Coloured noise replays with the intended PSD slope.

    Proves the whole chain end to end — ``noise`` synthesises, ``ArbitraryWaveform``
    renders, the driver uploads, the generator replays, the scope digitises — by
    checking the measured slope (white ~0, pink ~-10, brown ~-20 dB/decade). The
    front-end and DAC reconstruction bend the spectrum, hence a wide tolerance, but the
    colours land ~10 dB/decade apart so they remain unmistakable.
    """
    n = _require_arbitrary(generator)
    wf = _capture_noise_wideband(generator, scope, prof, noise_arbitrary(n, color, seed=seed))
    f_lo, f_hi = prof.noise_slope_band_hz
    slope = spectrum.psd_slope_db_per_decade(wf, f_lo=f_lo, f_hi=f_hi, nperseg=2048)
    assert slope == pytest.approx(PSD_SLOPE_DB_PER_DECADE[color], abs=prof.noise_slope_abs)


def check_builtin_noise_is_broadband(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    amplitude_vpp: float = 4.0,
) -> None:
    """The instrument's *native* noise source is broadband and flat.

    Strictly better than replaying a synthesised noise buffer where the hardware offers
    it: a built-in noise generator is continuous rather than frozen and periodic, and its
    bandwidth is the instrument's own (60 MHz on a DG1062Z) rather than the arbitrary
    buffer's replay rate. Nothing is uploaded, so there is no reconstruction filter
    bending the spectrum and no comb from a repeating buffer.

    Checked as noise rather than as a shape: Gaussian amplitude statistics (crest well
    above a sine's 1.41), a spectrum spread across bins rather than concentrated in one,
    and a PSD slope near flat — white, by definition.
    """
    _require_shape(generator, WaveShape.NOISE)
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.NOISE,
            frequency_hz=prof.noise_replay_hz, amplitude_vpp=amplitude_vpp,
        )
    )
    # Noise has no period to put on screen, so the window is chosen for bandwidth: the
    # same short, deep capture the arbitrary path uses, giving ~10 MSa/s.
    wf = capture(
        scope, expected_hz=prof.noise_replay_hz, span_vpp=amplitude_vpp,
        cycles=1.2, memory_depth=12_000,
    )
    features = describe(wf, psd=True)
    assert features.crest_factor > prof.noise_crest_min
    assert features.spectral_flatness > prof.noise_flatness_min
    f_lo, f_hi = prof.noise_slope_band_hz
    slope = spectrum.psd_slope_db_per_decade(wf, f_lo=f_lo, f_hi=f_hi, nperseg=2048)
    assert slope == pytest.approx(
        PSD_SLOPE_DB_PER_DECADE[NoiseColor.WHITE], abs=prof.noise_slope_abs
    )


def check_white_arb_noise_is_broadband(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    seed: int = 11,
) -> None:
    """White noise is measurably noise, not a tone: high crest and no dominant peak."""
    n = _require_arbitrary(generator)
    wf = _capture_noise_wideband(
        generator, scope, prof, noise_arbitrary(n, NoiseColor.WHITE, seed=seed)
    )
    features = describe(wf, psd=True)
    assert features.crest_factor > prof.noise_crest_min  # Gaussian peaks above a sine
    assert features.spectral_flatness > prof.noise_flatness_min  # broadband, not one bin


def check_band_limited_noise(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    low_cycles: int,
    high_cycles: int,
    seed: int = 3,
) -> None:
    """Band-limited noise confines its energy to [low, high] x the replay rate.

    Moving the band moves the measured energy — the sweepable narrow-band stimulus. An
    arbitrary's *broadband* replay is compromised by the reconstruction filter, but
    within a window the noise is accurate.
    """
    n = _require_arbitrary(generator)
    wave = band_limited_noise_arbitrary(
        n, low_cycles=low_cycles, high_cycles=high_cycles, seed=seed
    )
    wf = _capture_noise_wideband(generator, scope, prof, wave)

    band_lo = low_cycles * prof.noise_replay_hz
    band_hi = high_cycles * prof.noise_replay_hz
    spec = spectrum.welch_psd(wf, nperseg=4096)
    freqs, psd = spec.frequencies_hz, spec.values
    in_band = (freqs >= band_lo) & (freqs <= band_hi)
    out_band = (freqs > band_hi * 1.5) & (freqs < wf.sample_rate_hz * 0.45)
    centroid = float(np.sum(freqs[in_band] * psd[in_band]) / np.sum(psd[in_band]))
    rejection_db = 10.0 * float(np.log10(psd[in_band].mean() / psd[out_band].mean()))

    assert band_lo <= centroid <= band_hi  # energy centred inside the intended window
    assert rejection_db > prof.band_rejection_db_min  # band-limited, not broadband


# -- spectral accuracy: validates hwtools.analysis.spectrum on real captures --------


def _capture_sine_deep(
    generator: SignalGenerator,
    scope: Oscilloscope,
    *,
    frequency_hz: float,
    amplitude_vpp: float = 4.0,
) -> Waveform:
    """A deep, many-period capture of a sine — fine bins, low leakage."""
    _set_sine(generator, frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp)
    return capture(scope, expected_hz=frequency_hz, span_vpp=amplitude_vpp, deep=True)


def check_spectral_peak_tracks(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    frequency_hz: float,
) -> None:
    """``peak_frequency`` tracks the set frequency across decades."""
    wf = _capture_sine_deep(generator, scope, frequency_hz=frequency_hz)
    assert spectrum.peak_frequency(wf) == pytest.approx(
        frequency_hz, rel=prof.spectral_frequency_rel
    )


def check_amplitude_spectrum_reads_true_vpp(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    amplitude_vpp: float,
    frequency_hz: float = 1_000.0,
) -> None:
    """A flattop-windowed amplitude spectrum reads a tone's amplitude as Vpp/2.

    Reads slightly low in practice, partly because generators tend to output a percent
    or two under the set amplitude — which is why the tolerance is part of the profile.
    """
    wf = _capture_sine_deep(
        generator, scope, frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp
    )
    peak = spectrum.amplitude_spectrum(wf, window="flattop").peak()
    assert peak is not None
    _, amplitude = peak
    assert amplitude == pytest.approx(amplitude_vpp / 2.0, rel=prof.spectral_amplitude_rel)


def check_sub_bin_interpolation_beats_nearest_bin(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    frequency_hz: float = 12_345.0,
) -> None:
    """A tone deliberately between FFT bins is recovered to well under half a bin —
    the payoff of the parabolic sub-bin refinement in ``peak_frequency``."""
    wf = _capture_sine_deep(generator, scope, frequency_hz=frequency_hz)
    est = spectrum.peak_frequency(wf)
    assert est is not None
    bin_hz = 1.0 / wf.duration_s
    assert abs(est - frequency_hz) < 0.5 * bin_hz  # better than nearest-bin worst case
    assert abs(est - frequency_hz) < prof.spectral_frequency_rel * frequency_hz


# -- harmonic fingerprints: every shape against its Fourier series ------------------


def _relative_harmonics(wf: Waveform, fundamental_hz: float, n: int) -> list[float]:
    amps = spectrum.harmonic_amplitudes(wf, fundamental_hz, n)
    assert amps[0] > 0, "no energy at the fundamental"
    return [a / amps[0] for a in amps]  # r[0]=1, r[1]=2nd harmonic, r[2]=3rd, ...


def _assert_fourier(model: Fourier, wf: Waveform, f0: float) -> None:
    r = _relative_harmonics(wf, model.fundamental_mult * f0, model.n_check)
    for n in range(2, model.n_check + 1):
        theory, measured = model.coeff(n), r[n - 1]
        if theory < model.floor:
            assert measured < model.floor, f"h{n} should be ~0, got {measured:.3f}"
        else:
            assert measured == pytest.approx(theory, rel=model.rel_tol), (
                f"h{n}: measured {measured:.3f} vs Fourier {theory:.3f}"
            )


def _assert_envelope(model: Envelope, wf: Waveform, f0: float) -> None:
    r = _relative_harmonics(wf, f0, 6)
    if model.kind == "exp":
        # Periodic exponential: |h_n/h_1| = sqrt(b^2+(2pi)^2) / sqrt(b^2+(2pi*n)^2).
        # Fit b^2 from the 2nd harmonic (richer than a sawtooth => r[1] > 1/2), then
        # every higher harmonic must follow that one-parameter model.
        assert r[1] > 0.5, f"exp 2nd harmonic {r[1]:.3f} not richer than a sawtooth"
        ratio_sq = r[1] ** 2
        beta_sq = _TWO_PI**2 * (4 * ratio_sq - 1) / (1 - ratio_sq)

        def predict(n: int) -> float:
            return math.sqrt(beta_sq + _TWO_PI**2) / math.sqrt(beta_sq + (_TWO_PI * n) ** 2)

        for n in (3, 4, 5):
            assert r[n - 1] == pytest.approx(predict(n), rel=0.15), (
                f"exp h{n}: measured {r[n - 1]:.3f} vs fitted exponential {predict(n):.3f}"
            )
    elif model.kind == "steep":  # Lorenz: decays faster than 1/n, monotonically
        assert r[1] < model.h2_max, f"h2 {r[1]:.3f} not below {model.h2_max}"
        assert r[2] < model.h3_max, f"h3 {r[2]:.3f} not below {model.h3_max}"
        assert r[1] > r[2] > r[3], f"harmonics not monotone: {r[1:4]}"
    elif model.kind == "flat_band":  # sinc -> ~rectangular spectrum
        assert sum(1 for x in r[1:4] if 0.7 <= x <= 1.4) >= 3
    else:
        raise AssertionError(f"unhandled envelope {model.kind!r}")


def _assert_harmonics(model: HarmonicModel, wf: Waveform, f0: float) -> None:
    if isinstance(model, Fourier):
        _assert_fourier(model, wf, f0)
    elif isinstance(model, Envelope):
        _assert_envelope(model, wf, f0)
    # None: broadband — no harmonic structure to assert.


def check_shape_spectral_fingerprint(
    generator: SignalGenerator,
    scope: Oscilloscope,
    prof: SiggenProfile,
    *,
    shape: WaveShape,
    f0: float,
    amplitude_vpp: float = 4.0,
    overrides: Mapping[WaveShape, ShapeSpec] | None = None,
) -> None:
    """One built-in shape against the harmonic amplitudes its Fourier series predicts.

    Not hand-tuned ranges: the theory is closed-form (square odd harmonics at 1/n,
    triangle at 1/n^2, half-wave even at (4/pi)/(n^2-1), ...) and the bench matches it
    to a few tenths of a dB, so these tolerances are genuinely tight.

    ``overrides`` replaces the :data:`SPECS` entry for a shape. This is not a tolerance
    escape hatch — it exists because a *vendor-neutral* shape name can cover genuinely
    different waveforms. Two real examples, both measured: Rigol's ``SINC`` is spikier
    than the JDS6600's (crest 3.17 vs ~2.1), and Rigol's ``ABSSINE`` treats the set
    frequency as the rectified waveform's repetition rate where the JDS6600 treats it as
    the underlying sine's, moving the fundamental from 2*f0 to f0. In both cases the
    *theory* is unchanged; only which waveform the name denotes differs.
    """
    _require_shape(generator, shape)
    spec = (overrides or {}).get(shape) or SPECS[shape]
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=shape, frequency_hz=f0, amplitude_vpp=amplitude_vpp
        )
    )
    wf = capture(
        scope,
        expected_hz=f0,
        span_vpp=amplitude_vpp,
        cycles=HARMONIC_CYCLES[f0],
        memory_depth=HARMONIC_MEMORY,
        trigger_level_v=spec.trigger_v,
        settle_s=0.35,
    )
    features = describe(wf, psd=True)

    crest = features.crest_factor
    if isinstance(spec.crest, tuple):  # empirical band (no closed form)
        lo, hi = spec.crest
        assert lo <= crest <= hi, f"crest {crest:.2f} outside {spec.crest} for {shape.value}"
    else:  # theoretical crest factor
        assert crest == pytest.approx(spec.crest, abs=prof.crest_theory_abs), (
            f"crest {crest:.2f} != theory {spec.crest:.3f} for {shape.value}"
        )

    if spec.harmonics is None:  # noise: no clean fundamental at f0
        assert features.peak_hz is None or not (0.95 * f0 <= features.peak_hz <= 1.05 * f0)
    elif spec.peak_mult is not None:
        assert features.peak_hz == pytest.approx(spec.peak_mult * f0, rel=0.02)

    _assert_harmonics(spec.harmonics, wf, f0)
