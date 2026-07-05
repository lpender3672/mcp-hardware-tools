"""Cross-instrument HIL: validate the frequency-analysis tools against every built-in
waveform shape, at two decades (1 kHz and 100 kHz).

The JDS6600 is the ground-truth source, and each shape is checked against the
**harmonic amplitudes predicted by its Fourier series** — not hand-tuned ranges:

* sine .......... only the fundamental
* square/pulse/cmos  odd harmonics, |h_n/h_1| = 1/n
* triangle ...... odd harmonics, 1/n^2
* pos/neg step .. sawtooth, all harmonics 1/n
* half-wave ..... even harmonics, (4/pi)/(n^2-1); odd (n>=3) absent
* full-wave ..... fundamental at 2*f0; harmonics 3/(4m^2-1)
* multi-tone .... five equal tones
* exp rise/decay  periodic exponential, 1/sqrt(beta^2+(2*pi*n)^2) with beta fitted
                  from the 2nd harmonic and the rest verified against it
* lorenz ........ super-polynomial decay (faster than 1/n, monotone)
* sinc .......... flat (rectangular) band of low harmonics

The bench matches theory to a few tenths of a dB (square h3 = -9.5 vs 1/3 = -9.54;
triangle h5 = -28.0 vs 1/25 = -27.96), so the tolerances are genuinely tight.

Capture is chosen for a clean spectrum: **no AUTO** — arm SINGLE on an edge and
deep-read one triggered frame; **many periods** on screen (a 12-division window at a
1-2-5 timebase spans an integer number of periods at these frequencies, so the
capture is near-coherent) with deep memory, keeping bin width and leakage small.
Trigger levels are per shape: 0 V for symmetric shapes, +1 V for the logic-level /
rectified shapes that sit above ground. DC has no edge and no spectrum, so it is
checked separately at the driver level.

    uv run pytest -m cfg_siggen_1ch -s
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from hwtools.analysis import spectrum
from hwtools.analysis.describe import describe
from hwtools.drivers.joyit import JDS6600
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.model.waveform import Waveform
from tests.hardware._acquire import acquire_one_shot

SCOPE_CH = ChannelId.CH1
GEN_CH = SigGenChannel.CH1
_MEM = 1_200_000  # deep — churn plenty of samples for precision
# Per fundamental: a 1-2-5 timebase whose 12-div window spans many whole periods.
_TB = {
    1_000.0: TimebaseConfig(scale_s_per_div=10e-3),  # 120 ms -> 120 cycles
    100_000.0: TimebaseConfig(scale_s_per_div=200e-6),  # 2.4 ms -> 240 cycles
}
FREQS = list(_TB)
_TWO_PI = 2.0 * math.pi


# -- theoretical harmonic models (|h_n / h_1| from the Fourier series) ---------


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


def _full_wave(m: int) -> float:
    return 3.0 / (4 * m * m - 1)  # harmonics m of the 2*f0 fundamental


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
    """A harmonic pattern without a fixed closed form (parameter-dependent shapes)."""

    kind: str  # "exp" | "steep" | "flat_band"


HarmonicModel = Fourier | Envelope | None  # None -> broadband, no harmonic check


@dataclass(frozen=True)
class ShapeSpec:
    trigger_v: float
    crest: tuple[float, float]
    harmonics: HarmonicModel
    peak_mult: float | None = 1.0  # expected peak_hz / f0, or None to skip


SPECS: dict[WaveShape, ShapeSpec] = {
    WaveShape.SINE: ShapeSpec(0.0, (1.35, 1.55), Fourier(_pure)),
    WaveShape.PARTIAL_SINE: ShapeSpec(0.0, (1.35, 1.55), Fourier(_pure)),
    WaveShape.SQUARE: ShapeSpec(0.0, (0.95, 1.15), Fourier(_square)),
    WaveShape.PULSE: ShapeSpec(0.0, (0.95, 1.15), Fourier(_square)),
    WaveShape.CMOS: ShapeSpec(1.0, (0.95, 1.20), Fourier(_square)),
    WaveShape.TRIANGLE: ShapeSpec(0.0, (1.65, 1.85), Fourier(_triangle)),
    WaveShape.POS_STEP: ShapeSpec(0.0, (1.45, 1.80), Fourier(_sawtooth)),
    WaveShape.NEG_STEP: ShapeSpec(0.0, (1.45, 1.80), Fourier(_sawtooth)),
    WaveShape.HALF_WAVE: ShapeSpec(1.0, (1.20, 1.55), Fourier(_half_wave)),
    WaveShape.FULL_WAVE: ShapeSpec(
        1.0, (1.50, 1.95), Fourier(_full_wave, fundamental_mult=2, n_check=4), peak_mult=2.0
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
_AC_SHAPES = list(SPECS)  # every shape except DC (handled separately)


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
        assert r[1] < 0.45  # 2nd below the sawtooth's 1/2
        assert r[2] < 0.20  # 3rd well below 1/3
        assert r[1] > r[2] > r[3]
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


def _capture(
    gen: JDS6600, scope: DS1054Z, shape: WaveShape, spec: ShapeSpec, f0: float
) -> Waveform:
    gen.configure_channel(
        SignalGeneratorConfig(channel=GEN_CH, waveform=shape, frequency_hz=f0, amplitude_vpp=4.0)
    )
    time.sleep(0.35)
    for ch in (ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
        scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    scope.configure_channel(
        ChannelConfig(channel=SCOPE_CH, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=1.0)
    )
    scope.configure_timebase(_TB[f0])
    scope.configure_acquire(AcquireConfig(memory_depth=_MEM))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=SCOPE_CH, level_v=spec.trigger_v, slope=Slope.RISING),
            sweep=SweepMode.SINGLE,
        )
    )
    return acquire_one_shot(scope, [SCOPE_CH], _TB[f0]).waveforms[SCOPE_CH]


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize("f0", FREQS)
@pytest.mark.parametrize("shape", _AC_SHAPES, ids=lambda s: s.value)
def test_shape_spectral_fingerprint(
    generator: JDS6600, live_scope: DS1054Z, shape: WaveShape, f0: float
) -> None:
    spec = SPECS[shape]
    wf = _capture(generator, live_scope, shape, spec, f0)
    features = describe(wf, psd=True)

    lo, hi = spec.crest
    crest = features.crest_factor
    assert lo <= crest <= hi, f"crest {crest:.2f} outside {spec.crest} for {shape.value}"

    if spec.harmonics is None:  # noise: no clean fundamental at f0
        assert features.peak_hz is None or not (0.95 * f0 <= features.peak_hz <= 1.05 * f0)
    elif spec.peak_mult is not None:
        assert features.peak_hz == pytest.approx(spec.peak_mult * f0, rel=0.02)

    _assert_harmonics(spec.harmonics, wf, f0)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
def test_dc_shape_sets_and_reads_back(generator: JDS6600) -> None:
    """DC has no edge to trigger on and no spectrum; verify the driver selects it."""
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.DC, frequency_hz=1_000.0, amplitude_vpp=1.0
        )
    )
    assert generator.read_channel(GEN_CH).waveform is WaveShape.DC
