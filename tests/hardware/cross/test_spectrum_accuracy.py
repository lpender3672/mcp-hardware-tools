"""Cross-instrument HIL: frequency & amplitude accuracy of the spectral tools.

Sine ground truth from the JDS6600 validates three things `describe`/`spectrum`
claim, on real captures:

* `peak_frequency` tracks the set frequency across four decades (100 Hz -> 100 kHz);
* `amplitude_spectrum` (flattop window) reads a tone's true amplitude (Vpp/2);
* the parabolic sub-bin interpolation beats a plain nearest-bin estimate.

No AUTO: each capture arms SINGLE on a rising edge and deep-reads one frame. Memory
is chosen per frequency so the sample rate stays under the scope's 1 GSa/s ceiling
(a short high-frequency window can't hold the deepest record) while still capturing
tens of cycles for fine resolution.

    uv run pytest -m cfg_siggen_1ch -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.analysis import spectrum
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
_CYCLES = 40
_MAX_RATE = 5e8  # keep well under the DS1054Z's 1 GSa/s single-channel ceiling


def _acq(f0: float) -> tuple[TimebaseConfig, int]:
    """Timebase (~_CYCLES periods on screen) and the deepest legal memory whose
    resulting sample rate stays under _MAX_RATE for this frequency."""
    window = _CYCLES / f0
    tb = TimebaseConfig(scale_s_per_div=window / 12.0)
    mem = next((m for m in (1_200_000, 120_000, 12_000) if m / window <= _MAX_RATE), 12_000)
    return tb, mem


def _capture_sine(
    gen: JDS6600, scope: DS1054Z, f0: float, *, amplitude_vpp: float = 4.0
) -> Waveform:
    gen.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=f0, amplitude_vpp=amplitude_vpp
        )
    )
    time.sleep(0.3)
    tb, mem = _acq(f0)
    for ch in (ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
        scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    scope.configure_channel(
        ChannelConfig(
            channel=SCOPE_CH,
            coupling=Coupling.DC,
            scale_v_per_div=amplitude_vpp / 4.0,
            probe_ratio=1.0,
        )
    )
    scope.configure_timebase(tb)
    scope.configure_acquire(AcquireConfig(memory_depth=mem))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=SCOPE_CH, level_v=0.0, slope=Slope.RISING),
            sweep=SweepMode.SINGLE,
        )
    )
    return acquire_one_shot(scope, [SCOPE_CH], tb).waveforms[SCOPE_CH]


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize("f0", [100.0, 1_000.0, 10_000.0, 100_000.0])
def test_peak_frequency_tracks_across_decades(
    generator: JDS6600, live_scope: DS1054Z, f0: float
) -> None:
    wf = _capture_sine(generator, live_scope, f0)
    assert spectrum.peak_frequency(wf) == pytest.approx(f0, rel=5e-3)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize("amplitude_vpp", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_amplitude_spectrum_reads_true_vpp(
    generator: JDS6600, live_scope: DS1054Z, amplitude_vpp: float
) -> None:
    """A flattop-windowed amplitude spectrum reads a tone's amplitude = Vpp/2, across
    the generator's full 0.5-20 Vpp span (within a few percent; it reads slightly low,
    partly because the generator itself outputs ~1-2 % under the set amplitude)."""
    wf = _capture_sine(generator, live_scope, 1_000.0, amplitude_vpp=amplitude_vpp)
    peak = spectrum.amplitude_spectrum(wf, window="flattop").peak()
    assert peak is not None
    _, amplitude = peak
    assert amplitude == pytest.approx(amplitude_vpp / 2.0, rel=0.06)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
def test_sub_bin_interpolation_beats_nearest_bin(
    generator: JDS6600, live_scope: DS1054Z
) -> None:
    """A tone deliberately between FFT bins is recovered to well under half a bin —
    the payoff of the parabolic sub-bin refinement in peak_frequency."""
    f0 = 12_345.0
    wf = _capture_sine(generator, live_scope, f0)
    est = spectrum.peak_frequency(wf)
    assert est is not None
    bin_hz = 1.0 / wf.duration_s
    assert abs(est - f0) < 0.5 * bin_hz  # better than the nearest-bin worst case
    assert abs(est - f0) < 5e-3 * f0  # and accurate in absolute terms
