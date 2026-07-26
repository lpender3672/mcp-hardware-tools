"""Cross-instrument HIL — L1: an uploaded analytic arb replays as the right shape.

Upload an analytic sine/triangle/square into a slot, play it on the generator, and
measure it on the scope. The gate is the **crest factor** (`describe`) — a cheap,
amplitude-invariant shape discriminator: sine ~=sqrt(2) 1.414, triangle ~=sqrt(3)
1.732, square ~=1.0. The three are well separated and correctly ordered
(square < sine < triangle), which is what proves the upload→replay chain reproduces
*shape*, not just amplitude.

Bench-observed (JDS6600 -> DS1054Z, 1 kHz, 4 Vpp, high-Z):

* sine   crest ~1.43, triangle ~1.75 — within ~0.02 of ideal;
* square crest ~1.16 and Vpp overshoots (~4.5 on a 4.0 request) — the DAC
  reconstruction filter rings on the fast edges. That is a real fidelity limit of
  arb square replay, hence the looser square band here; L2/L3 (harmonics / coherence
  vs the built-in) quantify it properly.

    uv run pytest -m hardware -s

Leaves CH1 on a benign 1 kHz 2 Vpp built-in sine.
"""

from __future__ import annotations

import math

import pytest

from hwtools.analysis.describe import describe
from hwtools.drivers.joyit import JDS6600
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.model.timebase import TimebaseConfig
from hwtools.synth import analytic_arbitrary
from tests.hardware._acquire import acquire_repeating

SCOPE_CH = ChannelId.CH1
GEN_CH = SigGenChannel.CH1
_TB = TimebaseConfig(scale_s_per_div=2e-3)  # ~24 cycles of 1 kHz on screen

# (shape, theoretical crest factor, tolerance). Crest = peak/rms: sine sqrt(2),
# triangle sqrt(3), square 1. The square band is wide to admit the DAC edge-ringing
# overshoot while staying clear of the sine band (sqrt(2) - 0.1 = 1.31 > 1.3).
_SHAPES = [
    (WaveShape.SINE, math.sqrt(2), 0.1),
    (WaveShape.TRIANGLE, math.sqrt(3), 0.1),
    (WaveShape.SQUARE, 1.0, 0.3),
]


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize(("shape", "ideal_crest", "tol"), _SHAPES)
def test_analytic_arb_replays_with_expected_shape(
    generator: JDS6600,
    live_scope: DS1054Z,
    shape: WaveShape,
    ideal_crest: float,
    tol: float,
) -> None:
    caps = generator.capabilities
    assert caps.arb_length is not None
    slot = caps.arb_slots
    n = caps.arb_length.representative_length()
    generator.upload_arbitrary(slot, analytic_arbitrary(shape, points=n))
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, frequency_hz=1_000.0, amplitude_vpp=4.0, arb_slot=slot, enabled=True
        )
    )
    live_scope.configure_channel(
        ChannelConfig(channel=SCOPE_CH, coupling=Coupling.DC, scale_v_per_div=2.0, probe_ratio=1.0)
    )
    live_scope.configure_timebase(_TB)
    live_scope.configure_acquire(AcquireConfig())

    wf = acquire_repeating(live_scope, [SCOPE_CH], _TB, deep=False).waveforms[SCOPE_CH]
    features = describe(wf, psd=True)

    assert features.peak_hz == pytest.approx(1_000.0, rel=0.02)  # replays at the set rate
    assert features.periodicity > 0.8  # a stable, repeating waveform
    assert features.crest_factor == pytest.approx(ideal_crest, abs=tol)  # the right shape
