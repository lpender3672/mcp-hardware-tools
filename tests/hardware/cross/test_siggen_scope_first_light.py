"""Cross-instrument HIL: the JDS6600 drives, the DS1054Z measures.

This is the *strong* validation the generator's own read-backs can't give — the
siggen's only observable is what you wrote (a tautology), but its actual output is
ground-truthed here by an independent instrument digitising it. It closes the
HIL plan's dropped "H1 — JDS6600 online" and is the anchor Thread A's noise suite
builds on (drive a known signal, characterise it with `describe`).

Wiring: JDS6600 CH1 -> scope CH1 via BNC coax (**probe ratio 1x**, not a 10x probe).

Bench-confirmed facts pinned here:
* frequency tracks the request to <0.1 % (centi-Hz DDS);
* **amplitude reads as true Vpp on the scope's 1 MΩ high-Z input** — the JDS6600
  does *not* double into high-Z, so 4.0 Vpp set reads 4.0 Vpp measured (no 50 Ω
  load-factor correction is needed in the model);
* a SINE reads back as a clean sine (crest factor ~=sqrt(2), high periodicity).

    uv run pytest -m hardware -s

Leaves CH1 on a benign 1 kHz 2 Vpp sine.
"""

from __future__ import annotations

import pytest

from hwtools.analysis.describe import describe
from hwtools.drivers.joyit import JDS6600
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape
from hwtools.model.timebase import TimebaseConfig
from tests.hardware._acquire import acquire_repeating

SCOPE_CH = ChannelId.CH1
GEN_CH = SigGenChannel.CH1
_TB = TimebaseConfig(scale_s_per_div=500e-6)  # 6 ms window -> ~60 cycles at 10 kHz


def _view_ch1(scope: DS1054Z, scale_v_per_div: float) -> None:
    scope.configure_channel(
        ChannelConfig(
            channel=SCOPE_CH, coupling=Coupling.DC, scale_v_per_div=scale_v_per_div, probe_ratio=1.0
        )
    )
    scope.configure_timebase(_TB)
    scope.configure_acquire(AcquireConfig())


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
def test_sine_frequency_and_amplitude_measured_on_scope(
    generator: JDS6600, live_scope: DS1054Z
) -> None:
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=10_000.0, amplitude_vpp=4.0
        )
    )
    _view_ch1(live_scope, scale_v_per_div=2.0)  # 4 Vpp over 8 div

    wf = acquire_repeating(live_scope, [SCOPE_CH], _TB, deep=False).waveforms[SCOPE_CH]
    features = describe(wf, psd=True)

    assert features.peak_hz == pytest.approx(10_000.0, rel=0.02)
    assert features.vpp == pytest.approx(4.0, rel=0.1)  # true Vpp into high-Z, no doubling
    assert features.crest_factor == pytest.approx(1.414, abs=0.15)  # a real sine
    assert features.periodicity > 0.8


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize("frequency_hz", [1_000.0, 50_000.0])
def test_frequency_tracks_the_request(
    generator: JDS6600, live_scope: DS1054Z, frequency_hz: float
) -> None:
    """Two decades of frequency read back on the scope within a few percent."""
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, waveform=WaveShape.SINE, frequency_hz=frequency_hz, amplitude_vpp=4.0
        )
    )
    # ~40 cycles on screen at whatever frequency, so `describe` resolves the tone.
    tb = TimebaseConfig(scale_s_per_div=(40.0 / frequency_hz) / 12.0)
    live_scope.configure_channel(
        ChannelConfig(channel=SCOPE_CH, coupling=Coupling.DC, scale_v_per_div=2.0, probe_ratio=1.0)
    )
    live_scope.configure_timebase(tb)
    live_scope.configure_acquire(AcquireConfig())

    wf = acquire_repeating(live_scope, [SCOPE_CH], tb, deep=False).waveforms[SCOPE_CH]
    assert describe(wf, psd=True).peak_hz == pytest.approx(frequency_hz, rel=0.05)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
def test_dc_offset_measured_on_scope(generator: JDS6600, live_scope: DS1054Z) -> None:
    """A DC bias set on the generator shows up as the captured mean level."""
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH,
            waveform=WaveShape.SINE,
            frequency_hz=10_000.0,
            amplitude_vpp=2.0,
            offset_v=1.0,
        )
    )
    _view_ch1(live_scope, scale_v_per_div=1.0)

    wf = acquire_repeating(live_scope, [SCOPE_CH], _TB, deep=False).waveforms[SCOPE_CH]
    assert describe(wf).dc_level == pytest.approx(1.0, abs=0.2)
