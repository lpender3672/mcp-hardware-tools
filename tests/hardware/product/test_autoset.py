"""HIL: autoset converges on the real scope in one recommendation.

The headline self-correction test on metal. From a deliberately clipped and
mistriggered start, against the Pico square on CH2, ``autoset`` must take a wide
measurement, make a single recommendation, and land on a usable capture that
recovers the true signal — correct scale (centred, ~60% fill, not clipped),
timebase (a few periods on screen), and trigger level (the source midline).

Validates recommend states R2/R5/R8 + AS1 and the offset-centring convention
(screen window centred on the signal). Excluded from CI.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.analysis import measure
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session.loop import autoset

CH2 = ChannelId.CH2
SQUARE_HZ = 1_000


@pytest.mark.hardware
def test_autoset_converges_on_real_scope(harness: SerialHarness, live_scope: DS1054Z) -> None:
    harness.start_square(SQUARE_HZ, duty_pct=50)
    time.sleep(0.3)
    try:
        result = autoset(
            live_scope,
            # clipped (0.1 V/div) and badly triggered (9 V, out of range) start:
            channels={
                CH2: ChannelConfig(
                    channel=CH2, coupling=Coupling.DC, scale_v_per_div=0.1, probe_ratio=10.0
                )
            },
            timebase=TimebaseConfig(scale_s_per_div=2e-3),
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=9.0, slope=Slope.RISING),
                sweep=SweepMode.NORMAL,
            ),
        )
    finally:
        harness.stop()

    wf = result.capture.waveforms[CH2]
    config = result.setup.channels[CH2]
    freq = measure.frequency(wf)
    assert wf.saturation is not None
    screen_centre = (wf.saturation[0] + wf.saturation[1]) / 2.0
    signal_mid = (wf.vmin + wf.vmax) / 2.0

    fill = result.assessment.channels[CH2].fill_fraction
    trig_level = result.setup.trigger.trigger.level_v
    centre_err = screen_centre - signal_mid
    print(
        f"\n[autoset-hil] converged={result.converged} widen={result.widen_steps} "
        f"scale={config.scale_v_per_div:.2f} vpp={wf.vpp:.2f} fill={fill:.2f} freq={freq} "
        f"tb={result.setup.timebase.scale_s_per_div * 1e6:.0f}us "
        f"trig={trig_level:.2f} centre_err={centre_err:+.2f}"
    )

    # Converged in one recommendation, usable, not clipped.
    assert result.converged
    assert result.assessment.usable
    assert result.widen_steps == 0
    assert not result.assessment.channels[CH2].clipping
    # Recovered the true ~5.9 Vpp signal from the clipped 0.1 V/div start (R2).
    assert wf.vpp > 5.0
    assert 0.4 < result.assessment.channels[CH2].fill_fraction < 0.75
    # Timebase shows a few periods of the measured frequency (R5).
    assert freq is not None and abs(freq - SQUARE_HZ) / SQUARE_HZ < 0.1
    assert 2e-4 < result.setup.timebase.scale_s_per_div < 5e-4
    # Trigger level sits within the signal (R8) and the window is centred on it.
    assert wf.vmin < result.setup.trigger.trigger.level_v < wf.vmax
    assert abs(screen_centre - signal_mid) < 0.3 * wf.vpp
