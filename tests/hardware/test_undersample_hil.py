"""HIL: the loop detects and resolves undersampling on the real scope.

A fast (50 kHz) square at a slow timebase gives ~2 samples per period — visibly
undersampled. capture_until_usable must flag it (judge) and speed the timebase up
(suggest_adjustment B1) until the wave is resolved.

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
from hwtools.session.loop import capture_until_usable

CH2 = ChannelId.CH2
_START_TIMEBASE = 1e-3  # 1 ms/div -> ~2 samples/period at 50 kHz


@pytest.mark.hardware
def test_loop_resolves_undersampling_on_real_scope(
    harness: SerialHarness, live_scope: DS1054Z
) -> None:
    harness.start_square(50_000, duty_pct=50)
    time.sleep(0.3)
    try:
        result = capture_until_usable(
            live_scope,
            channels={
                CH2: ChannelConfig(
                    channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
            },
            timebase=TimebaseConfig(scale_s_per_div=_START_TIMEBASE),
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=1.5, slope=Slope.RISING),
                sweep=SweepMode.NORMAL,
            ),
            max_iterations=8,
        )
    finally:
        harness.stop()

    wf = result.capture.waveforms[CH2]
    spp = measure.samples_per_period(wf)
    print(
        f"\n[undersample-hil] converged={result.converged} "
        f"final dt={wf.dt_s * 1e6:.1f}us spp={spp}"
    )

    assert result.converged
    assert result.assessment.bandwidth_ok  # resolved
    assert wf.dt_s < _START_TIMEBASE / 100  # timebase was sped up (dt = scale/100)
    assert any("undersampled" in a.reason for a in result.adjustments)
