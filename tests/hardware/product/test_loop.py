"""HIL: the iterative capture_until_usable loop converges on the real scope.

Complements the single-stage autoset HIL test by exercising the iterative path on
metal: from a clipped start with a NORMAL-sweep trigger set out of range, the loop
must grow the vertical scale (suggest_adjustment A1) and move the trigger level to
the source midline (T1), reaching a usable, triggered capture (L1).

Notable: the feared NORMAL-sweep "untriggered capture returns stale data" problem
does not occur on the DS1000Z — it returns the live acquisition, so the loop can
read the source midline to fix the trigger even before it triggers.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session.loop import capture_until_usable

CH2 = ChannelId.CH2


@pytest.mark.hardware
def test_capture_until_usable_converges_normal_sweep(
    harness: SerialHarness, live_scope: DS1054Z
) -> None:
    harness.start_square(1_000, duty_pct=50)
    time.sleep(0.3)
    try:
        result = capture_until_usable(
            live_scope,
            channels={
                CH2: ChannelConfig(
                    channel=CH2, coupling=Coupling.DC, scale_v_per_div=0.1, probe_ratio=10.0
                )
            },
            timebase=TimebaseConfig(scale_s_per_div=5e-4),
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=9.0, slope=Slope.RISING),
                sweep=SweepMode.NORMAL,
            ),
            max_iterations=12,
        )
    finally:
        harness.stop()

    wf = result.capture.waveforms[CH2]
    reasons = " | ".join(a.reason for a in result.adjustments)
    print(f"\n[loop-hil] converged={result.converged} iters={result.iterations} | {reasons}")

    assert result.converged
    assert result.assessment.usable
    assert result.capture.triggered
    assert not result.assessment.channels[CH2].clipping
    assert wf.vpp > 5.0  # recovered the true signal from the clipped start
    # Both the clipping (A1) and trigger-level (T1) heuristics had to fire.
    assert any("clipping" in a.reason for a in result.adjustments)
    assert any("trigger level" in a.reason for a in result.adjustments)
