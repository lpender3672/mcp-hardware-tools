"""HIL: single-shot acquisition on the real scope.

Validates the SINGLE acquisition path on metal: arm a single acquisition, let the
trigger fire (here on the Pico square on CH2 as a convenient edge source),
capture exactly one frame, and confirm the scope is STOPped afterwards rather than
free-running. This is the mechanism a true non-repeating event needs — set up
correctly, arm once, catch it.

(A genuinely non-repeating source — a one-shot Pico pulse — is a future addition;
the arm/wait/capture/stop mechanism is identical and is what this verifies.)

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session.loop import capture_single

CH2 = ChannelId.CH2


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_single_shot_captures_one_frame_and_stops(
    harness: SerialHarness, live_scope: DS1054Z
) -> None:
    harness.start_square(1_000, duty_pct=50)
    time.sleep(0.3)
    try:
        result = capture_single(
            live_scope,
            channels={
                CH2: ChannelConfig(
                    channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
            },
            timebase=TimebaseConfig(scale_s_per_div=2e-4),
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=1.5, slope=Slope.RISING),
                sweep=SweepMode.SINGLE,
            ),
        )
        # After a single acquisition the scope is stopped, not free-running.
        post_status = live_scope.trigger_status()
    finally:
        harness.stop()

    print(f"\n[single-shot-hil] triggered={result.triggered} post_status={post_status.value}")

    assert result.triggered
    assert result.capture is not None
    assert result.capture.waveforms[CH2].vpp > 3.0  # the event was captured
    assert post_status is TriggerStatus.STOP  # single-shot stopped, did not re-arm
