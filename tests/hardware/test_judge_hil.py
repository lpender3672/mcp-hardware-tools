"""HIL: the judge detects clipping correctly against the real digitiser.

This is the first recommender-layer behaviour validated on metal. It already
caught a real bug (the ADC digitises ~+/-5 divisions, not the +/-4 the sim
modelled), now fixed by carrying the true saturation rails from the preamble.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.analysis.judge import judge_capture
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig

# The harness ~3.3 V logic signal (with overshoot) is ~5.5 Vpp on the scope, so a
# small V/div clips and a large V/div shows it whole.
_CASES = [(0.1, True), (0.2, True), (1.0, False), (2.0, False)]


@pytest.mark.hardware
def test_judge_clipping_matches_real_digitiser(harness: SerialHarness, live_scope: DS1054Z) -> None:
    harness.start_uart_stream(0xA5, baud=9600)
    time.sleep(0.4)

    live_scope.configure_timebase(TimebaseConfig(scale_s_per_div=2e-3))
    live_scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=1.5, slope=Slope.RISING),
            sweep=SweepMode.AUTO,
        )
    )

    try:
        for scale, expect_clip in _CASES:
            config = ChannelConfig(
                channel=ChannelId.CH1,
                coupling=Coupling.DC,
                scale_v_per_div=scale,
                probe_ratio=10.0,
            )
            live_scope.configure_channel(config)
            live_scope.run()
            time.sleep(0.3)
            cap = live_scope.capture([ChannelId.CH1])
            wf = cap.waveforms[ChannelId.CH1]
            quality = judge_capture(cap, {ChannelId.CH1: config}, live_scope.capabilities)

            assert wf.saturation is not None  # the real driver always reports rails
            low, high = wf.saturation
            print(
                f"\n[judge-hil] {scale} V/div vpp={wf.vpp:.2f} "
                f"sat=({low:.2f}, {high:.2f}) "
                f"clip={quality.clipping[ChannelId.CH1]} (expect {expect_clip})"
            )
            assert quality.clipping[ChannelId.CH1] is expect_clip
    finally:
        harness.stop()
