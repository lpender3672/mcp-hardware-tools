"""HIL: the Pico square-wave source produces the commanded frequency and duty.

The harness PWM square on GP1 (CH2) is the clean periodic ground-truth source for
the recommender HIL tests, so validate it directly: command a frequency/duty,
capture it, and confirm the measured values match.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hwtools.analysis import measure
from hwtools.decode.threshold import threshold
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from tests.hardware._acquire import acquire_one_shot

CH2 = ChannelId.CH2


@pytest.mark.hardware
@pytest.mark.parametrize(("freq_hz", "duty_pct"), [(1_000, 50), (5_000, 25), (2_000, 75)])
def test_square_source_matches_command(
    harness: SerialHarness, live_scope: DS1054Z, freq_hz: int, duty_pct: int
) -> None:
    harness.start_square(freq_hz, duty_pct=duty_pct)
    time.sleep(0.3)
    try:
        for ch in (ChannelId.CH1, ChannelId.CH3, ChannelId.CH4):
            live_scope.configure_channel(
                ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False)
            )
        live_scope.configure_channel(
            ChannelConfig(channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0)
        )
        live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))  # legal for 1 channel
        tb = TimebaseConfig(scale_s_per_div=(4.0 / freq_hz) / 12.0)
        live_scope.configure_timebase(tb)
        live_scope.configure_trigger(
            TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=1.5, slope=Slope.RISING),
                sweep=SweepMode.SINGLE,
            )
        )
        # The square triggers reliably on its rising edge: one-shot deep capture.
        wf = acquire_one_shot(live_scope, [CH2], tb).waveforms[CH2]
    finally:
        harness.stop()

    measured_freq = measure.frequency(wf)
    trace = threshold(wf, level_v=(wf.vmin + wf.vmax) / 2.0, hysteresis_v=wf.vpp * 0.2)
    measured_duty = 100.0 * np.count_nonzero(trace.levels) / trace.n

    print(
        f"\n[square-hil] req {freq_hz}Hz {duty_pct}% "
        f"-> {measured_freq:.0f}Hz {measured_duty:.0f}%"
    )

    assert measured_freq is not None
    assert abs(measured_freq - freq_hz) / freq_hz < 0.05
    assert abs(measured_duty - duty_pct) < 6.0
