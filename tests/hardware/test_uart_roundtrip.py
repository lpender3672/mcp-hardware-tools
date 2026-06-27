"""Supervised HIL vertical slice: harness firmware -> scope -> decode round-trip.

The whole product loop in miniature, end to end on real hardware. The RP2350
harness firmware (``firmware/``, flashed via BOOTSEL) free-runs a PIO UART-TX
emitting a known byte on GP0; the scope captures CH1 and the software decoder
must recover that byte. Excluded from CI (``hardware`` marker):

    uv run pytest -m hardware -s

Prerequisites: the firmware is built and flashed, and scope CH1 is on GP0
(pin 1) at 10x, GND on pin 3. Override the scope address with
``HWTOOLS_SCOPE_HOST``.
"""

from __future__ import annotations

import os
import time
from collections import Counter

import pytest

from hwtools.decode.threshold import threshold
from hwtools.decode.uart import UartParams, decode_uart
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig

SCOPE_HOST = os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")

# Must match the harness firmware (firmware/src/main.rs).
KNOWN_BYTE = 0xA5
BAUD = 9600


@pytest.mark.hardware
def test_firmware_uart_byte_round_trips_through_scope() -> None:
    scope = DS1054Z.over_tcp(SCOPE_HOST)
    with scope:
        scope.configure_channel(
            ChannelConfig(
                channel=ChannelId.CH1,
                coupling=Coupling.DC,
                scale_v_per_div=1.0,
                probe_ratio=10.0,
            )
        )
        # 2 ms/div -> ~5 samples/bit at 9600 and several bytes in the window.
        scope.configure_timebase(TimebaseConfig(scale_s_per_div=2e-3))
        scope.configure_trigger(
            TriggerConfig(
                trigger=EdgeTrigger(source=ChannelId.CH1, level_v=1.5, slope=Slope.FALLING),
                sweep=SweepMode.AUTO,
            )
        )
        scope.run()
        time.sleep(0.5)
        cap = scope.capture([ChannelId.CH1])

    wf = cap.waveforms[ChannelId.CH1]
    mid = (wf.vmin + wf.vmax) / 2.0
    trace = threshold(wf, level_v=mid, hysteresis_v=wf.vpp * 0.2)
    values = [w.value for w in decode_uart(trace, UartParams(baud=BAUD))]
    counts = Counter(values)

    print(f"\n[uart-roundtrip] vpp={wf.vpp:.2f}V bytes={len(values)} dist={dict(counts)}")

    assert values, "no UART words decoded from the capture"
    most_common_value, most_common_n = counts.most_common(1)[0]
    assert most_common_value == KNOWN_BYTE
    # The known byte must dominate (a single window-edge partial is tolerated).
    assert most_common_n / len(values) >= 0.7
