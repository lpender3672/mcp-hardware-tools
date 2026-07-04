"""Supervised HIL vertical slice: host commands harness -> scope -> decode.

The whole product loop in miniature, end to end on real hardware and now fully
host-driven: the host tells the RP2350 harness firmware to stream a known UART
byte on GP0 (CH1), the scope captures it, and the software decoder must recover
that byte. Excluded from CI (``hardware`` marker):

    uv run pytest -m hardware -s

Prerequisites: the harness firmware is flashed and CH1 is on GP0 (pin 1) at 10x,
GND on pin 3. Override addresses with ``HWTOOLS_SCOPE_HOST`` / ``HWTOOLS_PICO_PORT``.
"""

from __future__ import annotations

import os
import time
from collections import Counter

import pytest

from hwtools.decode.threshold import threshold
from hwtools.decode.uart import UartParams, decode_uart
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350.serial_harness import SerialHarness, find_pico_port
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from tests.hardware._acquire import acquire_one_shot

SCOPE_HOST = os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")

KNOWN_BYTE = 0xA5
BAUD = 9600


@pytest.mark.hardware
def test_host_commanded_uart_round_trips_through_scope() -> None:
    port = os.environ.get("HWTOOLS_PICO_PORT") or find_pico_port()
    if port is None:
        pytest.skip("no Raspberry Pi serial device found")

    harness = SerialHarness(port)
    with harness:
        assert "rp2350" in harness.idn()
        harness.start_uart_stream(KNOWN_BYTE, baud=BAUD)
        time.sleep(0.3)

        scope = DS1054Z.over_tcp(SCOPE_HOST)
        try:
            with scope:
                for ch in (ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
                    scope.configure_channel(
                        ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False)
                    )
                scope.configure_channel(
                    ChannelConfig(
                        channel=ChannelId.CH1,
                        coupling=Coupling.DC,
                        scale_v_per_div=1.0,
                        probe_ratio=10.0,
                    )
                )
                scope.configure_acquire(AcquireConfig(memory_depth=12_000))  # legal for 1 channel
                # 2 ms/div -> deep 12k pts ~= 2 us/sample, ~50 samples/bit at 9600.
                tb = TimebaseConfig(scale_s_per_div=2e-3)
                scope.configure_timebase(tb)
                scope.configure_trigger(
                    TriggerConfig(
                        trigger=EdgeTrigger(source=ChannelId.CH1, level_v=1.5, slope=Slope.FALLING),
                        sweep=SweepMode.SINGLE,
                    )
                )
                # One UART frame is a one-shot event (the stream repeats, triggers fast).
                cap = acquire_one_shot(scope, [ChannelId.CH1], tb)
        finally:
            harness.stop()

    wf = cap.waveforms[ChannelId.CH1]
    mid = (wf.vmin + wf.vmax) / 2.0
    trace = threshold(wf, level_v=mid, hysteresis_v=wf.vpp * 0.2)
    values = [w.value for w in decode_uart(trace, UartParams(baud=BAUD))]
    counts = Counter(values)

    print(f"\n[uart-roundtrip] vpp={wf.vpp:.2f}V bytes={len(values)} dist={dict(counts)}")

    assert values, "no UART words decoded from the capture"
    most_common_value, most_common_n = counts.most_common(1)[0]
    assert most_common_value == KNOWN_BYTE
    assert most_common_n / len(values) >= 0.7
