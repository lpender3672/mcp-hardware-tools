"""HIL: SPI decode round-trips a known transaction through the real scope.

The harness bit-bangs a fixed SPI transaction (clk GP2/CH1, mosi GP3/CH2, cs
GP4/CH3); the scope captures all three lines, and decode_spi must recover the
known bytes. Closes the SPI half of the decode layer's hardware-validation gap.

Probe: CH1->GP2 (clk), CH2->GP3 (mosi), CH3->GP4 (cs), GND->pin 3.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time
from itertools import pairwise

import pytest

from hwtools.decode.spi import SpiParams, decode_spi
from hwtools.decode.threshold import threshold
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.model.waveform import Waveform
from tests.hardware._acquire import acquire_single

CLK, MOSI, CS = ChannelId.CH1, ChannelId.CH2, ChannelId.CH3
# Must match firmware/common/src/protocol.rs SPI_TEST_BYTES.
EXPECTED = [0xA5, 0x3C]


def _digital(wf: Waveform) -> object:
    return threshold(wf, level_v=(wf.vmin + wf.vmax) / 2.0, hysteresis_v=max(wf.vpp * 0.2, 0.2))


@pytest.mark.hardware
def test_spi_decode_round_trips(harness: SerialHarness, live_scope: DS1054Z) -> None:
    harness.start_spi()
    time.sleep(0.3)
    try:
        live_scope.configure_channel(
            ChannelConfig(channel=ChannelId.CH4, scale_v_per_div=1.0, enabled=False)
        )
        for ch in (CLK, MOSI, CS):
            live_scope.configure_channel(
                ChannelConfig(
                    channel=ch, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
            )
        live_scope.configure_acquire(AcquireConfig(memory_depth=30_000))  # legal for 3 channels
        live_scope.configure_timebase(TimebaseConfig(scale_s_per_div=3e-4))
        live_scope.configure_trigger(
            TriggerConfig(
                trigger=EdgeTrigger(source=CS, level_v=1.5, slope=Slope.FALLING),  # CS assert
                sweep=SweepMode.SINGLE,
            )
        )
        cap = acquire_single(live_scope, [CLK, MOSI, CS])  # single-shot, deep RAW read
    finally:
        harness.stop()

    words = decode_spi(
        _digital(cap.waveforms[CLK]),  # type: ignore[arg-type]
        mosi=_digital(cap.waveforms[MOSI]),  # type: ignore[arg-type]
        cs=_digital(cap.waveforms[CS]),  # type: ignore[arg-type]
        params=SpiParams(cpol=0, cpha=0, msb_first=True, cs_active_low=True),
    )
    values = [w.mosi for w in words]
    print(f"\n[spi-hil] decoded mosi={[hex(v) for v in values if v is not None]}")

    # The known transaction appears as a framed pair (a leading partial frame from
    # the pre-trigger window is expected and ignored).
    assert (EXPECTED[0], EXPECTED[1]) in list(pairwise(values))
