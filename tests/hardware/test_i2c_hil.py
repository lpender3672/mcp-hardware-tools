"""HIL: I2C decode round-trips a known transaction through the real scope.

The harness bit-bangs a fixed I2C write (scl GP2/CH1, sda GP3/CH2) — START,
address+W, data bytes (each self-ACKed), STOP. The scope captures both lines and
decode_i2c must recover the address and payload. Closes the I2C half of the
decode layer's hardware-validation gap.

Probe: CH1->GP2 (scl), CH2->GP3 (sda), GND->pin 3.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time

import pytest

from hwtools.decode.i2c import decode_i2c
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

SCL, SDA = ChannelId.CH1, ChannelId.CH2
# Must match firmware/common/src/protocol.rs I2C_TEST_ADDR / I2C_TEST_DATA.
EXPECTED_ADDR = 0x50
EXPECTED_DATA = [0xDE, 0xAD]


def _digital(wf: Waveform) -> object:
    return threshold(wf, level_v=(wf.vmin + wf.vmax) / 2.0, hysteresis_v=max(wf.vpp * 0.2, 0.2))


@pytest.mark.hardware
def test_i2c_decode_round_trips(harness: SerialHarness, live_scope: DS1054Z) -> None:
    harness.start_i2c()
    time.sleep(0.3)
    try:
        # Only SCL/SDA on: keep exactly two analog channels enabled so the legal
        # memory-depth set is the 2-channel one (and a stale 3rd channel can't make
        # the depth illegal and beep the scope).
        for ch in (ChannelId.CH3, ChannelId.CH4):
            live_scope.configure_channel(
                ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False)
            )
        for ch in (SCL, SDA):
            live_scope.configure_channel(
                ChannelConfig(
                    channel=ch, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
            )
        # 1 ms/div -> 12 ms window spans ~3.7 transaction periods, so a full
        # START..STOP is always present whatever SDA edge the trigger caught.
        # 60k-point deep memory over 12 ms is ~0.2 us/sample (~100 samples per SCL
        # phase): the on-screen NORMal trace is only 1200 points (~10 us/sample),
        # too coarse to resolve the SDA-mid-low transitions and aliases them into
        # phantom START/STOPs. 60000 is a legal 2-channel record length; capture()
        # defaults to deep=True so the full memory downloads over the raw socket.
        live_scope.configure_acquire(AcquireConfig(memory_depth=60_000))
        live_scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3))
        live_scope.configure_trigger(
            TriggerConfig(
                trigger=EdgeTrigger(source=SDA, level_v=1.5, slope=Slope.FALLING),
                sweep=SweepMode.SINGLE,
            )
        )
        cap = acquire_single(live_scope, [SCL, SDA])  # single-shot, deep RAW read
    finally:
        harness.stop()

    transactions = decode_i2c(
        _digital(cap.waveforms[SDA]),  # type: ignore[arg-type]
        _digital(cap.waveforms[SCL]),  # type: ignore[arg-type]
    )
    summary = [(hex(t.address) if t.address is not None else None, t.data) for t in transactions]
    print(f"\n[i2c-hil] transactions={summary}")

    matching = [t for t in transactions if t.address == EXPECTED_ADDR and t.data == EXPECTED_DATA]
    assert matching, f"no matching transaction in {summary}"
    assert matching[0].read is False
    assert all(b.ack for b in matching[0].bytes)
