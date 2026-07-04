"""HIL: the ScopeSession tool facade end-to-end on the real scope.

Proves the agent-facing surface works on metal: acquire a real frame into the
store, then run the dense lens (describe) and triage over the *stored* frame by
handle, and decode a real SPI transaction through the session — the same path an
MCP client drives.

Wiring: CH2 square for the analog lens; CH1/CH2/CH3 = SPI clk/mosi/cs for decode.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time
from itertools import pairwise

import pytest

from hwtools.decode.spi import SpiParams
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session import ScopeSession

CH1, CH2, CH3 = ChannelId.CH1, ChannelId.CH2, ChannelId.CH3


@pytest.mark.hardware
def test_session_acquire_describe_triage_on_real_square(
    harness: SerialHarness, live_scope: DS1054Z
) -> None:
    harness.start_square(1_000, duty_pct=50)
    time.sleep(0.3)
    session = ScopeSession(live_scope)
    # Known channel state: only CH2 on, so the memory depth is legal for 1 channel.
    for ch in (CH1, CH3, ChannelId.CH4):
        live_scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    try:
        result = session.acquire(
            channels={
                CH2: ChannelConfig(
                    channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
            },
            timebase=TimebaseConfig(scale_s_per_div=5e-4),
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH2, level_v=1.5, slope=Slope.RISING)
            ),
            acquire_cfg=AcquireConfig(memory_depth=120_000),
            sweep=SweepMode.SINGLE,
        )
    finally:
        harness.stop()

    assert result.triggered
    cid = result.capture_id
    assert cid is not None

    # Dense lens over the stored frame, by handle.
    char = session.describe(CH2, capture_id=cid, value_bins=32, joint=True, joint_bins=16)
    print(
        f"\n[session-hil] peak={char.peak_hz} periodicity={char.periodicity:.2f} "
        f"crest={char.crest_factor:.2f} n_levels={char.n_levels} edges/s={char.edge_rate_hz:.0f}"
    )
    assert char.peak_hz is not None and abs(char.peak_hz - 1_000.0) / 1_000.0 < 0.1
    assert char.periodicity > 0.5
    assert char.value_histogram is not None and char.joint is not None

    # Triage the same stored frame; a clean logic square is periodic and low-flatness.
    verdict = session.triage(capture_id=cid)[CH2]
    assert verdict.signal_class.value in ("digital", "sine", "modulated")

    # The one-shot frame is kept (unrepeatable), visible in the audit.
    assert any(i.capture_id == cid and i.kept for i in session.list_captures())


@pytest.mark.hardware
def test_session_decode_spi_round_trips(harness: SerialHarness, live_scope: DS1054Z) -> None:
    harness.start_spi()
    time.sleep(0.3)
    session = ScopeSession(live_scope)
    try:
        live_scope.configure_channel(
            ChannelConfig(channel=ChannelId.CH4, scale_v_per_div=1.0, enabled=False)
        )
        result = session.acquire(
            channels={
                ch: ChannelConfig(
                    channel=ch, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0
                )
                for ch in (CH1, CH2, CH3)
            },
            timebase=TimebaseConfig(scale_s_per_div=5e-4),  # 1-2-5 valid; 300us would snap
            trigger=TriggerConfig(
                trigger=EdgeTrigger(source=CH3, level_v=1.5, slope=Slope.FALLING)  # CS assert
            ),
            acquire_cfg=AcquireConfig(memory_depth=30_000),
            sweep=SweepMode.SINGLE,
        )
    finally:
        harness.stop()

    assert result.triggered
    words = session.decode_spi(
        clk=CH1, mosi=CH2, cs=CH3, params=SpiParams(cpol=0, cpha=0, msb_first=True),
        capture_id=result.capture_id or "latest",
    )
    values = [w.mosi for w in words]
    print(f"\n[session-hil] spi mosi={[hex(v) for v in values if v is not None]}")
    assert (0xA5, 0x3C) in list(pairwise(values))
