"""The ScopeSession facade over the simulated scope: acquire into the store, then
run every lens (describe/triage/judge/recommend/decode) over the stored frame by
handle, plus store management and the MCP server wiring."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.drivers.simulated import SimulatedScope, Sine, Square
from hwtools.drivers.simulated.signals import SimSignal
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode
from hwtools.model.reading import AcquireResult
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session import ScopeSession
from hwtools.tools.server import build_server

CH1, CH2, CH3 = ChannelId.CH1, ChannelId.CH2, ChannelId.CH3


def _session(signal: SimSignal | None = None) -> ScopeSession:
    sig = signal or Sine(amplitude_v=1.0, frequency_hz=1_000.0)
    return ScopeSession(SimulatedScope({CH1: sig}))


def _acquire(
    session: ScopeSession, *, level: float = 0.0, sweep: SweepMode = SweepMode.SINGLE
) -> AcquireResult:
    return session.acquire(
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.3)},
        timebase=TimebaseConfig(scale_s_per_div=200e-6),
        trigger=TriggerConfig(
            trigger=EdgeTrigger(source=CH1, level_v=level, slope=Slope.RISING)
        ),
        sweep=sweep,
    )


def test_acquire_stores_frame_and_returns_reading() -> None:
    session = _session()
    result = _acquire(session)
    assert result.triggered
    assert result.capture_id == "cap-1"
    assert result.channels[CH1].vpp > 0
    assert len(session.store) == 1


def test_lenses_run_over_the_stored_frame_by_handle() -> None:
    session = _session()
    result = _acquire(session)
    cid = result.capture_id
    assert cid is not None

    char = session.describe(CH1, capture_id=cid, value_bins=16, joint=True, joint_bins=8)
    assert char.peak_hz is not None
    assert char.value_histogram is not None and len(char.value_histogram.counts) == 16
    assert char.joint is not None

    verdicts = session.triage(capture_id=cid)
    assert CH1 in verdicts

    # "latest" resolves without threading the id.
    assert session.describe(CH1).channel is CH1


def test_describe_missing_channel_raises_loudly() -> None:
    session = _session()
    _acquire(session)
    with pytest.raises(KeyError, match="CH2 not in capture"):
        session.describe(ChannelId.CH2)


def test_recommend_is_advisory_over_a_measurement_frame() -> None:
    session = _session()
    _acquire(session, sweep=SweepMode.AUTO)
    setup = session.recommend(
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=5.0)},
        timebase=TimebaseConfig(scale_s_per_div=1e-3),
        trigger=TriggerConfig(trigger=EdgeTrigger(source=CH1, level_v=0.0)),
    )
    assert setup.channels[CH1].scale_v_per_div > 0  # sized to fill; nothing applied


def test_store_management() -> None:
    session = _session()
    _acquire(session, sweep=SweepMode.AUTO)  # ephemeral
    r2 = _acquire(session, sweep=SweepMode.SINGLE)  # one-shot -> kept
    assert r2.capture_id is not None
    infos = session.list_captures()
    assert any(i.kept for i in infos)
    session.keep(r2.capture_id, label="golden")
    session.drop(r2.capture_id)
    assert all(i.capture_id != r2.capture_id for i in session.list_captures())


def test_digital_decode_over_a_stored_square() -> None:
    # A clean logic-level square thresholds and yields transitions the decoders use.
    session = ScopeSession(SimulatedScope({CH1: Square(amplitude_v=1.6, frequency_hz=1_000.0)}))
    session.acquire(
        channels={CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.5)},
        timebase=TimebaseConfig(scale_s_per_div=2e-3),
        trigger=TriggerConfig(trigger=EdgeTrigger(source=CH1, level_v=0.0)),
        sweep=SweepMode.SINGLE,
    )
    trace = session._digital("latest", CH1)  # exercising the auto-thresholder
    assert int(np.count_nonzero(np.diff(trace.levels))) > 0


def test_session_capabilities_judge_and_clear() -> None:
    session = _session()
    assert session.capabilities.n_channels == 4  # forwards the scope's capabilities
    _acquire(session, sweep=SweepMode.AUTO)
    result = session.judge({CH1: ChannelConfig(channel=CH1, scale_v_per_div=0.3)})
    assert CH1 in result.channels  # re-judges a stored frame against a config
    session.clear()
    assert len(session.store) == 0


def test_build_server_registers_the_tool_surface() -> None:
    session = _session()
    server = build_server(session)
    import anyio

    names = {t.name for t in anyio.run(server.list_tools)}
    assert {"acquire", "describe", "triage", "decode_uart", "decode_spi", "decode_i2c"} <= names


def test_server_tool_bodies_execute_over_the_session() -> None:
    """Drive each MCP tool through call_tool against a sim-backed session, so the
    server binding (not just registration) is exercised and locked against regression."""
    import anyio

    sq = Square(amplitude_v=1.6, frequency_hz=1_000.0)
    session = ScopeSession(SimulatedScope({CH1: sq, CH2: sq, CH3: sq}))
    server = build_server(session)

    def run(name: str, args: dict[str, object]) -> object:
        return anyio.run(server.call_tool, name, args)

    # acquire (AUTO always yields a frame on the sim) -> stores a frame
    run("acquire", {"channel": 1, "scale_v_per_div": 0.5, "timebase_s_per_div": 2e-3,
                    "trigger_level_v": 0.0, "sweep": "AUTO"})
    assert len(session.store) == 1
    run("describe", {"channel": 1, "value_bins": 8, "psd": True})
    run("triage", {})
    run("decode_uart", {"channel": 1, "baud": 1000.0})
    run("list_captures", {})

    # A multi-channel frame for the SPI/I2C tool bodies (the acquire tool is
    # single-channel, so stage it through the session directly).
    session.acquire(
        channels={ch: ChannelConfig(channel=ch, scale_v_per_div=0.5) for ch in (CH1, CH2, CH3)},
        timebase=TimebaseConfig(scale_s_per_div=2e-3),
        trigger=TriggerConfig(trigger=EdgeTrigger(source=CH1, level_v=0.0)),
        sweep=SweepMode.SINGLE,
    )
    run("decode_spi", {"clk": 1, "mosi": 2, "cs": 3})
    run("decode_i2c", {"sda": 2, "scl": 1})

    # store-management tool bodies
    cid = session.store.latest_id()
    assert cid is not None
    run("keep_capture", {"capture_id": cid, "label": "golden"})
    assert session.store.info(cid).kept
    run("drop_capture", {"capture_id": cid})
    assert all(i.capture_id != cid for i in session.list_captures())
