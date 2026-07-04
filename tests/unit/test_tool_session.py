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
from hwtools.tools.server import build_server
from hwtools.tools.session import ScopeSession

CH1 = ChannelId.CH1


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


def test_build_server_registers_the_tool_surface() -> None:
    session = _session()
    server = build_server(session)
    import anyio

    names = {t.name for t in anyio.run(server.list_tools)}
    assert {"acquire", "describe", "triage", "decode_uart", "decode_spi", "decode_i2c"} <= names
