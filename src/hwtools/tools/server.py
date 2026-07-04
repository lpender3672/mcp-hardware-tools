"""The MCP server — a thin binding of :class:`ScopeSession` onto agent-callable tools.

Every tool builds the model value objects from agent-friendly scalar arguments and
delegates to the session; the real behaviour lives in :mod:`hwtools.tools.session`.
``build_server`` takes a session so it can be driven by the simulated scope in
tests; ``main`` wires a real DS1054Z from the environment and serves over stdio.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from hwtools.decode.spi import SpiParams
from hwtools.decode.uart import Parity, UartParams
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.session import ScopeSession


def build_server(session: ScopeSession, *, name: str = "hwtools-scope") -> FastMCP:
    """Wire a :class:`ScopeSession` onto MCP tools. Pure — no I/O at build time."""
    mcp = FastMCP(name)

    @mcp.tool()
    def acquire(
        channel: int,
        scale_v_per_div: float,
        timebase_s_per_div: float,
        trigger_level_v: float,
        trigger_slope: str = "RISING",
        sweep: str = "SINGLE",
        probe_ratio: float = 10.0,
        coupling: str = "DC",
    ) -> dict[str, object]:
        """Acquire one channel and return the decision-ready reading (handle + measurements).

        ``sweep`` SINGLE captures one triggered event (honest miss if it never
        triggers); AUTO free-runs for a persistent signal. The returned
        ``capture_id`` addresses the stored frame for describe/decode/triage.
        """
        ch = ChannelId(channel)
        cfg = {
            ch: ChannelConfig(
                channel=ch,
                scale_v_per_div=scale_v_per_div,
                probe_ratio=probe_ratio,
                coupling=Coupling(coupling),
            )
        }
        trig = TriggerConfig(
            trigger=EdgeTrigger(source=ch, level_v=trigger_level_v, slope=Slope(trigger_slope))
        )
        result = session.acquire(
            channels=cfg,
            timebase=TimebaseConfig(scale_s_per_div=timebase_s_per_div),
            trigger=trig,
            sweep=SweepMode(sweep),
        )
        return result.model_dump()

    @mcp.tool()
    def describe(
        channel: int,
        capture_id: str = "latest",
        value_bins: int | None = None,
        time_bins: int | None = None,
        psd: bool = False,
        freq_bins: int | None = None,
        joint: bool = False,
        joint_bins: int | None = None,
    ) -> dict[str, object]:
        """Dense characterisation of a stored channel: feature vector + opt-in views."""
        return session.describe(
            ChannelId(channel),
            capture_id=capture_id,
            value_bins=value_bins,
            time_bins=time_bins,
            psd=psd,
            freq_bins=freq_bins,
            joint=joint,
            joint_bins=joint_bins,
        ).model_dump()

    @mcp.tool()
    def triage(capture_id: str = "latest") -> dict[str, object]:
        """Classify every channel of a stored frame (unknown-signal front door)."""
        return {ch.name: c.model_dump() for ch, c in session.triage(capture_id=capture_id).items()}

    @mcp.tool()
    def decode_uart(
        channel: int,
        baud: float,
        capture_id: str = "latest",
        bits: int = 8,
        parity: str = "NONE",
        stop_bits: float = 1.0,
    ) -> list[dict[str, object]]:
        """Decode UART on a stored channel into words."""
        params = UartParams(baud=baud, bits=bits, parity=Parity(parity), stop_bits=stop_bits)
        words = session.decode_uart(ChannelId(channel), params=params, capture_id=capture_id)
        return [w.model_dump() for w in words]

    @mcp.tool()
    def decode_spi(
        clk: int,
        mosi: int | None = None,
        miso: int | None = None,
        cs: int | None = None,
        cpol: int = 0,
        cpha: int = 0,
        bits: int = 8,
        capture_id: str = "latest",
    ) -> list[dict[str, object]]:
        """Decode SPI on stored channels into words."""
        words = session.decode_spi(
            clk=ChannelId(clk),
            mosi=ChannelId(mosi) if mosi is not None else None,
            miso=ChannelId(miso) if miso is not None else None,
            cs=ChannelId(cs) if cs is not None else None,
            params=SpiParams(cpol=cpol, cpha=cpha, bits=bits),
            capture_id=capture_id,
        )
        return [w.model_dump() for w in words]

    @mcp.tool()
    def decode_i2c(sda: int, scl: int, capture_id: str = "latest") -> list[dict[str, object]]:
        """Decode I2C on stored channels into transactions."""
        txns = session.decode_i2c(sda=ChannelId(sda), scl=ChannelId(scl), capture_id=capture_id)
        return [t.model_dump() for t in txns]

    @mcp.tool()
    def list_captures() -> list[dict[str, object]]:
        """Audit the capture store: id, size, kept, provenance for every live frame."""
        return [i.__dict__ for i in session.list_captures()]

    @mcp.tool()
    def drop_capture(capture_id: str) -> str:
        """Explicitly reclaim one stored frame."""
        session.drop(capture_id)
        return f"dropped {capture_id}"

    @mcp.tool()
    def keep_capture(capture_id: str, label: str | None = None) -> str:
        """Promote a frame to kept (exempt from ephemeral auto-eviction)."""
        session.keep(capture_id, label=label)
        return f"kept {capture_id}"

    return mcp


def main() -> None:  # pragma: no cover - process entry point
    """Wire a real DS1054Z from the environment and serve over stdio."""
    from hwtools.drivers.rigol.ds1054z import DS1054Z

    host = os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")
    scope = DS1054Z.over_tcp(host)
    scope.connect()
    server = build_server(ScopeSession(scope))
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
