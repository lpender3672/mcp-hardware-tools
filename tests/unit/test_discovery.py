"""LAN instrument discovery: IDN parsing, probing and the port-sweep fallback.

No hardware and no real network: a loopback TCP server stands in for an instrument's
raw-SCPI socket, and the VXI-11 broadcast is patched out.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Iterator

import pytest

from hwtools import discovery
from hwtools.discovery import (
    FoundInstrument,
    discover_instruments,
    open_instrument,
    parse_idn,
    probe,
)
from hwtools.drivers.rigol import DG1062, DS1054Z
from hwtools.transport.raw_tcp import RawTcpTransport
from hwtools.transport.visa import VisaTransport

_IDN = "RIGOL TECHNOLOGIES,DS1054Z,DS1ZA123456789,00.04.04.SP4"


@pytest.fixture
def fake_instrument() -> Iterator[int]:
    """A loopback server answering ``*IDN?`` like a Rigol raw socket; yields its port."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    srv.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(1.0)
                try:
                    if conn.recv(64).strip() == b"*IDN?":
                        conn.sendall(_IDN.encode() + b"\n")
                except OSError:
                    pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield srv.getsockname()[1]
    stop.set()
    t.join(1.0)
    srv.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_parse_idn_splits_four_fields() -> None:
    assert parse_idn(_IDN) == ("RIGOL TECHNOLOGIES", "DS1054Z", "DS1ZA123456789", "00.04.04.SP4")


def test_parse_idn_keeps_commas_in_firmware() -> None:
    assert parse_idn("ACME,X1,SN1,1.0,build 7") == ("ACME", "X1", "SN1", "1.0,build 7")


@pytest.mark.parametrize("reply", ["", "hello", "a,b", "ACME,,SN,1"])
def test_parse_idn_rejects_non_idn(reply: str) -> None:
    assert parse_idn(reply) is None


def test_probe_host_port_identifies(fake_instrument: int) -> None:
    found = probe(f"127.0.0.1:{fake_instrument}", timeout_s=1.0)
    assert found is not None
    assert (found.model, found.serial) == ("DS1054Z", "DS1ZA123456789")
    assert found.raw_port == fake_instrument
    assert found.resource == "TCPIP0::127.0.0.1::INSTR"
    assert found.idn == _IDN


def test_probe_nothing_listening_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "_idn_vxi11", lambda resource, timeout_s: None)
    assert probe("127.0.0.1", timeout_s=0.3, ports=(_free_port(),)) is None


def test_probe_rejects_non_tcpip_resource() -> None:
    assert probe("USB0::0x1AB1::0x04CE::DS1ZA::INSTR") is None


def test_discover_falls_back_to_sweep(
    monkeypatch: pytest.MonkeyPatch, fake_instrument: int
) -> None:
    monkeypatch.setattr(discovery, "_broadcast_hosts", lambda wait_s: [])
    found = discover_instruments(
        timeout_s=1.0,
        sweep_fallback=True,
        networks=[ipaddress.IPv4Network("127.0.0.1/32")],
        ports=(fake_instrument,),
    )
    assert [f.serial for f in found] == ["DS1ZA123456789"]


def test_discover_never_sweeps_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    # A /24 port sweep looks like a port scan on a shared network: opt-in only.
    def no_sweep(networks: object) -> list[str]:
        raise AssertionError("swept without sweep_fallback=True")

    monkeypatch.setattr(discovery, "_broadcast_hosts", lambda wait_s: [])
    monkeypatch.setattr(discovery, "_sweep_hosts", no_sweep)
    assert discover_instruments() == []


def test_skip_hosts_are_never_contacted(
    monkeypatch: pytest.MonkeyPatch, fake_instrument: int
) -> None:
    # An instrument in use must not be probed: a second client on its raw socket
    # interleaves with the owner's session.
    monkeypatch.setattr(discovery, "_broadcast_hosts", lambda wait_s: ["127.0.0.1"])
    probed: list[str] = []
    real_probe = discovery.probe

    def spy(address: str, **kw: object) -> FoundInstrument | None:
        probed.append(address)
        return real_probe(address, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(discovery, "probe", spy)
    assert (
        discover_instruments(timeout_s=1.0, ports=(fake_instrument,), skip_hosts={"127.0.0.1"})
        == []
    )
    assert probed == []


def test_skip_hosts_are_left_out_of_a_sweep(
    monkeypatch: pytest.MonkeyPatch, fake_instrument: int
) -> None:
    monkeypatch.setattr(discovery, "_broadcast_hosts", lambda wait_s: [])
    found = discover_instruments(
        timeout_s=1.0,
        sweep_fallback=True,
        networks=[ipaddress.IPv4Network("127.0.0.1/32")],
        ports=(fake_instrument,),
        skip_hosts=["127.0.0.1"],
    )
    assert found == []


def test_discover_dedupes_by_serial(monkeypatch: pytest.MonkeyPatch, fake_instrument: int) -> None:
    # The same instrument reported twice by the broadcast (two interfaces) is one entry.
    monkeypatch.setattr(discovery, "_broadcast_hosts", lambda wait_s: ["127.0.0.1", "127.0.0.1"])
    found = discover_instruments(timeout_s=1.0, ports=(fake_instrument,))
    assert len(found) == 1


def test_sweep_refuses_wide_networks() -> None:
    with pytest.raises(ValueError, match="wider than"):
        discovery._sweep_hosts([ipaddress.IPv4Network("10.0.0.0/16")])


def _found(model: str, raw_port: int | None) -> FoundInstrument:
    host = "10.0.0.5"
    return FoundInstrument(host, f"TCPIP0::{host}::INSTR", "RIGOL", model, "SN", "1", raw_port)


@pytest.mark.parametrize(
    ("model", "cls"), [("DS1054Z", DS1054Z), ("MSO1104Z", DS1054Z), ("DG1062Z", DG1062)]
)
def test_open_instrument_picks_driver(model: str, cls: type) -> None:
    raw = open_instrument(_found(model, 5555))
    assert isinstance(raw, cls)
    assert isinstance(raw._t, RawTcpTransport)
    visa = open_instrument(_found(model, None))
    assert isinstance(visa._t, VisaTransport)


def test_open_instrument_unknown_model() -> None:
    with pytest.raises(ValueError, match="no driver"):
        open_instrument(_found("SDG2042X", 5025))
