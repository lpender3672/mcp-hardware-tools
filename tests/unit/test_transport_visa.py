"""VisaTransport and the discovery probes must never close pyvisa's shared session.

pyvisa gives every ResourceManager in a process one shared session, so a
``ResourceManager.close()`` anywhere tears down every other live instrument link
— e.g. a discovery probe killing the generator mid-sweep. These tests run against
:mod:`tests.fixtures.fake_pyvisa`, which models that sharing; the first one pins
the premise against the real library.
"""

from __future__ import annotations

import importlib.util

import pytest

from hwtools import discovery
from hwtools.transport.visa import VisaTransport
from tests.fixtures import fake_pyvisa

GEN = "TCPIP0::10.0.0.49::INSTR"
SCOPE = "TCPIP0::10.0.0.7::INSTR"


@pytest.mark.skipif(importlib.util.find_spec("pyvisa") is None, reason="needs pyvisa")
def test_pyvisa_resource_managers_share_one_session() -> None:
    import pyvisa

    assert pyvisa.ResourceManager("@py") is pyvisa.ResourceManager("@py")


def test_closing_one_transport_leaves_another_open(monkeypatch: pytest.MonkeyPatch) -> None:
    rm = fake_pyvisa.install(monkeypatch)
    gen, scope = VisaTransport(GEN), VisaTransport(SCOPE)
    gen.open()
    scope.open()
    scope.close()
    assert gen.query("*IDN?").startswith("RIGOL")
    assert rm._shared is not None


def test_close_clears_state_when_the_session_is_already_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rm = fake_pyvisa.install(monkeypatch)
    t = VisaTransport(GEN)
    t.open()
    rm._shared.resources[0].fail_close = True  # type: ignore[union-attr]
    t.close()  # best effort: a dead link must not make teardown raise
    with pytest.raises(RuntimeError, match="not open"):
        t.query("*IDN?")
    t.close()  # idempotent


def test_reopen_releases_the_previous_session(monkeypatch: pytest.MonkeyPatch) -> None:
    rm = fake_pyvisa.install(monkeypatch)
    t = VisaTransport(GEN)
    t.open()
    t.open()
    first, second = rm._shared.resources  # type: ignore[union-attr]
    assert first.closed
    assert not second.closed


def test_vxi11_probe_leaves_live_sessions_open(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyvisa.install(monkeypatch)
    gen = VisaTransport(GEN)
    gen.open()
    assert discovery._idn_vxi11(SCOPE, 0.1) == fake_pyvisa.IDN
    assert gen.query("*IDN?").startswith("RIGOL")


def test_broadcast_leaves_live_sessions_open(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyvisa.install(monkeypatch)
    gen = VisaTransport(GEN)
    gen.open()
    discovery._broadcast_hosts(0.1)
    assert gen.query("*IDN?").startswith("RIGOL")
