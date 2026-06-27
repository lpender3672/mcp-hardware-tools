"""Shared fixtures for hardware-in-the-loop tests.

Bench addresses come from the environment (with bench defaults). Fixtures skip
rather than error when a device is absent, so a partial bench still runs what it
can. All tests using these are gated by the ``hardware`` marker.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness, find_pico_port


@pytest.fixture(scope="session")
def scope_host() -> str:
    return os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")


@pytest.fixture
def live_scope(scope_host: str) -> Iterator[DS1054Z]:
    """A connected DS1054Z; skips if it can't be reached."""
    scope = DS1054Z.over_tcp(scope_host)
    try:
        scope.connect()
    except OSError as exc:
        pytest.skip(f"scope at {scope_host} unreachable: {exc}")
    try:
        yield scope
    finally:
        scope.disconnect()


@pytest.fixture
def harness() -> Iterator[SerialHarness]:
    """The connected RP2350 harness; skips if no device is found."""
    port = os.environ.get("HWTOOLS_PICO_PORT") or find_pico_port()
    if port is None:
        pytest.skip("no harness device found (set HWTOOLS_PICO_PORT)")
    device = SerialHarness(port)
    try:
        device.open()
    except OSError as exc:
        pytest.skip(f"harness at {port} could not be opened: {exc}")
    try:
        yield device
    finally:
        device.close()
