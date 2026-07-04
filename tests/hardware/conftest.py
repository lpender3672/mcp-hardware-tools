"""Shared fixtures for hardware-in-the-loop tests.

Bench addresses come from the environment (with bench defaults). Fixtures skip
rather than error when a device is absent, so a partial bench still runs what it
can. All tests using these are gated by the ``hardware`` marker.

**Bench-rig markers.** Only one physical wiring is on the bench at a time, so a test
that needs a *specific* rig also carries a ``cfg_*`` marker (registered in
``pyproject.toml``) naming it — applied individually, per test, since sibling tests
in one module can need different rigs. Run the subset for the rig you have wired::

    uv run pytest -m cfg_siggen_1ch      # 1 siggen channel -> 1 scope channel
    uv run pytest -m cfg_digital         # MCU/PIO digital lines -> scope CH1-4

Most driver and contract tests carry no ``cfg_*`` marker: they only need the
instrument plugged in (USB/LAN), not a particular output wiring. The exception is a
driver test that needs a specific *stimulus* — the DS1054Z pathways need a signal on
CH2, so they carry ``cfg_digital`` like the rest of the digital rig.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hwtools.drivers.joyit import JDS6600, find_jds6600_port
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness, find_pico_port
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig, WaveShape


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
def live_generator() -> Iterator[JDS6600]:
    """The connected JDS6600 signal generator; skips if no device is found."""
    port = os.environ.get("HWTOOLS_JDS6600_PORT") or find_jds6600_port()
    if port is None:
        pytest.skip("no JDS6600 found (set HWTOOLS_JDS6600_PORT)")
    gen = JDS6600(port)
    try:
        gen.connect()
    except OSError as exc:
        pytest.skip(f"JDS6600 at {port} could not be opened: {exc}")
    try:
        yield gen
    finally:
        gen.disconnect()


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


@pytest.fixture
def generator(live_generator: JDS6600) -> Iterator[JDS6600]:
    """The live JDS6600, restored to a benign 1 kHz 2 Vpp sine on CH1 after each test."""
    try:
        yield live_generator
    finally:
        live_generator.configure_channel(
            SignalGeneratorConfig(
                channel=SigGenChannel.CH1,
                waveform=WaveShape.SINE,
                frequency_hz=1_000.0,
                amplitude_vpp=2.0,
            )
        )
