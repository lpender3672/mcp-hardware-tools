"""Fixtures shared across the whole test tree.

The DG1062Z session link lives here (rather than in ``tests/hardware/``) so both the
hardware pathways tests *and* the cross-directory signal-generator contract can share
a single connection. This instrument wedges under connect/disconnect churn, so one
session-scoped link is essential — see :func:`live_dg1062`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hwtools.drivers.rigol import DG1062


@pytest.fixture(scope="session")
def dg1062_resource() -> str:
    return os.environ.get("HWTOOLS_DG1062_RESOURCE", "TCPIP0::192.168.64.49::INSTR")


@pytest.fixture(scope="session")
def live_dg1062(dg1062_resource: str) -> Iterator[DG1062]:
    """A connected DG1062Z over VISA (VXI-11); skips if unreachable.

    **Session-scoped — one link for the whole run.** Two hard-won facts drive this:
    (1) this DG1062Z's network stack wedges under repeated connect/disconnect churn
    (needing a power-cycle), so a single long-lived link is used; (2) an arbitrary
    upload is a *single* large binary write — the raw SCPI socket's input path jams on
    that, whereas VXI-11 handles it. (The DS1054Z stays on the raw socket: its large
    transfers are *chunked reads*, which the raw socket does fine.)
    """
    import pyvisa

    gen = DG1062.over_visa(dg1062_resource)
    try:
        gen.connect()
    except (OSError, pyvisa.errors.VisaIOError) as exc:
        pytest.skip(f"DG1062Z at {dg1062_resource} unreachable: {exc}")
    try:
        yield gen
    finally:
        gen.disconnect()
