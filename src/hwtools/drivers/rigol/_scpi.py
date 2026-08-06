"""Shared transport plumbing for Rigol SCPI instruments.

Both the DS1054Z scope and the DG1062Z generator speak SCPI over the same
:class:`~hwtools.transport.base.Transport` seam and share the same connection
boilerplate: build over the fast raw socket (port 5555) or a pyvisa resource,
open/close the link, and read ``*IDN?``. This mixin holds that boilerplate once so
each driver carries only its instrument-specific command set.

Mixed in *before* the instrument ABC (``class DG1062(ScpiTransportMixin,
SignalGenerator)``) so its concrete ``connect``/``disconnect``/``idn`` satisfy the
ABC's abstract methods — method resolution finds the mixin ahead of the ABC.
"""

from __future__ import annotations

from typing import Self

from hwtools.transport.base import Transport
from hwtools.transport.raw_tcp import RIGOL_RAW_PORT, RawTcpTransport
from hwtools.transport.visa import VisaTransport


class ScpiTransportMixin:
    """Connection boilerplate for a SCPI instrument driven over a :class:`Transport`."""

    _t: Transport

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    @classmethod
    def over_tcp(cls, host: str, port: int = RIGOL_RAW_PORT) -> Self:
        """Build a driver using the fast raw SCPI socket (Rigol port 5555)."""
        return cls(RawTcpTransport(host, port))

    @classmethod
    def over_visa(cls, resource: str) -> Self:
        """Build a driver using a pyvisa resource string."""
        return cls(VisaTransport(resource))

    def connect(self) -> None:
        self._t.open()

    def disconnect(self) -> None:
        self._t.close()

    def idn(self) -> str:
        return self._t.query("*IDN?")

    def _check_error(self) -> None:
        """Drain ``:SYSTem:ERRor?`` and raise if the queue held any error.

        SCPI instruments queue command errors (bad header, out-of-range value)
        rather than failing the write, so a silent typo desyncs our model from the
        instrument. Reading the queue after a batch surfaces that loudly. A single
        batch can queue *several* errors, so we read until the terminating
        ``0,"No error"`` (bounded, so a wedged queue can't spin forever) and report
        them all. The DS1054Z deliberately does **not** call this — querying the
        error queue around a deep-memory reallocation blocks and jams its raw
        socket — but the DG1062Z has no such hazard.
        """
        errors: list[str] = []
        for _ in range(32):
            code, _, message = self._t.query(":SYSTem:ERRor?").partition(",")
            if int(code) == 0:  # 0,"No error" marks the end of the queue
                break
            errors.append(f"{code},{message.strip()}")
        if errors:
            raise RuntimeError("instrument reported error(s): " + "; ".join(errors))
