"""VISA transport via pyvisa (pure-Python ``pyvisa-py`` backend).

General-purpose fallback for USB-TMC or LXI/VXI-11 instruments. For DS1000Z bulk
waveform pulls, prefer :class:`~hwtools.transport.raw_tcp.RawTcpTransport`.

pyvisa gives every ``ResourceManager`` in a process one *shared* session, and
closing it closes every resource opened through it — another instrument's live
link included. A transport therefore only ever closes its own resource and
leaves the manager to pyvisa's exit handler.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from hwtools.transport.base import Transport
from hwtools.transport.block import decode_definite_block, encode_definite_block

if TYPE_CHECKING:
    import pyvisa


class VisaTransport(Transport):
    """SCPI over a pyvisa resource, using the pure-Python ``@py`` backend."""

    def __init__(self, resource: str, *, timeout_ms: int = 5000) -> None:
        self._resource = resource
        self._timeout_ms = timeout_ms
        self._inst: pyvisa.resources.MessageBasedResource | None = None

    def open(self) -> None:
        import pyvisa

        self.close()  # a reopen must not strand the previous session
        inst = pyvisa.ResourceManager("@py").open_resource(self._resource)
        inst.timeout = self._timeout_ms
        self._inst = inst  # type: ignore[assignment]

    def close(self) -> None:
        """Close this transport's resource (never the shared manager). Best
        effort and idempotent: a session whose link already died may refuse to
        close, and teardown must still leave the transport closed."""
        inst, self._inst = self._inst, None
        if inst is not None:
            with contextlib.suppress(Exception):
                inst.close()

    @property
    def _connected(self) -> pyvisa.resources.MessageBasedResource:
        if self._inst is None:
            raise RuntimeError("transport is not open; call open() first")
        return self._inst

    def write(self, command: str) -> None:
        self._connected.write(command)

    def query(self, command: str) -> str:
        return self._connected.query(command).strip()

    def query_block(self, command: str) -> bytes:
        self._connected.write(command)
        return decode_definite_block(self._connected.read_raw())

    def write_block(self, command: str, payload: bytes) -> None:
        self._connected.write_raw(command.encode("ascii") + encode_definite_block(payload))
