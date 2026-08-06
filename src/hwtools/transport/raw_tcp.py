"""Raw SCPI-over-TCP transport.

Rigol DS1000Z (and many LXI instruments) expose a raw SCPI socket — port 5555 on
Rigol — that is markedly faster than VXI-11 for bulk waveform transfers. Commands
are newline-terminated ASCII; block replies use IEEE 488.2 definite-length
framing, read exactly via :func:`~hwtools.transport.block.read_definite_block`.
"""

from __future__ import annotations

import socket

from hwtools.transport.base import Transport
from hwtools.transport.block import encode_definite_block, read_definite_block

RIGOL_RAW_PORT = 5555


class RawTcpTransport(Transport):
    """SCPI over a plain TCP socket."""

    # Deep-memory reads and :ACQuire:MDEPth changes (the scope reallocates its
    # capture memory) can take several seconds; a short timeout fires mid-reply and
    # desyncs the byte stream, which then jams the raw socket. Be generous.
    def __init__(self, host: str, port: int = RIGOL_RAW_PORT, *, timeout_s: float = 15.0) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def open(self) -> None:
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def _connected(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("transport is not open; call open() first")
        return self._sock

    def write(self, command: str) -> None:
        self._connected.sendall(command.encode("ascii") + b"\n")

    def query(self, command: str) -> str:
        self.write(command)
        return self._recv_line().decode("ascii").strip()

    def query_block(self, command: str) -> bytes:
        self.write(command)
        payload = read_definite_block(self._recv_exact)
        self._recv_line()  # consume the trailing newline after the block
        return payload

    def write_block(self, command: str, payload: bytes) -> None:
        self._connected.sendall(
            command.encode("ascii") + encode_definite_block(payload) + b"\n"
        )

    def _recv_exact(self, n: int) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self._connected.recv(remaining)
            if not chunk:
                raise ConnectionError("connection closed mid-read")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_line(self) -> bytes:
        out = bytearray()
        while not out.endswith(b"\n"):
            chunk = self._connected.recv(1)
            if not chunk:
                break
            out += chunk
        return bytes(out)
