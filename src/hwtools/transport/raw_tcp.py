"""Raw SCPI-over-TCP transport.

Rigol DS1000Z (and many LXI instruments) expose a raw SCPI socket — port 5555 on
Rigol — that is markedly faster than VXI-11 for bulk waveform transfers. Commands
are newline-terminated ASCII; block replies use IEEE 488.2 definite-length
framing, read exactly via :func:`~hwtools.transport.block.read_definite_block`.

The socket is one byte stream, so a read that times out or a reply that stops
mid-way leaves the *previous* command's bytes queued for the next one. A faulted
link is therefore dropped at once: every later call raises ConnectionError until
the transport is reopened, rather than returning another command's reply.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator

from hwtools.transport.base import Transport
from hwtools.transport.block import encode_definite_block, read_definite_block

RIGOL_RAW_PORT = 5555

# TCP keepalive: probe an idle link after this long, then every interval, so a
# peer that vanished (power, cable, a wedged stack) is noticed on an idle link
# rather than by the next command's full read timeout.
_KEEPALIVE_IDLE_S = 10
_KEEPALIVE_INTERVAL_S = 3
_KEEPALIVE_COUNT = 3


def _enable_keepalive(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    with contextlib.suppress(OSError, AttributeError):  # the timing is best effort
        if hasattr(socket, "SIO_KEEPALIVE_VALS"):  # Windows
            sock.ioctl(
                socket.SIO_KEEPALIVE_VALS,
                (1, _KEEPALIVE_IDLE_S * 1000, _KEEPALIVE_INTERVAL_S * 1000),
            )
        else:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, _KEEPALIVE_IDLE_S)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, _KEEPALIVE_INTERVAL_S)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _KEEPALIVE_COUNT)


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
        self._fault: BaseException | None = None  # why the link was dropped

    def open(self) -> None:
        self.close()
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        _enable_keepalive(sock)
        self._sock = sock

    def close(self) -> None:
        """Close the link; idempotent and never raises. ``shutdown`` first, so the
        instrument sees an orderly end of the session."""
        sock, self._sock = self._sock, None
        self._fault = None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    @property
    def _connected(self) -> socket.socket:
        if self._sock is None:
            if self._fault is not None:
                raise ConnectionError(
                    f"link to {self._host}:{self._port} was dropped "
                    f"({type(self._fault).__name__}: {self._fault}); reopen it"
                )
            raise RuntimeError("transport is not open; call open() first")
        return self._sock

    @contextlib.contextmanager
    def _exchange(self) -> Iterator[None]:
        """Guard one command/reply exchange: an I/O error, or a reply that is not
        where the stream should be, drops the link before re-raising."""
        try:
            yield
        except (OSError, ValueError) as exc:
            if self._sock is not None:
                self.close()
                self._fault = exc
            raise

    def write(self, command: str) -> None:
        with self._exchange():
            self._connected.sendall(command.encode("ascii") + b"\n")

    def query(self, command: str) -> str:
        with self._exchange():
            self.write(command)
            return self._recv_line().decode("ascii").strip()

    def query_block(self, command: str) -> bytes:
        with self._exchange():
            self.write(command)
            payload = read_definite_block(self._recv_exact)
            self._recv_line()  # consume the trailing newline after the block
            return payload

    def write_block(self, command: str, payload: bytes) -> None:
        with self._exchange():
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
                raise ConnectionError("connection closed mid-reply")
            out += chunk
        return bytes(out)
