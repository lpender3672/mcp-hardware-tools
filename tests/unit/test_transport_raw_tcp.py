"""RawTcpTransport fails loudly and cleanly when the link drops.

A raw SCPI socket carries one byte stream: once a read times out or the peer
vanishes mid-reply, whatever arrives next belongs to the *previous* command. So a
faulted link is dropped at once and every later call raises ConnectionError (an
OSError, i.e. reconnect-worthy) instead of returning stale bytes. A loopback
server stands in for the instrument.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from hwtools.transport.raw_tcp import RawTcpTransport

_IDN = b"RIGOL TECHNOLOGIES,DS1054Z,DS1ZA123456789,00.04.04.SP4\n"

Handler = Callable[[socket.socket], None]


class _Server:
    """Accepts connections and runs ``handler`` on each; records them."""

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen()
        self._srv.settimeout(0.1)
        self.port: int = self._srv.getsockname()[1]
        self.connections: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                continue
            self.connections.append(conn)
            threading.Thread(target=self._handler, args=(conn,), daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self._srv.close()
        for conn in self.connections:
            conn.close()


@pytest.fixture
def serve() -> Iterator[Callable[[Handler], _Server]]:
    servers: list[_Server] = []

    def start(handler: Handler) -> _Server:
        servers.append(_Server(handler))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def _answer_idn(conn: socket.socket, delay_s: float = 0.0) -> None:
    with conn:
        try:
            while conn.recv(64):
                time.sleep(delay_s)
                conn.sendall(_IDN)
        except OSError:
            pass


def test_query_round_trips(serve: Callable[[Handler], _Server]) -> None:
    srv = serve(_answer_idn)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    assert t.query("*IDN?") == _IDN.decode().strip()
    t.close()


def test_keepalive_is_enabled(serve: Callable[[Handler], _Server]) -> None:
    srv = serve(_answer_idn)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    assert t._sock is not None
    assert t._sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
    t.close()


def test_peer_closing_mid_reply_raises(serve: Callable[[Handler], _Server]) -> None:
    def half_reply(conn: socket.socket) -> None:
        with conn:
            conn.recv(64)
            conn.sendall(b"RIGOL TECHNOL")

    srv = serve(half_reply)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    with pytest.raises(ConnectionError):
        t.query("*IDN?")


def test_a_timed_out_reply_is_never_read_as_the_next_one(
    serve: Callable[[Handler], _Server],
) -> None:
    srv = serve(lambda conn: _answer_idn(conn, delay_s=0.3))
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=0.1)
    t.open()
    with pytest.raises(TimeoutError):
        t.query("*IDN?")
    time.sleep(0.4)  # the late reply has now arrived on the old stream
    with pytest.raises(ConnectionError):
        t.query("*IDN?")


def test_a_desynced_block_drops_the_link(serve: Callable[[Handler], _Server]) -> None:
    def not_a_block(conn: socket.socket) -> None:
        with conn:
            conn.recv(64)
            conn.sendall(b"garbage\n")
            time.sleep(0.5)

    srv = serve(not_a_block)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    with pytest.raises(ValueError):
        t.query_block(":WAV:DATA?")
    with pytest.raises(ConnectionError):
        t.query("*IDN?")


def test_reopen_after_a_fault_works(serve: Callable[[Handler], _Server]) -> None:
    srv = serve(_answer_idn)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    srv.connections[0].shutdown(socket.SHUT_RDWR)  # the instrument drops the link
    with pytest.raises(ConnectionError):
        t.query("*IDN?")
    t.open()
    assert t.query("*IDN?").startswith("RIGOL")
    t.close()


def test_reopen_closes_the_previous_connection(serve: Callable[[Handler], _Server]) -> None:
    # A Rigol raw socket serves one client at a time: a stranded connection
    # blocks every new one until the instrument is power-cycled.
    ended = threading.Event()

    def until_eof(conn: socket.socket) -> None:
        with conn:
            conn.settimeout(2.0)
            try:
                if conn.recv(64) == b"":
                    ended.set()
            except OSError:
                pass

    srv = serve(until_eof)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    t.open()
    assert ended.wait(2.0)  # the first connection saw an orderly EOF
    t.close()


def test_close_is_idempotent_and_leaves_a_clear_error(
    serve: Callable[[Handler], _Server],
) -> None:
    srv = serve(_answer_idn)
    t = RawTcpTransport("127.0.0.1", srv.port, timeout_s=1.0)
    t.open()
    t.close()
    t.close()
    with pytest.raises(RuntimeError, match="not open"):
        t.query("*IDN?")
