"""In-memory transport for transcript-style tests.

Feed it the replies an instrument would give, drive the driver, then assert on
the exact commands it sent (``.log``). ``query_block`` returns the payload bytes
directly — the IEEE block framing is the real transports' concern, tested
separately.
"""

from __future__ import annotations

from hwtools.transport.base import Transport


class FakeTransport(Transport):
    """A scripted :class:`Transport` recording every command it is sent."""

    def __init__(
        self,
        queries: dict[str, str] | None = None,
        blocks: dict[str, bytes] | None = None,
    ) -> None:
        self._queries = queries or {}
        self._blocks = blocks or {}
        self.log: list[str] = []
        self.written_blocks: list[tuple[str, bytes]] = []  # (command, payload) for write_block
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def write(self, command: str) -> None:
        self.log.append(command)

    def query(self, command: str) -> str:
        self.log.append(command)
        if command not in self._queries:
            raise KeyError(f"FakeTransport has no scripted reply for query {command!r}")
        return self._queries[command]

    def query_block(self, command: str) -> bytes:
        self.log.append(command)
        if command not in self._blocks:
            raise KeyError(f"FakeTransport has no scripted block for {command!r}")
        return self._blocks[command]

    def write_block(self, command: str, payload: bytes) -> None:
        self.log.append(command)
        self.written_blocks.append((command, payload))
