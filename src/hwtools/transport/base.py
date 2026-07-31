"""The byte-level link to an instrument — the seam that makes drivers testable.

A driver speaks SCPI strings through a :class:`Transport`; it never touches a
socket or VISA session directly. That lets the same driver run against a real
instrument (:class:`~hwtools.transport.raw_tcp.RawTcpTransport`,
:class:`~hwtools.transport.visa.VisaTransport`) or against a
:class:`~hwtools.transport.fake.FakeTransport` carrying a recorded transcript, so
command serialization and response parsing are unit-tested with no hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class Transport(ABC):
    """A synchronous request/response link carrying SCPI text and block data."""

    @abstractmethod
    def open(self) -> None:
        """Establish the link."""

    @abstractmethod
    def close(self) -> None:
        """Tear down the link."""

    @abstractmethod
    def write(self, command: str) -> None:
        """Send a command with no reply."""

    @abstractmethod
    def query(self, command: str) -> str:
        """Send a command and read its single text reply (newline-terminated)."""

    @abstractmethod
    def query_block(self, command: str) -> bytes:
        """Send a command and read an IEEE 488.2 definite-length block.

        Returns the *payload* bytes only — the ``#<n><len>`` header and trailing
        newline are stripped by the transport.
        """

    @abstractmethod
    def write_block(self, command: str, payload: bytes) -> None:
        """Send ``command`` immediately followed by ``payload`` as an IEEE 488.2
        definite-length block.

        The transport frames the ``#<n><len>`` header around ``payload`` (the inverse
        of :meth:`query_block`), so the caller passes the raw bytes. Used to download
        binary waveform data (e.g. a Rigol ``DATA:DAC16`` arbitrary upload)."""

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
