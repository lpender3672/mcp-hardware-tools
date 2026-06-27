"""The harness MCU abstraction — a digital DUT that emits known stimuli.

Development scaffolding, not part of the product surface: its only job is to put
*known* signals on the wire so the scope tooling and decoders can be validated
against ground truth. Today it streams a repeating UART byte; SPI/I²C emission
grow here behind the same interface as the firmware contract is filled in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class DigitalDUT(ABC):
    """An MCU that can be told to emit a known digital stimulus."""

    @abstractmethod
    def open(self) -> None:
        """Connect to the device."""

    @abstractmethod
    def close(self) -> None:
        """Disconnect, leaving any active stimulus running on the device."""

    @abstractmethod
    def start_uart_stream(self, value: int, *, baud: int, tx_pin: int = 0) -> None:
        """Continuously transmit ``value`` (one byte) as 8N1 UART on ``tx_pin``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop any active stimulus."""

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
