"""The harness MCU abstraction — a digital DUT that emits known stimuli.

Development scaffolding, not part of the product surface: its only job is to put
*known* signals on the wire so the scope tooling and decoders can be validated
against ground truth. It mirrors the firmware command contract (see
``firmware/common/src/protocol.rs``): identify, stream a UART byte, stop, and
reboot into the bootloader for scripted reflashing.
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
    def idn(self) -> str:
        """Return the device's identity banner."""

    @abstractmethod
    def start_uart_stream(self, value: int, *, baud: int) -> None:
        """Continuously transmit ``value`` (one byte) as 8N1 UART at ``baud``."""

    @abstractmethod
    def start_square(self, freq_hz: int, *, duty_pct: int = 50) -> None:
        """Output a square wave at ``freq_hz`` with ``duty_pct`` percent duty."""

    @abstractmethod
    def stop(self) -> None:
        """Stop any active stimulus (the line returns to idle)."""

    @abstractmethod
    def reboot_to_bootloader(self) -> None:
        """Reboot the device into its USB bootloader for reflashing."""

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
