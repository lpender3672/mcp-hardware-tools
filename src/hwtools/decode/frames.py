"""Typed results produced by the protocol decoders.

These are the structured frames the agent reasons over — the whole point of
decoding in software rather than reading the scope screen. Pure data; no
instrument or decode logic lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UartWord(BaseModel):
    """One UART character recovered from a line."""

    model_config = ConfigDict(frozen=True)

    value: int = Field(ge=0, description="Decoded word value (data bits assembled).")
    t_start_s: float = Field(description="Time of the start-bit edge.")
    framing_error: bool = Field(default=False, description="Stop bit was not at idle level.")
    parity_error: bool = Field(default=False, description="Parity bit did not match.")


class SpiWord(BaseModel):
    """One word clocked on SPI, captured on both data lines."""

    model_config = ConfigDict(frozen=True)

    mosi: int | None = Field(default=None, ge=0, description="Controller-out word (if MOSI given).")
    miso: int | None = Field(default=None, ge=0, description="Peripheral-out word (if MISO given).")
    t_start_s: float = Field(description="Time of this word's first sampled bit.")


class I2cByte(BaseModel):
    """One byte on an I²C bus plus the ACK/NAK that followed it."""

    model_config = ConfigDict(frozen=True)

    value: int = Field(ge=0, le=0xFF)
    ack: bool = Field(description="True if the receiver pulled SDA low (ACK).")
    is_address: bool = Field(default=False, description="True for the address byte of a transfer.")
    read: bool | None = Field(default=None, description="R/W bit, set only on the address byte.")
    t_start_s: float = Field(description="Time of this byte's first data bit.")


class I2cTransaction(BaseModel):
    """One START-to-STOP transfer on an I²C bus."""

    model_config = ConfigDict(frozen=True)

    address: int | None = Field(default=None, ge=0, le=0x7F, description="7-bit target address.")
    read: bool | None = Field(default=None, description="Transfer direction from the address byte.")
    bytes: list[I2cByte] = Field(default_factory=list)
    t_start_s: float = Field(description="Time of the START condition.")

    @property
    def data(self) -> list[int]:
        """Payload byte values after the address byte."""
        return [b.value for b in self.bytes if not b.is_address]
