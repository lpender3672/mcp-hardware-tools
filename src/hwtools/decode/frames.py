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
