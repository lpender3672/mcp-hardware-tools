"""Asynchronous serial (UART) decode.

Operates on a single thresholded line. Standard idle-high TTL framing by default:
idle high, a falling start bit, ``bits`` data bits (LSB first), optional parity,
then one or more stop bits at idle. Each data/stop bit is sampled at its centre,
measured from the start edge, which is robust to the threshold sampling grid.

Pure: a :class:`~hwtools.model.waveform.DigitalTrace` in, typed
:class:`~hwtools.decode.frames.UartWord` list out.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from hwtools.decode.frames import UartWord
from hwtools.model.waveform import DigitalTrace


class Parity(StrEnum):
    NONE = "NONE"
    EVEN = "EVEN"
    ODD = "ODD"


class UartParams(BaseModel):
    """Framing parameters needed to decode a UART line."""

    model_config = ConfigDict(frozen=True)

    baud: float = Field(gt=0)
    bits: int = Field(default=8, ge=5, le=9)
    parity: Parity = Parity.NONE
    stop_bits: float = Field(default=1.0, gt=0)
    lsb_first: bool = True
    idle_high: bool = True


def decode_uart(trace: DigitalTrace, params: UartParams) -> list[UartWord]:
    """Recover UART words from a thresholded line."""
    levels = trace.levels
    n = int(levels.size)
    if n == 0:
        return []

    samples_per_bit = (1.0 / params.baud) / trace.dt_s
    idle = params.idle_high
    words: list[UartWord] = []

    i = 0
    while i < n - 1:
        # A start bit is a transition away from idle.
        if bool(levels[i]) == idle and bool(levels[i + 1]) != idle:
            start_edge = i + 1
            word = _decode_frame(trace, start_edge, samples_per_bit, params)
            if word is None:
                i = start_edge  # not enough samples for a full frame
                break
            words.append(word)
            # Resume at the stop bit (idle/mark), so the *next* start edge is
            # still ahead and detectable — crucial when frames are back-to-back.
            i = _resume_index(start_edge, samples_per_bit, params)
        else:
            i += 1

    return words


def _resume_index(start_edge: int, samples_per_bit: float, params: UartParams) -> int:
    """Index at the first stop bit — where decoding resumes the start-edge hunt."""
    parity_bits = 0 if params.parity is Parity.NONE else 1
    bits_to_stop = 1 + params.bits + parity_bits  # start + data + parity
    return round(start_edge + bits_to_stop * samples_per_bit)


def _decode_frame(
    trace: DigitalTrace, start_edge: int, samples_per_bit: float, params: UartParams
) -> UartWord | None:
    n = int(trace.levels.size)
    high_is_one = params.idle_high  # TTL: idle/mark high == logic 1

    def bit_at(bit_index: float) -> bool | None:
        """Logic level (True=1) at the centre of the bit ``bit_index`` from start edge."""
        idx = round(start_edge + (bit_index + 0.5) * samples_per_bit)
        if idx >= n:
            return None
        level = bool(trace.levels[idx])
        return level if high_is_one else (not level)

    value = 0
    for b in range(params.bits):
        bit = bit_at(1 + b)  # bit 0 is the start bit
        if bit is None:
            return None
        if bit:
            pos = b if params.lsb_first else (params.bits - 1 - b)
            value |= 1 << pos

    parity_error = False
    next_bit = 1 + params.bits
    if params.parity is not Parity.NONE:
        pbit = bit_at(next_bit)
        if pbit is None:
            return None
        ones = bin(value).count("1") + (1 if pbit else 0)
        expected_even = ones % 2 == 0
        parity_error = expected_even if params.parity is Parity.ODD else not expected_even
        next_bit += 1

    # Stop bit(s) sit at idle/mark, i.e. logic 1; anything else is a framing error.
    stop = bit_at(next_bit)
    framing_error = stop is not None and not stop

    return UartWord(
        value=value,
        t_start_s=trace.t0_s + start_edge * trace.dt_s,
        framing_error=framing_error,
        parity_error=parity_error,
    )
