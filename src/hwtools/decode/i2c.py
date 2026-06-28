"""I²C decode.

A state machine over the SDA/SCL pair: START is SDA falling while SCL is high,
STOP is SDA rising while SCL is high, and data bits are latched on SCL rising
edges (8 data bits MSB-first, then a 9th clock for ACK/NAK). The first byte after
a START is the 7-bit address plus the R/W bit. Pure: aligned :class:`DigitalTrace`
inputs, typed :class:`~hwtools.decode.frames.I2cTransaction` list out.

A repeated START (without an intervening STOP) is treated as the end of the
current transaction and the beginning of a new one.

START/STOP detection is *debounced*: SCL must be stably high for a guard window
on both sides of the SDA edge. At coarse capture resolution an SDA edge that
lands a sample or two away from an SCL transition would otherwise masquerade as
a spurious START/STOP and truncate the real transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hwtools.decode.frames import I2cByte, I2cTransaction
from hwtools.model.waveform import DigitalTrace

#: Samples SCL must hold high either side of an SDA edge to count as START/STOP.
START_STOP_GUARD = 2


@dataclass
class _Builder:
    t_start_s: float
    address: int | None = None
    read: bool | None = None
    bytes: list[I2cByte] = field(default_factory=list)

    def finish(self) -> I2cTransaction:
        return I2cTransaction(
            address=self.address, read=self.read, bytes=self.bytes, t_start_s=self.t_start_s
        )


def decode_i2c(
    sda: DigitalTrace, scl: DigitalTrace, *, start_stop_guard: int = START_STOP_GUARD
) -> list[I2cTransaction]:
    """Recover I²C transactions from the SDA/SCL pair."""
    if sda.n != scl.n:
        raise ValueError("sda and scl must share the same sample grid")

    scl_high = np.asarray(scl.levels, dtype=np.bool_)

    def scl_stable_high(i: int) -> bool:
        """True if SCL is high for ``start_stop_guard`` samples either side of the
        SDA edge between ``i-1`` and ``i``. Edges too near the capture boundary to
        confirm are rejected (a START/STOP there is a partial frame anyway)."""
        g = start_stop_guard
        if i - g < 0 or i + g > scl_high.size:
            return False
        return bool(scl_high[i - g : i + g].all())

    transactions: list[I2cTransaction] = []
    builder: _Builder | None = None
    cur_byte = 0
    bit_count = 0
    byte_start_s = 0.0
    expecting_ack = False
    is_address = True

    def time_at(i: int) -> float:
        return sda.t0_s + i * sda.dt_s

    for i in range(1, sda.n):
        scl_prev, scl_cur = bool(scl.levels[i - 1]), bool(scl.levels[i])
        sda_prev, sda_cur = bool(sda.levels[i - 1]), bool(sda.levels[i])

        # START / STOP: an SDA edge while SCL is held *stably* high.
        if scl_prev and scl_cur and sda_prev != sda_cur and scl_stable_high(i):
            if not sda_cur:  # SDA falling -> START (or repeated START)
                if builder is not None and builder.bytes:
                    transactions.append(builder.finish())
                builder = _Builder(t_start_s=time_at(i))
                cur_byte = bit_count = 0
                expecting_ack = False
                is_address = True
            else:  # SDA rising -> STOP
                if builder is not None and builder.bytes:
                    transactions.append(builder.finish())
                builder = None
            continue

        # Data/ACK latched on SCL rising edges, only inside a transaction.
        if builder is not None and scl_cur and not scl_prev:
            if not expecting_ack:
                if bit_count == 0:
                    byte_start_s = time_at(i)
                cur_byte = (cur_byte << 1) | (1 if sda_cur else 0)
                bit_count += 1
                if bit_count == 8:
                    expecting_ack = True
            else:
                ack = not sda_cur  # ACK = receiver pulls SDA low
                if is_address:
                    builder.address = cur_byte >> 1
                    builder.read = bool(cur_byte & 1)
                    builder.bytes.append(
                        I2cByte(
                            value=cur_byte,
                            ack=ack,
                            is_address=True,
                            read=builder.read,
                            t_start_s=byte_start_s,
                        )
                    )
                    is_address = False
                else:
                    builder.bytes.append(
                        I2cByte(value=cur_byte, ack=ack, t_start_s=byte_start_s)
                    )
                cur_byte = bit_count = 0
                expecting_ack = False

    return transactions
