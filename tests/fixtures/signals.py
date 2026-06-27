"""Generators that build known waveforms/traces with ground-truth properties.

Decoders are validated by encoding a known payload here, then asserting the
decoder recovers it. This is the no-hardware stand-in for the JDS6600+Pico
harness that will later supply ground truth on the bench.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hwtools.decode.uart import Parity
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import DigitalTrace, Waveform


def bits_to_trace(
    bit_values: list[bool],
    *,
    samples_per_bit: int,
    dt_s: float,
    t0_s: float = 0.0,
    channel: ChannelId = ChannelId.CH1,
) -> DigitalTrace:
    """Expand a logic-bit sequence into an oversampled :class:`DigitalTrace`."""
    levels = np.repeat(np.asarray(bit_values, dtype=np.bool_), samples_per_bit)
    return DigitalTrace(channel=channel, levels=levels, t0_s=t0_s, dt_s=dt_s)


def trace_to_waveform(
    trace: DigitalTrace, *, low_v: float = 0.0, high_v: float = 3.3
) -> Waveform:
    """Map a logic trace to an analog waveform (for threshold round-trip tests)."""
    samples: npt.NDArray[np.float64] = np.where(trace.levels, high_v, low_v).astype(np.float64)
    return Waveform(channel=trace.channel, samples=samples, t0_s=trace.t0_s, dt_s=trace.dt_s)


def uart_bit_sequence(
    values: list[int],
    *,
    bits: int = 8,
    parity: Parity = Parity.NONE,
    stop_bits: int = 1,
    lsb_first: bool = True,
    idle_high: bool = True,
    idle_pad_bits: int = 2,
) -> list[bool]:
    """Build the logic-level bit sequence for a UART transmission of ``values``."""
    idle = idle_high
    seq: list[bool] = [idle] * idle_pad_bits
    for value in values:
        data_bits = [bool((value >> b) & 1) for b in range(bits)]
        if not lsb_first:
            data_bits.reverse()
        seq.append(not idle)  # start bit (space)
        # Logic 1 maps to the idle/mark physical level, logic 0 to its inverse.
        seq.extend(idle if bit else (not idle) for bit in data_bits)
        if parity is not Parity.NONE:
            ones = sum(data_bits)
            even = ones % 2 == 0
            parity_one = (not even) if parity is Parity.EVEN else even
            seq.append(idle if parity_one else (not idle))
        seq.extend([idle] * stop_bits)  # stop bit(s) at mark
    seq.extend([idle] * idle_pad_bits)
    return seq


def uart_trace(
    values: list[int],
    *,
    baud: float,
    samples_per_bit: int = 20,
    bits: int = 8,
    parity: Parity = Parity.NONE,
    stop_bits: int = 1,
    lsb_first: bool = True,
    idle_high: bool = True,
    channel: ChannelId = ChannelId.CH1,
) -> DigitalTrace:
    """Build a complete UART :class:`DigitalTrace` carrying ``values``."""
    seq = uart_bit_sequence(
        values,
        bits=bits,
        parity=parity,
        stop_bits=stop_bits,
        lsb_first=lsb_first,
        idle_high=idle_high,
    )
    dt_s = (1.0 / baud) / samples_per_bit
    return bits_to_trace(seq, samples_per_bit=samples_per_bit, dt_s=dt_s, channel=channel)
