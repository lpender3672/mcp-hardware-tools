"""Generators that build known waveforms/traces with ground-truth properties.

Decoders are validated by encoding a known payload here, then asserting the
decoder recovers it. This is the no-hardware stand-in for the JDS6600+Pico
harness that will later supply ground truth on the bench.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hwtools.decode.spi import sample_edge_is_rising
from hwtools.decode.threshold import find_edges
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


def spi_traces(
    mosi_words: list[int],
    *,
    miso_words: list[int] | None = None,
    bits: int = 8,
    cpol: int = 0,
    cpha: int = 0,
    msb_first: bool = True,
    sph: int = 4,
    with_cs: bool = False,
    dt_s: float = 1e-6,
) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace | None, DigitalTrace | None]:
    """Build aligned clk/mosi/miso(/cs) traces carrying ``mosi_words``.

    Data is held stable across each bit's sampling edge by filling the midpoints
    between consecutive sampling edges, so the decoder reads exactly the intended
    bit regardless of the CPOL/CPHA convention. Returns ``(clk, mosi, miso, cs)``.
    """
    n_bits = len(mosi_words) * bits
    idle = bool(cpol)
    halves: list[bool] = [idle] * sph  # leading idle pad
    for _ in range(n_bits):
        halves.extend([not idle] * sph)
        halves.extend([idle] * sph)
    halves.extend([idle] * sph)  # trailing idle pad
    clk_arr = np.asarray(halves, dtype=np.bool_)
    total = clk_arr.size

    clk = DigitalTrace(channel=ChannelId.CH1, levels=clk_arr, t0_s=0.0, dt_s=dt_s)
    want_rising = sample_edge_is_rising(cpol, cpha)
    sample_idx = [e.index for e in find_edges(clk) if e.rising == want_rising]
    assert len(sample_idx) == n_bits, "clock did not produce the expected sampling edges"

    def build(words: list[int]) -> npt.NDArray[np.bool_]:
        arr = np.zeros(total, dtype=np.bool_)
        for k, s in enumerate(sample_idx):
            bit_in_word = k % bits
            bit_pos = (bits - 1 - bit_in_word) if msb_first else bit_in_word
            val = bool((words[k // bits] >> bit_pos) & 1)
            lo = 0 if k == 0 else (sample_idx[k - 1] + s) // 2 + 1
            hi = total if k == n_bits - 1 else (s + sample_idx[k + 1]) // 2 + 1
            arr[lo:hi] = val
        return arr

    mosi = DigitalTrace(channel=ChannelId.CH2, levels=build(mosi_words), t0_s=0.0, dt_s=dt_s)
    miso = (
        DigitalTrace(channel=ChannelId.CH3, levels=build(miso_words), t0_s=0.0, dt_s=dt_s)
        if miso_words is not None
        else None
    )

    cs = None
    if with_cs:
        cs_arr = np.ones(total, dtype=np.bool_)  # idle high (deasserted)
        cs_arr[sph : total - sph] = False  # asserted (low) across the clocked region
        cs = DigitalTrace(channel=ChannelId.CH4, levels=cs_arr, t0_s=0.0, dt_s=dt_s)

    return clk, mosi, miso, cs


def i2c_traces(
    address: int,
    data: list[int],
    *,
    read: bool = False,
    ack: bool = True,
    sph: int = 4,
    dt_s: float = 1e-6,
) -> tuple[DigitalTrace, DigitalTrace]:
    """Build an aligned SDA/SCL pair for one START..STOP transfer.

    Emits the address byte (``address<<1 | r/w``) then each data byte, every byte
    followed by an ACK/NAK. Returns ``(sda, scl)``.
    """
    sda_bits: list[bool] = []
    scl_bits: list[bool] = []

    def seg(scl_level: bool, sda_level: bool) -> None:
        scl_bits.extend([scl_level] * sph)
        sda_bits.extend([sda_level] * sph)

    ack_level = not ack  # ACK pulls SDA low (logic 0)
    byte_values = [(address << 1) | (1 if read else 0), *data]

    seg(True, True)  # idle
    seg(True, False)  # START: SDA falls while SCL high
    for value in byte_values:
        for b in range(8):
            bit = bool((value >> (7 - b)) & 1)  # MSB first
            seg(False, bit)  # set data while SCL low
            seg(True, bit)  # SCL high: bit is sampled
        seg(False, ack_level)  # ACK setup while SCL low
        seg(True, ack_level)  # SCL high: ACK sampled
    seg(False, False)  # prepare STOP: SDA low while SCL low
    seg(True, False)  # SCL high, SDA still low
    seg(True, True)  # STOP: SDA rises while SCL high
    seg(True, True)  # idle

    sda = DigitalTrace(
        channel=ChannelId.CH1, levels=np.asarray(sda_bits, dtype=np.bool_), t0_s=0.0, dt_s=dt_s
    )
    scl = DigitalTrace(
        channel=ChannelId.CH2, levels=np.asarray(scl_bits, dtype=np.bool_), t0_s=0.0, dt_s=dt_s
    )
    return sda, scl
