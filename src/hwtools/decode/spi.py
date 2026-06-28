"""SPI decode.

Samples the data line(s) on the active clock edge selected by CPOL/CPHA, packs
bits into words (MSB-first by default), and — when a chip-select line is given —
only counts clocks while the device is selected, resetting partial words between
selections. Pure: aligned :class:`DigitalTrace` inputs, typed
:class:`~hwtools.decode.frames.SpiWord` list out.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hwtools.decode.frames import SpiWord
from hwtools.decode.threshold import find_edges
from hwtools.model.waveform import DigitalTrace


class SpiParams(BaseModel):
    """SPI framing parameters."""

    model_config = ConfigDict(frozen=True)

    cpol: int = Field(default=0, ge=0, le=1, description="Clock idle level.")
    cpha: int = Field(default=0, ge=0, le=1, description="Clock phase (which edge samples).")
    bits: int = Field(default=8, ge=1, le=32)
    msb_first: bool = True
    cs_active_low: bool = True


def sample_edge_is_rising(cpol: int, cpha: int) -> bool:
    """Which clock-edge direction latches data, from CPOL/CPHA.

    The leading edge rises when the clock idles low (CPOL=0); CPHA=0 samples that
    leading edge, CPHA=1 the trailing one. (Modes 0 and 3 sample rising.)
    """
    leading_is_rising = cpol == 0
    samples_leading = cpha == 0
    return leading_is_rising if samples_leading else not leading_is_rising


def decode_spi(
    clk: DigitalTrace,
    *,
    mosi: DigitalTrace | None = None,
    miso: DigitalTrace | None = None,
    cs: DigitalTrace | None = None,
    params: SpiParams | None = None,
) -> list[SpiWord]:
    """Recover SPI words from a clock line plus one or both data lines."""
    if mosi is None and miso is None:
        raise ValueError("decode_spi needs at least one of mosi/miso")
    p = params or SpiParams()

    want_rising = sample_edge_is_rising(p.cpol, p.cpha)
    sample_edges = [e for e in find_edges(clk) if e.rising == want_rising]

    words: list[SpiWord] = []
    mosi_acc = 0
    miso_acc = 0
    count = 0
    word_start_s = 0.0
    prev_idx: int | None = None

    for edge in sample_edges:
        idx = edge.index
        if cs is not None:
            asserted = (not bool(cs.levels[idx])) if p.cs_active_low else bool(cs.levels[idx])
            if not asserted:
                count = mosi_acc = miso_acc = 0  # drop any partial word
                prev_idx = idx
                continue
            # A new transaction starts whenever CS deasserted since the previous
            # clock edge — even across an idle gap where no clock edge fires, so
            # word boundaries realign per CS assertion rather than running on.
            if prev_idx is not None:
                gap = cs.levels[prev_idx:idx]
                deasserted_between = bool(gap.any()) if p.cs_active_low else bool((~gap).any())
                if deasserted_between:
                    count = mosi_acc = miso_acc = 0
            prev_idx = idx

        if count == 0:
            word_start_s = edge.time_s
        bit_pos = (p.bits - 1 - count) if p.msb_first else count
        if mosi is not None and bool(mosi.levels[idx]):
            mosi_acc |= 1 << bit_pos
        if miso is not None and bool(miso.levels[idx]):
            miso_acc |= 1 << bit_pos
        count += 1

        if count == p.bits:
            words.append(
                SpiWord(
                    mosi=mosi_acc if mosi is not None else None,
                    miso=miso_acc if miso is not None else None,
                    t_start_s=word_start_s,
                )
            )
            count = mosi_acc = miso_acc = 0

    return words
