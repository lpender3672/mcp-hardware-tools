"""SPI decode: all four modes, bit order, CS gating, round-trip."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hwtools.decode.spi import SpiParams, decode_spi, sample_edge_is_rising
from hwtools.model.waveform import DigitalTrace
from tests.fixtures.signals import spi_traces


def test_cs_reasserts_realign_word_boundaries() -> None:
    # Regression (found on hardware): a partial transaction, then an idle gap
    # (CS deasserted with the clock idle, so no clock edge fires), then a full
    # transaction. The decoder must drop the partial and reframe on the new CS
    # assertion rather than running the bit stream on.
    sph = 4
    clk_f, mosi_f, _, cs_f = spi_traces([0xA5, 0x3C], with_cs=True, sph=sph)
    assert cs_f is not None
    partial_clk = np.tile(np.concatenate([np.zeros(sph, bool), np.ones(sph, bool)]), 3)
    partial = partial_clk.size
    gap = 8

    def join(lead_clk: bool, frame: DigitalTrace, cs: bool) -> DigitalTrace:
        levels = np.concatenate(
            [
                np.full(partial, lead_clk, dtype=bool) if cs else partial_clk,
                np.full(gap, cs, dtype=bool),
                frame.levels,
            ]
        )
        return DigitalTrace(channel=frame.channel, levels=levels, t0_s=0.0, dt_s=frame.dt_s)

    clk = join(False, clk_f, cs=False)
    mosi = join(True, mosi_f, cs=False)  # garbage during the partial
    cs = join(False, cs_f, cs=True)  # CS deasserted (high) during the gap

    words = decode_spi(clk, mosi=mosi, cs=cs, params=SpiParams(cs_active_low=True))
    assert [w.mosi for w in words] == [0xA5, 0x3C]


def test_mode_sampling_edges() -> None:
    assert sample_edge_is_rising(0, 0) is True  # mode 0
    assert sample_edge_is_rising(0, 1) is False  # mode 1
    assert sample_edge_is_rising(1, 0) is False  # mode 2
    assert sample_edge_is_rising(1, 1) is True  # mode 3


def test_decode_single_byte_mode0() -> None:
    clk, mosi, _, _ = spi_traces([0xA5])
    words = decode_spi(clk, mosi=mosi)
    assert [w.mosi for w in words] == [0xA5]


def test_decode_both_data_lines() -> None:
    clk, mosi, miso, _ = spi_traces([0x12, 0x34], miso_words=[0xAB, 0xCD])
    words = decode_spi(clk, mosi=mosi, miso=miso)
    assert [(w.mosi, w.miso) for w in words] == [(0x12, 0xAB), (0x34, 0xCD)]


def test_requires_a_data_line() -> None:
    clk, _, _, _ = spi_traces([0x00])
    try:
        decode_spi(clk)
    except ValueError:
        return
    raise AssertionError("expected ValueError when no data line is given")


def test_cs_gates_decoding() -> None:
    clk, mosi, _, cs = spi_traces([0xAA, 0x55], with_cs=True)
    params = SpiParams(cs_active_low=True)
    assert [w.mosi for w in decode_spi(clk, mosi=mosi, cs=cs, params=params)] == [0xAA, 0x55]


@given(
    words=st.lists(st.integers(min_value=0, max_value=255), min_size=1, max_size=6),
    cpol=st.sampled_from([0, 1]),
    cpha=st.sampled_from([0, 1]),
    msb_first=st.booleans(),
)
def test_spi_round_trip(words: list[int], cpol: int, cpha: int, msb_first: bool) -> None:
    clk, mosi, _, _ = spi_traces(words, cpol=cpol, cpha=cpha, msb_first=msb_first)
    params = SpiParams(cpol=cpol, cpha=cpha, msb_first=msb_first)
    decoded = decode_spi(clk, mosi=mosi, params=params)
    assert [w.mosi for w in decoded] == words
