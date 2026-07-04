"""UART decode: directed cases, error flags, and a hypothesis round-trip."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hwtools.decode.threshold import threshold
from hwtools.decode.uart import Parity, UartParams, decode_uart
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import DigitalTrace
from tests.fixtures.signals import trace_to_waveform, uart_trace


def test_decode_single_byte_0xa5() -> None:
    trace = uart_trace([0xA5], baud=115200)
    words = decode_uart(trace, UartParams(baud=115200))
    assert [w.value for w in words] == [0xA5]
    assert words[0].framing_error is False


def test_decode_msb_first() -> None:
    trace = uart_trace([0xA5], baud=9600, lsb_first=False)
    words = decode_uart(trace, UartParams(baud=9600, lsb_first=False))
    assert [w.value for w in words] == [0xA5]


def test_decode_empty_trace_returns_nothing() -> None:
    empty = DigitalTrace(
        channel=ChannelId.CH1, levels=np.array([], dtype=np.bool_), t0_s=0, dt_s=1e-6
    )
    assert decode_uart(empty, UartParams(baud=115200)) == []


def test_even_parity_accepts_matching_word() -> None:
    trace = uart_trace([0x3C], baud=19200, parity=Parity.EVEN)
    words = decode_uart(trace, UartParams(baud=19200, parity=Parity.EVEN))
    assert [w.value for w in words] == [0x3C]
    assert words[0].parity_error is False


def test_through_analog_threshold_pipeline() -> None:
    # Full path: logic -> analog volts -> threshold -> decode.
    trace = uart_trace([0x55, 0xAA], baud=57600)
    analog = trace_to_waveform(trace, low_v=0.0, high_v=3.3)
    redigitized = threshold(analog, level_v=1.65, hysteresis_v=0.4)
    words = decode_uart(redigitized, UartParams(baud=57600))
    assert [w.value for w in words] == [0x55, 0xAA]


@given(
    values=st.lists(st.integers(min_value=0, max_value=255), max_size=8),
    baud=st.sampled_from([9600, 57600, 115200]),
    lsb_first=st.booleans(),
    parity=st.sampled_from(list(Parity)),
)
def test_uart_round_trip(values: list[int], baud: int, lsb_first: bool, parity: Parity) -> None:
    trace = uart_trace(values, baud=baud, lsb_first=lsb_first, parity=parity)
    words = decode_uart(trace, UartParams(baud=baud, lsb_first=lsb_first, parity=parity))
    assert [w.value for w in words] == values
    assert all(not w.framing_error and not w.parity_error for w in words)
