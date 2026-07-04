"""I²C decode: address/RW, ACK, multi-byte payload, round-trip."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hwtools.decode.i2c import decode_i2c
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import DigitalTrace
from tests.fixtures.signals import i2c_traces


def _trace(channel: ChannelId, bits: list[int]) -> DigitalTrace:
    return DigitalTrace(
        channel=channel, levels=np.asarray(bits, dtype=np.bool_), t0_s=0.0, dt_s=1e-6
    )


def test_marginal_sda_edge_near_scl_glitch_is_not_start_stop() -> None:
    # Regression (found on hardware): at coarse capture resolution an SDA edge can
    # land on a one- or two-sample SCL "high" sliver next to an SCL transition.
    # Without debounce the old decoder read that as a spurious START/STOP and
    # truncated the real transaction. SCL is high at indices 2-3 only; the SDA
    # fall there must be rejected, leaving no transaction.
    scl = _trace(ChannelId.CH1, [0, 0, 1, 1, 0, 0])
    sda = _trace(ChannelId.CH2, [1, 1, 1, 0, 0, 0])
    assert decode_i2c(sda, scl) == []


def test_write_transaction_address_and_data() -> None:
    sda, scl = i2c_traces(0x50, [0xDE, 0xAD], read=False)
    txns = decode_i2c(sda, scl)
    assert len(txns) == 1
    txn = txns[0]
    assert txn.address == 0x50
    assert txn.read is False
    assert txn.data == [0xDE, 0xAD]
    assert all(b.ack for b in txn.bytes)


def test_read_bit_is_recovered() -> None:
    sda, scl = i2c_traces(0x3C, [0x01], read=True)
    txn = decode_i2c(sda, scl)[0]
    assert txn.address == 0x3C
    assert txn.read is True


def test_nak_is_flagged() -> None:
    sda, scl = i2c_traces(0x50, [0x00], ack=False)
    txn = decode_i2c(sda, scl)[0]
    assert all(not b.ack for b in txn.bytes)


@given(
    address=st.integers(min_value=0, max_value=0x7F),
    data=st.lists(st.integers(min_value=0, max_value=255), max_size=5),
    read=st.booleans(),
)
def test_i2c_round_trip(address: int, data: list[int], read: bool) -> None:
    sda, scl = i2c_traces(address, data, read=read)
    txn = decode_i2c(sda, scl)[0]
    assert txn.address == address
    assert txn.read is read
    assert txn.data == data
