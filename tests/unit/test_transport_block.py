"""IEEE 488.2 definite-length block parsing — buffered and streamed."""

from __future__ import annotations

import pytest

from hwtools.transport.block import decode_definite_block, read_definite_block


def test_decode_buffered_block() -> None:
    assert decode_definite_block(b"#800000003ABC") == b"ABC"
    assert decode_definite_block(b"#9000000004\x01\x02\x03\x04") == b"\x01\x02\x03\x04"


def test_decode_rejects_non_block() -> None:
    with pytest.raises(ValueError):
        decode_definite_block(b"ABC")


def test_read_streamed_block_pulls_exact_lengths() -> None:
    stream = bytearray(b"#3005hello")
    reads: list[int] = []

    def recv_exact(n: int) -> bytes:
        reads.append(n)
        chunk = bytes(stream[:n])
        del stream[:n]
        return chunk

    assert read_definite_block(recv_exact) == b"hello"
    # header byte, ndigits byte, the length digits, then exactly the payload
    assert reads == [1, 1, 3, 5]
