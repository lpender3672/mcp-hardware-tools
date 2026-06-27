"""IEEE 488.2 definite-length block parsing.

Block responses look like ``#<ndigits><length><payload>``: a ``#``, one digit
giving the number of length digits, that many digits giving the payload byte
count, then the payload. Used by every transport that pulls waveform data.
"""

from __future__ import annotations

from collections.abc import Callable


def decode_definite_block(raw: bytes) -> bytes:
    """Extract the payload from a complete definite-length block in ``raw``."""
    if not raw.startswith(b"#"):
        raise ValueError(f"not a definite-length block: {raw[:8]!r}")
    ndigits = int(raw[1:2])
    if ndigits == 0:
        raise ValueError("indefinite-length blocks (#0) are not supported")
    length = int(raw[2 : 2 + ndigits])
    start = 2 + ndigits
    return raw[start : start + length]


def read_definite_block(recv_exact: Callable[[int], bytes]) -> bytes:
    """Stream a definite-length block using a ``recv_exact(n) -> n bytes`` reader.

    Lets a socket read exactly the header then exactly the payload, without
    guessing how much to receive.
    """
    if recv_exact(1) != b"#":
        raise ValueError("stream is not positioned at a definite-length block")
    ndigits = int(recv_exact(1))
    if ndigits == 0:
        raise ValueError("indefinite-length blocks (#0) are not supported")
    length = int(recv_exact(ndigits))
    return recv_exact(length)
