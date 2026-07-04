"""Layer 4 (pure) — protocol decode over captured sample arrays.

No instrument awareness: every function here takes model value objects and
returns typed frames, so the bulk of the correctness is unit-testable against
synthetic waveforms with no hardware attached.
"""

from hwtools.decode.frames import I2cByte, I2cTransaction, SpiWord, UartWord
from hwtools.decode.i2c import decode_i2c
from hwtools.decode.spi import SpiParams, decode_spi, sample_edge_is_rising
from hwtools.decode.threshold import Edge, find_edges, threshold
from hwtools.decode.uart import Parity, UartParams, decode_uart

__all__ = [
    "Edge",
    "I2cByte",
    "I2cTransaction",
    "Parity",
    "SpiParams",
    "SpiWord",
    "UartParams",
    "UartWord",
    "decode_i2c",
    "decode_spi",
    "decode_uart",
    "find_edges",
    "sample_edge_is_rising",
    "threshold",
]
