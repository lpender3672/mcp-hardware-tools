"""Layer 4 (pure) — protocol decode over captured sample arrays.

No instrument awareness: every function here takes model value objects and
returns typed frames, so the bulk of the correctness is unit-testable against
synthetic waveforms with no hardware attached.
"""

from hwtools.decode.frames import UartWord
from hwtools.decode.threshold import Edge, find_edges, threshold
from hwtools.decode.uart import Parity, UartParams, decode_uart

__all__ = [
    "Edge",
    "Parity",
    "UartParams",
    "UartWord",
    "decode_uart",
    "find_edges",
    "threshold",
]
