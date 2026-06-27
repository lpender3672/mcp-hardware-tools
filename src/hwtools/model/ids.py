"""Closed enumerations shared across the model.

No magic strings cross a layer boundary: anywhere a value is one-of-a-fixed-set
(a channel, a coupling, an edge slope) it is one of these enums. Drivers are the
only place these map to/from instrument-specific wire strings.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class ChannelId(IntEnum):
    """Analog input channel, 1-indexed to match scope front-panel labelling."""

    CH1 = 1
    CH2 = 2
    CH3 = 3
    CH4 = 4


class Coupling(StrEnum):
    """Input coupling for a channel."""

    AC = "AC"
    DC = "DC"
    GND = "GND"


class Slope(StrEnum):
    """Edge direction for an edge trigger."""

    RISING = "RISING"
    FALLING = "FALLING"
    EITHER = "EITHER"


class SweepMode(StrEnum):
    """How the trigger re-arms between acquisitions."""

    AUTO = "AUTO"
    NORMAL = "NORMAL"
    SINGLE = "SINGLE"


class AcqType(StrEnum):
    """Acquisition processing applied as samples are captured."""

    NORMAL = "NORMAL"
    AVERAGE = "AVERAGE"
    PEAK = "PEAK"
    HIGH_RES = "HIGH_RES"
