"""Simulated instruments — synthetic backends implementing the real interfaces.

A first-class :class:`SimulatedScope` makes the whole capture/judge/adjust loop
runnable in CI with no hardware.
"""

from hwtools.drivers.simulated.scope import DEFAULT_CAPABILITIES, SimulatedScope
from hwtools.drivers.simulated.signals import Dc, SimSignal, Sine, Square

__all__ = ["DEFAULT_CAPABILITIES", "Dc", "SimSignal", "SimulatedScope", "Sine", "Square"]
