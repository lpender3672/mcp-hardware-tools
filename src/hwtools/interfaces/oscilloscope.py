"""The oscilloscope abstraction.

A thin, synchronous "set this, read that" contract. Every method speaks in
`hwtools.model` value objects, never instrument wire strings — so the
self-correcting loop above reasons over typed state and the same loop drives a
Rigol, a Keysight, a Teledyne, or the simulated scope without changing.

This is an M0 skeleton: the method set is fixed here so tests and the loop can be
written against it; concrete behaviour and the full model types arrive in later
milestones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class Oscilloscope(ABC):
    """Driver-agnostic control surface for a digital storage oscilloscope."""

    # -- connection -----------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Open the link to the instrument."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link to the instrument."""

    @abstractmethod
    def idn(self) -> str:
        """Return the instrument identification string."""

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()
