"""The oscilloscope abstraction.

A thin, synchronous "set this, read that" contract. Every method speaks in
`hwtools.model` value objects, never instrument wire strings — so the
self-correcting loop above reasons over typed state and the same loop drives a
Rigol, a Keysight, a Teledyne, or the simulated scope without changing.

Acquisition is split deliberately: configure_* set up the instrument, the
run/stop/single controls arm it, and capture() downloads whatever the last
acquisition produced. The loop composes these (e.g. single() then capture()).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from hwtools.model.acquire import AcquireConfig
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig


class Oscilloscope(ABC):
    """Driver-agnostic control surface for a digital storage oscilloscope."""

    # -- identity & connection ------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> ScopeCapabilities:
        """Static limits/feature set for this instrument model."""

    @abstractmethod
    def connect(self) -> None:
        """Open the link to the instrument."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link to the instrument."""

    @abstractmethod
    def idn(self) -> str:
        """Return the instrument identification string."""

    # -- configuration --------------------------------------------------------

    @abstractmethod
    def configure_channel(self, config: ChannelConfig) -> None:
        """Apply a channel's coupling, scale, offset, probe, etc."""

    @abstractmethod
    def configure_timebase(self, config: TimebaseConfig) -> None:
        """Apply the horizontal scale/offset/mode."""

    @abstractmethod
    def configure_trigger(self, config: TriggerConfig) -> None:
        """Apply the trigger setup."""

    @abstractmethod
    def configure_acquire(self, config: AcquireConfig) -> None:
        """Apply the acquisition processing/memory setup."""

    @abstractmethod
    def autoscale(self) -> None:
        """Let the instrument pick a first-guess setup."""

    # -- run control ----------------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """Start continuous acquisition."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition."""

    @abstractmethod
    def single(self) -> None:
        """Arm a single acquisition."""

    @abstractmethod
    def force_trigger(self) -> None:
        """Force a trigger now."""

    @abstractmethod
    def trigger_status(self) -> TriggerStatus:
        """Report the current trigger-system state."""

    # -- readout --------------------------------------------------------------

    @abstractmethod
    def capture(self, channels: Sequence[ChannelId], *, deep: bool = True) -> Capture:
        """Download the last acquisition for the given channels.

        ``deep`` (default) reads the full acquisition memory — the samples the
        instrument actually captured, governed by the configured memory depth.
        ``deep=False`` reads only the decimated on-screen trace: far fewer points
        and far faster, but it aliases fast edges, so it's for a quick look only.
        """

    # -- context manager ------------------------------------------------------

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
