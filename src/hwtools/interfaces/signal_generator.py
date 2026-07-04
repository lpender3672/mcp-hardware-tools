"""The signal-generator abstraction — the output-side twin of :class:`Oscilloscope`.

A thin, synchronous "set this, read that" contract speaking only in
:mod:`hwtools.model` value objects, never instrument wire strings. Configuration is
per-channel and idempotent: :meth:`configure_channel` applies a whole
:class:`SignalGeneratorConfig` at once, and :meth:`read_channel` reads it back — the
honest self-consistency check on a device whose only observable is what you wrote.
The *real* validation of the output lives one instrument over: drive the siggen,
measure it on a scope (see the cross-instrument HIL tests).

Output enable is separated from configuration because on real hardware it is a
shared, both-channels-at-once register; a driver read-modify-writes it so enabling
one channel does not disturb the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from hwtools.model.siggen import (
    ArbitraryWaveform,
    SigGenCapabilities,
    SigGenChannel,
    SignalGeneratorConfig,
)


class SignalGenerator(ABC):
    """Driver-agnostic control surface for a function/arbitrary waveform generator."""

    # -- identity & connection ------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> SigGenCapabilities:
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
    def configure_channel(self, config: SignalGeneratorConfig) -> None:
        """Apply a channel's waveform, frequency, amplitude, offset, duty and output
        state in one shot. Raises if any value exceeds the instrument's limits."""

    @abstractmethod
    def enable_output(self, channel: SigGenChannel, on: bool) -> None:
        """Turn one channel's output on/off without disturbing the other channel."""

    @abstractmethod
    def set_phase_deg(self, degrees: float) -> None:
        """Set the phase of CH2 relative to CH1, in degrees (0..360)."""

    # -- arbitrary waveforms --------------------------------------------------

    @abstractmethod
    def upload_arbitrary(self, slot: int, wave: ArbitraryWaveform) -> None:
        """Store a user-defined waveform into a slot. Raises if the instrument has no
        arbitrary support, the slot is out of range, or the point count is wrong."""

    @abstractmethod
    def read_arbitrary(self, slot: int) -> ArbitraryWaveform:
        """Read back the waveform stored in a slot (normalised to [-1, 1])."""

    # -- readback -------------------------------------------------------------

    @abstractmethod
    def read_channel(self, channel: SigGenChannel) -> SignalGeneratorConfig:
        """Read back a channel's current setup from the instrument."""

    @abstractmethod
    def output_enabled(self, channel: SigGenChannel) -> bool:
        """Report whether a channel's output is currently driven."""

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
