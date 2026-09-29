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
    def upload_arbitrary(
        self,
        channel: SigGenChannel,
        wave: ArbitraryWaveform,
        *,
        sample_rate_hz: float | None = None,
        slot: int | None = None,
    ) -> None:
        """Store a user-defined waveform, addressed per the instrument's arb model.

        The two families differ (:attr:`SigGenCapabilities.arb_addressing`):

        * **VOLATILE** (e.g. Rigol DG1000Z): the waveform goes to ``channel``'s live
          buffer and is selected for output immediately; ``slot`` must be ``None``.
        * **SLOT** (e.g. JDS6600): the waveform is stored in the device-global
          ``slot`` (1..``arb_slots``), later selected for a channel via
          :attr:`SignalGeneratorConfig.arb_slot`; ``channel`` is unused for storage.

        ``sample_rate_hz`` is the playback clock for instruments that stream the
        buffer point by point (:attr:`SigGenCapabilities.arb_sample_rate`); the output
        then repeats at :func:`~hwtools.model.siggen.arb_repetition_hz`. Omit it to use
        the instrument's documented default — which the driver still states explicitly
        on the wire, so the rate never depends on leftover instrument state. Passing a
        rate to an instrument that has none is an error, not a no-op.

        Raises if the instrument has no arbitrary support, the point count is wrong,
        the rate is unsupported or out of range, or ``slot`` disagrees with the
        addressing mode.

        **Selecting the arbitrary waveform is an output-mode change**, not a
        :class:`~hwtools.model.siggen.WaveShape`: a subsequent
        :meth:`configure_channel` switches the channel back to a built-in waveform and
        discards the arbitrary output.
        """

    @abstractmethod
    def read_arbitrary(
        self, channel: SigGenChannel, *, slot: int | None = None
    ) -> ArbitraryWaveform:
        """Read back a stored waveform (normalised to [-1, 1]).

        ``channel`` and ``slot`` address it the same way as :meth:`upload_arbitrary`.
        """

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
