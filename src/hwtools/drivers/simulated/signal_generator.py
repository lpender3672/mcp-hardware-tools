"""A simulated signal generator over the same :class:`SignalGenerator` contract.

Its only observable is what you wrote — like the real device — so it stores each
channel's config (and any uploaded arbitrary slot) and echoes it back. That is
enough to hold the sim to the same behavioural contract as the real driver in CI,
with no hardware.
"""

from __future__ import annotations

from hwtools.interfaces.signal_generator import SignalGenerator
from hwtools.model.siggen import (
    ArbAddressing,
    ArbitraryWaveform,
    ArbLength,
    SigGenCapabilities,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
    check_arbitrary_length,
    check_within,
    resolve_arb_sample_rate,
)

DEFAULT_CAPABILITIES = SigGenCapabilities(
    model_name="SimulatedSignalGenerator",
    n_channels=2,
    max_frequency_hz=15e6,
    max_amplitude_vpp=20.0,
    max_offset_v=9.99,
    waveforms=tuple(WaveShape),
    arb_slots=60,
    arb_length=ArbLength.fixed(2048),
    arb_code_levels=4096,
)


class SimulatedSignalGenerator(SignalGenerator):
    """A hardware-free signal generator that stores and echoes its configuration."""

    def __init__(self, *, capabilities: SigGenCapabilities = DEFAULT_CAPABILITIES) -> None:
        self._caps = capabilities
        self._channels: dict[SigGenChannel, SignalGeneratorConfig] = {}
        self._arb: dict[int, ArbitraryWaveform] = {}  # SLOT storage: slot -> waveform
        self._volatile: dict[SigGenChannel, ArbitraryWaveform] = {}  # VOLATILE: channel -> buffer
        self._phase_deg = 0.0
        self._connected = False

    @property
    def capabilities(self) -> SigGenCapabilities:
        return self._caps

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def idn(self) -> str:
        return self._caps.model_name

    def configure_channel(self, config: SignalGeneratorConfig) -> None:
        check_within(config, self._caps)
        self._channels[config.channel] = config

    def enable_output(self, channel: SigGenChannel, on: bool) -> None:
        current = self._channels.get(channel)
        if current is None:
            raise RuntimeError(f"channel {channel} has not been configured")
        self._channels[channel] = current.model_copy(update={"enabled": on})

    def set_phase_deg(self, degrees: float) -> None:
        if not 0.0 <= degrees <= 360.0:
            raise ValueError("phase must be within 0..360 degrees")
        self._phase_deg = degrees

    def upload_arbitrary(
        self,
        channel: SigGenChannel,
        wave: ArbitraryWaveform,
        *,
        sample_rate_hz: float | None = None,
        slot: int | None = None,
    ) -> None:
        check_arbitrary_length(wave.n, self._caps)
        # Called for its validation, not its value: the rate has no effect on a stored
        # buffer, but a rate this profile cannot accept must still be refused so the sim
        # holds callers to the same contract a real instrument does.
        resolve_arb_sample_rate(sample_rate_hz, self._caps)
        if self._caps.arb_addressing is ArbAddressing.VOLATILE:
            self._reject_slot(slot)
            self._volatile[channel] = wave
        else:
            self._arb[self._require_slot(slot)] = wave

    def read_arbitrary(
        self, channel: SigGenChannel, *, slot: int | None = None
    ) -> ArbitraryWaveform:
        if self._caps.arb_addressing is ArbAddressing.VOLATILE:
            self._reject_slot(slot)
            wave = self._volatile.get(channel)
            if wave is None:
                raise RuntimeError(f"channel {channel} has no volatile arbitrary waveform")
            return wave
        wave = self._arb.get(self._require_slot(slot))
        if wave is None:
            raise RuntimeError(f"arbitrary slot {slot} is empty")
        return wave

    def _require_slot(self, slot: int | None) -> int:
        if slot is None:
            raise ValueError(f"this instrument needs a slot=1..{self._caps.arb_slots}")
        if not 1 <= slot <= self._caps.arb_slots:
            raise ValueError(f"arbitrary slot must be 1..{self._caps.arb_slots}; got {slot}")
        return slot

    @staticmethod
    def _reject_slot(slot: int | None) -> None:
        if slot is not None:
            raise ValueError("volatile addressing takes no slot; pass slot=None")

    def read_channel(self, channel: SigGenChannel) -> SignalGeneratorConfig:
        config = self._channels.get(channel)
        if config is None:
            raise RuntimeError(f"channel {channel} has not been configured")
        return config

    def output_enabled(self, channel: SigGenChannel) -> bool:
        config = self._channels.get(channel)
        return bool(config and config.enabled)
