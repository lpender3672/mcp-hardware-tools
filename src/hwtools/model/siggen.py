"""Signal-generator value objects — the typed surface a siggen driver speaks.

Mirrors the scope side (:mod:`hwtools.model.channel` / ``capability``): vendor-neutral
enums and frozen configs in SI base units, with the unit baked into each field name.
A concrete driver (e.g. :class:`~hwtools.drivers.joyit.jds6600.JDS6600`) is the only
place these map to instrument wire codes.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SigGenChannel(IntEnum):
    """Signal-generator output channel, 1-indexed to match the front panel."""

    CH1 = 1
    CH2 = 2


class WaveShape(StrEnum):
    """A built-in waveform shape.

    Vendor-neutral names; the driver maps each to its instrument's code. Arbitrary
    (user-uploaded) waveforms are intentionally absent — that is a separate,
    under-specified upload path (JDS6600 ``a``/``b``), deferred to a later pass.
    """

    SINE = "SINE"
    SQUARE = "SQUARE"
    PULSE = "PULSE"
    TRIANGLE = "TRIANGLE"
    PARTIAL_SINE = "PARTIAL_SINE"
    CMOS = "CMOS"
    DC = "DC"
    HALF_WAVE = "HALF_WAVE"
    FULL_WAVE = "FULL_WAVE"
    POS_STEP = "POS_STEP"
    NEG_STEP = "NEG_STEP"
    NOISE = "NOISE"
    EXP_RISE = "EXP_RISE"
    EXP_DECAY = "EXP_DECAY"
    MULTI_TONE = "MULTI_TONE"
    SINC = "SINC"
    LORENZ = "LORENZ"


class SignalGeneratorConfig(BaseModel):
    """A single output channel's setup: shape, frequency, level, offset, duty.

    Amplitude is peak-to-peak (the instrument's front-panel knob). Structural bounds
    only are enforced here (frequency > 0, amplitude ≥ 0, duty a percentage);
    device-specific limits (max frequency/amplitude, offset range) are the driver's
    to enforce against its :class:`SigGenCapabilities`, so it can throw a descriptive
    error rather than silently clamp.
    """

    model_config = ConfigDict(frozen=True)

    channel: SigGenChannel
    waveform: WaveShape = WaveShape.SINE
    frequency_hz: float = Field(gt=0)
    amplitude_vpp: float = Field(ge=0, description="Peak-to-peak amplitude, volts.")
    offset_v: float = Field(default=0.0, description="DC bias added to the output, volts.")
    duty_pct: float = Field(default=50.0, ge=0, le=100, description="Duty cycle, percent.")
    enabled: bool = Field(default=True, description="Whether the output is driven.")
    arb_slot: int | None = Field(
        default=None,
        description="Play a stored arbitrary slot instead of ``waveform``; None = built-in.",
    )


class ArbitraryWaveform(BaseModel):
    """One period of a user-defined waveform, normalised to [-1, 1].

    Vendor-neutral: samples are the *shape* only, independent of any instrument's
    point count or DAC resolution — the driver resamples/quantises to its hardware
    (e.g. the JDS6600's 2048 points at 12-bit) and refuses a length its device can't
    take rather than silently resampling. Output amplitude/offset are applied by the
    channel config on top, so a full-scale ±1 buffer spans the configured Vpp.
    """

    model_config = ConfigDict(frozen=True)

    samples: tuple[float, ...] = Field(description="Normalised samples in [-1, 1], one period.")

    @field_validator("samples")
    @classmethod
    def _in_unit_range(cls, samples: tuple[float, ...]) -> tuple[float, ...]:
        if not samples:
            raise ValueError("an arbitrary waveform needs at least one sample")
        if any(not -1.0 <= s <= 1.0 for s in samples):
            raise ValueError("arbitrary-waveform samples must be normalised to [-1, 1]")
        return samples

    @property
    def n(self) -> int:
        return len(self.samples)

    @classmethod
    def normalised(cls, samples: npt.ArrayLike) -> ArbitraryWaveform:
        """Build from raw samples, scaling the peak magnitude to full-scale ±1.

        The convenience path for turning any generated buffer (a noise realisation,
        a hand-built shape) into an uploadable waveform. Raises on an all-zero input,
        which has no meaningful full-scale.
        """
        values = np.asarray(samples, dtype=np.float64)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak == 0.0:
            raise ValueError("cannot normalise an all-zero (or empty) sample buffer")
        return cls(samples=tuple(values / peak))


class SigGenCapabilities(BaseModel):
    """Static limits/feature set for a signal-generator model.

    The loop reads these to stay in range (don't ask for a frequency the instrument
    can't reach) and to know which waveforms exist without hard-coding a vendor.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    n_channels: int = Field(gt=0)
    max_frequency_hz: float = Field(gt=0)
    max_amplitude_vpp: float = Field(gt=0)
    max_offset_v: float = Field(gt=0, description="Symmetric bias limit: |offset| ≤ this.")
    waveforms: tuple[WaveShape, ...] = ()
    # Arbitrary-waveform support (all zero = none): number of storable slots, the
    # exact point count each slot holds, and the DAC code resolution (levels).
    arb_slots: int = Field(default=0, ge=0)
    arb_points: int = Field(default=0, ge=0)
    arb_code_levels: int = Field(default=0, ge=0)

    @property
    def channels(self) -> tuple[SigGenChannel, ...]:
        return tuple(SigGenChannel(i) for i in range(1, self.n_channels + 1))

    def has_channel(self, channel: SigGenChannel) -> bool:
        return int(channel) <= self.n_channels


def check_within(config: SignalGeneratorConfig, caps: SigGenCapabilities) -> None:
    """Raise ``ValueError`` if ``config`` asks for anything ``caps`` cannot deliver.

    The shared enforcement point for every driver: a request outside the
    instrument's limits fails loudly and descriptively rather than being silently
    clamped, so the caller (human or agent) sees the divergence and can correct it.
    """
    if not caps.has_channel(config.channel):
        raise ValueError(f"{caps.model_name} has no channel {config.channel}")
    if config.arb_slot is not None:
        if caps.arb_slots == 0:
            raise ValueError(f"{caps.model_name} has no arbitrary-waveform support")
        if not 1 <= config.arb_slot <= caps.arb_slots:
            raise ValueError(
                f"arbitrary slot must be 1..{caps.arb_slots}; got {config.arb_slot}"
            )
    elif caps.waveforms and config.waveform not in caps.waveforms:
        raise ValueError(f"{caps.model_name} does not support the {config.waveform} waveform")
    if config.frequency_hz > caps.max_frequency_hz:
        raise ValueError(
            f"frequency {config.frequency_hz:g} Hz exceeds the "
            f"{caps.max_frequency_hz:g} Hz limit of the {caps.model_name}"
        )
    if config.amplitude_vpp > caps.max_amplitude_vpp:
        raise ValueError(
            f"amplitude {config.amplitude_vpp:g} Vpp exceeds the "
            f"{caps.max_amplitude_vpp:g} Vpp limit of the {caps.model_name}"
        )
    if abs(config.offset_v) > caps.max_offset_v:
        raise ValueError(
            f"offset {config.offset_v:g} V exceeds the ±{caps.max_offset_v:g} V "
            f"limit of the {caps.model_name}"
        )
