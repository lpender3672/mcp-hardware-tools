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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class PlaybackMode(StrEnum):
    """A device's output generation/repetition mode, beyond plain continuous output.

    Declarative only: :class:`SigGenCapabilities` advertises which modes an instrument
    supports so the loop can *refuse* an unsupported one (a burst on the
    continuous-only JDS6600) rather than silently ignore it — per the tool-design
    philosophy. Driver support for each mode is wired in as a consumer needs it; the
    JDS6600 is ``{CONTINUOUS}``, a Rigol DG1000Z adds the rest.
    """

    CONTINUOUS = "CONTINUOUS"
    BURST = "BURST"  # N-cycle / infinite / gated pulse trains — single-shot is N=1
    SWEEP = "SWEEP"
    MODULATION = "MODULATION"


class ArbAddressing(StrEnum):
    """How an instrument addresses its arbitrary-waveform storage.

    The two families need different call shapes, so a driver advertises which it is
    (:attr:`SigGenCapabilities.arb_addressing`) and callers/​the contract branch on it:

    * ``SLOT`` — a bank of numbered, device-global slots (the JDS6600's 60-entry DDS
      wavetable). A waveform is uploaded to a slot, and a channel *selects* it for
      output via :attr:`SignalGeneratorConfig.arb_slot`.
    * ``VOLATILE`` — one live buffer *per channel* (a Rigol DG1000Z). A waveform is
      uploaded to a channel and plays immediately; there is no numbered bank
      (``arb_slots == 0``) and no slot to select.
    """

    SLOT = "SLOT"
    VOLATILE = "VOLATILE"


# Shapes for which a config field carries no physical meaning: noise has no frequency,
# and DC has neither frequency nor peak-to-peak amplitude. Declared here because
# SignalGeneratorConfig needs them to know when a zero frequency is legitimate rather
# than a mistake, and check_within needs them to skip the lower limits.
_NO_FREQUENCY = frozenset({WaveShape.DC, WaveShape.NOISE})
_NO_AMPLITUDE = frozenset({WaveShape.DC})


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
    # ``allow_inf_nan=False`` on every field whose range is a real physical limit. A NaN
    # otherwise passes *every* bounds comparison, since ``nan > limit`` is False, and
    # reaches the instrument as the literal "nan" — `offset_v`, which has no bounds of
    # its own, was exactly that hole. Note it is per-field, not model-wide:
    # `output_load_ohms` legitimately uses ``inf`` to mean high-Z.
    #
    # ge rather than gt on the frequency: DC and noise have no frequency, and 0 is how
    # that is said. Any other shape must be positive — enforced by _frequency_applies
    # below, so the weaker field bound does not let a real mistake through.
    frequency_hz: float = Field(ge=0, allow_inf_nan=False)
    amplitude_vpp: float = Field(
        ge=0, allow_inf_nan=False, description="Peak-to-peak amplitude, volts."
    )
    offset_v: float = Field(
        default=0.0, allow_inf_nan=False, description="DC bias added to the output, volts."
    )
    duty_pct: float = Field(
        default=50.0, ge=0, le=100, allow_inf_nan=False, description="Duty cycle, percent."
    )
    enabled: bool = Field(default=True, description="Whether the output is driven.")
    arb_slot: int | None = Field(
        default=None,
        description="Play a stored arbitrary slot instead of ``waveform``; None = built-in. "
        "Only valid on SLOT-addressed instruments (see :class:`ArbAddressing`).",
    )
    output_load_ohms: float = Field(
        default=float("inf"),
        gt=0,
        description="Load impedance the output drives into, ohms; ``inf`` = high-Z. "
        "Load-referred instruments (Rigol) scale amplitude/offset by this, so it must "
        "match the real load (a high-Z scope) or the delivered level is wrong; "
        "fixed-source instruments (JDS6600) do not compensate and ignore it.",
    )

    @model_validator(mode="after")
    def _frequency_applies(self) -> SignalGeneratorConfig:
        """Only DC and noise may carry a zero frequency, because only they have none.

        Lets a driver read a channel back honestly — a DC channel's ``:FREQuency``
        register is meaningless, and reporting ``0.0`` says so, where inventing some
        plausible-looking value would quietly lie about what the instrument is doing.
        """
        if self.frequency_hz == 0.0 and self.waveform not in _NO_FREQUENCY:
            raise ValueError(
                f"{self.waveform} needs a frequency above 0 Hz; only "
                f"{'/'.join(sorted(s.value for s in _NO_FREQUENCY))} may be 0"
            )
        return self


class ArbitraryWaveform(BaseModel):
    """One period of a user-defined waveform, normalised to [-1, 1].

    Vendor-neutral: samples are the *shape* only, independent of any instrument's
    point count or DAC resolution — the driver resamples/quantises to its hardware
    (the JDS6600's fixed 2048 points at 12-bit, or a Rigol DG1000Z's 8..16384 points
    at 14-bit) and refuses a length its device can't take rather than silently
    resampling. Output amplitude/offset are applied by the channel config on top, so
    a full-scale ±1 buffer spans the configured Vpp.

    Backed by a read-only float64 ``ndarray`` (not a Python tuple) so a deep buffer —
    tens of thousands of points on a modern AWG — stays compact and its bounds check
    stays vectorised rather than a per-sample Python loop.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    samples: np.ndarray = Field(description="Normalised samples in [-1, 1], one period.")

    @field_validator("samples", mode="before")
    @classmethod
    def _as_unit_array(cls, value: npt.ArrayLike) -> np.ndarray:
        arr = np.array(value, dtype=np.float64)  # own a fresh, contiguous copy
        if arr.ndim != 1:
            raise ValueError("an arbitrary waveform must be a 1-D sample buffer")
        if arr.size == 0:
            raise ValueError("an arbitrary waveform needs at least one sample")
        if not bool(np.all((arr >= -1.0) & (arr <= 1.0))):
            raise ValueError("arbitrary-waveform samples must be normalised to [-1, 1]")
        arr.setflags(write=False)  # immutable contents, matching the frozen model
        return arr

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArbitraryWaveform) and np.array_equal(
            self.samples, other.samples
        )

    def __hash__(self) -> int:
        return hash(self.samples.tobytes())

    @property
    def n(self) -> int:
        return int(self.samples.size)

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
        return cls(samples=values / peak)


class ArbLength(BaseModel):
    """The point-count constraint on an instrument's arbitrary-waveform buffer.

    One shape covers both device families rather than forking into "fixed" vs
    "variable" waveform types:

    * **Fixed-length** (the JDS6600's DDS wavetable is always exactly 2048 points)
      sets ``min_points == max_points`` — :attr:`is_fixed` is simply that equality.
    * **Variable-length** (a Rigol DG1000Z accepts 8..16384 points per remote
      download) sets a real range, with an optional :attr:`granularity` the length
      must be a multiple of.

    ``max_points`` is the largest buffer a *single* upload accepts — not necessarily
    the device's total arb memory (the DG1000Z streams beyond 16 kpts via multi-packet
    ``DAC16`` transfers, a separate driver capability). Keeping the two distinct means
    :meth:`accepts` refuses what a plain upload cannot deliver, loudly, instead of the
    driver silently truncating.
    """

    model_config = ConfigDict(frozen=True)

    min_points: int = Field(gt=0)
    max_points: int = Field(gt=0)
    granularity: int = Field(default=1, gt=0, description="Length must be a multiple of this.")

    @model_validator(mode="after")
    def _consistent(self) -> ArbLength:
        if self.max_points < self.min_points:
            raise ValueError(
                f"max_points ({self.max_points}) must be >= min_points ({self.min_points})"
            )
        if self.min_points % self.granularity or self.max_points % self.granularity:
            raise ValueError(
                f"min/max points must be multiples of the granularity ({self.granularity})"
            )
        return self

    @classmethod
    def fixed(cls, points: int) -> ArbLength:
        """A fixed-length buffer of exactly ``points`` samples (min == max)."""
        return cls(min_points=points, max_points=points)

    @property
    def is_fixed(self) -> bool:
        return self.min_points == self.max_points

    def accepts(self, n: int) -> bool:
        """Whether an ``n``-point buffer is uploadable in a single transfer."""
        return self.min_points <= n <= self.max_points and n % self.granularity == 0

    def representative_length(self, target: int = 4096) -> int:
        """A valid length near ``target`` for tests / first-light.

        The fixed value on a fixed-length device; otherwise ``target`` clamped into
        range and snapped down onto the granularity grid — so callers get one length
        that works on both families without hard-coding a device's point count.
        """
        if self.is_fixed:
            return self.min_points
        n = max(self.min_points, min(target, self.max_points))
        n -= n % self.granularity
        return max(self.min_points, n)

    def describe(self) -> str:
        if self.is_fixed:
            return f"exactly {self.min_points} points"
        step = "" if self.granularity == 1 else f" in steps of {self.granularity}"
        return f"{self.min_points}..{self.max_points} points{step}"


class ArbSampleRate(BaseModel):
    """The playback-clock constraint on a sample-rate-clocked arbitrary buffer.

    Some AWGs clock an arbitrary waveform out *point by point* at a settable sample
    rate rather than at a frequency (a Rigol DG1000Z in ``SRATe`` output mode). The
    waveform's repetition frequency is then a consequence of the rate and the buffer
    depth — see :func:`arb_repetition_hz` — not something you set directly.

    Instruments whose arbitrary output is frequency-clocked or fixed (the JDS6600's
    DDS wavetable) advertise :attr:`SigGenCapabilities.arb_sample_rate` as ``None``,
    and passing a rate to such a driver is refused rather than ignored.
    """

    model_config = ConfigDict(frozen=True)

    min_hz: float = Field(gt=0, description="Slowest settable playback rate, Sa/s.")
    max_hz: float = Field(gt=0, description="Fastest settable playback rate, Sa/s.")
    default_hz: float = Field(gt=0, description="The instrument's power-on rate, Sa/s.")

    @model_validator(mode="after")
    def _consistent(self) -> ArbSampleRate:
        if self.max_hz < self.min_hz:
            raise ValueError(f"max_hz ({self.max_hz:g}) must be >= min_hz ({self.min_hz:g})")
        if not self.min_hz <= self.default_hz <= self.max_hz:
            raise ValueError(
                f"default_hz ({self.default_hz:g}) must lie within "
                f"{self.min_hz:g}..{self.max_hz:g} Sa/s"
            )
        return self

    def accepts(self, hz: float) -> bool:
        return self.min_hz <= hz <= self.max_hz

    def describe(self) -> str:
        return f"{self.min_hz:g}..{self.max_hz:g} Sa/s"


def arb_repetition_hz(sample_rate_hz: float, n_points: int) -> float:
    """The output repetition frequency of an ``n_points`` buffer clocked at a rate.

    The buffer is one period, played point by point, so it repeats at
    ``sample_rate_hz / n_points``. Shared by drivers and tests so the relationship is
    stated once rather than re-derived (and mis-derived) at each call site.
    """
    if n_points <= 0:
        raise ValueError("an arbitrary buffer needs at least one point")
    return sample_rate_hz / n_points


class SigGenCapabilities(BaseModel):
    """Static limits/feature set for a signal-generator model.

    The loop reads these to stay in range (don't ask for a frequency the instrument
    can't reach) and to know which waveforms exist without hard-coding a vendor.

    Both *floors* and ceilings are modelled: real instruments clamp a below-minimum
    frequency or amplitude to their lower limit **silently** (the DG1000Z guide says
    so explicitly for ``:FREQuency``, ``:VOLTage`` and ``:VOLTage:OFFSet``), which is
    exactly the divergence this project refuses to let pass unremarked.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    n_channels: int = Field(gt=0)
    max_frequency_hz: float = Field(gt=0)
    max_amplitude_vpp: float = Field(gt=0)
    max_offset_v: float = Field(gt=0, description="Symmetric bias limit: |offset| ≤ this.")
    # Lower limits, below which the instrument silently clamps. 0 = "unspecified/no
    # floor", so a profile that has not characterised its floor keeps today's behaviour.
    min_frequency_hz: float = Field(
        default=0.0, ge=0, description="Slowest settable frequency; 0 = unspecified."
    )
    min_amplitude_vpp: float = Field(
        default=0.0, ge=0, description="Smallest settable amplitude, Vpp; 0 = unspecified."
    )
    waveforms: tuple[WaveShape, ...] = ()
    # Output generation/repetition modes the device supports (declarative; drivers wire
    # each mode in as needed). Every generator does at least CONTINUOUS.
    playback_modes: frozenset[PlaybackMode] = Field(
        default_factory=lambda: frozenset({PlaybackMode.CONTINUOUS})
    )
    # Arbitrary-waveform support: how storage is addressed (SLOT vs VOLATILE), the
    # number of numbered slots (SLOT only), the per-upload length constraint (None =
    # no arb support), and the DAC code resolution (levels).
    arb_addressing: ArbAddressing = ArbAddressing.SLOT
    arb_slots: int = Field(default=0, ge=0)
    arb_length: ArbLength | None = None
    arb_code_levels: int = Field(default=0, ge=0)
    # Settable playback clock for the arbitrary buffer; None = the rate is fixed or
    # frequency-derived and a caller may not set it (see :class:`ArbSampleRate`).
    arb_sample_rate: ArbSampleRate | None = None

    @model_validator(mode="after")
    def _arb_consistent(self) -> SigGenCapabilities:
        if self.arb_slots > 0 and self.arb_length is None:
            raise ValueError(
                f"{self.model_name}: arb_slots > 0 requires an arb_length constraint"
            )
        if self.arb_addressing is ArbAddressing.SLOT and self.arb_length is not None:
            if self.arb_slots == 0:
                raise ValueError(
                    f"{self.model_name}: SLOT-addressed arbitrary support needs arb_slots > 0"
                )
        elif self.arb_addressing is ArbAddressing.VOLATILE and self.arb_slots > 0:
            raise ValueError(
                f"{self.model_name}: VOLATILE-addressed instruments have no numbered slots; "
                f"set arb_slots=0"
            )
        if self.arb_sample_rate is not None and self.arb_length is None:
            raise ValueError(
                f"{self.model_name}: an arbitrary sample rate needs arbitrary support "
                f"(set arb_length)"
            )
        if self.min_frequency_hz > self.max_frequency_hz:
            raise ValueError(
                f"{self.model_name}: min_frequency_hz ({self.min_frequency_hz:g}) exceeds "
                f"max_frequency_hz ({self.max_frequency_hz:g})"
            )
        if self.min_amplitude_vpp > self.max_amplitude_vpp:
            raise ValueError(
                f"{self.model_name}: min_amplitude_vpp ({self.min_amplitude_vpp:g}) exceeds "
                f"max_amplitude_vpp ({self.max_amplitude_vpp:g})"
            )
        return self

    @property
    def channels(self) -> tuple[SigGenChannel, ...]:
        return tuple(SigGenChannel(i) for i in range(1, self.n_channels + 1))

    def has_channel(self, channel: SigGenChannel) -> bool:
        return int(channel) <= self.n_channels

    def supports_arbitrary(self) -> bool:
        return self.arb_length is not None


def check_within(config: SignalGeneratorConfig, caps: SigGenCapabilities) -> None:
    """Raise ``ValueError`` if ``config`` asks for anything ``caps`` cannot deliver.

    The shared enforcement point for every driver: a request outside the
    instrument's limits fails loudly and descriptively rather than being silently
    clamped, so the caller (human or agent) sees the divergence and can correct it.

    Checks both ends of each range. The floors matter as much as the ceilings: an
    instrument asked for less than its minimum frequency or amplitude typically pins
    the value to that minimum and says nothing, which is precisely the silent
    divergence this function exists to prevent.
    """
    if not caps.has_channel(config.channel):
        raise ValueError(f"{caps.model_name} has no channel {config.channel}")
    if config.arb_slot is not None:
        if not caps.supports_arbitrary():
            raise ValueError(f"{caps.model_name} has no arbitrary-waveform support")
        if caps.arb_addressing is ArbAddressing.VOLATILE:
            raise ValueError(
                f"{caps.model_name} addresses arbitrary waveforms per channel (volatile); "
                f"upload with upload_arbitrary(channel, wave) rather than selecting a slot"
            )
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
    if config.waveform not in _NO_FREQUENCY and config.frequency_hz < caps.min_frequency_hz:
        raise ValueError(
            f"frequency {config.frequency_hz:g} Hz is below the "
            f"{caps.min_frequency_hz:g} Hz floor of the {caps.model_name}, which would "
            f"silently clamp it"
        )
    if config.amplitude_vpp > caps.max_amplitude_vpp:
        raise ValueError(
            f"amplitude {config.amplitude_vpp:g} Vpp exceeds the "
            f"{caps.max_amplitude_vpp:g} Vpp limit of the {caps.model_name}"
        )
    if config.waveform not in _NO_AMPLITUDE and config.amplitude_vpp < caps.min_amplitude_vpp:
        raise ValueError(
            f"amplitude {config.amplitude_vpp:g} Vpp is below the "
            f"{caps.min_amplitude_vpp:g} Vpp floor of the {caps.model_name}, which would "
            f"silently clamp it"
        )
    if abs(config.offset_v) > caps.max_offset_v:
        raise ValueError(
            f"offset {config.offset_v:g} V exceeds the ±{caps.max_offset_v:g} V "
            f"limit of the {caps.model_name}"
        )


def check_arbitrary_length(n_points: int, caps: SigGenCapabilities) -> None:
    """Raise ``ValueError`` if an ``n_points`` arbitrary buffer is not uploadable.

    The shared enforcement point every driver calls before an ``upload_arbitrary``:
    a length the device can't take in a single transfer fails loudly and descriptively
    (naming the accepted range) rather than being silently truncated or resampled.
    """
    if caps.arb_length is None:
        raise ValueError(f"{caps.model_name} has no arbitrary-waveform support")
    if not caps.arb_length.accepts(n_points):
        raise ValueError(
            f"{caps.model_name} takes arbitrary waveforms of "
            f"{caps.arb_length.describe()}; got {n_points}"
        )


def resolve_arb_sample_rate(
    sample_rate_hz: float | None, caps: SigGenCapabilities
) -> float | None:
    """Validate a requested arbitrary playback rate and return the effective one.

    The shared enforcement point every driver calls from ``upload_arbitrary``:

    * the instrument has no settable rate (``caps.arb_sample_rate is None``) — a
      caller-supplied rate is *refused*, not quietly dropped, and ``None`` is returned
      to mean "not applicable";
    * a rate was omitted — the instrument's documented default is returned, so the
      driver can state it on the wire instead of inheriting whatever the front panel
      or a previous run happened to leave behind;
    * a rate is out of range — raises, naming the accepted span.
    """
    limits = caps.arb_sample_rate
    if limits is None:
        if sample_rate_hz is not None:
            raise ValueError(
                f"{caps.model_name} plays arbitrary waveforms at a fixed/derived rate; "
                f"it has no settable sample rate (pass sample_rate_hz=None)"
            )
        return None
    if sample_rate_hz is None:
        return limits.default_hz
    if not limits.accepts(sample_rate_hz):
        raise ValueError(
            f"{caps.model_name} clocks arbitrary waveforms at {limits.describe()}; "
            f"got {sample_rate_hz:g} Sa/s"
        )
    return sample_rate_hz
