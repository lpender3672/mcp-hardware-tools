"""Signal triage — the unknown-capture entry point.

Consumes the always-on *feature vector* from :mod:`hwtools.analysis.describe`
(never the raw frame) and routes it to a signal class plus decode/process hints.
The classifier is deliberately a **rule-based skeleton** behind a stable
signature: the "yet-to-be-decided algorithm" (a learned model, a richer
feature set) can replace :func:`classify` without touching any caller.

``triage`` composes describe -> classify over a capture: one call turns an unknown
frame into "here is what it is, and here is the primitive to reach for next".
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from hwtools.analysis.describe import Characterization, describe
from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId


class SignalClass(StrEnum):
    """The coarse kind of a signal — enough to pick the next processing step."""

    FLAT_DC = "flat_dc"
    SINE = "sine"
    DIGITAL = "digital"  # square / logic — the decoders' domain
    NOISE = "noise"
    MODULATED = "modulated"  # periodic but not a clean tone or square
    UNKNOWN = "unknown"


class Classification(BaseModel):
    """A per-channel verdict plus the hints a decode/process step needs."""

    model_config = ConfigDict(frozen=True)

    channel: ChannelId
    signal_class: SignalClass
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    est_symbol_rate_hz: float | None = Field(
        default=None, description="Edge/symbol rate seed for a decoder (UART baud, SPI clk)."
    )
    suggested: str = Field(default="", description="The primitive to reach for next.")


# Thresholds for the rule-based skeleton (tuned to the describe feature scales).
_FLAT_VPP_FRAC = 1e-3  # vpp below this fraction of |dc| (or absolute) is flat
_NOISE_FLATNESS = 0.3  # spectral flatness above this is noise-like
_TONE_FLATNESS = 0.1  # below this a single tone dominates
_PERIODIC = 0.5  # autocorrelation strength above this is periodic
_SQUARE_CREST = 1.2  # crest factor below this is square-like (~1)
_SINE_CREST = 1.7  # crest factor around 1.41 is sine-like


def classify(features: Characterization) -> Classification:
    """Route a feature vector to a signal class and decode/process hints."""
    ch = features.channel
    flat_ref = max(abs(features.dc_level), 1.0)
    if features.vpp <= _FLAT_VPP_FRAC * flat_ref:
        return _verdict(ch, SignalClass.FLAT_DC, 0.95, "flat: vpp ~ 0", suggested="measure")

    if features.spectral_flatness >= _NOISE_FLATNESS and features.periodicity < _PERIODIC:
        return _verdict(
            ch, SignalClass.NOISE, 0.8,
            f"broadband (flatness={features.spectral_flatness:.2f}), aperiodic",
            suggested="describe(psd=True)",
        )

    is_square = features.n_levels == 2 and features.crest_factor < _SQUARE_CREST
    if is_square and features.edge_rate_hz > 0:
        return _verdict(
            ch, SignalClass.DIGITAL, 0.85,
            f"two levels, crest~1 (edges={features.edge_rate_hz:.0f}/s)",
            symbol_rate=features.edge_rate_hz / 2.0,  # ~half the edges are symbol boundaries
            suggested="decode_uart / decode_spi / decode_i2c",
        )

    if (
        features.periodicity >= _PERIODIC
        and features.spectral_flatness < _TONE_FLATNESS
        and _SQUARE_CREST <= features.crest_factor <= _SINE_CREST
        and features.peak_hz is not None
    ):
        return _verdict(
            ch, SignalClass.SINE, 0.85,
            f"single tone at {features.peak_hz:.0f} Hz, crest~1.41",
            suggested="recommend / measure",
        )

    if features.periodicity >= _PERIODIC:
        return _verdict(
            ch, SignalClass.MODULATED, 0.5,
            "periodic but not a clean tone or square",
            suggested="describe(joint=True, psd=True)",
        )

    return _verdict(
        ch, SignalClass.UNKNOWN, 0.3, "no rule matched", suggested="describe(joint=True)"
    )


def triage(
    capture: Capture,
    *,
    channels: ChannelId | None = None,
) -> dict[ChannelId, Classification]:
    """Describe then classify every channel of a capture (the unknown-frame front door).

    Pass ``channels`` to triage a single channel; otherwise all channels in the
    capture are triaged. Returns one :class:`Classification` per channel.
    """
    targets = [channels] if channels is not None else list(capture.waveforms)
    out: dict[ChannelId, Classification] = {}
    for ch in targets:
        wf = capture.waveforms.get(ch)
        if wf is None:
            continue
        out[ch] = classify(describe(wf))
    return out


def cross_channel_hint(classifications: Mapping[ChannelId, Classification]) -> str | None:
    """A protocol guess from how many channels carry a digital signal together.

    Two digital lines suggest I2C (scl/sda); three suggest SPI (clk/mosi/cs); one a
    UART. A coarse steer for which decoder to reach for — the decoders confirm.
    """
    digital = [c for c in classifications.values() if c.signal_class is SignalClass.DIGITAL]
    n = len(digital)
    if n >= 3:
        return "3+ digital lines -> likely SPI (clk/mosi/cs)"
    if n == 2:
        return "2 digital lines -> likely I2C (scl/sda)"
    if n == 1:
        return "1 digital line -> likely UART"
    return None


def _verdict(
    channel: ChannelId,
    signal_class: SignalClass,
    confidence: float,
    reason: str,
    *,
    symbol_rate: float | None = None,
    suggested: str = "",
) -> Classification:
    return Classification(
        channel=channel,
        signal_class=signal_class,
        confidence=confidence,
        reason=reason,
        est_symbol_rate_hz=symbol_rate,
        suggested=suggested,
    )
