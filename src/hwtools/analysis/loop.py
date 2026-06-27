"""The convergence loop — the product.

``capture_until_usable`` drives *any* :class:`~hwtools.interfaces.oscilloscope.Oscilloscope`
(simulated or real) from a first-guess configuration to a usable capture:
configure -> capture -> judge -> adjust -> re-capture, until the capture is
usable or a step budget runs out. The same loop logic works on the simulated
scope (CI) and the bench, because it only speaks the interface and the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from hwtools.analysis.adjust import suggest_adjustment
from hwtools.analysis.judge import judge_capture
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.quality import Adjustment, CaptureQuality
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig


@dataclass(frozen=True)
class LoopResult:
    """Outcome of a convergence run."""

    capture: Capture
    quality: CaptureQuality
    iterations: int
    converged: bool
    adjustments: list[Adjustment] = field(default_factory=list)


def capture_until_usable(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    max_iterations: int = 8,
) -> LoopResult:
    """Converge on a usable capture, adjusting the setup as needed."""
    current_channels = dict(channels)
    current_timebase = timebase
    current_trigger = trigger
    for config in current_channels.values():
        scope.configure_channel(config)
    scope.configure_timebase(current_timebase)
    scope.configure_trigger(current_trigger)
    scope.configure_acquire(acquire or AcquireConfig())

    targets = list(current_channels)
    adjustments: list[Adjustment] = []

    for iteration in range(max_iterations):
        scope.run()
        capture = scope.capture(targets)
        quality = judge_capture(capture, current_channels, scope.capabilities)
        adjustment = suggest_adjustment(
            quality,
            capture,
            current_channels,
            current_timebase,
            current_trigger,
            scope.capabilities,
        )
        # Stop when there is nothing left to improve; converged iff also usable.
        if adjustment.is_empty():
            return LoopResult(capture, quality, iteration, quality.usable, adjustments)

        adjustments.append(adjustment)
        for channel, config in adjustment.channels.items():
            current_channels[channel] = config
            scope.configure_channel(config)
        if adjustment.timebase is not None:
            current_timebase = adjustment.timebase
            scope.configure_timebase(current_timebase)
        if adjustment.trigger is not None:
            current_trigger = adjustment.trigger
            scope.configure_trigger(current_trigger)

    scope.run()
    capture = scope.capture(targets)
    quality = judge_capture(capture, current_channels, scope.capabilities)
    return LoopResult(capture, quality, max_iterations, quality.usable, adjustments)
