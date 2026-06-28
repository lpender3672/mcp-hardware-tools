"""The convergence loop — the product.

``capture_until_usable`` drives *any* :class:`~hwtools.interfaces.oscilloscope.Oscilloscope`
(simulated or real) from a first-guess configuration to a usable capture:
configure -> capture -> judge -> adjust -> re-capture, until the capture is
usable or a step budget runs out. The same loop logic works on the simulated
scope (CI) and the bench, because it only speaks the interface and the model.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from hwtools.analysis.adjust import suggest_adjustment
from hwtools.analysis.judge import judge_capture
from hwtools.analysis.recommend import Setup, recommend_setup
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.quality import Adjustment, CaptureQuality
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig

# Trigger states that mean a single acquisition has completed and a frame is ready.
_CAPTURED_STATES = (TriggerStatus.STOP, TriggerStatus.TRIGGERED)

# Default time to wait for a real trigger before forcing a frame.
_DEFAULT_ACQUIRE_WAIT_S = 0.5


def _acquire(
    scope: Oscilloscope,
    targets: list[ChannelId],
    *,
    wait_s: float,
    poll_interval_s: float = 0.02,
) -> Capture:
    """Arm one acquisition and return a single frame — never free-running (AUTO).

    Arms SINGLE, waits up to ``wait_s`` for a real trigger, and if none comes
    forces a frame (:TFORce) so the tool always has data to reason over. Pass
    ``wait_s=0`` for a blind measurement (force immediately — the trigger level
    isn't known yet); pass a positive wait when the configured trigger should fire.

    NOTE: on real hardware that changes vertical scale between iterations this needs
    the two-phase arm-then-trigger wait proven in ``tests/hardware/_acquire.py``
    (otherwise a just-armed scope still reports the previous STOP and a stale frame
    is read, rescaled by the new scale). The simulated scope resolves ``single()``
    synchronously with no stale frame, so the unit/loop tests don't exercise it;
    wire the two-phase wait here when validating autoset/loop on the bench.
    """
    scope.single()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in _CAPTURED_STATES:
            return scope.capture(targets)
        time.sleep(poll_interval_s)
    scope.force_trigger()
    return scope.capture(targets)


@dataclass(frozen=True)
class LoopResult:
    """Outcome of a convergence run."""

    capture: Capture
    quality: CaptureQuality
    iterations: int
    converged: bool
    adjustments: list[Adjustment] = field(default_factory=list)


@dataclass(frozen=True)
class AutosetResult:
    """Outcome of a single-stage autoset."""

    capture: Capture
    quality: CaptureQuality
    setup: Setup
    widen_steps: int
    converged: bool


@dataclass(frozen=True)
class SingleShotResult:
    """Outcome of a single-shot acquisition."""

    triggered: bool
    capture: Capture | None = None
    quality: CaptureQuality | None = None


def capture_single(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    poll_timeout_s: float = 2.0,
    poll_interval_s: float = 0.02,
) -> SingleShotResult:
    """Capture one non-repeating event: set up, arm SINGLE, wait, read one frame.

    Unlike the convergence loops this cannot iterate — a single-shot event happens
    once, so the configuration must be right *before* arming. It applies the given
    setup, arms a single acquisition, polls the trigger status until the frame is
    captured (or ``poll_timeout_s`` elapses), then downloads and judges it.
    """
    configured = dict(channels)
    for config in configured.values():
        scope.configure_channel(config)
    scope.configure_timebase(timebase)
    scope.configure_trigger(trigger)
    scope.configure_acquire(acquire or AcquireConfig())
    targets = list(configured)

    scope.single()
    deadline = time.monotonic() + poll_timeout_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in _CAPTURED_STATES:
            capture = scope.capture(targets)
            quality = judge_capture(capture, configured, scope.capabilities)
            return SingleShotResult(triggered=True, capture=capture, quality=quality)
        time.sleep(poll_interval_s)
    return SingleShotResult(triggered=False)


def autoset(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    wide_scale_v_per_div: float = 5.0,
    max_widen: int = 3,
    poll_timeout_s: float = _DEFAULT_ACQUIRE_WAIT_S,
) -> AutosetResult:
    """Configure the scope in one recommendation stage.

    Takes a deliberately wide measurement so the signal isn't clipped, makes a
    single :func:`~hwtools.analysis.recommend.recommend_setup` recommendation
    (scale/offset/timebase/trigger for every channel at once), applies it, and
    returns the resulting capture. The only iteration is widening the
    *measurement* if a very large signal still clips at the initial wide scale —
    never to creep toward a target.
    """
    targets = list(channels)
    scope.configure_timebase(timebase)
    scope.configure_acquire(acquire or AcquireConfig())
    scope.configure_trigger(trigger)

    wide_scale = wide_scale_v_per_div
    measure_channels = {
        ch: cfg.model_copy(update={"scale_v_per_div": wide_scale, "offset_v": 0.0})
        for ch, cfg in channels.items()
    }
    widen_steps = 0
    while True:
        for cfg in measure_channels.values():
            scope.configure_channel(cfg)
        # Blind measurement: the trigger level isn't known yet, so force a frame
        # (never wait on a trigger that may not fire, never free-run on AUTO).
        measurement = _acquire(scope, targets, wait_s=0.0)
        if widen_steps >= max_widen or not any(
            measurement.waveforms[ch].is_clipped for ch in targets
        ):
            break
        wide_scale *= 4.0
        measure_channels = {
            ch: cfg.model_copy(update={"scale_v_per_div": wide_scale, "offset_v": 0.0})
            for ch, cfg in measure_channels.items()
        }
        widen_steps += 1

    setup = recommend_setup(
        measurement,
        channels=measure_channels,
        timebase=timebase,
        trigger=trigger,
        capabilities=scope.capabilities,
        wide_scale_v_per_div=wide_scale,
    )
    for cfg in setup.channels.values():
        scope.configure_channel(cfg)
    scope.configure_timebase(setup.timebase)
    scope.configure_trigger(setup.trigger)
    final = _acquire(scope, targets, wait_s=poll_timeout_s)
    quality = judge_capture(final, setup.channels, scope.capabilities)
    return AutosetResult(final, quality, setup, widen_steps, quality.usable)


def capture_until_usable(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    max_iterations: int = 8,
    poll_timeout_s: float = _DEFAULT_ACQUIRE_WAIT_S,
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
        capture = _acquire(scope, targets, wait_s=poll_timeout_s)
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

    capture = _acquire(scope, targets, wait_s=poll_timeout_s)
    quality = judge_capture(capture, current_channels, scope.capabilities)
    return LoopResult(capture, quality, max_iterations, quality.usable, adjustments)
