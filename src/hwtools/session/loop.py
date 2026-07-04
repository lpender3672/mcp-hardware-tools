"""Convergence wrappers over the acquire primitive — **experimental**.

The product's real surface is the primitives (:func:`hwtools.session.acquire`,
``judge``, ``recommend``): the agent drives the loop, reading a decision-ready
:class:`~hwtools.model.reading.AcquireResult` and choosing the next move. These
functions bake that policy into a fast, deterministic Python loop instead — useful
as a cheap convenience for the routine case, and as a deterministic thing to test
against the simulated scope, but **not** the main path. They are kept experimental.

* :func:`capture_single` — one triggered SINGLE acquisition (honest miss, no force).
* :func:`autoset` — one wide measurement → one :func:`recommend_setup` → apply.
* :func:`capture_until_usable` — iterate judge→adjust on cheap free-run frames.

All three build on the single :func:`~hwtools.session.acquire.acquire` primitive;
they hold an internal :class:`~hwtools.session.store.CaptureStore` unless the caller
threads one in (to keep the final frame addressable).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from hwtools.analysis.adjust import suggest_adjustment
from hwtools.analysis.recommend import Setup, recommend_setup
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.adjustment import Adjustment
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, SweepMode
from hwtools.model.reading import AcquireResult
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig
from hwtools.session.acquire import acquire as _acquire
from hwtools.session.store import CaptureStore


@dataclass(frozen=True)
class LoopResult:
    """Outcome of a convergence run."""

    capture: Capture
    assessment: AcquireResult
    iterations: int
    converged: bool
    adjustments: list[Adjustment] = field(default_factory=list)


@dataclass(frozen=True)
class AutosetResult:
    """Outcome of a single-stage autoset."""

    capture: Capture
    assessment: AcquireResult
    setup: Setup
    widen_steps: int
    converged: bool


@dataclass(frozen=True)
class SingleShotResult:
    """Outcome of a single-shot acquisition."""

    triggered: bool
    capture: Capture | None = None
    assessment: AcquireResult | None = None


def capture_single(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    store: CaptureStore | None = None,
    trigger_timeout_s: float | None = None,
) -> SingleShotResult:
    """Capture one non-repeating event: arm SINGLE, wait, deep-read once.

    Cannot iterate — the event happens once, so the setup must be right before
    arming. On a missed trigger it reports ``triggered=False`` with no capture; it
    never forces a frame. The frame (if any) is stored *kept* (one-shot safety).
    """
    store = store or CaptureStore()
    frame = _acquire(
        scope,
        store,
        channels=channels,
        timebase=timebase,
        trigger=trigger,
        acquire_cfg=acquire,
        sweep=SweepMode.SINGLE,
        trigger_timeout_s=trigger_timeout_s,
    )
    return SingleShotResult(
        triggered=frame.result.triggered, capture=frame.capture, assessment=frame.result
    )


def autoset(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    wide_scale_v_per_div: float = 5.0,
    max_widen: int = 3,
    store: CaptureStore | None = None,
) -> AutosetResult:
    """Configure the scope in one recommendation stage, for a repeating signal.

    Takes a deliberately wide free-run measurement so the signal isn't clipped,
    makes one :func:`~hwtools.analysis.recommend.recommend_setup` recommendation,
    applies it, and returns the resulting frame. The only iteration is widening the
    *measurement* if a very large signal still clips at the initial wide scale.
    """
    store = store or CaptureStore()
    wide_scale = wide_scale_v_per_div
    measure_channels = {
        ch: cfg.model_copy(update={"scale_v_per_div": wide_scale, "offset_v": 0.0})
        for ch, cfg in channels.items()
    }
    widen_steps = 0
    while True:
        frame = _acquire(
            scope,
            store,
            channels=measure_channels,
            timebase=timebase,
            trigger=trigger,
            acquire_cfg=acquire,
            sweep=SweepMode.AUTO,
        )
        measurement = frame.capture
        assert measurement is not None  # AUTO always yields a frame
        if widen_steps >= max_widen or not any(
            wf.is_clipped for wf in measurement.waveforms.values()
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
    final = _acquire(
        scope,
        store,
        channels=setup.channels,
        timebase=setup.timebase,
        trigger=setup.trigger,
        acquire_cfg=acquire,
        sweep=SweepMode.AUTO,
    )
    assert final.capture is not None
    return AutosetResult(
        capture=final.capture,
        assessment=final.result,
        setup=setup,
        widen_steps=widen_steps,
        converged=final.result.usable,
    )


def capture_until_usable(
    scope: Oscilloscope,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire: AcquireConfig | None = None,
    max_iterations: int = 8,
    store: CaptureStore | None = None,
) -> LoopResult:
    """Converge on a usable capture of a repeating signal, adjusting as needed.

    Free-runs (AUTO) so the scope always yields a frame — nothing is forced — and
    iterates configure→capture→judge→adjust on cheap screen frames.
    """
    store = store or CaptureStore()
    current_channels = dict(channels)
    current_timebase = timebase
    current_trigger = trigger
    adjustments: list[Adjustment] = []

    frame = _acquire(
        scope,
        store,
        channels=current_channels,
        timebase=current_timebase,
        trigger=current_trigger,
        acquire_cfg=acquire,
        sweep=SweepMode.AUTO,
    )
    for iteration in range(max_iterations):
        assert frame.capture is not None
        adjustment = suggest_adjustment(
            frame.result,
            frame.capture,
            current_channels,
            current_timebase,
            current_trigger,
            scope.capabilities,
        )
        # Stop when there is nothing left to improve; converged iff also usable.
        if adjustment.is_empty():
            return LoopResult(
                frame.capture, frame.result, iteration, frame.result.usable, adjustments
            )

        adjustments.append(adjustment)
        for channel, config in adjustment.channels.items():
            current_channels[channel] = config
        if adjustment.timebase is not None:
            current_timebase = adjustment.timebase
        if adjustment.trigger is not None:
            current_trigger = adjustment.trigger

        frame = _acquire(
            scope,
            store,
            channels=current_channels,
            timebase=current_timebase,
            trigger=current_trigger,
            acquire_cfg=acquire,
            sweep=SweepMode.AUTO,
        )

    assert frame.capture is not None
    return LoopResult(
        frame.capture, frame.result, max_iterations, frame.result.usable, adjustments
    )
