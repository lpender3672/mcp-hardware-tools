"""The one honest acquisition primitive.

Collapses the old one-shot/repeating split into a single function whose *policy*
is a parameter (``sweep``), because the choice of strategy belongs to the caller
(the agent), not the mechanism:

* **SINGLE / NORMAL** — a triggered one-shot. Arm, two-phase wait (arm, then
  trigger), deep-read the frozen frame, store it *kept* (an unrepeatable event is
  never an auto-eviction victim). On a missed trigger it returns ``triggered=False``
  with no frame — it never forces a trigger and never fabricates data.
* **AUTO** — a free-running measurement of a persistent signal. The scope's
  auto-trigger always yields a frame (no forcing), so a convergence loop can read a
  cheap screen frame, judge, adjust, re-read. Stored *ephemeral*: the next acquire
  reclaims it.

Waits scale with the timebase — a frame takes about one acquisition window
(horizontal divisions x s/div) to fill — plus a small fixed latency constant.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from hwtools.analysis.judge import judge_capture
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, SweepMode, TriggerStatus
from hwtools.model.reading import AcquireResult
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig
from hwtools.session.store import CaptureStore

# A single acquisition has completed and a frame is ready.
_CAPTURED_STATES = (TriggerStatus.STOP, TriggerStatus.TRIGGERED)
# The scope has accepted the arm (left any previous STOP) and is acquiring.
_ARMED_STATES = (TriggerStatus.WAIT, TriggerStatus.RUN, TriggerStatus.AUTO)

# One-shot timing. Two separate concerns, previously (wrongly) conflated:
#   * arming — leaving the previous STOP — takes a few SCPI round-trips, bounded.
#   * the trigger *occurring* is NOT bounded by the timebase (a rare event may take
#     a while); only the frame *fill* scales with the window. So the trigger budget
#     is a generous fixed wall-clock plus a window allowance for slow timebases.
_ARM_TIMEOUT_S = 1.0  # wait up to this for the scope to arm (leave the old STOP)
_TRIGGER_TIMEOUT_S = 2.0  # generous fixed budget for a trigger to actually occur
_FILL_WINDOWS = 2.0  # plus this many windows, so slow timebases have time to fill
_SETTLE_LATENCY_S = 0.02  # free-run: command/settle latency for a fresh frame
_POLL_INTERVAL_S = 0.02


@dataclass(frozen=True)
class AcquiredFrame:
    """One acquisition's outcome: the agent-facing result plus the in-process frame.

    ``result`` is the decision-ready view (serialized to the agent, carries the
    handle). ``capture`` is the frame itself for in-process callers (the loop feeds
    it to ``suggest_adjustment``); it is ``None`` on a missed one-shot trigger.
    """

    result: AcquireResult
    capture: Capture | None
    capture_id: str | None


def _window_s(scope: Oscilloscope, timebase: TimebaseConfig) -> float:
    return scope.capabilities.horizontal_divisions * timebase.scale_s_per_div


def _wait_until(
    scope: Oscilloscope, states: tuple[TriggerStatus, ...], timeout_s: float
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in states:
            return True
        time.sleep(_POLL_INTERVAL_S)
    return False


def acquire(
    scope: Oscilloscope,
    store: CaptureStore,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    acquire_cfg: AcquireConfig | None = None,
    sweep: SweepMode = SweepMode.SINGLE,
    deep: bool | None = None,
    keep: bool | None = None,
    label: str | None = None,
    trigger_timeout_s: float | None = None,
) -> AcquiredFrame:
    """Acquire one frame under ``sweep``, store it, and return the judged result.

    ``deep``/``keep`` default by mode: a triggered one-shot deep-reads and is kept;
    a free-run measurement reads the cheap screen frame and is ephemeral. Pass them
    explicitly to override (e.g. an ``AUTO`` deep read once the setup is good).
    ``trigger_timeout_s`` overrides how long a one-shot waits for its trigger (the
    default scales a generous fixed budget with the timebase).
    """
    # Reconfigure while STOPPED. The DS1000Z binds Memory Depth = Sample Rate x
    # (timebase x 12) on the *running* acquisition, so changing the timebase while
    # it free-runs (e.g. autoset's wide 2 ms -> applied 500 us) conflicts with the
    # live depth and beeps "Parameter limited!" before the next MDEPth AUTO
    # re-settles. Stopping first lets configure_acquire's :RUN start clean at the
    # new timebase. (Harmless on the simulated scope.)
    scope.stop()
    configured = dict(channels)
    for config in configured.values():
        scope.configure_channel(config)
    scope.configure_timebase(timebase)
    scope.configure_acquire(acquire_cfg or AcquireConfig())
    scope.configure_trigger(trigger.model_copy(update={"sweep": sweep}))
    targets = list(configured)

    one_shot = sweep in (SweepMode.SINGLE, SweepMode.NORMAL)
    read_deep = (deep if deep is not None else one_shot)

    if one_shot:
        capture = _acquire_triggered(
            scope, targets, timebase, deep=read_deep, trigger_timeout_s=trigger_timeout_s
        )
        if capture is None:
            # Honest miss: no frame stored, no fabricated data.
            result = AcquireResult(triggered=False, timebase=timebase, notes=["not triggered"])
            return AcquiredFrame(result=result, capture=None, capture_id=None)
        keep_frame = keep if keep is not None else True
        keep_reason = "one-shot" if keep_frame else None
    else:
        capture = _acquire_free_run(scope, targets, timebase, deep=read_deep)
        keep_frame = keep if keep is not None else False
        keep_reason = None

    capture_id = store.put(
        capture, keep=keep_frame, label=label, keep_reason=keep_reason, sweep=sweep
    )
    result = judge_capture(capture, configured, scope.capabilities, timebase).with_capture_id(
        capture_id
    )
    return AcquiredFrame(result=result, capture=capture, capture_id=capture_id)


def _acquire_triggered(
    scope: Oscilloscope,
    targets: list[ChannelId],
    timebase: TimebaseConfig,
    *,
    deep: bool,
    trigger_timeout_s: float | None = None,
) -> Capture | None:
    """Arm one acquisition and read the triggered frame, or ``None`` on timeout.

    Two-phase wait — first for the scope to *arm* (leave any previous STOP), then
    for the *new* acquisition to complete — so a just-changed vertical scale cannot
    return a stale frame rescaled by the new increment. Never forces a trigger.
    """
    scope.single()
    _wait_until(scope, _ARMED_STATES, _ARM_TIMEOUT_S)
    if trigger_timeout_s is None:
        trigger_timeout_s = _TRIGGER_TIMEOUT_S + _FILL_WINDOWS * _window_s(scope, timebase)
    if not _wait_until(scope, _CAPTURED_STATES, trigger_timeout_s):
        return None
    return scope.capture(targets, deep=deep)


def _acquire_free_run(
    scope: Oscilloscope, targets: list[ChannelId], timebase: TimebaseConfig, *, deep: bool
) -> Capture:
    """Free-run (AUTO) and read a fresh frame of a persistent signal — never forced."""
    scope.run()
    time.sleep(_window_s(scope, timebase) + _SETTLE_LATENCY_S)
    return scope.capture(targets, deep=deep)
