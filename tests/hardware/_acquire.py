"""Shared HIL acquisition helpers — one-shot vs repeating, never forcing a trigger.

Mirrors ``hwtools.session.loop``: a one-shot capture arms SINGLE and deep-reads
the triggered frame (or times out); a repeating capture free-runs in AUTO and
reads a fresh frame the scope auto-triggers. Waits scale with the timebase — a
frame takes ~one acquisition window (12 div x s/div) to fill.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.timebase import TimebaseConfig

_CAPTURED = (TriggerStatus.STOP, TriggerStatus.TRIGGERED)
_ARMED = (TriggerStatus.WAIT, TriggerStatus.RUN, TriggerStatus.AUTO)
# See hwtools.session.acquire: the trigger *occurring* isn't bounded by the
# timebase, so the budget is a generous fixed wall-clock plus a window allowance.
_ARM_TIMEOUT_S = 1.0
_TRIGGER_TIMEOUT_S = 2.0
_FILL_WINDOWS = 2.0
_SETTLE_LATENCY_S = 0.02
_POLL_S = 0.02


def _window_s(scope: Oscilloscope, timebase: TimebaseConfig) -> float:
    return scope.capabilities.horizontal_divisions * timebase.scale_s_per_div


def _wait_until(scope: Oscilloscope, states: tuple[TriggerStatus, ...], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in states:
            return True
        time.sleep(_POLL_S)
    return False


def acquire_one_shot(
    scope: Oscilloscope, channels: Sequence[ChannelId], timebase: TimebaseConfig
) -> Capture:
    """Arm SINGLE, wait for the trigger (timeout ~2 windows), deep-read one frame.

    Two-phase wait — arm, then trigger — so a scale change can't return a stale
    frame. Raises :class:`TimeoutError` if the trigger never fires; never forces.
    """
    scope.single()
    # The arm must be confirmed, not assumed. STOP counts as "captured", and a scope
    # left in STOP by the previous single-shot satisfies the trigger wait below
    # instantly — returning the *previous* frame as though it were fresh. Failing here
    # turns that silent staleness into a visible error.
    if not _wait_until(scope, _ARMED, _ARM_TIMEOUT_S):
        raise TimeoutError(
            f"scope did not arm within {_ARM_TIMEOUT_S:g}s of :SINGle "
            f"(status {scope.trigger_status()}); a capture now would return a stale frame"
        )
    timeout_s = _TRIGGER_TIMEOUT_S + _FILL_WINDOWS * _window_s(scope, timebase)
    if not _wait_until(scope, _CAPTURED, timeout_s):
        raise TimeoutError("single acquisition did not trigger within the budget")
    return scope.capture(channels)


def acquire_repeating(
    scope: Oscilloscope,
    channels: Sequence[ChannelId],
    timebase: TimebaseConfig,
    *,
    deep: bool = False,
) -> Capture:
    """Free-run (AUTO sweep) and read a fresh frame of a persistent signal.

    The scope's auto-trigger always yields a frame — nothing is forced — but a
    frame at the current settings takes ~one window to acquire, so wait that long.
    """
    scope.run()
    time.sleep(_window_s(scope, timebase) + _SETTLE_LATENCY_S)
    return scope.capture(channels, deep=deep)
