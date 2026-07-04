"""Shared HIL acquisition helpers — one-shot vs repeating, never forcing a trigger.

Mirrors ``hwtools.session.loop``: a one-shot capture arms SINGLE and deep-reads
the triggered frame (or times out); a repeating capture free-runs in AUTO and
reads a fresh frame the scope auto-triggers. Waits scale with the timebase — a
frame takes ~one acquisition window (12 div x s/div) to fill.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.timebase import TimebaseConfig

_CAPTURED = (TriggerStatus.STOP, TriggerStatus.TRIGGERED)
_ARMED = (TriggerStatus.WAIT, TriggerStatus.RUN, TriggerStatus.AUTO)
_TIMEOUT_WINDOWS = 2.0
_TRIGGER_LATENCY_S = 0.05
_SETTLE_LATENCY_S = 0.02
_POLL_S = 0.02


def _window_s(scope: DS1054Z, timebase: TimebaseConfig) -> float:
    return scope.capabilities.horizontal_divisions * timebase.scale_s_per_div


def _wait_until(scope: DS1054Z, states: tuple[TriggerStatus, ...], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in states:
            return True
        time.sleep(_POLL_S)
    return False


def acquire_one_shot(
    scope: DS1054Z, channels: Sequence[ChannelId], timebase: TimebaseConfig
) -> Capture:
    """Arm SINGLE, wait for the trigger (timeout ~2 windows), deep-read one frame.

    Two-phase wait — arm, then trigger — so a scale change can't return a stale
    frame. Raises :class:`TimeoutError` if the trigger never fires; never forces.
    """
    scope.single()
    _wait_until(scope, _ARMED, _TRIGGER_LATENCY_S)
    timeout_s = _TIMEOUT_WINDOWS * _window_s(scope, timebase) + _TRIGGER_LATENCY_S
    if not _wait_until(scope, _CAPTURED, timeout_s):
        raise TimeoutError("single acquisition did not trigger within the window budget")
    return scope.capture(channels)


def acquire_repeating(
    scope: DS1054Z, channels: Sequence[ChannelId], timebase: TimebaseConfig, *, deep: bool = False
) -> Capture:
    """Free-run (AUTO sweep) and read a fresh frame of a persistent signal.

    The scope's auto-trigger always yields a frame — nothing is forced — but a
    frame at the current settings takes ~one window to acquire, so wait that long.
    """
    scope.run()
    time.sleep(_window_s(scope, timebase) + _SETTLE_LATENCY_S)
    return scope.capture(channels, deep=deep)
