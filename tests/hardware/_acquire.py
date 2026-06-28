"""Shared HIL acquisition helper.

The tool's capture methodology is single-shot, never free-running (AUTO): arm one
acquisition, wait for the trigger, then deep-read the frozen frame. These HIL tests
all need that same arm-and-wait, so it lives here once.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, TriggerStatus

_CAPTURED = (TriggerStatus.STOP, TriggerStatus.TRIGGERED)
_ARMED = (TriggerStatus.WAIT, TriggerStatus.RUN, TriggerStatus.AUTO)


def acquire_single(
    scope: DS1054Z,
    channels: Sequence[ChannelId],
    *,
    timeout_s: float = 2.0,
    poll_s: float = 0.02,
    force_if_idle: bool = False,
) -> Capture:
    """Arm a SINGLE acquisition, wait for the trigger, then deep-read it.

    ``force_if_idle`` forces a frame (:TFORce) if no trigger arrives within
    ``timeout_s`` — for blind captures of a possibly-idle line. Without it, a
    missed trigger leaves an empty record and the deep read raises (surfacing
    that the signal never crossed the trigger, rather than hiding it).
    """
    scope.single()
    # Two-phase wait. After :SINGle the scope still reports the *previous* STOP for
    # a moment; reading then yields the stale frame (rescaled by the new vertical
    # scale -> phantom out-of-range volts). So first wait for it to ARM (leave the
    # old STOP), then wait for the *new* acquisition to complete.
    arm_deadline = time.monotonic() + 1.0
    while time.monotonic() < arm_deadline:
        if scope.trigger_status() in _ARMED:
            break
        time.sleep(poll_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if scope.trigger_status() in _CAPTURED:
            return scope.capture(channels)
        time.sleep(poll_s)
    if force_if_idle:
        scope.force_trigger()
        # Wait for the forced acquisition to complete before reading.
        force_deadline = time.monotonic() + 1.0
        while time.monotonic() < force_deadline:
            if scope.trigger_status() in _CAPTURED:
                break
            time.sleep(poll_s)
    return scope.capture(channels)
