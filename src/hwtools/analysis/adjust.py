"""Propose a configuration change from a capture's shortcomings.

The second half of the self-correcting brain: given an
:class:`~hwtools.model.reading.AcquireResult` and the capture/config it came
from, suggest an :class:`~hwtools.model.adjustment.Adjustment` the loop applies
before re-capturing. Pure — no instrument access.

Heuristics:
* clipping        -> grow the vertical scale and recentre the channel,
* low screen fill -> shrink the vertical scale to fill the screen,
* not triggered   -> move the trigger level to the source's midline,
* undersampled    -> speed up the timebase.
"""

from __future__ import annotations

from collections.abc import Mapping

from hwtools.model.adjustment import Adjustment
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.reading import AcquireResult
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig

# Target screen fill (fraction of vertical divisions) when zooming in.
_TARGET_FILL_FRAC = 0.6
# Only zoom in when fill is below this.
_LOW_FILL_FRAC = 0.3


def suggest_adjustment(
    result: AcquireResult,
    capture: Capture,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig | None,
    capabilities: ScopeCapabilities,
) -> Adjustment:
    """Suggest config changes that move ``result`` toward usable."""
    divisions_v = capabilities.vertical_divisions
    new_channels: dict[ChannelId, ChannelConfig] = {}
    reasons: list[str] = []

    for channel, config in channels.items():
        wf = capture.waveforms.get(channel)
        if wf is None:
            continue
        reading = result.channels.get(channel)
        midline = (wf.vmax + wf.vmin) / 2.0

        if reading is not None and reading.clipping:
            new_scale = config.scale_v_per_div * 2.0
            new_channels[channel] = config.model_copy(
                update={"scale_v_per_div": new_scale, "offset_v": -midline}
            )
            reasons.append(f"{channel.name}: clipping, scale->{new_scale:g} V/div")
        elif (
            reading is not None
            and reading.fill_fraction < _LOW_FILL_FRAC
            and wf.vpp > 0
        ):
            new_scale = wf.vpp / (_TARGET_FILL_FRAC * divisions_v)
            if new_scale < config.scale_v_per_div * 0.9:
                new_channels[channel] = config.model_copy(
                    update={"scale_v_per_div": new_scale, "offset_v": -midline}
                )
                reasons.append(f"{channel.name}: low fill, scale->{new_scale:g} V/div")

    new_trigger: TriggerConfig | None = None
    if trigger is not None:
        source_wf = capture.waveforms.get(trigger.source)
        if source_wf is not None:
            current = trigger.trigger.level_v
            # Fix the trigger level when the frame didn't trigger OR the level sits
            # outside the signal's range (so it never would). Under AUTO free-run the
            # frame always reports triggered, so an out-of-range level is only caught
            # by the range check — without it the loop leaves the trigger unusable.
            out_of_range = not (source_wf.vmin < current < source_wf.vmax)
            if not result.triggered or out_of_range:
                level = (source_wf.vmax + source_wf.vmin) / 2.0
                # Only adjust if it meaningfully moves the level (else the loop churns).
                if abs(level - current) > 0.05 * max(source_wf.vpp, 1e-6):
                    new_trigger = trigger.model_copy(
                        update={"trigger": trigger.trigger.model_copy(update={"level_v": level})}
                    )
                    reasons.append(f"trigger level->{level:g} V")

    new_timebase: TimebaseConfig | None = None
    if not result.bandwidth_ok:
        new_timebase = timebase.model_copy(
            update={"scale_s_per_div": timebase.scale_s_per_div / 2.0}
        )
        reasons.append("undersampled, timebase halved")

    return Adjustment(
        channels=new_channels,
        timebase=new_timebase,
        trigger=new_trigger,
        reason="; ".join(reasons),
    )
