"""Single-stage setup recommendation.

Given one *measurement* capture (taken wide enough not to clip), compute the
whole instrument setup at once — vertical scale and offset per channel, the
timebase, and the trigger level — rather than nudging toward it over many
captures. This is the recommendation stage the convergence loop is built on.

A channel that is still clipping in the measurement can't be sized (its true
amplitude is unknown), so the only safe single recommendation for it is to open
the vertical scale wide; one more measurement then yields the real numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hwtools.analysis import measure
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig


@dataclass(frozen=True)
class Setup:
    """A complete recommended instrument configuration."""

    channels: dict[ChannelId, ChannelConfig]
    timebase: TimebaseConfig
    trigger: TriggerConfig


def recommend_setup(
    capture: Capture,
    *,
    channels: Mapping[ChannelId, ChannelConfig],
    timebase: TimebaseConfig,
    trigger: TriggerConfig,
    capabilities: ScopeCapabilities,
    target_fill_frac: float = 0.6,
    periods_on_screen: float = 4.0,
    wide_scale_v_per_div: float = 5.0,
) -> Setup:
    """Recommend a full setup from one measurement capture."""
    divisions_v = capabilities.vertical_divisions
    divisions_h = capabilities.horizontal_divisions

    new_channels: dict[ChannelId, ChannelConfig] = {}
    for channel, config in channels.items():
        wf = capture.waveforms.get(channel)
        if wf is None:
            new_channels[channel] = config
            continue
        if wf.is_clipped:
            # Unknown true amplitude: open wide so the next measurement sees it.
            new_channels[channel] = config.model_copy(
                update={"scale_v_per_div": wide_scale_v_per_div, "offset_v": 0.0}
            )
            continue
        midline = (wf.vmax + wf.vmin) / 2.0
        scale = (
            max(wf.vpp / (target_fill_frac * divisions_v), 1e-4)
            if wf.vpp > 1e-9
            else config.scale_v_per_div
        )
        new_channels[channel] = config.model_copy(
            update={"scale_v_per_div": scale, "offset_v": -midline}
        )

    source_wf = capture.waveforms.get(trigger.source)

    new_timebase = timebase
    if source_wf is not None:
        freq = measure.frequency(source_wf)
        if freq is not None and freq > 0:
            window = periods_on_screen / freq
            new_timebase = timebase.model_copy(update={"scale_s_per_div": window / divisions_h})

    new_trigger = trigger
    if source_wf is not None:
        level = (source_wf.vmax + source_wf.vmin) / 2.0
        new_trigger = trigger.model_copy(
            update={"trigger": trigger.trigger.model_copy(update={"level_v": level})}
        )

    return Setup(channels=new_channels, timebase=new_timebase, trigger=new_trigger)
