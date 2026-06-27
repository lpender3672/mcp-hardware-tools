"""Judge a capture: is it usable, and if not, why not.

Maps a :class:`~hwtools.model.capture.Capture` (plus the vertical configuration
and instrument grid it was taken with) to a decision-ready
:class:`~hwtools.model.quality.CaptureQuality`. This is half the brain of the
self-correcting loop; :mod:`hwtools.analysis.adjust` is the other half.
"""

from __future__ import annotations

from collections.abc import Mapping

from hwtools.analysis import measure
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId
from hwtools.model.quality import CaptureQuality

# A channel is "clipping" when it reaches within this fraction of a screen rail.
_RAIL_MARGIN_FRAC = 0.01
# Below this fraction of full screen the signal is too small to be useful.
_LOW_FILL_FRAC = 0.1
# Fewer samples per period than this and the timebase is too fast for the signal.
_MIN_SAMPLES_PER_PERIOD = 2.5


def judge_capture(
    capture: Capture,
    channels: Mapping[ChannelId, ChannelConfig],
    capabilities: ScopeCapabilities,
) -> CaptureQuality:
    """Assess a capture against the configuration it was taken with."""
    clipping: dict[ChannelId, bool] = {}
    fill_fraction: dict[ChannelId, float] = {}
    notes: list[str] = []
    bandwidth_ok = True

    for channel, wf in capture.waveforms.items():
        config = channels.get(channel)
        if config is None:
            continue

        full_scale = config.scale_v_per_div * capabilities.vertical_divisions

        if wf.saturation is not None:
            # Exact: a sample at the digitiser's rail means real clipping.
            clipped = wf.is_clipped
        else:
            # Fallback: model the rails from the configured screen window.
            halfspan = full_scale / 2.0
            rail_high = -config.offset_v + halfspan
            rail_low = -config.offset_v - halfspan
            margin = _RAIL_MARGIN_FRAC * full_scale
            clipped = wf.vmax >= rail_high - margin or wf.vmin <= rail_low + margin
        clipping[channel] = clipped
        fill = wf.vpp / full_scale if full_scale > 0 else 0.0
        fill_fraction[channel] = fill

        if clipped:
            notes.append(f"{channel.name} clipping at the rails")
        elif fill < _LOW_FILL_FRAC:
            notes.append(f"{channel.name} fills only {fill:.0%} of the screen")

        spp = measure.samples_per_period(wf)
        if spp is not None and spp < _MIN_SAMPLES_PER_PERIOD:
            bandwidth_ok = False
            notes.append(f"{channel.name} undersampled ({spp:.1f} samples/period)")

    if not capture.triggered:
        notes.append("not triggered")

    return CaptureQuality(
        triggered=capture.triggered,
        clipping=clipping,
        fill_fraction=fill_fraction,
        bandwidth_ok=bandwidth_ok,
        notes=notes,
    )
