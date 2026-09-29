"""Scope-side measurement for cross-instrument HIL tests.

Takes an :class:`Oscilloscope`, not a concrete driver — the scope is as much an
interface as the generator, so nothing here knows it is talking to a DS1054Z.

**Triggered single-shot, never free-running.** ``acquire_repeating`` returns whatever
frame the scope last auto-triggered, which after a just-changed generator setting can
still be the *previous* signal — a stale-frame flake that reads exactly like a driver
bug. Arming SINGLE and waiting for a real edge cannot do that.

Scaling comes from what the caller expects, not an autoset: ``expected_hz`` puts ~40
cycles on screen so ``describe`` has enough periods to resolve the tone, and
``span_vpp`` scales the channel to ~4 of its 8 vertical divisions for the expected
level. A signal far from expectation therefore lands clipped or tiny, which is what we
want — the assertion should fail loudly rather than be rescued by autoscaling.
"""

from __future__ import annotations

import time

from hwtools.analysis.describe import Characterization, describe
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from hwtools.model.waveform import Waveform
from tests.hardware._acquire import acquire_one_shot

SCOPE_CH = ChannelId.CH1
CYCLES_ON_SCREEN = 40.0
# A generator needs a moment after a config change before its output is the new one;
# arming the scope sooner just captures the transition.
SETTLE_S = 0.3
# Legal for a single enabled channel on the DS1054Z, and deep enough for a clean FFT.
MEMORY_DEPTH = 12_000
# Deepest-first, for spectral work where bin width and leakage matter.
DEEP_MEMORY_OPTIONS = (1_200_000, 120_000, 12_000)
# Keep well under the DS1054Z's 1 GSa/s single-channel ceiling: a short high-frequency
# window cannot hold the deepest record, and asking for it silently gets a lesser one.
MAX_SAMPLE_RATE_HZ = 5e8


def timebase_for(expected_hz: float, *, cycles: float = CYCLES_ON_SCREEN) -> TimebaseConfig:
    """A timebase putting roughly ``cycles`` periods of ``expected_hz`` on screen."""
    return TimebaseConfig(scale_s_per_div=(cycles / expected_hz) / 12.0)


def deepest_memory_under(
    window_s: float,
    *,
    max_sample_rate_hz: float = MAX_SAMPLE_RATE_HZ,
    options: tuple[int, ...] = DEEP_MEMORY_OPTIONS,
) -> int:
    """The deepest offered memory whose resulting sample rate stays under the ceiling.

    Sample rate is depth / window, so a short window forces a shallower record. Falls
    back to the shallowest option rather than failing — that is still a valid capture,
    just a coarser one.
    """
    return next(
        (m for m in options if m / window_s <= max_sample_rate_hz),
        options[-1],
    )


def capture(
    scope: Oscilloscope,
    *,
    expected_hz: float,
    span_vpp: float,
    channel: ChannelId = SCOPE_CH,
    trigger_level_v: float = 0.0,
    cycles: float = CYCLES_ON_SCREEN,
    settle_s: float = SETTLE_S,
    memory_depth: int | None = None,
    deep: bool = False,
) -> Waveform:
    """Settle, scale the scope for the expected signal, then triggered single-shot.

    ``trigger_level_v`` matters for an offset signal: the level must sit inside the
    waveform's excursion or the single shot never triggers. Defaults to 0 V, right for
    anything centred there.

    ``deep`` picks the deepest record the sample-rate ceiling allows for this window,
    for spectral work where bin width and leakage matter; ``memory_depth`` overrides
    both and asks for a specific depth.
    """
    time.sleep(settle_s)
    tb = timebase_for(expected_hz, cycles=cycles)
    if memory_depth is None:
        memory_depth = (
            deepest_memory_under(cycles / expected_hz) if deep else MEMORY_DEPTH
        )
    # Only the channel under test stays on — legal memory depths depend on how many
    # channels are enabled, so leaving others on silently caps the depth.
    for other in (ChannelId.CH1, ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
        if other is not channel:
            scope.configure_channel(
                ChannelConfig(channel=other, scale_v_per_div=1.0, enabled=False)
            )
    scope.configure_channel(
        ChannelConfig(
            channel=channel,
            coupling=Coupling.DC,
            scale_v_per_div=span_vpp / 4.0,
            probe_ratio=1.0,  # BNC coax, not a 10x probe
        )
    )
    scope.configure_timebase(tb)
    scope.configure_acquire(AcquireConfig(memory_depth=memory_depth))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=channel, level_v=trigger_level_v, slope=Slope.RISING),
            sweep=SweepMode.SINGLE,
        )
    )
    return acquire_one_shot(scope, [channel], tb).waveforms[channel]


def measure(
    scope: Oscilloscope,
    *,
    expected_hz: float,
    span_vpp: float,
    channel: ChannelId = SCOPE_CH,
    trigger_level_v: float = 0.0,
    cycles: float = CYCLES_ON_SCREEN,
    settle_s: float = SETTLE_S,
    memory_depth: int | None = None,
    deep: bool = False,
) -> Characterization:
    """:func:`capture`, then :func:`~hwtools.analysis.describe.describe` with the PSD."""
    return describe(
        capture(
            scope,
            expected_hz=expected_hz,
            span_vpp=span_vpp,
            channel=channel,
            trigger_level_v=trigger_level_v,
            cycles=cycles,
            settle_s=settle_s,
            memory_depth=memory_depth,
            deep=deep,
        ),
        psd=True,
    )
