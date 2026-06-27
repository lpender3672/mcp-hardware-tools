"""A simulated oscilloscope over synthetic signals.

Implements the same :class:`Oscilloscope` interface as the real driver, so the
whole self-correcting loop runs in CI with no bench. It renders true signals
(:mod:`hwtools.drivers.simulated.signals`) through the current vertical/timebase/
trigger configuration, reproducing the behaviours the loop must handle:

* **clipping** when the vertical scale is too small for the signal,
* **trigger miss** when the trigger level is outside the signal's range,
* a captured window of ``timebase_scale * horizontal_divisions`` seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from hwtools.drivers.simulated.signals import SimSignal
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Slope, SweepMode, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig
from hwtools.model.waveform import Waveform

DEFAULT_CAPABILITIES = ScopeCapabilities(
    model_name="SimulatedScope",
    n_channels=4,
    max_sample_rate_hz=1e9,
    analog_bandwidth_hz=100e6,
    vertical_divisions=8,
    horizontal_divisions=12,
    memory_depths=(1200,),
    trigger_kinds=("edge",),
)


class SimulatedScope(Oscilloscope):
    """An oscilloscope backed by synthetic signals."""

    def __init__(
        self,
        signals: Mapping[ChannelId, SimSignal],
        *,
        capabilities: ScopeCapabilities = DEFAULT_CAPABILITIES,
        noise_v: float = 0.0,
        n_points: int = 1200,
        adc_overscan: float = 1.275,
        seed: int = 0,
    ) -> None:
        self._signals = dict(signals)
        self._caps = capabilities
        self._noise_v = noise_v
        self._n_points = n_points
        # The digitiser captures beyond the on-screen graticule: the DS1000Z
        # digitises ~+/-5 divisions, not the +/-4 of the 8-div screen (divergence
        # #1). 1.275 * (8/2) ~= 5.1 div, matching the bench.
        self._adc_overscan = adc_overscan
        self._rng = np.random.default_rng(seed)
        self._channels: dict[ChannelId, ChannelConfig] = {}
        self._timebase = TimebaseConfig(scale_s_per_div=1e-3)
        self._trigger: TriggerConfig | None = None
        self._acquire = AcquireConfig()
        self._connected = False
        self._status = TriggerStatus.STOP

    # -- identity & connection ------------------------------------------------

    @property
    def capabilities(self) -> ScopeCapabilities:
        return self._caps

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def idn(self) -> str:
        return f"hwtools,{self._caps.model_name},SIM0,0.1"

    # -- configuration --------------------------------------------------------

    def configure_channel(self, config: ChannelConfig) -> None:
        self._channels[config.channel] = config

    def configure_timebase(self, config: TimebaseConfig) -> None:
        self._timebase = config

    def configure_trigger(self, config: TriggerConfig) -> None:
        self._trigger = config

    def configure_acquire(self, config: AcquireConfig) -> None:
        self._acquire = config

    def autoscale(self) -> None:
        """Best-effort instrument autoscale from the known signals."""
        divisions_v = self._caps.vertical_divisions
        for channel, signal in self._signals.items():
            lo, hi = signal.vrange
            span = max(hi - lo, 1e-3)
            scale = max(span / (divisions_v - 2), 1e-3)  # fill ~6 of 8 divisions
            mid = (hi + lo) / 2.0
            self._channels[channel] = ChannelConfig(
                channel=channel, scale_v_per_div=scale, offset_v=-mid
            )

    # -- run control ----------------------------------------------------------

    def run(self) -> None:
        self._status = TriggerStatus.AUTO

    def stop(self) -> None:
        self._status = TriggerStatus.STOP

    def single(self) -> None:
        # Arm a single acquisition: it completes (STOP) once the trigger would
        # fire on the signal, otherwise it waits forever (WAIT).
        self._status = TriggerStatus.STOP if self._would_trigger() else TriggerStatus.WAIT

    def force_trigger(self) -> None:
        self._status = TriggerStatus.AUTO

    def _would_trigger(self) -> bool:
        if self._trigger is None:
            return False
        signal = self._signals.get(self._trigger.trigger.source)
        if signal is None:
            return False
        low, high = signal.vrange
        return low <= self._trigger.trigger.level_v <= high

    def trigger_status(self) -> TriggerStatus:
        return self._status

    # -- readout --------------------------------------------------------------

    def capture(self, channels: Sequence[ChannelId]) -> Capture:
        window_s = self._timebase.scale_s_per_div * self._caps.horizontal_divisions
        dt_s = window_s / self._n_points
        triggered, t0_s = self._resolve_trigger(window_s, dt_s)
        self._status = self._status_for(triggered)

        times = t0_s + np.arange(self._n_points, dtype=np.float64) * dt_s
        waveforms = {ch: self._render(ch, times, dt_s, t0_s) for ch in channels}
        return Capture(
            waveforms=waveforms,
            trigger_status=self._status,
            sample_rate_hz=1.0 / dt_s,
            memory_depth=self._n_points,
        )

    def _render(
        self, channel: ChannelId, times: npt.NDArray[np.float64], dt_s: float, t0_s: float
    ) -> Waveform:
        config = self._channels.get(channel) or ChannelConfig(
            channel=channel, scale_v_per_div=1.0
        )
        signal = self._signals.get(channel)
        true_v = (
            signal.sample(times)
            if signal is not None
            else np.zeros(self._n_points, dtype=np.float64)
        )
        if self._noise_v > 0:
            true_v = true_v + self._rng.normal(0.0, self._noise_v, size=true_v.shape)

        # The digitiser saturates a little beyond the screen (see _adc_overscan).
        halfspan = (
            config.scale_v_per_div * (self._caps.vertical_divisions / 2.0) * self._adc_overscan
        )
        lo = -config.offset_v - halfspan
        hi = -config.offset_v + halfspan
        samples = np.clip(true_v, lo, hi)
        return Waveform(
            channel=channel, samples=samples, t0_s=t0_s, dt_s=dt_s, saturation=(lo, hi)
        )

    def _resolve_trigger(self, window_s: float, dt_s: float) -> tuple[bool, float]:
        """Return (triggered, window start time), aligning a trigger to centre."""
        centre_offset = self._timebase.offset_s
        if self._trigger is None:
            return (False, centre_offset - window_s / 2.0)

        edge = self._trigger.trigger
        signal = self._signals.get(edge.source)
        if signal is None:
            return (False, centre_offset - window_s / 2.0)

        lo, hi = signal.vrange
        if not (lo <= edge.level_v <= hi):
            return (False, centre_offset - window_s / 2.0)

        crossing = self._first_crossing(signal, edge.level_v, edge.slope, window_s, dt_s)
        if crossing is None:
            return (False, centre_offset - window_s / 2.0)
        return (True, crossing + centre_offset - window_s / 2.0)

    @staticmethod
    def _first_crossing(
        signal: SimSignal, level: float, slope: Slope, window_s: float, dt_s: float
    ) -> float | None:
        search = np.arange(0.0, max(2.0 * window_s, dt_s), dt_s, dtype=np.float64)
        values = signal.sample(search)
        above = values >= level
        delta = np.diff(above.astype(np.int8))
        if slope is Slope.RISING:
            idx = np.flatnonzero(delta == 1)
        elif slope is Slope.FALLING:
            idx = np.flatnonzero(delta == -1)
        else:
            idx = np.flatnonzero(delta != 0)
        return float(search[int(idx[0]) + 1]) if idx.size else None

    def _status_for(self, triggered: bool) -> TriggerStatus:
        if triggered:
            return TriggerStatus.TRIGGERED
        sweep = self._trigger.sweep if self._trigger is not None else SweepMode.AUTO
        return TriggerStatus.AUTO if sweep is SweepMode.AUTO else TriggerStatus.WAIT
