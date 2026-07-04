"""The agent-facing tool surface — one stateful facade over scope + store + analysis.

Every operation the agent performs goes through a :class:`ScopeSession`: it holds
the one piece of state that must persist across tool calls (the
:class:`~hwtools.session.store.CaptureStore`) and the live scope, and composes the
pure analysis lenses over stored frames *by handle*. The MCP server
(:mod:`hwtools.tools.server`) is a thin binding over this class; all the real
behaviour — and its tests — live here.

The design line from ``docs/signal-infrastructure-plan.md`` holds: acquisition
returns the lean decision view (:class:`AcquireResult`); everything else runs over
a stored frame referenced by ``capture_id`` (default ``"latest"``), so a one-shot
is captured once and inspected many ways. ``recommend`` stays advisory — it returns
a :class:`Setup` the agent may apply, never a hidden loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hwtools.analysis.classify import Classification, triage
from hwtools.analysis.describe import Characterization, describe
from hwtools.analysis.judge import judge_capture
from hwtools.analysis.recommend import Setup, recommend_setup
from hwtools.decode.frames import I2cTransaction, SpiWord, UartWord
from hwtools.decode.i2c import decode_i2c
from hwtools.decode.spi import SpiParams, decode_spi
from hwtools.decode.threshold import threshold
from hwtools.decode.uart import UartParams, decode_uart
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, SweepMode
from hwtools.model.reading import AcquireResult
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig
from hwtools.model.waveform import DigitalTrace, Waveform
from hwtools.session.acquire import acquire as _acquire
from hwtools.session.store import CaptureInfo, CaptureStore

#: Default hysteresis for auto-thresholding a line to logic levels (fraction of vpp).
_HYSTERESIS_FRAC = 0.2
_MIN_HYSTERESIS_V = 0.2


class ScopeSession:
    """Stateful agent-facing facade: acquire into a store, run lenses by handle."""

    def __init__(self, scope: Oscilloscope, *, store: CaptureStore | None = None) -> None:
        self._scope = scope
        self._store = store or CaptureStore()

    @property
    def store(self) -> CaptureStore:
        return self._store

    @property
    def capabilities(self) -> ScopeCapabilities:
        return self._scope.capabilities

    # -- acquisition ----------------------------------------------------------

    def acquire(
        self,
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
    ) -> AcquireResult:
        """Acquire one frame into the store and return the lean decision view.

        The stored frame is addressed by ``result.capture_id`` for every follow-up
        lens (``describe``/``decode_*``/``triage``) — no re-capture.
        """
        frame = _acquire(
            self._scope,
            self._store,
            channels=channels,
            timebase=timebase,
            trigger=trigger,
            acquire_cfg=acquire_cfg,
            sweep=sweep,
            deep=deep,
            keep=keep,
            label=label,
            trigger_timeout_s=trigger_timeout_s,
        )
        return frame.result

    # -- lenses over a stored frame ------------------------------------------

    def describe(
        self,
        channel: ChannelId,
        *,
        capture_id: str = "latest",
        percentiles: Sequence[float] | None = None,
        value_bins: int | None = None,
        time_bins: int | None = None,
        psd: bool = False,
        freq_bins: int | None = None,
        joint: bool = False,
        joint_bins: int | None = None,
    ) -> Characterization:
        """Dense characterisation of one channel of a stored frame."""
        wf = self._waveform(capture_id, channel)
        kwargs = {} if percentiles is None else {"percentiles": percentiles}
        return describe(
            wf,
            value_bins=value_bins,
            time_bins=time_bins,
            psd=psd,
            freq_bins=freq_bins,
            joint=joint,
            joint_bins=joint_bins,
            **kwargs,
        )

    def triage(self, *, capture_id: str = "latest") -> dict[ChannelId, Classification]:
        """Classify every channel of a stored frame (the unknown-signal front door)."""
        return triage(self._store.get(capture_id))

    def judge(
        self,
        channels: Mapping[ChannelId, ChannelConfig],
        *,
        capture_id: str = "latest",
        timebase: TimebaseConfig | None = None,
    ) -> AcquireResult:
        """Re-judge a stored frame against a channel configuration."""
        return judge_capture(
            self._store.get(capture_id), channels, self._scope.capabilities, timebase
        )

    def recommend(
        self,
        *,
        channels: Mapping[ChannelId, ChannelConfig],
        timebase: TimebaseConfig,
        trigger: TriggerConfig,
        capture_id: str = "latest",
    ) -> Setup:
        """Advise a full setup from a stored measurement frame (the agent applies it)."""
        return recommend_setup(
            self._store.get(capture_id),
            channels=channels,
            timebase=timebase,
            trigger=trigger,
            capabilities=self._scope.capabilities,
        )

    # -- decode over a stored frame ------------------------------------------

    def decode_uart(
        self, channel: ChannelId, *, params: UartParams, capture_id: str = "latest"
    ) -> list[UartWord]:
        return decode_uart(self._digital(capture_id, channel), params)

    def decode_spi(
        self,
        *,
        clk: ChannelId,
        mosi: ChannelId | None = None,
        miso: ChannelId | None = None,
        cs: ChannelId | None = None,
        params: SpiParams | None = None,
        capture_id: str = "latest",
    ) -> list[SpiWord]:
        return decode_spi(
            self._digital(capture_id, clk),
            mosi=self._digital(capture_id, mosi) if mosi is not None else None,
            miso=self._digital(capture_id, miso) if miso is not None else None,
            cs=self._digital(capture_id, cs) if cs is not None else None,
            params=params,
        )

    def decode_i2c(
        self, *, sda: ChannelId, scl: ChannelId, capture_id: str = "latest"
    ) -> list[I2cTransaction]:
        return decode_i2c(
            self._digital(capture_id, sda), self._digital(capture_id, scl)
        )

    # -- store management -----------------------------------------------------

    def list_captures(self) -> list[CaptureInfo]:
        return self._store.list()

    def keep(self, capture_id: str, *, label: str | None = None) -> None:
        self._store.keep(capture_id, label=label)

    def drop(self, capture_id: str) -> None:
        self._store.drop(capture_id)

    def clear(self) -> None:
        self._store.clear()

    # -- internals ------------------------------------------------------------

    def _waveform(self, capture_id: str, channel: ChannelId) -> Waveform:
        capture = self._store.get(capture_id)
        wf = capture.waveforms.get(channel)
        if wf is None:
            present = ", ".join(c.name for c in capture.waveforms)
            raise KeyError(f"{channel.name} not in capture {capture_id!r}; present: {present}")
        return wf

    def _digital(self, capture_id: str, channel: ChannelId) -> DigitalTrace:
        """Auto-threshold a channel to logic levels at its midline with hysteresis."""
        wf = self._waveform(capture_id, channel)
        level = (wf.vmin + wf.vmax) / 2.0
        return threshold(
            wf, level_v=level, hysteresis_v=max(wf.vpp * _HYSTERESIS_FRAC, _MIN_HYSTERESIS_V)
        )
