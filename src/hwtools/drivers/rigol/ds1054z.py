"""Rigol DS1054Z oscilloscope driver.

A thin SCPI wrapper translating typed model objects to/from the DS1000Z command
set. No agent logic. Bulk waveform reads default to the raw TCP socket (port
5555), which is far faster than VXI-11.

Waveform scaling (BYTE format): the preamble gives x/y increment, origin and
reference, and a sample byte converts to volts as
``(raw - yorigin - yreference) * yincrement`` at time
``xorigin + i * xincrement``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.capability import ScopeCapabilities
from hwtools.model.capture import Capture
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import AcqType, ChannelId, Slope, SweepMode, TriggerCoupling, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import TriggerConfig
from hwtools.model.waveform import Waveform
from hwtools.transport.base import Transport
from hwtools.transport.raw_tcp import RIGOL_RAW_PORT, RawTcpTransport
from hwtools.transport.visa import VisaTransport

_SLOPE = {Slope.RISING: "POSitive", Slope.FALLING: "NEGative", Slope.EITHER: "RFALl"}
_SWEEP = {SweepMode.AUTO: "AUTO", SweepMode.NORMAL: "NORMal", SweepMode.SINGLE: "SINGle"}
_TRIG_COUPLING = {
    TriggerCoupling.DC: "DC",
    TriggerCoupling.AC: "AC",
    TriggerCoupling.LF_REJECT: "LFReject",
    TriggerCoupling.HF_REJECT: "HFReject",
}
_ACQ_TYPE = {
    AcqType.NORMAL: "NORMal",
    AcqType.AVERAGE: "AVERages",
    AcqType.PEAK: "PEAK",
    AcqType.HIGH_RES: "HRESolution",
}
_TRIG_STATUS = {
    "TD": TriggerStatus.TRIGGERED,
    "WAIT": TriggerStatus.WAIT,
    "RUN": TriggerStatus.RUN,
    "AUTO": TriggerStatus.AUTO,
    "STOP": TriggerStatus.STOP,
}

# Legal :ACQuire:MDEPth values depend on how many analog channels are enabled —
# the memory is shared, so the per-channel record shortens as channels are added
# (DS1000Z programming guide, :ACQuire:MDEPth). Asking for a value outside the set
# for the active channel count is a command error that beeps the scope and desyncs
# the raw socket, so the driver validates against this table before sending.
_MEMORY_DEPTHS: dict[int, tuple[int, ...]] = {
    1: (12_000, 120_000, 1_200_000, 12_000_000, 24_000_000),
    2: (6_000, 60_000, 600_000, 6_000_000, 12_000_000),
    3: (3_000, 30_000, 300_000, 3_000_000, 6_000_000),
    4: (3_000, 30_000, 300_000, 3_000_000, 6_000_000),
}

_CAPABILITIES = ScopeCapabilities(
    model_name="Rigol DS1054Z",
    n_channels=4,
    max_sample_rate_hz=1e9,
    analog_bandwidth_hz=50e6,
    vertical_divisions=8,
    horizontal_divisions=12,
    memory_depths=_MEMORY_DEPTHS[1],  # single-channel set (the deepest available)
    trigger_kinds=("edge",),
    decoder_kinds=("uart", "spi", "i2c"),
)


class DS1054Z(Oscilloscope):
    """Driver for the Rigol DS1054Z over any :class:`Transport`."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    @classmethod
    def over_tcp(cls, host: str, port: int = RIGOL_RAW_PORT) -> DS1054Z:
        """Build a driver using the fast raw SCPI socket."""
        return cls(RawTcpTransport(host, port))

    @classmethod
    def over_visa(cls, resource: str) -> DS1054Z:
        """Build a driver using a pyvisa resource string."""
        return cls(VisaTransport(resource))

    @property
    def capabilities(self) -> ScopeCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        self._t.open()

    def disconnect(self) -> None:
        self._t.close()

    def idn(self) -> str:
        return self._t.query("*IDN?")

    # -- configuration --------------------------------------------------------

    def configure_channel(self, config: ChannelConfig) -> None:
        n = int(config.channel)
        w = self._t.write
        w(f":CHANnel{n}:DISPlay {_onoff(config.enabled)}")
        w(f":CHANnel{n}:PROBe {config.probe_ratio:g}")  # before SCALe: scale is probe-referred
        w(f":CHANnel{n}:COUPling {config.coupling.value}")
        w(f":CHANnel{n}:SCALe {config.scale_v_per_div:g}")
        w(f":CHANnel{n}:OFFSet {config.offset_v:g}")
        w(f":CHANnel{n}:BWLimit {'20M' if config.bandwidth_limit else 'OFF'}")
        w(f":CHANnel{n}:INVert {_onoff(config.invert)}")

    def configure_timebase(self, config: TimebaseConfig) -> None:
        w = self._t.write
        w(f":TIMebase:MODE {config.mode.value}")
        w(f":TIMebase:MAIN:SCALe {config.scale_s_per_div:g}")
        w(f":TIMebase:MAIN:OFFSet {config.offset_s:g}")

    def configure_trigger(self, config: TriggerConfig) -> None:
        w = self._t.write
        w(":TRIGger:MODE EDGE")
        w(f":TRIGger:EDGe:SOURce CHANnel{int(config.source)}")
        w(f":TRIGger:EDGe:SLOPe {_SLOPE[config.trigger.slope]}")
        w(f":TRIGger:EDGe:LEVel {config.trigger.level_v:g}")
        w(f":TRIGger:SWEep {_SWEEP[config.sweep]}")
        w(f":TRIGger:COUPling {_TRIG_COUPLING[config.coupling]}")

    def configure_acquire(self, config: AcquireConfig) -> None:
        w = self._t.write
        w(f":ACQuire:TYPE {_ACQ_TYPE[config.type]}")
        if config.type is AcqType.AVERAGE:
            w(f":ACQuire:AVERages {config.averages}")
        # :ACQuire:MDEPth is honoured only while the scope is RUNNING (DS1000Z
        # programming guide; confirmed in the field) — set it while stopped and the
        # scope silently ignores it. We :RUN first; the caller re-arms with
        # run()/single() afterwards. Never query :SYSTem:ERRor? around a depth
        # change: that read blocks indefinitely and jams the raw socket.
        if config.memory_depth is None:
            w(":RUN")
            w(":ACQuire:MDEPth AUTO")
            return
        n = self._enabled_analog_count()
        valid = _MEMORY_DEPTHS[n]
        if config.memory_depth not in valid:
            raise ValueError(
                f"memory_depth {config.memory_depth} is not a legal record length with "
                f"{n} analog channel(s) enabled; choose one of {valid} (or None for AUTO). "
                f"Memory is shared across enabled channels."
            )
        w(":RUN")
        w(f":ACQuire:MDEPth {config.memory_depth}")

    def _enabled_analog_count(self) -> int:
        """How many analog channels are currently displayed (min 1).

        Queried from the instrument rather than tracked, so a channel left on by
        earlier setup still counts — the legal memory-depth set depends on it.
        """
        on = 0
        for n in range(1, _CAPABILITIES.n_channels + 1):
            if self._t.query(f":CHANnel{n}:DISPlay?").strip() in ("1", "ON"):
                on += 1
        return max(1, on)

    def autoscale(self) -> None:
        self._t.write(":AUToscale")

    # -- run control ----------------------------------------------------------

    def run(self) -> None:
        self._t.write(":RUN")

    def stop(self) -> None:
        self._t.write(":STOP")

    def single(self) -> None:
        self._t.write(":SINGle")

    def force_trigger(self) -> None:
        self._t.write(":TFORce")

    def trigger_status(self) -> TriggerStatus:
        reply = self._t.query(":TRIGger:STATus?").strip().upper()
        if reply not in _TRIG_STATUS:
            raise ValueError(f"unexpected trigger status {reply!r}")
        return _TRIG_STATUS[reply]

    # -- readout --------------------------------------------------------------

    #: Max points the DS1000Z returns from one ``:WAVeform:DATA?`` in RAW/BYTE.
    _RAW_CHUNK = 250_000

    def capture(self, channels: Sequence[ChannelId], *, deep: bool = True) -> Capture:
        # RAW reads come from the frozen acquisition memory, so the scope must be
        # stopped first; the trigger status is then latched at that frozen state.
        if deep:
            self.stop()
        sample_rate_hz = float(self._t.query(":ACQuire:SRATe?"))
        reader = self._read_waveform_raw if deep else self._read_waveform
        waveforms = {ch: reader(ch) for ch in channels}
        return Capture(
            waveforms=waveforms,
            trigger_status=self.trigger_status(),
            sample_rate_hz=sample_rate_hz,
        )

    def _read_waveform_raw(self, channel: ChannelId) -> Waveform:
        """Read the full acquisition memory, paged in :attr:`_RAW_CHUNK` blocks.

        The on-screen ``NORMal`` trace is only 1200 decimated points — coarse
        enough to alias fast logic edges. RAW returns every captured sample (up to
        the memory depth), which is what decode/measure actually need.

        Raises if there is no deep record (RAW points == 0): that means the
        acquisition was free-running (AUTO) or never triggered. The tool always
        arms a SINGLE triggered acquisition before a deep capture, so an empty
        record is a real error to surface — not something to paper over with the
        coarse screen trace.
        """
        w = self._t.write
        w(f":WAVeform:SOURce CHANnel{int(channel)}")
        w(":WAVeform:MODE RAW")
        w(":WAVeform:FORMat BYTE")
        pre = _Preamble.parse(self._t.query(":WAVeform:PREamble?"))
        if pre.points <= 0:
            raise RuntimeError(
                f"no deep acquisition in memory for CH{int(channel)} (RAW points=0). "
                f"Arm a SINGLE triggered capture and wait for the trigger before a deep "
                f"read; deep memory is not filled in AUTO/free-running sweep."
            )
        out = bytearray()
        start = 1
        while start <= pre.points:
            stop = min(start + self._RAW_CHUNK - 1, pre.points)
            w(f":WAVeform:STARt {start}")
            w(f":WAVeform:STOP {stop}")
            out += self._t.query_block(":WAVeform:DATA?")
            start = stop + 1
        if len(out) != pre.points:
            raise OSError(
                f"deep read of CH{int(channel)} returned {len(out)} of "
                f"{pre.points} points (memory short-read)"
            )
        return self._scale(channel, bytes(out), pre)

    def _read_waveform(self, channel: ChannelId) -> Waveform:
        w = self._t.write
        w(f":WAVeform:SOURce CHANnel{int(channel)}")
        w(":WAVeform:MODE NORMal")
        w(":WAVeform:FORMat BYTE")
        pre = _Preamble.parse(self._t.query(":WAVeform:PREamble?"))
        raw = self._t.query_block(":WAVeform:DATA?")
        return self._scale(channel, raw, pre)

    @staticmethod
    def _scale(channel: ChannelId, raw: bytes, pre: _Preamble) -> Waveform:
        codes = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        volts = (codes - pre.yorigin - pre.yreference) * pre.yincrement
        # BYTE codes span 0..255; the extremes are the true digitiser saturation
        # rails, so the analysis can tell genuine clipping from a tall-but-fit signal.
        rail_a = (0.0 - pre.yorigin - pre.yreference) * pre.yincrement
        rail_b = (255.0 - pre.yorigin - pre.yreference) * pre.yincrement
        saturation = (min(rail_a, rail_b), max(rail_a, rail_b))
        return Waveform(
            channel=channel,
            samples=volts,
            t0_s=pre.xorigin,
            dt_s=pre.xincrement,
            saturation=saturation,
        )


def _onoff(value: bool) -> str:
    return "ON" if value else "OFF"


class _Preamble:
    """The ten-field ``:WAVeform:PREamble?`` reply needed to scale samples."""

    __slots__ = (
        "points",
        "xincrement",
        "xorigin",
        "xreference",
        "yincrement",
        "yorigin",
        "yreference",
    )

    def __init__(self, fields: list[str]) -> None:
        self.points = int(fields[2])  # samples available for the active source/mode
        self.xincrement = float(fields[4])
        self.xorigin = float(fields[5])
        self.xreference = float(fields[6])
        self.yincrement = float(fields[7])
        self.yorigin = float(fields[8])
        self.yreference = float(fields[9])

    @classmethod
    def parse(cls, reply: str) -> _Preamble:
        fields = reply.strip().split(",")
        if len(fields) != 10:
            raise ValueError(f"expected 10 preamble fields, got {len(fields)}: {reply!r}")
        return cls(fields)
