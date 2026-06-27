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

_CAPABILITIES = ScopeCapabilities(
    model_name="Rigol DS1054Z",
    n_channels=4,
    max_sample_rate_hz=1e9,
    analog_bandwidth_hz=50e6,
    vertical_divisions=8,
    horizontal_divisions=12,
    memory_depths=(12_000, 120_000, 1_200_000, 12_000_000, 24_000_000),
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
        w(f":ACQuire:MDEPth {config.memory_depth if config.memory_depth is not None else 'AUTO'}")

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

    def capture(self, channels: Sequence[ChannelId]) -> Capture:
        sample_rate_hz = float(self._t.query(":ACQuire:SRATe?"))
        waveforms = {ch: self._read_waveform(ch) for ch in channels}
        return Capture(
            waveforms=waveforms,
            trigger_status=self.trigger_status(),
            sample_rate_hz=sample_rate_hz,
        )

    def _read_waveform(self, channel: ChannelId) -> Waveform:
        w = self._t.write
        w(f":WAVeform:SOURce CHANnel{int(channel)}")
        w(":WAVeform:MODE NORMal")
        w(":WAVeform:FORMat BYTE")
        pre = _Preamble.parse(self._t.query(":WAVeform:PREamble?"))
        raw = self._t.query_block(":WAVeform:DATA?")
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

    __slots__ = ("xincrement", "xorigin", "xreference", "yincrement", "yorigin", "yreference")

    def __init__(self, fields: list[str]) -> None:
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
