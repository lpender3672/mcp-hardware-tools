"""Joy-IT JDS6600 DDS signal-generator driver over its USB-CDC serial channel.

Speaks the JDS6600 line protocol (``docs/JT-JDS6600-Communication-protocol.pdf``,
corrected against the bench — the manual has typos): 115200 8N1, one command per
line framed ``:<op><fn>=<data>.\\r\\n``. ``w`` writes a parameter and the device
replies ``:ok``; ``r`` reads one back as ``:r<fn>=<data>.``. This realises the
:class:`SignalGenerator` contract, so it is driven from transcript unit tests (via
:meth:`from_serial`) with no hardware, and its *output* is validated against a real
scope in the cross-instrument HIL suite.

Wire encodings (all bench-confirmed on a JDS6600-15):

* **frequency** — centi-Hz with unit code 0: ``:w23=<round(hz*100)>,0.`` (a read of
  1 MHz returns ``100000000,0``). Writing always in unit code 0 sidesteps the
  manual's inconsistent multi-unit table.
* **amplitude** — millivolt Vpp: ``round(vpp*1000)`` (``5000`` = 5.0 Vpp).
* **offset (bias)** — ``round(v*100)+1000`` (``1000`` = 0 V, range ±9.99 V).
* **duty** — tenths of a percent: ``round(pct*10)`` (``500`` = 50.0 %).
* **phase** — tenths of a degree: ``round(deg*10)``.

Function codes: 20 output-enable (both channels, ``a,b``), 21/22 waveform,
23/24 frequency, 25/26 amplitude, 27/28 bias, 29/30 duty (per channel — the
manual's shared-``w29`` claim is wrong), 31 CH2-vs-CH1 phase. The per-channel codes
are ``base + (channel - 1)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from hwtools.interfaces.signal_generator import SignalGenerator
from hwtools.model.siggen import (
    ArbAddressing,
    ArbitraryWaveform,
    ArbLength,
    PlaybackMode,
    SigGenCapabilities,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
    check_arbitrary_length,
    check_within,
    resolve_arb_sample_rate,
)

# CH340 USB-serial bridge the JDS6600 enumerates behind. NOTE: the CH340 is a
# generic bridge used by many devices, so this is a best-effort match, not a
# guaranteed-unique JDS6600 identifier (unlike the harness's dedicated VID:PID).
JDS6600_USB_VID = 0x1A86
JDS6600_USB_PID = 0x7523

# Bias code is offset in hundredths of a volt, biased so 1000 == 0 V.
_BIAS_ZERO = 1000
_BIAS_PER_VOLT = 100

_SHAPE_CODE: dict[WaveShape, int] = {
    WaveShape.SINE: 0,
    WaveShape.SQUARE: 1,
    WaveShape.PULSE: 2,
    WaveShape.TRIANGLE: 3,
    WaveShape.PARTIAL_SINE: 4,
    WaveShape.CMOS: 5,
    WaveShape.DC: 6,
    WaveShape.HALF_WAVE: 7,
    WaveShape.FULL_WAVE: 8,
    WaveShape.POS_STEP: 9,
    WaveShape.NEG_STEP: 10,
    WaveShape.NOISE: 11,
    WaveShape.EXP_RISE: 12,
    WaveShape.EXP_DECAY: 13,
    WaveShape.MULTI_TONE: 14,
    WaveShape.SINC: 15,
    WaveShape.LORENZ: 16,
}
_CODE_SHAPE: dict[int, WaveShape] = {code: wf for wf, code in _SHAPE_CODE.items()}

# Arbitrary waveforms: 60 slots, 2048 points each, 12-bit codes (0..4095) sent as
# ASCII CSV via the ``a`` (write) / ``b`` (read) operators. Bench-confirmed against a
# ``:b01=`` read. Slot N is selected for output as waveform code 100 + N.
_ARB_SLOTS = 60
_ARB_POINTS = 2048
_ARB_CODE_LEVELS = 4096
_ARB_CODE_MAX = _ARB_CODE_LEVELS - 1
# A channel plays arbitrary slot N when its waveform code is 100 + N (w21=101 = arb 01).
_ARB_WAVE_BASE = 100

_CAPABILITIES = SigGenCapabilities(
    model_name="Joy-IT JDS6600-15",
    n_channels=2,
    max_frequency_hz=15e6,  # JDS6600-15; the device clamps writes to 1_500_000_000 cHz
    max_amplitude_vpp=20.0,  # device clamps amplitude writes to 20_000 mV
    max_offset_v=9.99,
    waveforms=tuple(_SHAPE_CODE),
    playback_modes=frozenset({PlaybackMode.CONTINUOUS}),  # pure DDS: continuous loop only
    arb_addressing=ArbAddressing.SLOT,  # 60 device-global numbered slots
    arb_slots=_ARB_SLOTS,
    arb_length=ArbLength.fixed(_ARB_POINTS),  # fixed 2048-point DDS wavetable
    arb_code_levels=_ARB_CODE_LEVELS,
)


@runtime_checkable
class SerialLike(Protocol):
    """The slice of ``serial.Serial`` this driver needs (so it can be faked)."""

    def write(self, data: bytes) -> int | None: ...
    def readline(self) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def find_jds6600_port() -> str | None:
    """Return the serial port of the first attached CH340 bridge, if any.

    Best-effort: matches the CH340 VID:PID, which the JDS6600 uses but so do many
    other devices — confirm with :meth:`idn` if more than one CH340 is present.
    """
    from serial.tools import list_ports

    for port in list_ports.comports():
        if port.vid == JDS6600_USB_VID and port.pid == JDS6600_USB_PID:
            return str(port.device)
    return None


class JDS6600(SignalGenerator):
    """Driver for the Joy-IT JDS6600 over a serial command channel."""

    def __init__(self, port: str, *, baud: int = 115200, timeout_s: float = 1.0) -> None:
        self._port = port
        self._baud = baud
        self._timeout_s = timeout_s
        self._serial: SerialLike | None = None

    @classmethod
    def from_serial(cls, serial: SerialLike) -> JDS6600:
        """Build a driver around an already-open serial-like object (for tests)."""
        gen = cls(port="<injected>")
        gen._serial = serial
        return gen

    # -- connection -----------------------------------------------------------

    @property
    def capabilities(self) -> SigGenCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        if self._serial is None:
            import serial as pyserial

            self._serial = pyserial.Serial(self._port, self._baud, timeout=self._timeout_s)
        self._serial.reset_input_buffer()

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    @property
    def _link(self) -> SerialLike:
        if self._serial is None:
            raise RuntimeError("generator is not connected; call connect() first")
        return self._serial

    def idn(self) -> str:
        """Identify as ``JDS6600-<model> SN:<serial>`` from function codes 00/01."""
        model = self._query(0)
        serial = self._query(1)
        return f"JDS6600-{model} SN:{serial}"

    # -- serial framing -------------------------------------------------------

    def _transact(self, line: str) -> str:
        self._link.write(line.encode("ascii") + b"\r\n")
        return self._link.readline().decode("ascii", errors="replace").strip()

    def _command(self, op: str, code: int, data: str) -> str:
        """Frame and send one ``:<op><code>=<data>.`` line, returning the raw reply."""
        return self._transact(f":{op}{code:02d}={data}.")

    def _write(self, fn: int, data: str) -> None:
        """Send a ``:w<fn>=<data>.`` and require the ``:ok`` acknowledgement."""
        reply = self._command("w", fn, data)
        if reply.lstrip(":").lower() != "ok":
            raise RuntimeError(f":w{fn:02d}={data}. rejected: {reply!r}")

    def _query(self, fn: int) -> str:
        """Send a ``:r<fn>=0.`` and return the echoed data field (sans trailing dot)."""
        reply = self._command("r", fn, "0")
        prefix = f":r{fn:02d}="
        if not reply.startswith(prefix):
            raise RuntimeError(f":r{fn:02d}=0. returned unexpected reply: {reply!r}")
        return reply[len(prefix) :].rstrip(".")

    # -- configuration --------------------------------------------------------

    def configure_channel(self, config: SignalGeneratorConfig) -> None:
        check_within(config, _CAPABILITIES)
        ch = config.channel
        # Shape (21/22): an arb slot is selected as code 100 + slot; else a built-in.
        if config.arb_slot is not None:
            self._write(20 + ch, str(_ARB_WAVE_BASE + config.arb_slot))
        else:
            self._write(20 + ch, str(_SHAPE_CODE[config.waveform]))
        self._write(22 + ch, f"{round(config.frequency_hz * 100)},0")  # 23 / 24 frequency
        self._write(24 + ch, str(round(config.amplitude_vpp * 1000)))  # 25 / 26 amplitude
        self._write(26 + ch, str(round(config.offset_v * _BIAS_PER_VOLT) + _BIAS_ZERO))  # 27/28
        self._write(28 + ch, str(round(config.duty_pct * 10)))  # 29 / 30 duty
        self.enable_output(ch, config.enabled)

    def enable_output(self, channel: SigGenChannel, on: bool) -> None:
        # Output-enable (fn 20) is a shared both-channels register: read the current
        # pair, flip only this channel's bit, write it back.
        ch1, ch2 = self._output_pair()
        if channel is SigGenChannel.CH1:
            ch1 = on
        else:
            ch2 = on
        self._write(20, f"{int(ch1)},{int(ch2)}")

    def set_phase_deg(self, degrees: float) -> None:
        if not 0.0 <= degrees <= 360.0:
            raise ValueError("phase must be within 0..360 degrees")
        self._write(31, str(round(degrees * 10)))

    # -- arbitrary waveforms --------------------------------------------------

    def upload_arbitrary(
        self,
        channel: SigGenChannel,
        wave: ArbitraryWaveform,
        *,
        sample_rate_hz: float | None = None,
        slot: int | None = None,
    ) -> None:
        # SLOT-addressed: slots are device-global, so `channel` is unused here; a
        # channel selects a slot for output via SignalGeneratorConfig.arb_slot.
        slot = self._require_slot(slot)
        check_arbitrary_length(wave.n, _CAPABILITIES)
        # A DDS wavetable: the slot plays at the channel's configured frequency, so
        # there is no playback clock to set. Refuses a rate rather than ignoring it.
        resolve_arb_sample_rate(sample_rate_hz, _CAPABILITIES)
        codes = np.rint((wave.samples + 1.0) / 2.0 * _ARB_CODE_MAX).astype(np.int64)
        reply = self._command("a", slot, ",".join(map(str, codes.tolist())))
        if reply.lstrip(":").lower() != "ok":
            raise RuntimeError(f"arbitrary upload to slot {slot} rejected: {reply!r}")

    def read_arbitrary(
        self, channel: SigGenChannel, *, slot: int | None = None
    ) -> ArbitraryWaveform:
        slot = self._require_slot(slot)
        reply = self._command("b", slot, "0")
        prefix = f":b{slot:02d}="
        if not reply.startswith(prefix):
            raise RuntimeError(f"arbitrary read of slot {slot} unexpected reply: {reply!r}")
        codes = np.array(
            [int(x) for x in reply[len(prefix) :].rstrip(".").split(",") if x], dtype=np.float64
        )
        return ArbitraryWaveform(samples=codes / _ARB_CODE_MAX * 2.0 - 1.0)

    def _require_slot(self, slot: int | None) -> int:
        if slot is None:
            raise ValueError(
                f"the JDS6600 addresses arbitrary waveforms by global slot; "
                f"pass slot=1..{_ARB_SLOTS}"
            )
        self._check_slot(slot)
        return slot

    def _check_slot(self, slot: int) -> None:
        if not 1 <= slot <= _ARB_SLOTS:
            raise ValueError(f"arbitrary slot must be 1..{_ARB_SLOTS}; got {slot}")

    # -- readback -------------------------------------------------------------

    def read_channel(self, channel: SigGenChannel) -> SignalGeneratorConfig:
        ch = channel
        wcode = int(self._query(20 + ch))
        # Codes >= 100 mean an arbitrary slot is selected; the built-in waveform
        # field is then a don't-care (kept at its default) and arb_slot governs.
        arb_slot = wcode - _ARB_WAVE_BASE if wcode >= _ARB_WAVE_BASE else None
        waveform = WaveShape.SINE if arb_slot is not None else _CODE_SHAPE[wcode]
        freq_val, _unit = self._query(22 + ch).split(",")
        amplitude_vpp = int(self._query(24 + ch)) / 1000
        offset_v = (int(self._query(26 + ch)) - _BIAS_ZERO) / _BIAS_PER_VOLT
        duty_pct = int(self._query(28 + ch)) / 10
        return SignalGeneratorConfig(
            channel=channel,
            waveform=waveform,
            frequency_hz=int(freq_val) / 100,
            amplitude_vpp=amplitude_vpp,
            offset_v=offset_v,
            duty_pct=duty_pct,
            enabled=self.output_enabled(channel),
            arb_slot=arb_slot,
        )

    def output_enabled(self, channel: SigGenChannel) -> bool:
        ch1, ch2 = self._output_pair()
        return ch1 if channel is SigGenChannel.CH1 else ch2

    def _output_pair(self) -> tuple[bool, bool]:
        a, b = self._query(20).split(",")
        return bool(int(a)), bool(int(b))
