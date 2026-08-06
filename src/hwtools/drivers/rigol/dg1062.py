"""Rigol DG1062Z arbitrary/function-generator driver.

A thin SCPI wrapper translating typed :mod:`hwtools.model` objects to/from the
DG1000Z command set (``docs/DG1000Z_ProgrammingGuide_EN.pdf``, cited below by
section). No agent logic. It speaks over the same
:class:`~hwtools.transport.base.Transport` seam as the DS1054Z scope and shares its
connection plumbing via :class:`ScpiTransportMixin`.

**The write path restarts the output stage around discrete parameter writes**:

    :OUTPut OFF → settle → :FUNCtion:ARBitrary:MODE → :OUTPut:LOAD → :FUNCtion →
    :FREQuency → :VOLTage → :VOLTage:OFFSet → duty → settle → :OUTPut ON

Every element of that is load-bearing, and three of them were learned the hard way
rather than read off the page — see :data:`_OUTPUT_SETTLE_S` for the measurements.

*Why discrete writes and not* ``:APPLy`` (the guide's Ch3 "Method 2", not "Method 1"):

* **Omitted parameters are reset, not retained.** Ch1, *Symbol Description*, Square
  Brackets: "If the parameter is omitted, the instrument will set the parameter to
  its default." Every ``:APPLy:*`` verb ends in an optional ``<phase>`` defaulting to
  0°, so ``:APPLy:SINusoid <f>,<a>,<o>`` silently zeroes the start phase and fights
  :meth:`~DG1062.set_phase_deg`. Discrete writes touch only what they name.
* **Changing waveform type carries the old frequency over, then clamps it.**
  ``:FREQuency[:FIXed]`` (2-96): "the instrument still uses the frequency if the
  frequency is valid for the new waveform type; the instrument will display prompt
  message and set the frequency to the frequency upper limit of the new waveform type
  automatically if the frequency is invalid". Escaping that needs *type first, then
  every parameter rewritten* — an order ``:APPLy`` cannot express.

*Why the output is cycled:* a channel playing the volatile arbitrary **cannot be moved
onto a built-in waveform while its output is live**. Nothing in the command set
achieves it — not discrete writes, not ``:APPLy`` with every argument supplied, not
``:FUNCtion:ARBitrary:MODE FREQ``, not any delay. The registers *and the front panel*
switch to the new waveform while the output keeps emitting the old buffer, so no
read-back can detect it. Restarting the output stage is what reloads it.

*Why* ``:FUNCtion:ARBitrary:MODE FREQ`` *is still written:* not as the arbitrary
escape (it does not work as one), but because sample-rate mode leaves ``:FREQuency``
inert — "cannot set the frequency or period" (2-100) — so a channel returning from an
upload would otherwise hand its stale frequency to the type-change clamp above.

**Amplitude is output-load-referred.** The DG1000Z scales the delivered level to the
configured load; driving a high-Z scope at a 50 Ω setting doubles the real voltage.
The driver applies :attr:`SignalGeneratorConfig.output_load_ohms` (a high-Z default)
before the levels, because ``<amplitude>``'s range "is limited by the Impedance"
and ``<offset>``'s by the impedance *and* amplitude — so load → frequency →
amplitude → offset is the guide's own dependency order.
"""

from __future__ import annotations

import math
import time

import numpy as np

from hwtools.drivers.rigol._scpi import ScpiTransportMixin
from hwtools.interfaces.signal_generator import SignalGenerator
from hwtools.model.siggen import (
    ArbAddressing,
    ArbitraryWaveform,
    ArbLength,
    ArbSampleRate,
    PlaybackMode,
    SigGenCapabilities,
    SigGenChannel,
    SignalGeneratorConfig,
    WaveShape,
    check_arbitrary_length,
    check_within,
    resolve_arb_sample_rate,
)
from hwtools.transport.base import Transport

# WaveShape -> :FUNCtion[:SHAPe] <name>, spelled as the guide lists them (2-108). Only
# shapes the DG1062Z can produce appear; anything absent is refused by check_within
# (loud, per the tool-design philosophy) rather than silently substituted.
#
# TRIANGLE maps to RAMP, whose symmetry carries the shape: ":FUNCtion:RAMP:SYMMetry"
# (2-107) defines symmetry as "the percentage that the rising period of the ramp
# waveform takes up in the period", so 50 % is a symmetric triangle and 100 % a
# rising sawtooth. NB the guide contradicts itself here — ":APPLy:TRIangle" (2-80)
# claims "the triangle waveform is ramp waveform with 100% symmetry", which by that
# definition would be a sawtooth. We follow the symmetry definition and expose the
# knob as duty_pct rather than hard-coding either value.
_SHAPE_NAME: dict[WaveShape, str] = {
    WaveShape.SINE: "SINusoid",
    WaveShape.SQUARE: "SQUare",
    WaveShape.PULSE: "PULSe",
    WaveShape.TRIANGLE: "RAMP",
    WaveShape.DC: "DC",
    WaveShape.NOISE: "NOISe",
    WaveShape.SINC: "SINC",
    WaveShape.LORENZ: "LORENTZ",
    WaveShape.EXP_RISE: "EXPRISE",
    WaveShape.EXP_DECAY: "EXPFALL",
    WaveShape.HALF_WAVE: "ABSSINEHALF",
    WaveShape.FULL_WAVE: "ABSSINE",
}

# Reverse map for read-back. ":FUNCtion? ... returns a string, for example, SQU"
# (2-108) — i.e. the abbreviated form — but abbreviation is only *permitted*, not
# guaranteed, so both spellings are accepted for every shape that has two.
_NAME_SHAPE: dict[str, WaveShape] = {
    "SIN": WaveShape.SINE,
    "SINUSOID": WaveShape.SINE,
    "SQU": WaveShape.SQUARE,
    "SQUARE": WaveShape.SQUARE,
    "RAMP": WaveShape.TRIANGLE,
    "PULS": WaveShape.PULSE,
    "PULSE": WaveShape.PULSE,
    "NOIS": WaveShape.NOISE,
    "NOISE": WaveShape.NOISE,
    "DC": WaveShape.DC,
    "SINC": WaveShape.SINC,
    "LORENTZ": WaveShape.LORENZ,
    "EXPRISE": WaveShape.EXP_RISE,
    "EXPFALL": WaveShape.EXP_DECAY,
    "ABSSINE": WaveShape.FULL_WAVE,
    "ABSSINEHALF": WaveShape.HALF_WAVE,
}

# ":FUNCtion?" reply meaning "this channel is playing the volatile arbitrary buffer"
# — not a WaveShape, so read_channel cannot represent it and says so.
_VOLATILE_ARB_NAME = "USER"

# Per-shape maximum output frequency, from Table 2-1 (2-67, "Frequency ranges
# available for the different models and different waveforms"), DG1062Z column — the
# guide's authoritative table, which :FREQuency[:FIXed] points at. Shapes absent here
# have no frequency at all (DC; noise is specified as a 60 MHz *bandwidth*).
# The built-in named arbs (SINC, LORENTZ, ...) are "Arbitrary Waveform" -> 20 MHz.
_ARB_MAX_HZ = 20e6
_SHAPE_MAX_HZ: dict[WaveShape, float] = {
    WaveShape.SINE: 60e6,
    WaveShape.SQUARE: 25e6,
    WaveShape.PULSE: 25e6,
    WaveShape.TRIANGLE: 1e6,  # ramp
    WaveShape.SINC: _ARB_MAX_HZ,
    WaveShape.LORENZ: _ARB_MAX_HZ,
    WaveShape.EXP_RISE: _ARB_MAX_HZ,
    WaveShape.EXP_DECAY: _ARB_MAX_HZ,
    WaveShape.HALF_WAVE: _ARB_MAX_HZ,
    WaveShape.FULL_WAVE: _ARB_MAX_HZ,
}

# Table 2-1 gives every waveform a 1 Hz floor. (The per-command :APPLy tables say
# "1uHz" instead; Table 2-1 is the one :FREQuency[:FIXed] cites, and the tighter of
# two documented limits is the safe one to enforce.)
_MIN_FREQUENCY_HZ = 1.0
# ":VOLTage[:LEVel][:IMMediate][:AMPLitude]" (2-180): "The minimum of <amplitude> is
# 2mVpp"; below it the instrument sets the amplitude to its lower limit and says
# nothing, so the caps floor turns that into a loud refusal.
_MIN_AMPLITUDE_VPP = 0.002
# ":FUNCtion:PULSe:DCYCle" (2-101) documents 0.001 % to 99.999 %. The square duty
# range is only "limited by the waveform frequency" (2-108) and so cannot be
# pre-computed — that one is left to the instrument's error queue.
_PULSE_DUTY_MIN_PCT = 0.001
_PULSE_DUTY_MAX_PCT = 99.999
# ":FUNCtion:RAMP:SYMMetry" (2-107): 0 % to 100 %.
_SYMMETRY_MIN_PCT = 0.0
_SYMMETRY_MAX_PCT = 100.0
# Shapes with no duty-cycle/symmetry control at all; a non-default duty_pct for one of
# these is a caller mistake, so it is refused rather than quietly dropped.
_DUTY_SHAPES = frozenset({WaveShape.SQUARE, WaveShape.PULSE, WaveShape.TRIANGLE})
_DEFAULT_DUTY_PCT = 50.0
# Shapes carrying no frequency, whose :FREQuency register is a placeholder.
_NO_FREQUENCY_SHAPES = frozenset({WaveShape.DC, WaveShape.NOISE})

# Reconfiguring a channel requires *restarting its output stage*, not merely rewriting
# its registers. Two distinct hazards, both scope-measured on a DG1062Z (fw 03.01.12)
# and both invisible to read-back — which is exactly why they survive a driver's own
# round-trip checks, and why only a second instrument can catch them:
#
# 1. **A channel playing the volatile arbitrary cannot be talked out of it.** With the
#    output left on, *nothing* dislodges it: neither discrete :FUNCtion/:FREQuency/
#    :VOLTage writes, nor :APPLy:SINusoid with all four arguments, nor
#    :FUNCtion:ARBitrary:MODE FREQ, nor any settle. The registers and the front panel
#    both switch to the new waveform while the output keeps emitting the old buffer.
#    Cycling :OUTPut:STATe around the parameter writes fixes it (4/4), as does applying
#    the whole configuration twice (4/4).
# 2. **An enable that overtakes pending writes comes up on the previous waveform.** The
#    DG1000Z executes commands in an overlapped fashion — hence the guide documenting
#    *OPC? to "ensure synchronization" (2-35) — so :OUTPut:STATe ON needs a gap after
#    the parameter writes. Measured, 4 trials per gap:
#        0 ms -> 0/4   20 ms -> 0/4   50 ms -> 4/4   100 ms -> 4/4   200 ms -> 4/4
#
# So the write path is: output off, settle, parameters, settle, output on. Verified 8/8
# across both transitions that matter — arbitrary -> built-in (crest 1.43-1.45 for a
# sine) and built-in -> built-in (crest 1.01-1.02 for a square).
#
# Measured dead ends, recorded so they are not re-tried:
#   * *OPC? between the writes and the enable — a VXI-11 round-trip returns in a few
#     milliseconds, far short of the ~50 ms needed, so it lands long before the writes;
#   * waiting *after* the enable rather than before it (0/6 at every value up to 2 s);
#   * a longer pre-enable gap alone, with the output left on (still 0/4 out of arb mode).
#
# Overridable per instance: tests against a FakeTransport pass 0.0, having no output
# stage to restart.
_OUTPUT_SETTLE_S = 0.25

# ":OUTPut[<n>]:LOAD {<ohms>|INFinity|...}" (2-55): <ohms> is an *Integer*, 1 to 10k.
_LOAD_MIN_OHMS = 1
_LOAD_MAX_OHMS = 10_000
# ":OUTPut[<n>]:LOAD?" returns 9.900000E+37 when the load is set to INFinity (HighZ).
_HIGHZ_SENTINEL = 9e37

# 14-bit DAC: ":DATA:DAC" (2-174) gives the binary range as 0000..3FFF, and the
# decimal <value> range as 0 to 16383.
_ARB_CODE_LEVELS = 16384
_ARB_CODE_MAX = _ARB_CODE_LEVELS - 1

_CAPABILITIES = SigGenCapabilities(
    model_name="Rigol DG1062Z",
    n_channels=2,
    max_frequency_hz=60e6,  # sine ceiling (Table 2-1); per-shape limits enforced below
    min_frequency_hz=_MIN_FREQUENCY_HZ,
    # Amplitude/offset ceilings are datasheet figures (into high-Z; halve for 50 Ω) —
    # the programming guide only says the max "is limited by the Impedance and
    # Frequency/Period settings". Requests inside these still get checked by the
    # instrument, and _check_error surfaces a rejection.
    max_amplitude_vpp=20.0,
    min_amplitude_vpp=_MIN_AMPLITUDE_VPP,
    max_offset_v=10.0,
    waveforms=tuple(_SHAPE_NAME),
    # The guide carries full :BURSt, :SWEep and :MOD (AM/FM/PM/ASK/FSK/PSK/PWM)
    # subsystems for this family. Declarative only — drivers wire each mode in as a
    # consumer needs it — but advertising them keeps the loop from refusing a mode the
    # instrument actually has.
    playback_modes=frozenset(
        {
            PlaybackMode.CONTINUOUS,
            PlaybackMode.BURST,
            PlaybackMode.SWEEP,
            PlaybackMode.MODULATION,
        }
    ),
    arb_addressing=ArbAddressing.VOLATILE,  # one live buffer per channel, no numbered bank
    arb_slots=0,
    # ":DATA:DAC" (2-174) and ":DATA:POINts" (2-176): 8 points to 16k points.
    arb_length=ArbLength(min_points=8, max_points=16384),
    arb_code_levels=_ARB_CODE_LEVELS,
    # ":FUNCtion:ARBitrary:SRATe" (2-101): 1 uSa/s to 60 MSa/s, default 20 MSa/s.
    arb_sample_rate=ArbSampleRate(min_hz=1e-6, max_hz=60e6, default_hz=20e6),
)


def _num(value: float) -> str:
    """Format a real parameter for the wire at full precision.

    ``f"{x:g}"`` — the obvious choice — rounds to 6 significant digits, which turns a
    1 234 567 Hz request into 1 234 570 Hz. That error is then *invisible*: the
    instrument stores the truncated value and echoes it back, so every read-back
    assertion still passes. ``repr`` emits the shortest decimal string that round-trips
    the float exactly, and SCPI is built on IEEE 754 (Ch1, *SCPI Command Overview*).
    """
    return repr(float(value))


class DG1062(ScpiTransportMixin, SignalGenerator):
    """Driver for the Rigol DG1062Z over any :class:`Transport`.

    Connection plumbing (``over_tcp``/``over_visa``/``connect``/``disconnect``/
    ``idn``/``_check_error``) comes from :class:`ScpiTransportMixin`.

    ``output_settle_s`` is the gap either side of the output restart that
    :meth:`configure_channel` performs — see :data:`_OUTPUT_SETTLE_S` for the
    measurements behind it. Set it to ``0.0`` when driving a
    :class:`~hwtools.transport.fake.FakeTransport`, which has no output stage to restart
    and where the waits would only slow the suite down.
    """

    def __init__(
        self, transport: Transport, *, output_settle_s: float = _OUTPUT_SETTLE_S
    ) -> None:
        super().__init__(transport)
        self._output_settle_s = output_settle_s

    @property
    def capabilities(self) -> SigGenCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        super().connect()
        # Drop any stale queued error so the first _check_error reports only our own
        # doing, then make sure :VOLTage speaks the model's units — the front panel may
        # have left a channel in Vrms or dBm (":VOLTage:UNIT", 2-185).
        self._t.write("*CLS")
        for n in (1, 2):
            # Checked before writing, because *any* :SOUR2: command drags the front
            # panel's active channel to CH2. Writing both unconditionally makes the
            # display flip channels on every connect — and several times a second across
            # a test run, which looks alarming and is entirely avoidable. Normally both
            # already read VPP and nothing is sent.
            if self._t.query(f":SOUR{n}:VOLTage:UNIT?").strip().upper() != "VPP":
                self._t.write(f":SOUR{n}:VOLTage:UNIT VPP")

    # -- configuration --------------------------------------------------------

    def configure_channel(self, config: SignalGeneratorConfig) -> None:
        """Apply a whole channel config using the guide's discrete write path.

        The command order is the instrument's own dependency order, and every
        parameter is written explicitly so nothing survives from the previous
        waveform: an arbitrary-mode escape, then the load (amplitude/offset are
        referred to it), then type, frequency, amplitude, offset, duty.

        Start phase is deliberately *not* written — that belongs to
        :meth:`set_phase_deg`, and because these are discrete commands rather than
        ``:APPLy``, configuring a channel leaves the phase relationship intact.
        """
        check_within(config, _CAPABILITIES)
        self._check_shape_limits(config)
        n = int(config.channel)
        w = self._t.write
        shape = config.waveform

        # Drop the output before touching anything. See _OUTPUT_SETTLE_S: with the output
        # live, a channel playing the volatile arbitrary cannot be talked onto a built-in
        # waveform by any command sequence — it keeps emitting the old buffer while
        # reporting the new settings. Restarting the output stage around the writes is
        # what actually reloads it.
        w(f":OUTP{n}:STATe OFF")
        if config.enabled:
            # Only worth waiting for when something is going to be driven afterwards: a
            # channel being left off has no output stage state worth reloading.
            self._settle()
        # Leave sample-rate arbitrary mode, where ":FUNCtion:ARBitrary:MODE" (2-100) says
        # the frequency "cannot be set" — so :FREQuency below would otherwise be ignored
        # and the stale value handed to the type-change clamp. Harmless for a channel
        # already playing a basic waveform.
        w(f":SOUR{n}:FUNCtion:ARBitrary:MODE FREQ")
        # Load before the levels: <amplitude>'s range "is limited by the Impedance"
        # and <offset>'s by the impedance and amplitude (2-73 ff.).
        w(f":OUTP{n}:LOAD {self._load_arg(config.output_load_ohms)}")
        # Type before parameters. The carry-over clamp fires on this write; the
        # explicit writes that follow overwrite whatever it landed on.
        w(f":SOUR{n}:FUNCtion {_SHAPE_NAME[shape]}")
        if shape not in _NO_FREQUENCY_SHAPES:
            w(f":SOUR{n}:FREQuency {_num(config.frequency_hz)}")
        if shape is not WaveShape.DC:  # DC has no peak-to-peak amplitude
            w(f":SOUR{n}:VOLTage {_num(config.amplitude_vpp)}")
        w(f":SOUR{n}:VOLTage:OFFSet {_num(config.offset_v)}")
        self._apply_duty(n, config)
        # enable_output settles before switching on, so the enable cannot overtake the
        # writes above.
        self.enable_output(config.channel, config.enabled)
        self._check_error()

    def _settle(self) -> None:
        """Pause long enough for the instrument to act on what it has been sent."""
        if self._output_settle_s > 0.0:
            time.sleep(self._output_settle_s)

    def _apply_duty(self, n: int, config: SignalGeneratorConfig) -> None:
        """Route ``duty_pct`` to the per-shape command that carries it.

        Validated up-front by :meth:`_check_shape_limits`, so anything reaching here
        is a shape that has such a control.
        """
        duty = _num(config.duty_pct)
        if config.waveform is WaveShape.SQUARE:
            self._t.write(f":SOUR{n}:FUNCtion:SQUare:DCYCle {duty}")
        elif config.waveform is WaveShape.PULSE:
            self._t.write(f":SOUR{n}:FUNCtion:PULSe:DCYCle {duty}")
        elif config.waveform is WaveShape.TRIANGLE:
            self._t.write(f":SOUR{n}:FUNCtion:RAMP:SYMMetry {duty}")

    def enable_output(self, channel: SigGenChannel, on: bool) -> None:
        """Turn one channel's output on or off.

        Switching *on* settles first, so the enable cannot overtake whatever parameter
        writes preceded it — see :data:`_OUTPUT_SETTLE_S`. Keeping that here rather than
        in :meth:`configure_channel` means a caller who writes their own parameters and
        then calls this gets the same protection; switching off has no such hazard.
        """
        if on:
            self._settle()
        # ":OUTPut[<n>][:STATe]" (2-57) is per-channel and independent on the DG1000Z
        # (no shared register), so a plain write — no read-modify-write needed.
        self._t.write(f":OUTP{int(channel)}:STATe {'ON' if on else 'OFF'}")

    def set_phase_deg(self, degrees: float) -> None:
        """Set CH2's start phase relative to CH1, then align the two channels.

        ``:PHASe:SYNChronize`` (2-154) "re-configures the two channels to make them
        output according to the specified frequency and phase". Note the guide's
        caveat: the align operation "is invalid" while either channel is in modulation
        mode, which this driver does not currently enable.
        """
        if not 0.0 <= degrees <= 360.0:
            raise ValueError("phase must be within 0..360 degrees")
        self._t.write(f":SOUR1:PHASe {_num(0.0)}")
        self._t.write(f":SOUR2:PHASe {_num(degrees)}")
        self._t.write(":SOUR1:PHASe:SYNChronize")

    # -- arbitrary waveforms --------------------------------------------------

    def upload_arbitrary(
        self,
        channel: SigGenChannel,
        wave: ArbitraryWaveform,
        *,
        sample_rate_hz: float | None = None,
        slot: int | None = None,
    ) -> None:
        """Download ``wave`` to ``channel``'s volatile buffer and play it.

        Clocked in sample-rate mode, so the output repeats at
        :func:`~hwtools.model.siggen.arb_repetition_hz` — ``sample_rate_hz`` divided by
        the point count.
        """
        if slot is not None:
            raise ValueError(
                "the DG1062Z addresses arbitrary waveforms per channel (volatile); pass slot=None"
            )
        check_arbitrary_length(wave.n, _CAPABILITIES)
        rate = resolve_arb_sample_rate(sample_rate_hz, _CAPABILITIES)
        if rate is None:  # unreachable while _CAPABILITIES declares a settable rate
            raise RuntimeError(
                "the DG1062Z capabilities no longer advertise an arbitrary sample rate; "
                "sample-rate playback cannot be configured"
            )
        n = int(channel)
        w = self._t.write
        # Sample-rate mode *before* the download, because the mode decides whether the
        # point count survives: ":DATA:DAC" (2-174) extends a buffer of 8..8k points to
        # 8192 points "using the average interpolation mode" if the channel is in
        # frequency output mode, and leaves it "unchanged" in sample rate output mode.
        # Interpolation would break a faithful read-back round-trip.
        w(f":SOUR{n}:FUNCtion:ARBitrary:MODE SRATe")
        # State the clock rather than inheriting it. Left unset, playback runs at
        # whatever the front panel or a previous run happened to leave, so the output
        # frequency would be unpredictable.
        w(f":SOUR{n}:FUNCtion:ARBitrary:SRATe {_num(rate)}")
        # Binary :DATA:DAC to VOLATILE — the read-back-able buffer served by
        # :DATA:LOAD?. Not :DATA:DAC16, which targets the write-only DDRII memory
        # (2-173) and so cannot round-trip; not the ASCII :DATA[:DATA] float list,
        # which is far longer on the wire for a deep buffer. The download "switches the
        # specified channel to output volatile waveform automatically" (2-174), so no
        # separate select is needed.
        #
        # Deliberately *not* :APPLy:ARBitrary, even though Ch3 uses it: its optional
        # <amplitude>/<offset> would be omitted here and therefore reset to their 5 Vpp
        # / 0 VDC defaults (Ch1, Square Brackets), discarding the level a preceding
        # configure_channel established.
        codes = np.rint((wave.samples + 1.0) / 2.0 * _ARB_CODE_MAX)
        self._t.write_block(f":SOUR{n}:DATA:DAC VOLATILE,", codes.astype("<u2").tobytes())
        self._check_error()

    def read_arbitrary(
        self, channel: SigGenChannel, *, slot: int | None = None
    ) -> ArbitraryWaveform:
        if slot is not None:
            raise ValueError("the DG1062Z has no numbered arbitrary slots; pass slot=None")
        n = int(channel)
        # ":DATA:LOAD?" (2-176) is a two-step read: VOLATILE gives the number of data
        # packages, then each 1..count is fetched as its own definite-length block
        # ("#9000016384 denotes that 16K data is transmitted"). Parsed via float so a
        # scientific-notation count — the form nearly every other numeric query on this
        # instrument uses — does not blow up.
        count = int(float(self._t.query(f":SOUR{n}:DATA:LOAD? VOLATILE")))
        raw = bytearray()
        for k in range(1, count + 1):
            raw += self._t.query_block(f":SOUR{n}:DATA:LOAD? {k}")
        # The packages carry the same 14-bit codes the upload wrote, two bytes per
        # point. An odd total means that pairing is wrong, so fail loudly rather than
        # decode garbage.
        if len(raw) % 2:
            raise OSError(
                f"arbitrary read-back for CH{n} returned {len(raw)} bytes (odd); "
                f"expected 16-bit codes — the LOAD? encoding differs from the assumed uint16"
            )
        codes = np.frombuffer(bytes(raw), dtype="<u2").astype(np.float64)
        return ArbitraryWaveform(samples=codes / _ARB_CODE_MAX * 2.0 - 1.0)

    # -- readback -------------------------------------------------------------

    def read_channel(self, channel: SigGenChannel) -> SignalGeneratorConfig:
        """Read a channel's setup back with per-parameter queries.

        Not ``:APPLy?``, whose reply collapses every arbitrary waveform except DC to
        the single name ``USER`` (2-72) and so cannot tell SINC from LORENTZ.
        """
        n = int(channel)
        q = self._t.query
        name = q(f":SOUR{n}:FUNCtion?").strip().upper()
        if name == _VOLATILE_ARB_NAME:
            raise RuntimeError(
                f"CH{n} is playing its volatile arbitrary buffer, which is an output mode "
                f"rather than a WaveShape; read it with read_arbitrary(CH{n})"
            )
        if name not in _NAME_SHAPE:
            raise RuntimeError(
                f"CH{n} is playing {name!r}, a built-in waveform this driver does not map "
                f"to a WaveShape"
            )
        waveform = _NAME_SHAPE[name]
        # DC and noise have no frequency, and the model says so with 0.0. Reporting
        # whatever the meaningless :FREQuency register happens to hold — or a
        # plausible-looking substitute — would misrepresent the instrument.
        frequency_hz = (
            0.0
            if waveform in _NO_FREQUENCY_SHAPES
            else float(q(f":SOUR{n}:FREQuency?"))
        )
        return SignalGeneratorConfig(
            channel=channel,
            waveform=waveform,
            frequency_hz=frequency_hz,
            amplitude_vpp=float(q(f":SOUR{n}:VOLTage?")),
            offset_v=float(q(f":SOUR{n}:VOLTage:OFFSet?")),
            duty_pct=self._read_duty(n, waveform),
            enabled=self.output_enabled(channel),
            output_load_ohms=self._read_load(n),
        )

    def output_enabled(self, channel: SigGenChannel) -> bool:
        # ":OUTPut[<n>][:STATe]?" returns ON or OFF (2-57).
        return self._t.query(f":OUTP{int(channel)}:STATe?").strip().upper() in ("ON", "1")

    def _read_duty(self, n: int, waveform: WaveShape) -> float:
        if waveform is WaveShape.SQUARE:
            return float(self._t.query(f":SOUR{n}:FUNCtion:SQUare:DCYCle?"))
        if waveform is WaveShape.PULSE:
            return float(self._t.query(f":SOUR{n}:FUNCtion:PULSe:DCYCle?"))
        if waveform is WaveShape.TRIANGLE:
            # Really the ramp symmetry — queried, not assumed, so a front-panel change
            # shows up instead of being papered over with the model default.
            return float(self._t.query(f":SOUR{n}:FUNCtion:RAMP:SYMMetry?"))
        return _DEFAULT_DUTY_PCT  # not a duty-bearing shape; the model default

    def _read_load(self, n: int) -> float:
        ohms = float(self._t.query(f":OUTP{n}:LOAD?"))
        return math.inf if ohms >= _HIGHZ_SENTINEL else ohms

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _load_arg(ohms: float) -> str:
        if math.isinf(ohms):
            return "INFinity"
        if not _LOAD_MIN_OHMS <= ohms <= _LOAD_MAX_OHMS:
            raise ValueError(
                f"output load must be {_LOAD_MIN_OHMS:g}..{_LOAD_MAX_OHMS:g} ohms or inf "
                f"(high-Z); got {ohms:g}"
            )
        if not float(ohms).is_integer():
            # Ch1, Parameter Type 2: an Integer parameter must not be given a decimal,
            # "otherwise, errors will occur".
            raise ValueError(
                f"the DG1062Z takes an integer output load in ohms; got {ohms:g}"
            )
        return str(int(ohms))

    @staticmethod
    def _check_shape_limits(config: SignalGeneratorConfig) -> None:
        """Enforce the per-shape limits Table 2-1 and the duty commands document.

        ``check_within`` covers the model-wide caps; these are the narrower bounds that
        depend on which waveform was asked for, and they are checked before any I/O so
        an impossible request costs no instrument round-trip.
        """
        shape = config.waveform
        limit = _SHAPE_MAX_HZ.get(shape)
        if limit is not None and config.frequency_hz > limit:
            raise ValueError(
                f"{shape} on the DG1062Z is limited to {limit:g} Hz; "
                f"got {config.frequency_hz:g} Hz"
            )
        duty = config.duty_pct
        if shape not in _DUTY_SHAPES:
            if duty != _DEFAULT_DUTY_PCT:
                raise ValueError(
                    f"{shape} on the DG1062Z has no duty-cycle or symmetry control; "
                    f"got duty_pct={duty:g} (leave it at {_DEFAULT_DUTY_PCT:g})"
                )
        elif shape is WaveShape.PULSE and not _PULSE_DUTY_MIN_PCT <= duty <= _PULSE_DUTY_MAX_PCT:
            raise ValueError(
                f"pulse duty cycle must be {_PULSE_DUTY_MIN_PCT:g}..{_PULSE_DUTY_MAX_PCT:g} %; "
                f"got {duty:g} % (the attainable range narrows further with frequency)"
            )
        elif shape is WaveShape.TRIANGLE and not _SYMMETRY_MIN_PCT <= duty <= _SYMMETRY_MAX_PCT:
            raise ValueError(
                f"ramp symmetry must be {_SYMMETRY_MIN_PCT:g}..{_SYMMETRY_MAX_PCT:g} %; "
                f"got {duty:g} %"
            )
