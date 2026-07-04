"""HIL: every DS1054Z driver pathway exercised directly on the scope.

The unit suite covers the pure analysis; this module makes sure each *driver*
branch — the SCPI-touching code that can only be trusted on metal — has at least
one test that runs it against the real instrument: every trigger slope, trigger
coupling, acquisition type, the memory-depth validation (valid and illegal), the
shallow/deep read paths (including the multi-chunk RAW loop and the empty-record
guard), autoscale, force-trigger, and the trigger-status states.

Wiring: CH2 carries the Pico square; CH4 is left idle for the untriggered paths.

    uv run pytest -m hardware -s
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.rp2350 import SerialHarness
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import (
    AcqType,
    ChannelId,
    Coupling,
    Slope,
    SweepMode,
    TriggerCoupling,
    TriggerStatus,
)
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from tests.hardware._acquire import acquire_one_shot

CH2, CH4 = ChannelId.CH2, ChannelId.CH4
_TB = TimebaseConfig(scale_s_per_div=5e-4)
_CAPTURED_OR_AUTO = (TriggerStatus.STOP, TriggerStatus.TRIGGERED, TriggerStatus.AUTO)


@pytest.fixture
def square(harness: SerialHarness) -> Iterator[SerialHarness]:
    """The Pico square on CH2, running for the duration of a test."""
    harness.start_square(1_000, duty_pct=50)
    time.sleep(0.3)
    try:
        yield harness
    finally:
        harness.stop()


def _only_ch2(scope: DS1054Z) -> None:
    for ch in (ChannelId.CH1, ChannelId.CH3, CH4):
        scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    scope.configure_channel(
        ChannelConfig(channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0)
    )
    scope.configure_timebase(_TB)


def _edge(source: ChannelId, level: float, slope: Slope = Slope.RISING) -> TriggerConfig:
    return TriggerConfig(
        trigger=EdgeTrigger(source=source, level_v=level, slope=slope), sweep=SweepMode.SINGLE
    )


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_idn_and_capabilities(live_scope: DS1054Z) -> None:
    assert "RIGOL" in live_scope.idn()
    caps = live_scope.capabilities
    assert caps.n_channels == 4 and caps.vertical_divisions == 8


@pytest.mark.hardware
@pytest.mark.cfg_digital
@pytest.mark.parametrize("slope", [Slope.RISING, Slope.FALLING, Slope.EITHER])
def test_trigger_slopes_all_arm_and_fire(
    square: SerialHarness, live_scope: DS1054Z, slope: Slope
) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))
    live_scope.configure_trigger(_edge(CH2, 1.5, slope))
    cap = acquire_one_shot(live_scope, [CH2], _TB)  # raises if it never triggered
    assert cap.waveforms[CH2].vpp > 3.0


@pytest.mark.hardware
@pytest.mark.cfg_digital
@pytest.mark.parametrize(
    "coupling",
    [TriggerCoupling.DC, TriggerCoupling.AC, TriggerCoupling.LF_REJECT, TriggerCoupling.HF_REJECT],
)
def test_trigger_couplings_accepted(
    square: SerialHarness, live_scope: DS1054Z, coupling: TriggerCoupling
) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))
    live_scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=CH2, level_v=1.5), sweep=SweepMode.SINGLE, coupling=coupling
        )
    )
    cap = acquire_one_shot(live_scope, [CH2], _TB)
    assert cap.waveforms[CH2].vpp > 3.0


@pytest.mark.hardware
@pytest.mark.cfg_digital
@pytest.mark.parametrize("acq", [AcqType.NORMAL, AcqType.AVERAGE, AcqType.PEAK, AcqType.HIGH_RES])
def test_acquisition_types_configure_and_capture(
    square: SerialHarness, live_scope: DS1054Z, acq: AcqType
) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(type=acq, averages=4))
    live_scope.configure_trigger(_edge(CH2, 1.5))
    cap = acquire_one_shot(live_scope, [CH2], _TB)
    assert cap.waveforms[CH2].vpp > 3.0


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_memory_depth_valid_then_illegal_raises(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=120_000))  # legal for 1 channel
    # An illegal record length for the active channel count must raise, not beep the
    # scope and desync the socket.
    with pytest.raises(ValueError, match="not a legal record length"):
        live_scope.configure_acquire(AcquireConfig(memory_depth=5_000))


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_multi_chunk_deep_read(square: SerialHarness, live_scope: DS1054Z) -> None:
    """A deep read past the 250k RAW chunk exercises the paging loop + length check."""
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=1_200_000))  # > 4 chunks
    live_scope.configure_trigger(_edge(CH2, 1.5))
    cap = acquire_one_shot(live_scope, [CH2], _TB)
    assert cap.waveforms[CH2].n > 250_000  # more than one chunk was paged in


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_shallow_read_is_the_screen_trace(square: SerialHarness, live_scope: DS1054Z) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig())
    live_scope.run()
    time.sleep(_TB.scale_s_per_div * 12 + 0.05)
    shallow = live_scope.capture([CH2], deep=False).waveforms[CH2]
    assert 1000 <= shallow.n <= 1500  # the ~1200-point decimated screen trace
    assert shallow.vpp > 3.0


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_shallow_read_after_multichunk_deep_read_returns_full_screen(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    """Regression: a NORMal read after a multi-chunk RAW read must still return the
    full 1200-point screen trace.

    A RAW read pages the memory by setting :WAV:STARt/:STOP; the last batch leaves
    STARt well past 1200. Per the manual the NORMal STARt range is 1..1200, and a
    stale STARt > 1200 makes :WAV:DATA? return a single point. The screen read must
    pin STARt back to 1. (This is the driver-layer test that pins the bug that
    otherwise surfaced as a flat frame in the convergence loop.)
    """
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=1_200_000))  # forces multi-chunk
    live_scope.configure_trigger(_edge(CH2, 1.5))
    deep = acquire_one_shot(live_scope, [CH2], _TB).waveforms[CH2]
    assert deep.n > 250_000  # more than one RAW chunk was paged (STARt left high)

    shallow = live_scope.capture([CH2], deep=False).waveforms[CH2]
    assert 1000 <= shallow.n <= 1500  # full screen, not a single stale point
    assert shallow.vpp > 3.0


# NOTE: the empty-record RAW guard (RuntimeError "no deep acquisition") is not
# reachable on this DS1104Z — a deep read after an AUTO free-run still returns the
# live memory (points > 0), so there is no on-scope state that yields points == 0.
# The guard is covered deterministically instead by
# tests/unit/test_ds1054z.py::test_deep_capture_throws_when_no_deep_record.


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_timebase_snapped_to_grid_matches_the_scope(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    """A non-1-2-5 timebase is snapped by the driver so the scope never clamps it
    ("Parameter limited!"). Verify via the captured dt (from the preamble) that the
    scope really is at the snapped 500 us/div, not a value it picked by clamping."""
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig())
    live_scope.configure_timebase(TimebaseConfig(scale_s_per_div=3.33e-4))  # driver snaps -> 500 us
    live_scope.run()
    time.sleep(500e-6 * 12 + 0.05)
    live_scope.stop()
    wf = live_scope.capture([CH2], deep=False).waveforms[CH2]
    # Shallow dt = timebase/100; 500 us -> 5 us (333 us would be 3.33 us).
    assert wf.dt_s == pytest.approx(5e-6, rel=0.1)


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_non_grid_scale_and_offset_capture_cleanly(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    """Non-1-2-5 scale (0.8 V/div, from the loop's x2 growth) + an offset past the
    ±20 V limit must be sent as scope-valid values (VERNier / clamp) so acquisition
    succeeds without a "Parameter limited!" clamp."""
    for ch in (ChannelId.CH1, ChannelId.CH3, CH4):
        live_scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    live_scope.configure_channel(
        ChannelConfig(channel=CH2, coupling=Coupling.DC, scale_v_per_div=0.8, probe_ratio=10.0)
    )
    live_scope.configure_timebase(_TB)
    live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))
    live_scope.configure_trigger(_edge(CH2, 1.5))
    cap = acquire_one_shot(live_scope, [CH2], _TB)
    assert cap.waveforms[CH2].vpp > 3.0  # captured a real frame at the continuous scale


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_bandwidth_limit_and_invert_are_accepted(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    """Exercise the :CHANnel:BWLimit 20M and :CHANnel:INVert ON branches on metal:
    both must be accepted by configure_channel and a real frame still captured."""
    for ch in (ChannelId.CH1, ChannelId.CH3, CH4):
        live_scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    live_scope.configure_channel(
        ChannelConfig(
            channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=10.0,
            bandwidth_limit=True, invert=True,
        )
    )
    live_scope.configure_timebase(_TB)
    live_scope.configure_acquire(AcquireConfig())
    live_scope.run()
    time.sleep(_TB.scale_s_per_div * 12 + 0.05)
    live_scope.stop()
    wf = live_scope.capture([CH2], deep=False).waveforms[CH2]
    assert wf.vpp > 3.0  # a real frame with the 20 MHz BW-limit and invert engaged


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_autoscale_finds_the_signal(square: SerialHarness, live_scope: DS1054Z) -> None:
    _only_ch2(live_scope)
    live_scope.autoscale()
    time.sleep(1.0)  # the instrument autoset takes a moment
    live_scope.stop()
    wf = live_scope.capture([CH2], deep=False).waveforms[CH2]
    assert wf.vpp > 1.0  # autoscale brought the signal on screen


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_force_trigger_command_is_accepted(square: SerialHarness, live_scope: DS1054Z) -> None:
    """Exercise :TFORce on metal.

    force_trigger() is not in the acquisition path — on this DS1104Z :TFORce does
    not reliably drive a NORMAL/SINGLE acquisition to a captured state, which is
    exactly why the loop never relies on it. The pathway that matters here is that
    the command reaches the scope cleanly and leaves the socket healthy: after it,
    the driver can still read status and capture a frame.
    """
    _only_ch2(live_scope)
    live_scope.configure_trigger(_edge(CH2, 1.5))
    live_scope.single()
    time.sleep(0.1)
    live_scope.force_trigger()  # accepted without desyncing the raw socket
    time.sleep(0.1)
    # The socket is still healthy: status and a full *IDN? round-trip both answer.
    assert live_scope.trigger_status() in TriggerStatus
    assert "RIGOL" in live_scope.idn()


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_trigger_status_reports_armed_then_captured(
    square: SerialHarness, live_scope: DS1054Z
) -> None:
    _only_ch2(live_scope)
    live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))
    live_scope.configure_trigger(_edge(CH2, 1.5))
    live_scope.single()
    # It should reach a captured state on the running square within a short budget.
    deadline = time.monotonic() + 2.0
    captured = False
    while time.monotonic() < deadline:
        if live_scope.trigger_status() in (TriggerStatus.STOP, TriggerStatus.TRIGGERED):
            captured = True
            break
        time.sleep(0.02)
    assert captured


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_configured_values_are_saved_verbatim(square: SerialHarness, live_scope: DS1054Z) -> None:
    """The scope must store exactly the (snapped/clamped) values the driver sends —
    a write-then-read-back guard against the silent "Parameter limited!" divergence.

    Feeds off-grid / out-of-range inputs (1.25 V/div, 333 us, 9 V), then reads each
    setting back once (spaced, scope stopped) and asserts the scope holds the
    driver's *computed* value, not the raw input the scope would have clamped.
    """
    _only_ch2(live_scope)
    live_scope.configure_channel(
        ChannelConfig(channel=CH2, coupling=Coupling.DC, scale_v_per_div=1.25, probe_ratio=10.0)
    )
    live_scope.configure_timebase(TimebaseConfig(scale_s_per_div=3.33e-4))
    live_scope.configure_acquire(AcquireConfig(memory_depth=12_000))
    live_scope.configure_trigger(_edge(CH2, 9.0))  # 9 V -> clamped to 4.9 div at 1 V/div
    live_scope.stop()
    time.sleep(0.1)

    def readback(query: str) -> float:
        time.sleep(0.1)  # let the setting settle; one gentle query at a time
        return float(live_scope._t.query(query))

    assert readback(":CHANnel2:SCALe?") == pytest.approx(1.0, rel=0.01)  # snapped, not 1.25
    assert readback(":CHANnel2:OFFSet?") == pytest.approx(0.0, abs=0.02)
    assert readback(":TIMebase:MAIN:SCALe?") == pytest.approx(5e-4, rel=0.01)  # snapped, not 333us
    assert readback(":TRIGger:EDGe:LEVel?") == pytest.approx(4.9, abs=0.1)  # clamped, not 9


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_connect_and_capture_over_visa(scope_host: str) -> None:
    """The alternate VXI-11 transport (over_visa / VisaTransport) works on metal:
    every other HIL test uses the raw-TCP socket, so this is the only on-scope
    coverage of the VISA path. Skips if the instrument is not reachable over VXI-11.
    """
    scope = DS1054Z.over_visa(f"TCPIP::{scope_host}::INSTR")
    try:
        scope.connect()
    except Exception as exc:
        pytest.skip(f"VXI-11 not reachable at {scope_host}: {exc}")
    try:
        assert "RIGOL" in scope.idn()
        _only_ch2(scope)
        scope.configure_acquire(AcquireConfig())
        scope.run()
        time.sleep(_TB.scale_s_per_div * 12 + 0.05)
        scope.stop()
        wf = scope.capture([CH2], deep=False).waveforms[CH2]
        assert wf.n > 1  # a real frame came back over VISA
    finally:
        scope.disconnect()
