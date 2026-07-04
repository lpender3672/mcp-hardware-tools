"""Behavioural contract every Oscilloscope implementation must satisfy.

Parametrised over the simulated scope (always runs) and the real DS1054Z (only
under ``-m hardware``), so the same expectations hold the sim and the driver to
the same behaviour. Assertions are structural — no signal-specific values — so
they pass against whatever is on the real scope's CH1.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.drivers.simulated import SimulatedScope, Sine
from hwtools.interfaces.oscilloscope import Oscilloscope
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, Slope, SweepMode, TriggerStatus
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig


@pytest.fixture(
    params=[
        pytest.param("sim", id="simulated"),
        pytest.param("real", id="ds1054z", marks=pytest.mark.hardware),
    ]
)
def scope(request: pytest.FixtureRequest) -> Iterator[Oscilloscope]:
    instrument: Oscilloscope
    if request.param == "sim":
        instrument = SimulatedScope({ChannelId.CH1: Sine(amplitude_v=1.0, frequency_hz=1_000.0)})
    else:
        host = os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")
        instrument = DS1054Z.over_tcp(host)
    instrument.connect()
    try:
        yield instrument
    finally:
        instrument.disconnect()


def test_capabilities_are_sane(scope: Oscilloscope) -> None:
    caps = scope.capabilities
    assert caps.n_channels >= 1
    assert caps.max_sample_rate_hz > 0
    assert caps.analog_bandwidth_hz > 0
    assert ChannelId.CH1 in caps.channels


def test_idn_is_nonempty(scope: Oscilloscope) -> None:
    assert scope.idn().strip()


def test_configure_then_capture_returns_valid_waveform(scope: Oscilloscope) -> None:
    scope.configure_channel(
        ChannelConfig(
            channel=ChannelId.CH1,
            coupling=Coupling.DC,
            scale_v_per_div=1.0,
            probe_ratio=10.0,
        )
    )
    scope.configure_timebase(TimebaseConfig(scale_s_per_div=1e-3))
    scope.configure_trigger(
        TriggerConfig(
            trigger=EdgeTrigger(source=ChannelId.CH1, level_v=0.0, slope=Slope.RISING),
            sweep=SweepMode.AUTO,
        )
    )
    scope.configure_acquire(AcquireConfig())
    scope.run()

    cap = scope.capture([ChannelId.CH1])

    assert cap.channels == [ChannelId.CH1]
    assert cap.sample_rate_hz > 0
    assert isinstance(cap.trigger_status, TriggerStatus)
    wf = cap.waveforms[ChannelId.CH1]
    assert wf.n > 0
    assert wf.dt_s > 0
    assert np.all(np.isfinite(wf.samples))


def test_capture_after_stop_reports_status(scope: Oscilloscope) -> None:
    scope.configure_channel(ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=1.0))
    scope.stop()
    assert isinstance(scope.trigger_status(), TriggerStatus)
