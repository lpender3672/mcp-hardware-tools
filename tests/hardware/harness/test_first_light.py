"""Supervised hardware-in-the-loop first light.

Excluded from CI (``hardware`` marker). Run on the bench with a real DS1054Z:

    uv run pytest -m hardware -s

Validates the whole driver path end-to-end against the live instrument:
connect, configure CH1 (10x probe), run, download, and scale to volts.
Override the address with ``HWTOOLS_SCOPE_HOST`` if it differs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling, SweepMode
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.trigger import EdgeTrigger, TriggerConfig
from tests.hardware._acquire import acquire_repeating

HOST = os.environ.get("HWTOOLS_SCOPE_HOST", "192.168.1.214")


@pytest.mark.hardware
@pytest.mark.cfg_digital
def test_first_light_ch1() -> None:
    scope = DS1054Z.over_tcp(HOST)
    with scope:
        idn = scope.idn()
        assert "DS1" in idn, f"unexpected instrument: {idn!r}"

        # One channel only -> 12000 is a legal record length; keep the download small.
        for ch in (ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
            scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
        scope.configure_channel(
            ChannelConfig(
                channel=ChannelId.CH1,
                coupling=Coupling.DC,
                scale_v_per_div=1.0,
                probe_ratio=10.0,
            )
        )
        scope.configure_acquire(AcquireConfig(memory_depth=12_000))
        tb = TimebaseConfig(scale_s_per_div=1e-3)
        scope.configure_timebase(tb)
        scope.configure_trigger(
            TriggerConfig(
                trigger=EdgeTrigger(source=ChannelId.CH1, level_v=1.0),
                sweep=SweepMode.AUTO,
            )
        )
        # CH1 may be idle (nothing to trigger on), so free-run in AUTO and read a
        # fresh screen frame — the scope's auto-trigger always yields one.
        cap = acquire_repeating(scope, [ChannelId.CH1], tb)

    wf = cap.waveforms[ChannelId.CH1]
    print(
        f"\n[first-light] {idn}"
        f"\n  trigger={cap.trigger_status.value}  sample_rate={cap.sample_rate_hz:.3e} Hz"
        f"\n  n={wf.n}  dt={wf.dt_s:.3e}s  span={wf.duration_s:.3e}s"
        f"\n  vmin={wf.vmin:.3f}V  vmax={wf.vmax:.3f}V  vpp={wf.vpp:.3f}V"
    )

    assert wf.n > 0
    assert wf.dt_s > 0
    assert np.all(np.isfinite(wf.samples))
