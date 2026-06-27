"""Capture assembly, trigger-status semantics, quality/adjustment helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hwtools.model import (
    Adjustment,
    Capture,
    CaptureQuality,
    ChannelConfig,
    ChannelId,
    ScopeCapabilities,
    TriggerStatus,
    Waveform,
)


def _wf(channel: ChannelId) -> Waveform:
    return Waveform(channel=channel, samples=[0.0, 1.0], t0_s=0.0, dt_s=1e-6)


def test_capture_rejects_mismatched_channel_key() -> None:
    with pytest.raises(ValidationError):
        Capture(
            waveforms={ChannelId.CH2: _wf(ChannelId.CH1)},
            trigger_status=TriggerStatus.TRIGGERED,
            sample_rate_hz=1e6,
        )


@pytest.mark.parametrize(
    ("status", "triggered"),
    [
        (TriggerStatus.TRIGGERED, True),
        (TriggerStatus.AUTO, True),
        (TriggerStatus.WAIT, False),
        (TriggerStatus.STOP, False),
    ],
)
def test_capture_triggered_property(status: TriggerStatus, triggered: bool) -> None:
    cap = Capture(
        waveforms={ChannelId.CH1: _wf(ChannelId.CH1)},
        trigger_status=status,
        sample_rate_hz=1e6,
    )
    assert cap.triggered is triggered
    assert cap.channels == [ChannelId.CH1]


def test_quality_usable_requires_trigger_and_no_clipping() -> None:
    good = CaptureQuality(triggered=True, clipping={ChannelId.CH1: False})
    assert good.usable is True

    untriggered = CaptureQuality(triggered=False, clipping={ChannelId.CH1: False})
    assert untriggered.usable is False

    clipped = CaptureQuality(triggered=True, clipping={ChannelId.CH1: True})
    assert clipped.usable is False


def test_adjustment_emptiness() -> None:
    assert Adjustment().is_empty() is True
    nudge = Adjustment(
        channels={ChannelId.CH1: ChannelConfig(channel=ChannelId.CH1, scale_v_per_div=1.0)},
        reason="reduce gain to clear clipping",
    )
    assert nudge.is_empty() is False


def test_scope_capabilities_channels_and_membership() -> None:
    caps = ScopeCapabilities(
        model_name="DS1054Z",
        n_channels=4,
        max_sample_rate_hz=1e9,
        analog_bandwidth_hz=50e6,
    )
    assert caps.channels == (ChannelId.CH1, ChannelId.CH2, ChannelId.CH3, ChannelId.CH4)
    assert caps.has_channel(ChannelId.CH4) is True
