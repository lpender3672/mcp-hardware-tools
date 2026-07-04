"""Signal triage: the rule-based classifier routes each shape to its class, and
triage + the cross-channel hint steer toward the right decoder."""

from __future__ import annotations

import numpy as np

from hwtools.analysis.classify import SignalClass, classify, cross_channel_hint, triage
from hwtools.analysis.describe import describe
from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, TriggerStatus
from hwtools.model.waveform import Waveform

CH1, CH2, CH3 = ChannelId.CH1, ChannelId.CH2, ChannelId.CH3
_DT = 1e-6
_N = 20_000


def _wf(samples: np.ndarray, channel: ChannelId = CH1) -> Waveform:
    return Waveform(channel=channel, samples=samples, t0_s=0.0, dt_s=_DT)


def _t() -> np.ndarray:
    return np.arange(_N) * _DT


def _sine(ch: ChannelId = CH1) -> Waveform:
    return _wf(np.sin(2 * np.pi * 1_000.0 * _t()), ch)


def _square(ch: ChannelId = CH1) -> Waveform:
    return _wf(np.where((_t() * 1_000.0) % 1.0 < 0.5, 1.0, -1.0), ch)


def _noise(ch: ChannelId = CH1) -> Waveform:
    return _wf(np.random.default_rng(0).normal(0.0, 1.0, _N), ch)


def _dc(ch: ChannelId = CH1) -> Waveform:
    return _wf(np.full(_N, 2.0), ch)


def test_classifies_sine() -> None:
    c = classify(describe(_sine()))
    assert c.signal_class is SignalClass.SINE
    assert c.confidence > 0.5


def test_classifies_square_as_digital_with_symbol_rate() -> None:
    c = classify(describe(_square()))
    assert c.signal_class is SignalClass.DIGITAL
    assert c.est_symbol_rate_hz is not None and c.est_symbol_rate_hz > 0
    assert "decode" in c.suggested


def test_classifies_noise() -> None:
    c = classify(describe(_noise()))
    assert c.signal_class is SignalClass.NOISE


def test_classifies_dc() -> None:
    c = classify(describe(_dc()))
    assert c.signal_class is SignalClass.FLAT_DC


def test_triage_runs_every_channel() -> None:
    cap = Capture(
        waveforms={CH1: _square(CH1), CH2: _sine(CH2)},
        trigger_status=TriggerStatus.STOP,
        sample_rate_hz=1 / _DT,
    )
    result = triage(cap)
    assert result[CH1].signal_class is SignalClass.DIGITAL
    assert result[CH2].signal_class is SignalClass.SINE


def test_cross_channel_hint_counts_digital_lines() -> None:
    cap = Capture(
        waveforms={CH1: _square(CH1), CH2: _square(CH2), CH3: _square(CH3)},
        trigger_status=TriggerStatus.STOP,
        sample_rate_hz=1 / _DT,
    )
    hint = cross_channel_hint(triage(cap))
    assert hint is not None and "SPI" in hint

    two = Capture(
        waveforms={CH1: _square(CH1), CH2: _square(CH2)},
        trigger_status=TriggerStatus.STOP,
        sample_rate_hz=1 / _DT,
    )
    assert "I2C" in (cross_channel_hint(triage(two)) or "")
