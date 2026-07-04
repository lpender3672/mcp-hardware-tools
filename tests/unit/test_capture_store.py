"""The capture store: handles, provenance, and the anti-lose-track / anti-buildup
management guarantees (never-reuse ids, loud eviction, byte budget, ephemeral vs
kept tiers)."""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.model.capture import Capture
from hwtools.model.ids import ChannelId, SweepMode, TriggerStatus
from hwtools.model.waveform import Waveform
from hwtools.session.store import CaptureStore

CH1, CH2 = ChannelId.CH1, ChannelId.CH2


def _capture(
    n: int = 8, channels: tuple[ChannelId, ...] = (CH1,), *, triggered: bool = True
) -> Capture:
    wfs = {
        ch: Waveform(channel=ch, samples=np.zeros(n, dtype=np.float64), t0_s=0.0, dt_s=1e-6)
        for ch in channels
    }
    return Capture(
        waveforms=wfs,
        trigger_status=TriggerStatus.TRIGGERED if triggered else TriggerStatus.WAIT,
        sample_rate_hz=1e6,
    )


def test_ids_are_monotonic_and_never_reused() -> None:
    store = CaptureStore()
    a = store.put(_capture())
    b = store.put(_capture())
    assert (a, b) == ("cap-1", "cap-2")
    # Even after everything is dropped, the counter never rewinds.
    store.clear()
    c = store.put(_capture())
    assert c == "cap-3"


def test_latest_and_label_resolution() -> None:
    store = CaptureStore()
    store.put(_capture(), keep=True, label="first")
    second = store.put(_capture(), keep=True, label="second")
    assert store.get("latest") is store.get(second)
    assert store.get("first") is not None
    assert store.info("second").label == "second"


def test_evicted_or_unknown_id_raises_naming_live_captures() -> None:
    store = CaptureStore()
    kept = store.put(_capture(), keep=True)
    store.put(_capture())  # ephemeral; superseded next
    store.put(_capture())  # supersedes the previous ephemeral
    with pytest.raises(KeyError) as exc:
        store.get("cap-2")  # the first ephemeral, now evicted
    assert "cap-2" in str(exc.value)
    assert kept in str(exc.value)  # names what is still live


def test_fresh_put_drops_previous_ephemeral_but_keeps_kept() -> None:
    store = CaptureStore()
    kept = store.put(_capture(), keep=True)
    store.put(_capture())  # ephemeral cap-2
    store.put(_capture())  # ephemeral cap-3, drops cap-2
    live = {i.capture_id for i in store.list()}
    assert live == {kept, "cap-3"}


def test_keep_promotes_and_survives() -> None:
    store = CaptureStore()
    eph = store.put(_capture())
    store.keep(eph, label="precious")
    store.put(_capture())  # a fresh ephemeral must NOT evict the now-kept frame
    assert store.get(eph) is not None
    assert store.info(eph).kept is True
    assert store.info(eph).label == "precious"


def test_drop_and_clear() -> None:
    store = CaptureStore()
    a = store.put(_capture(), keep=True)
    store.put(_capture(), keep=True)
    store.drop(a)
    assert len(store) == 1
    store.clear()
    assert len(store) == 0


def test_clear_ephemeral_leaves_kept() -> None:
    store = CaptureStore()
    kept = store.put(_capture(), keep=True)
    store.put(_capture())  # ephemeral
    store.clear_ephemeral()
    assert {i.capture_id for i in store.list()} == {kept}


def test_single_frame_over_budget_raises() -> None:
    store = CaptureStore(budget_bytes=64)  # 8 float64 samples = 64 bytes
    with pytest.raises(ValueError, match=r"over the .* budget"):
        store.put(_capture(n=100))  # 800 bytes


def test_kept_frames_filling_budget_refuse_new_put_loudly() -> None:
    # Budget holds two 64-byte frames. Two kept frames fill it; a third must fail
    # loudly rather than silently evicting a kept frame.
    store = CaptureStore(budget_bytes=128)
    store.put(_capture(n=8), keep=True)
    store.put(_capture(n=8), keep=True)
    with pytest.raises(MemoryError, match="exceed"):
        store.put(_capture(n=8), keep=True)


def test_provenance_records_shape_and_kept_reason() -> None:
    store = CaptureStore()
    cid = store.put(
        _capture(n=8, channels=(CH1, CH2)),
        keep=True,
        keep_reason="one-shot",
        sweep=SweepMode.SINGLE,
    )
    info = store.info(cid)
    assert info.channels == (CH1, CH2)
    assert info.n_samples == 8
    assert info.bytes == 8 * 8 * 2  # 8 samples * 8 bytes * 2 channels
    assert info.kept and info.keep_reason == "one-shot"
    assert info.sweep is SweepMode.SINGLE
    assert store.resident_bytes() == info.bytes


def test_get_on_empty_store_raises() -> None:
    with pytest.raises(KeyError):
        CaptureStore().get("latest")
