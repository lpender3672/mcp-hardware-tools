"""The generalised arbitrary-waveform model: length constraint, buffer, length check.

Covers the reshape that lets one ``SignalGenerator`` contract span both a fixed-length
DDS wavetable (JDS6600, 2048 points) and a variable-length AWG (Rigol DG1000Z,
8..16384 points per upload) without forking into device-specific waveform types.
"""

from __future__ import annotations

import numpy as np
import pytest

from hwtools.model.siggen import (
    ArbitraryWaveform,
    ArbLength,
    PlaybackMode,
    SigGenCapabilities,
    check_arbitrary_length,
)

# -- ArbLength: fixed vs variable --------------------------------------------


def test_fixed_length_accepts_only_the_exact_count() -> None:
    length = ArbLength.fixed(2048)
    assert length.is_fixed
    assert length.accepts(2048)
    assert not length.accepts(2047)
    assert not length.accepts(4096)
    assert length.representative_length() == 2048
    assert length.describe() == "exactly 2048 points"


def test_variable_length_accepts_within_range_on_grid() -> None:
    length = ArbLength(min_points=8, max_points=16384)
    assert not length.is_fixed
    assert length.accepts(8)
    assert length.accepts(16384)
    assert length.accepts(4096)
    assert not length.accepts(7)  # below min
    assert not length.accepts(16385)  # above max


def test_granularity_requires_a_multiple() -> None:
    length = ArbLength(min_points=16, max_points=1024, granularity=8)
    assert length.accepts(16)
    assert length.accepts(1024)
    assert not length.accepts(20)  # not a multiple of 8
    assert "in steps of 8" in length.describe()


def test_representative_length_clamps_and_snaps_to_grid() -> None:
    length = ArbLength(min_points=16, max_points=1024, granularity=8)
    # default target 4096 is above max -> clamps to max (already on grid)
    assert length.representative_length() == 1024
    # a target between grid points snaps down onto the grid, staying >= min
    assert length.representative_length(target=100) == 96  # 100 -> 96 (multiple of 8)
    assert length.representative_length(target=1) == 16  # clamped up to min


def test_arblength_rejects_inconsistent_bounds() -> None:
    with pytest.raises(ValueError, match="max_points"):
        ArbLength(min_points=100, max_points=8)
    with pytest.raises(ValueError, match="granularity"):
        ArbLength(min_points=10, max_points=1000, granularity=8)  # 10 % 8 != 0


# -- check_arbitrary_length ---------------------------------------------------


def _caps(arb_length: ArbLength | None, *, slots: int = 4) -> SigGenCapabilities:
    return SigGenCapabilities(
        model_name="Test",
        n_channels=1,
        max_frequency_hz=1e6,
        max_amplitude_vpp=10.0,
        max_offset_v=5.0,
        arb_slots=slots if arb_length is not None else 0,
        arb_length=arb_length,
        arb_code_levels=16384 if arb_length is not None else 0,
    )


def test_check_length_passes_within_range() -> None:
    check_arbitrary_length(4096, _caps(ArbLength(min_points=8, max_points=16384)))


def test_check_length_rejects_out_of_range_with_descriptive_error() -> None:
    caps = _caps(ArbLength(min_points=8, max_points=16384))
    with pytest.raises(ValueError, match=r"8\.\.16384 points"):
        check_arbitrary_length(20000, caps)


def test_check_length_rejects_when_no_arb_support() -> None:
    with pytest.raises(ValueError, match="no arbitrary-waveform support"):
        check_arbitrary_length(2048, _caps(None))


def test_capabilities_require_length_when_slots_present() -> None:
    with pytest.raises(ValueError, match="requires an arb_length"):
        SigGenCapabilities(
            model_name="Bad",
            n_channels=1,
            max_frequency_hz=1e6,
            max_amplitude_vpp=10.0,
            max_offset_v=5.0,
            arb_slots=4,
            arb_length=None,
        )


def test_capabilities_default_to_continuous_playback() -> None:
    caps = _caps(None)
    assert caps.playback_modes == frozenset({PlaybackMode.CONTINUOUS})
    assert not caps.supports_arbitrary()


# -- ArbitraryWaveform: numpy-backed, immutable -------------------------------


def test_samples_are_a_readonly_float64_array() -> None:
    wave = ArbitraryWaveform(samples=[-1.0, 0.0, 1.0])
    assert isinstance(wave.samples, np.ndarray)
    assert wave.samples.dtype == np.float64
    assert wave.n == 3
    with pytest.raises(ValueError):  # read-only: contents can't be mutated in place
        wave.samples[0] = 0.5


def test_accepts_a_deep_buffer_cheaply() -> None:
    # A 16 kpt DG-scale buffer constructs and validates without a per-sample Python loop.
    big = np.linspace(-1.0, 1.0, 16384)
    wave = ArbitraryWaveform(samples=big)
    assert wave.n == 16384


def test_rejects_out_of_unit_range_and_bad_shape() -> None:
    with pytest.raises(ValueError, match="normalised"):
        ArbitraryWaveform(samples=[0.0, 1.5])
    with pytest.raises(ValueError, match="1-D"):
        ArbitraryWaveform(samples=[[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="at least one sample"):
        ArbitraryWaveform(samples=[])


def test_equality_and_hash_follow_the_samples() -> None:
    a = ArbitraryWaveform(samples=[-1.0, 0.0, 1.0])
    b = ArbitraryWaveform(samples=[-1.0, 0.0, 1.0])
    c = ArbitraryWaveform(samples=[-1.0, 0.5, 1.0])
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2  # a and b collapse, c is distinct


def test_normalised_scales_peak_to_full_scale() -> None:
    wave = ArbitraryWaveform.normalised([0.0, 0.25, -0.5])
    assert float(np.max(np.abs(wave.samples))) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="all-zero"):
        ArbitraryWaveform.normalised(np.zeros(8))
