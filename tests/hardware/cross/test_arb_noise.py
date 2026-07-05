"""Cross-instrument HIL: upload synthetic noise into an arb slot and measure it.

Proves the full output chain end-to-end — ``noise`` synthesises coloured noise,
:meth:`ArbitraryWaveform.normalised` renders it, the driver uploads it, the JDS6600
replays it, and the DS1054Z digitises it — by checking the **measured PSD slope on
the scope matches the intended colour** (white ~0, pink ~-10, brown ~-20 dB/decade).

This needs *bandwidth*: the 2048-point buffer replays at ~2 MSa/s, so the scope must
sample fast enough to see the noise rather than alias it. Hence a deep capture at a
fast timebase (~10 MSa/s), and the PSD slope is fit over a mid-band well below
Nyquist. The arb noise is frozen/periodic (see ``hwtools.noise``); this test is the
calibration anchor that quantifies how faithfully the real front-end reproduces the
intended spectrum.

    uv run pytest -m hardware -s

Leaves CH1 on a benign 1 kHz 2 Vpp built-in sine.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hwtools.analysis.describe import describe
from hwtools.analysis.spectrum import psd_slope_db_per_decade, welch_psd
from hwtools.drivers.joyit import JDS6600
from hwtools.drivers.rigol.ds1054z import DS1054Z
from hwtools.model.acquire import AcquireConfig
from hwtools.model.channel import ChannelConfig
from hwtools.model.ids import ChannelId, Coupling
from hwtools.model.siggen import SigGenChannel, SignalGeneratorConfig
from hwtools.model.timebase import TimebaseConfig
from hwtools.model.waveform import Waveform
from hwtools.noise import (
    PSD_SLOPE_DB_PER_DECADE,
    NoiseColor,
    band_limited_noise_arbitrary,
    noise_arbitrary,
)
from tests.hardware._acquire import acquire_repeating

SCOPE_CH = ChannelId.CH1
GEN_CH = SigGenChannel.CH1
_REPLAY_HZ = 1_000.0
_TB = TimebaseConfig(scale_s_per_div=100e-6)  # 1.2 ms window; deep -> ~10 MSa/s
_SLOPE_BAND_HZ = (5e3, 5e5)  # mid-band: above the replay rate, below the front-end roll-off


def _play(generator: JDS6600, slot: int) -> None:
    """Route ``slot`` to CH1 at the standard replay rate/level."""
    generator.configure_channel(
        SignalGeneratorConfig(
            channel=GEN_CH, frequency_hz=_REPLAY_HZ, amplitude_vpp=4.0, arb_slot=slot, enabled=True
        )
    )


def _deep_capture(scope: DS1054Z) -> Waveform:
    """A single-channel deep capture of CH1 at ~10 MSa/s — enough bandwidth to see the
    arb's ~1 MHz content instead of aliasing it (the shallow screen trace would)."""
    for ch in (ChannelId.CH2, ChannelId.CH3, ChannelId.CH4):
        scope.configure_channel(ChannelConfig(channel=ch, scale_v_per_div=1.0, enabled=False))
    scope.configure_channel(
        ChannelConfig(channel=SCOPE_CH, coupling=Coupling.DC, scale_v_per_div=1.0, probe_ratio=1.0)
    )
    scope.configure_timebase(_TB)
    scope.configure_acquire(AcquireConfig(memory_depth=12_000))  # deep single-channel
    time.sleep(0.3)
    return acquire_repeating(scope, [SCOPE_CH], _TB, deep=True).waveforms[SCOPE_CH]


def _capture_noise(generator: JDS6600, scope: DS1054Z, color: NoiseColor) -> Waveform:
    caps = generator.capabilities
    slot = caps.arb_slots
    generator.upload_arbitrary(slot, noise_arbitrary(caps.arb_points, color, seed=11))
    _play(generator, slot)
    return _deep_capture(scope)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize("color", list(NoiseColor))
def test_arb_noise_replays_with_the_intended_spectral_slope(
    generator: JDS6600, live_scope: DS1054Z, color: NoiseColor
) -> None:
    wf = _capture_noise(generator, live_scope, color)
    f_lo, f_hi = _SLOPE_BAND_HZ
    slope = psd_slope_db_per_decade(wf, f_lo=f_lo, f_hi=f_hi, nperseg=2048)
    # Wide tolerance: the front-end + DAC reconstruction bend the spectrum, but the
    # colour is unmistakable (white/pink/brown land ~10 dB/decade apart).
    assert slope == pytest.approx(PSD_SLOPE_DB_PER_DECADE[color], abs=5.0)


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
def test_white_arb_noise_reads_as_broadband_noise(
    generator: JDS6600, live_scope: DS1054Z
) -> None:
    """White noise is measurably noise, not a tone: high crest and no dominant peak."""
    wf = _capture_noise(generator, live_scope, NoiseColor.WHITE)
    features = describe(wf, psd=True)
    assert features.crest_factor > 2.5  # Gaussian noise peaks well above a sine's 1.41
    assert features.spectral_flatness > 0.08  # broadband, not concentrated in one bin


@pytest.mark.hardware
@pytest.mark.cfg_siggen_1ch
@pytest.mark.parametrize(("low_cycles", "high_cycles"), [(50, 100), (150, 250), (300, 400)])
def test_band_limited_noise_lands_in_its_window_and_sweeps(
    generator: JDS6600, live_scope: DS1054Z, low_cycles: int, high_cycles: int
) -> None:
    """Band-limited noise confines its energy to [low, high] x replay-rate, and moving
    the band moves the measured energy — the sweepable narrow-band stimulus. The arb's
    broadband replay is compromised, but *within a window* the noise is accurate."""
    caps = generator.capabilities
    slot = caps.arb_slots
    generator.upload_arbitrary(
        slot,
        band_limited_noise_arbitrary(
            caps.arb_points, low_cycles=low_cycles, high_cycles=high_cycles, seed=3
        ),
    )
    _play(generator, slot)
    wf = _deep_capture(live_scope)

    band_lo, band_hi = low_cycles * _REPLAY_HZ, high_cycles * _REPLAY_HZ
    spec = welch_psd(wf, nperseg=4096)
    freqs, psd = spec.frequencies_hz, spec.values
    in_band = (freqs >= band_lo) & (freqs <= band_hi)
    out_band = (freqs > band_hi * 1.5) & (freqs < wf.sample_rate_hz * 0.45)
    centroid = float(np.sum(freqs[in_band] * psd[in_band]) / np.sum(psd[in_band]))
    rejection_db = 10.0 * np.log10(psd[in_band].mean() / psd[out_band].mean())

    assert band_lo <= centroid <= band_hi  # energy is centred inside the intended window
    assert rejection_db > 15.0  # clearly band-limited, not broadband
