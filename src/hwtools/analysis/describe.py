"""The dense characterisation lens — the shared substrate for triage, interactive
debug, noise-robustness measurement, and (later) system-ID.

Two tiers over one stored frame (see ``docs/signal-infrastructure-plan.md``):

* an always-on **feature vector** — small, fixed, cheap — that :mod:`classify`
  reads to triage an unknown signal and the agent reads for a quick verdict
  (n_levels, edge rate, periodicity, duty, spectral flatness, crest factor, peak);
* opt-in **dense views** the agent pulls when it wants to *see* the signal: value
  histogram (shape / multimodality), per-time-bin stats (non-stationarity), PSD,
  and the joint time-vs-value grid (the single densest view — digital reads as two
  bands, a sine as a filled envelope, noise as a cloud).

Pure functions over a :class:`~hwtools.model.waveform.Waveform`; caller sets which
lenses at what resolution, so the default is cheap and density is opt-in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from hwtools.analysis import spectrum
from hwtools.model.ids import ChannelId
from hwtools.model.waveform import Waveform

DEFAULT_PERCENTILES = (1.0, 25.0, 50.0, 75.0, 99.0)
#: A histogram bin holding at least this fraction of samples is a discrete "level".
_LEVEL_FRACTION = 0.08
#: Hysteresis band (fraction of vpp) for counting edges without noise chatter.
_EDGE_HYSTERESIS_FRAC = 0.25


class ValueHistogram(BaseModel):
    """The amplitude PDF: bin edges and the sample count in each bin."""

    model_config = ConfigDict(frozen=True)

    edges: list[float] = Field(description="Bin edges, volts (length = bins + 1).")
    counts: list[int] = Field(description="Samples per bin (length = bins).")


class TimeBin(BaseModel):
    """Per-window stats over one slice of the record — catches non-stationarity."""

    model_config = ConfigDict(frozen=True)

    t_lo: float
    t_hi: float
    mean: float
    vpp: float
    percentiles: dict[str, float] = Field(default_factory=dict)


class PsdView(BaseModel):
    """A (down-binned) power spectral density."""

    model_config = ConfigDict(frozen=True)

    freqs_hz: list[float]
    power: list[float]


class JointHistogram(BaseModel):
    """The joint time-vs-value grid — the densest single view of a signal."""

    model_config = ConfigDict(frozen=True)

    t_edges: list[float] = Field(description="Time bin edges, seconds (length = t_bins + 1).")
    v_edges: list[float] = Field(description="Value bin edges, volts (length = v_bins + 1).")
    counts: list[list[int]] = Field(description="counts[t][v] sample counts (t_bins x v_bins).")


class Characterization(BaseModel):
    """Dense-lens output: the always-on feature vector plus any opt-in views."""

    model_config = ConfigDict(frozen=True)

    channel: ChannelId
    # -- feature vector (always present) --
    vpp: float
    dc_level: float = Field(description="Mean (DC) level, volts.")
    rms: float
    crest_factor: float = Field(
        description="Peak / AC-rms: ~1 square, ~1.41 sine, >3 noise — a cheap shape discriminator."
    )
    n_levels: int = Field(description="Discrete voltage levels occupied (1 DC, 2 binary digital).")
    peak_hz: float | None = Field(default=None, description="Dominant tone, or None if none.")
    peak_prominence: float = Field(default=0.0, description="Peak height over the spectral median.")
    spectral_flatness: float = Field(description="Wiener entropy: ~0 tonal, ~1 noise-like.")
    periodicity: float = Field(description="Autocorrelation peak strength, 0..1.")
    period_s: float | None = Field(default=None, description="Estimated period, or None if none.")
    edge_rate_hz: float = Field(description="Midline transitions per second (hysteretic).")
    duty: float | None = Field(default=None, description="Fraction of time above midline.")
    # -- opt-in dense views --
    percentiles: dict[str, float] | None = None
    value_histogram: ValueHistogram | None = None
    time_bins: list[TimeBin] | None = None
    psd: PsdView | None = None
    joint: JointHistogram | None = None


def describe(
    wf: Waveform,
    *,
    percentiles: Sequence[float] | None = DEFAULT_PERCENTILES,
    value_bins: int | None = None,
    time_bins: int | None = None,
    psd: bool = False,
    freq_bins: int | None = None,
    joint: bool = False,
    joint_bins: int | None = None,
) -> Characterization:
    """Characterise a waveform: always the feature vector, plus any requested views.

    ``percentiles`` (robust spread) default on and cheap; ``value_bins`` adds the
    amplitude PDF (shape); ``time_bins`` segments the record (non-stationarity);
    ``psd`` adds a Welch spectrum (``freq_bins`` down-bins it); ``joint`` adds the
    time-vs-value grid (``joint_bins`` sets both axes' resolution).
    """
    samples = wf.samples
    vmin, vmax, vpp = wf.vmin, wf.vmax, wf.vpp
    mean = float(np.mean(samples)) if wf.n else 0.0
    ac_rms = float(np.std(samples)) if wf.n else 0.0
    crest = (0.5 * vpp / ac_rms) if ac_rms > 0 else 0.0

    peak_hz, prominence = _dominant_tone(wf)
    flatness = spectrum.spectral_flatness(wf)
    periodicity, period_s = _periodicity(wf)
    edge_rate, duty = _edges_and_duty(wf)

    view_percentiles = _percentiles(samples, percentiles) if percentiles else None

    return Characterization(
        channel=wf.channel,
        vpp=vpp,
        dc_level=mean,
        rms=float(np.sqrt(np.mean(np.square(samples)))) if wf.n else 0.0,
        crest_factor=crest,
        n_levels=_n_levels(samples, vmin, vmax, vpp),
        peak_hz=peak_hz,
        peak_prominence=prominence,
        spectral_flatness=flatness,
        periodicity=periodicity,
        period_s=period_s,
        edge_rate_hz=edge_rate,
        duty=duty,
        percentiles=view_percentiles,
        value_histogram=_value_histogram(samples, vmin, vmax, value_bins) if value_bins else None,
        time_bins=_time_bins(wf, time_bins, percentiles) if time_bins else None,
        psd=_psd_view(wf, freq_bins) if psd else None,
        joint=_joint(wf, vmin, vmax, joint_bins) if joint else None,
    )


# -- feature vector helpers ---------------------------------------------------


def _dominant_tone(wf: Waveform) -> tuple[float | None, float]:
    """(peak frequency, prominence over the spectral median), or (None, 0)."""
    return spectrum.peak_frequency_and_prominence(wf)


def _periodicity(wf: Waveform) -> tuple[float, float | None]:
    """Strength (0..1) and period of the strongest self-repeat, via autocorrelation.

    Uses an FFT autocorrelation (O(n log n)) so it scales to deep frames.
    """
    x = wf.samples - np.mean(wf.samples)
    n = x.size
    if n < 16 or not np.any(x):
        return 0.0, None
    size = int(1 << (2 * n - 1).bit_length())  # next pow2 >= 2n-1, for linear autocorr
    f = np.fft.rfft(x, n=size)
    corr = np.fft.irfft(f * np.conj(f), n=size)[:n]
    if corr[0] <= 0:
        return 0.0, None
    corr = corr / corr[0]
    # Skip the main lobe: search past the first return below zero.
    below = np.flatnonzero(corr < 0.0)
    start = int(below[0]) if below.size else 1
    if start >= n:
        return 0.0, None
    peak_lag = start + int(np.argmax(corr[start:]))
    strength = float(corr[peak_lag])
    if strength <= 0:
        return 0.0, None
    return strength, peak_lag * wf.dt_s


def _edges_and_duty(wf: Waveform) -> tuple[float, float | None]:
    """Hysteretic midline-crossing rate (Hz) and fraction of time above midline.

    A Schmitt band about the midline suppresses noise chatter, so the edge rate
    reflects real transitions. Duty is the fraction of samples in the high state.
    """
    samples = wf.samples
    if wf.n < 2 or wf.vpp <= 0:
        return 0.0, None
    mid = (wf.vmin + wf.vmax) / 2.0
    band = _EDGE_HYSTERESIS_FRAC * wf.vpp / 2.0
    high, low = mid + band, mid - band

    # Vectorised Schmitt trigger: samples above `high` are definitely-high (+1),
    # below `low` definitely-low (-1); in-band samples hold the previous state.
    # Forward-filling the +/-1 markers reconstructs the latched state array, and a
    # state change is a transition — O(n) with no Python loop.
    markers = np.zeros(wf.n, dtype=np.int8)
    markers[samples > high] = 1
    markers[samples < low] = -1
    filled = markers.astype(np.float64)
    filled[markers == 0] = np.nan
    fill_idx = np.where(markers != 0, np.arange(wf.n), 0)
    np.maximum.accumulate(fill_idx, out=fill_idx)
    filled = filled[fill_idx]
    # Samples before the first marker inherit the initial state.
    filled[np.isnan(filled)] = 1.0 if samples[0] >= mid else -1.0

    transitions = int(np.count_nonzero(np.diff(filled) != 0))
    edge_rate = transitions / wf.duration_s if wf.duration_s > 0 else 0.0
    return edge_rate, float(np.count_nonzero(filled > 0)) / wf.n


def _n_levels(samples: npt.NDArray[np.float64], vmin: float, vmax: float, vpp: float) -> int:
    """Rough count of discrete voltage levels: clusters of the value histogram that
    each hold a large sample fraction (1 = DC/flat, 2 = binary digital)."""
    if vpp <= 1e-12:
        return 1
    hist, _ = np.histogram(samples, bins=64, range=(vmin, vmax))
    frac = hist / hist.sum()
    dominant = frac > _LEVEL_FRACTION
    # Count connected runs of dominant bins (adjacent bins = one level).
    levels = int(np.sum(dominant[1:] & ~dominant[:-1])) + int(dominant[0])
    return max(levels, 1)


def _percentiles(samples: npt.NDArray[np.float64], ps: Sequence[float]) -> dict[str, float]:
    if samples.size == 0:
        return {}
    values = np.percentile(samples, list(ps))
    return {_pkey(p): float(v) for p, v in zip(ps, values, strict=True)}


def _pkey(p: float) -> str:
    return f"{p:g}"


# -- dense view helpers -------------------------------------------------------


def _value_histogram(
    samples: npt.NDArray[np.float64], vmin: float, vmax: float, bins: int
) -> ValueHistogram:
    lo, hi = (vmin, vmax) if vmax > vmin else (vmin - 0.5, vmin + 0.5)
    counts, edges = np.histogram(samples, bins=bins, range=(lo, hi))
    return ValueHistogram(edges=[float(e) for e in edges], counts=[int(c) for c in counts])


def _time_bins(wf: Waveform, bins: int, percentiles: Sequence[float] | None) -> list[TimeBin]:
    out: list[TimeBin] = []
    edges = np.linspace(0, wf.n, bins + 1, dtype=int)
    for i in range(bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        seg = wf.samples[lo:hi]
        out.append(
            TimeBin(
                t_lo=wf.t0_s + lo * wf.dt_s,
                t_hi=wf.t0_s + hi * wf.dt_s,
                mean=float(np.mean(seg)),
                vpp=float(seg.max() - seg.min()),
                percentiles=_percentiles(seg, percentiles) if percentiles else {},
            )
        )
    return out


def _psd_view(wf: Waveform, freq_bins: int | None) -> PsdView:
    spec = spectrum.welch_psd(wf, nperseg=(2 * freq_bins if freq_bins else None))
    freqs, power = spec.frequencies_hz, spec.values
    if freq_bins and freqs.size > freq_bins:
        # Average into freq_bins contiguous groups to hit the requested resolution.
        idx = np.linspace(0, freqs.size, freq_bins + 1, dtype=int)
        groups = [(idx[i], idx[i + 1]) for i in range(freq_bins) if idx[i + 1] > idx[i]]
        freqs = np.array([freqs[a:b].mean() for a, b in groups])
        power = np.array([power[a:b].mean() for a, b in groups])
    return PsdView(freqs_hz=[float(f) for f in freqs], power=[float(p) for p in power])


def _joint(wf: Waveform, vmin: float, vmax: float, bins: int | None) -> JointHistogram:
    b = bins or 32
    lo, hi = (vmin, vmax) if vmax > vmin else (vmin - 0.5, vmin + 0.5)
    times = wf.time_axis()
    counts, t_edges, v_edges = np.histogram2d(
        times, wf.samples, bins=[b, b], range=[[times[0], times[-1]], [lo, hi]]
    )
    return JointHistogram(
        t_edges=[float(e) for e in t_edges],
        v_edges=[float(e) for e in v_edges],
        counts=[[int(c) for c in row] for row in counts.astype(int)],
    )
