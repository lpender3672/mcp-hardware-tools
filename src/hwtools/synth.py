"""Synthesise one period of an analytic waveform as an :class:`ArbitraryWaveform`.

The output side's counterpart to :mod:`hwtools.analysis` (which characterises
captured signals): here we *build* a normalised [-1, 1] buffer to upload to a
signal generator's arbitrary slot. Two uses:

* **validation** — replaying an analytic sine/triangle/square as an arb and
  measuring it back proves the upload path is faithful (the arb's spectrum/shape
  should match both the analytic ideal and the device's own built-in of that shape);
* **Thread A** — the same buffer is where signal + synthetic noise get composited
  for the noise-robustness suite (arb-replays-everything).

Shapes are full-scale (span exactly [-1, 1]); the generator's amplitude/offset
scale them at output. Crest factors are shape-invariant, which is what makes them a
cheap first validation gate: sine √2 ≈ 1.414, triangle √3 ≈ 1.732, square 1.0.
"""

from __future__ import annotations

import numpy as np

from hwtools.model.siggen import ArbitraryWaveform, WaveShape

_ANALYTIC = frozenset({WaveShape.SINE, WaveShape.TRIANGLE, WaveShape.SQUARE})


def analytic_arbitrary(
    shape: WaveShape, *, points: int, duty: float = 0.5
) -> ArbitraryWaveform:
    """One period of ``shape`` over ``points`` samples, normalised to [-1, 1].

    ``duty`` applies to :attr:`WaveShape.SQUARE` only. Raises for shapes without an
    analytic form here (upload a hand-built :class:`ArbitraryWaveform` for those).
    """
    if points <= 0:
        raise ValueError("points must be positive")
    if shape not in _ANALYTIC:
        raise ValueError(
            f"no analytic synthesiser for {shape}; supported: "
            f"{', '.join(sorted(s.value for s in _ANALYTIC))}"
        )
    phase = np.arange(points) / points  # one period, [0, 1)
    if shape is WaveShape.SINE:
        y = np.sin(2.0 * np.pi * phase)
    elif shape is WaveShape.TRIANGLE:
        # -1 at phase 0, +1 at the half-period, back to -1 — a symmetric triangle.
        y = np.where(phase < 0.5, -1.0 + 4.0 * phase, 3.0 - 4.0 * phase)
    else:  # SQUARE
        if not 0.0 < duty < 1.0:
            raise ValueError("duty must be within (0, 1)")
        y = np.where(phase < duty, 1.0, -1.0)
    y = np.clip(y, -1.0, 1.0)  # guard tiny FP overshoot against the [-1, 1] validator
    return ArbitraryWaveform(samples=tuple(float(v) for v in y))
