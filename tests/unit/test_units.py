"""Engineering-notation formatting (human-facing only)."""

from __future__ import annotations

import pytest

from hwtools.model.units import eng


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1500, "Hz", "1.5 kHz"),
        (0.0025, "V", "2.5 mV"),
        (1e6, "Hz", "1 MHz"),
        (3.3, "V", "3.3 V"),
        (-0.2, "V", "-200 mV"),
        (0, "s", "0 s"),
    ],
)
def test_eng_formats_with_si_prefix(value: float, unit: str, expected: str) -> None:
    assert eng(value, unit) == expected
