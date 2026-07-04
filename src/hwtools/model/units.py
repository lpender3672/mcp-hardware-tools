"""Small unit helpers shared across the model.

The model deliberately keeps quantities as plain ``float`` in SI base units, with
the unit baked into each field *name* (``level_v``, ``scale_s_per_div``,
``sample_rate_hz``). These helpers are only for human-facing rendering — error
messages, judgement notes, reprs — never for storage or comparison.
"""

from __future__ import annotations

# SI prefixes from femto to giga, by power-of-1000 exponent.
_PREFIXES: dict[int, str] = {
    -5: "f",
    -4: "p",
    -3: "n",
    -2: "µ",  # micro
    -1: "m",
    0: "",
    1: "k",
    2: "M",
    3: "G",
}


def eng(value: float, unit: str = "", *, places: int = 3) -> str:
    """Format ``value`` in engineering notation, e.g. ``eng(1500, "Hz") -> '1.5 kHz'``.

    Falls back to plain notation outside the supported prefix range.
    """
    if value == 0 or not _is_finite(value):
        return f"{value:.{places}g} {unit}".strip()

    exp = 0
    scaled = abs(value)
    while scaled >= 1000 and exp < max(_PREFIXES):
        scaled /= 1000
        exp += 1
    while scaled < 1 and exp > min(_PREFIXES):
        scaled *= 1000
        exp -= 1

    prefix = _PREFIXES.get(exp)
    if prefix is None:
        return f"{value:.{places}g} {unit}".strip()

    sign = "-" if value < 0 else ""
    return f"{sign}{scaled:.{places}g} {prefix}{unit}".strip()


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
