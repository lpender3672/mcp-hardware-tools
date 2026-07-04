"""The Oscilloscope ABC is a contract, not instantiable.

Lifecycle and behaviour are covered against a concrete driver in
``test_ds1054z.py`` using a FakeTransport.
"""

from __future__ import annotations

import pytest

from hwtools.interfaces.oscilloscope import Oscilloscope


def test_oscilloscope_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Oscilloscope()  # type: ignore[abstract]
