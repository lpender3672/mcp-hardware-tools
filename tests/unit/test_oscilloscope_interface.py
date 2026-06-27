"""M0 sanity: the Oscilloscope ABC is a contract, not instantiable, and a
minimal concrete implementation drives its context-manager lifecycle.
"""

from __future__ import annotations

import pytest

from hwtools.interfaces.oscilloscope import Oscilloscope


def test_oscilloscope_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Oscilloscope()  # type: ignore[abstract]


class _SpyScope(Oscilloscope):
    """Smallest implementation that satisfies the abstract method set."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def connect(self) -> None:
        self.events.append("connect")

    def disconnect(self) -> None:
        self.events.append("disconnect")

    def idn(self) -> str:
        return "SPY,scope,0,0"


def test_context_manager_connects_then_disconnects() -> None:
    scope = _SpyScope()
    with scope as s:
        assert s is scope
        assert s.idn().startswith("SPY")
    assert scope.events == ["connect", "disconnect"]
