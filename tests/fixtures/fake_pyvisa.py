"""A stand-in ``pyvisa`` module with the real library's session semantics.

pyvisa hands every ``ResourceManager("@py")`` in a process the *same* shared
session, and closing that manager closes every resource opened through it — its
own docstring warns that ``close()`` "will also terminate connections obtained
from other ResourceManager instances". :class:`FakeResourceManager` reproduces
exactly that, so a test can show one caller's cleanup killing another caller's
live instrument session. :func:`install` puts it in ``sys.modules``, which the
lazy ``import pyvisa`` inside the transport and discovery code then picks up.
"""

from __future__ import annotations

import sys
import types

import pytest

IDN = "RIGOL TECHNOLOGIES,DG1062Z,DG1ZA000000001,00.01.14"


class SessionClosedError(Exception):
    """Raised by a resource used after its session was closed."""


class FakeResource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.timeout = 0
        self.fail_close = False

    def _check(self) -> None:
        if self.closed:
            raise SessionClosedError(f"{self.name}: session closed")

    def close(self) -> None:
        if self.fail_close:
            raise SessionClosedError(f"{self.name}: link already dead")
        self.closed = True

    def query(self, command: str) -> str:
        self._check()
        return IDN + "\n" if command == "*IDN?" else "0\n"

    def write(self, command: str) -> None:
        self._check()

    def read_raw(self) -> bytes:
        self._check()
        return b"#10\n"

    def write_raw(self, data: bytes) -> None:
        self._check()


class FakeResourceManager:
    """One shared session per process, as pyvisa's ``@py`` backend has."""

    _shared: FakeResourceManager | None = None

    def __new__(cls, backend: str = "") -> FakeResourceManager:
        if cls._shared is None:
            rm = super().__new__(cls)
            rm.resources = []
            cls._shared = rm
        return cls._shared

    resources: list[FakeResource]

    def open_resource(self, name: str, open_timeout: int | None = None) -> FakeResource:
        res = FakeResource(name)
        self.resources.append(res)
        return res

    def list_resources(self, query: str = "?*::INSTR") -> tuple[str, ...]:
        return ()

    def close(self) -> None:
        for res in self.resources:
            res.closed = True
        type(self)._shared = None


def install(monkeypatch: pytest.MonkeyPatch) -> type[FakeResourceManager]:
    """Replace ``pyvisa`` with the fake for one test; returns the manager class."""
    FakeResourceManager._shared = None
    module = types.ModuleType("pyvisa")
    module.ResourceManager = FakeResourceManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvisa", module)
    return FakeResourceManager
