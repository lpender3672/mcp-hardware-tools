"""Layer 6 — the MCP tool surface.

Thin bindings over :class:`~hwtools.tools.session.ScopeSession`, which composes the
session store and pure analysis into the operations the agent calls. The behaviour
lives in the session; :mod:`hwtools.tools.server` is the MCP transport.
"""

from hwtools.tools.session import ScopeSession

__all__ = ["ScopeSession"]
