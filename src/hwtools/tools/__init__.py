"""Layer 6 — the MCP transport.

Exposes the :class:`~hwtools.session.scope_session.ScopeSession` (Layer 5) over
MCP. This package holds no behaviour of its own: :func:`~hwtools.tools.server.build_server`
binds a session's operations onto agent-callable tools, and ``main`` serves them
over stdio.
"""

from hwtools.tools.server import build_server, main

__all__ = ["build_server", "main"]
