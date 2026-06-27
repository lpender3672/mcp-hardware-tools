"""hwtools — self-correcting MCP control of bench instruments.

See the architecture plan: five strictly-separated layers (model → transport →
drivers → decode/analysis → tool surface), with the typed model at the bottom
that everything else depends on.
"""

__version__ = "0.0.0"
