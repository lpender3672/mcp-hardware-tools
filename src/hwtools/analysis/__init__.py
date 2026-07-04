"""Layer 4 (pure) — measurements and the judge/adjust brain of the loop.

Pure functions over value objects; no instrument or session state. The stateful
convergence wrappers that *drive* a scope live in :mod:`hwtools.session`.
"""

from hwtools.analysis import measure, spectrum
from hwtools.analysis.adjust import suggest_adjustment
from hwtools.analysis.classify import Classification, SignalClass, classify, triage
from hwtools.analysis.describe import Characterization, describe
from hwtools.analysis.judge import judge_capture
from hwtools.analysis.recommend import Setup, recommend_setup

__all__ = [
    "Characterization",
    "Classification",
    "Setup",
    "SignalClass",
    "classify",
    "describe",
    "judge_capture",
    "measure",
    "recommend_setup",
    "spectrum",
    "suggest_adjustment",
    "triage",
]
