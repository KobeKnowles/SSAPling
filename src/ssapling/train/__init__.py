"""SSAPling train.

Responsibility: Composing and running training procedures from registered,
user-extensible stages: interaction, learning, checkpointing, and monitoring;
run seeds, saved settings, and outputs.

Not its responsibility: Scoring definitions and final assessment.

May import: everything. Enforced by tests/test_layering.py.
"""

from .context import COUNTERS, Published, RunContext, derive_seed
from .errors import ProcedureStuckError
from .procedure import (
    Condition,
    Controller,
    Entry,
    Every,
    FixedController,
    Procedure,
    ReplayController,
)
from .runner import Runner
from .stages import Stage, learn, stages

__all__ = [
    "COUNTERS",
    "Condition",
    "Controller",
    "Entry",
    "Every",
    "FixedController",
    "Procedure",
    "ProcedureStuckError",
    "Published",
    "ReplayController",
    "RunContext",
    "Runner",
    "Stage",
    "derive_seed",
    "learn",
    "stages",
]
