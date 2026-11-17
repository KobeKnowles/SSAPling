"""SSAPling interaction.

Responsibility: The loop that runs episodes: driving environments, models, and
teachers under the chosen timing and execution, static solving, and writing
records.

Not its responsibility: Choosing instances, deciding when episodes run,
scoring.

May import: core, tasks, environments, teachers, models, data. Enforced by
tests/test_layering.py.
"""

from .episode import EpisodeSettings, run_episode, solve
from .errors import ComponentProcessError
from .hosting import in_process

__all__ = [
    "ComponentProcessError",
    "EpisodeSettings",
    "in_process",
    "run_episode",
    "solve",
]
