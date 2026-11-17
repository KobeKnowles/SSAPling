"""Every registry in one place.

Each area owns its registry (``ssapling.tasks.registry`` and so on); this
module collects them so a downstream package can register everything from
one import::

    from ssapling import registries as r

    def register_components():
        r.tasks.register("addition", AdditionTask)
        r.teachers.register("digit_tutor", DigitTutor)
        r.models.register("lookup", LookupModel)

More registries are added here as their areas are built.
"""

from .data import codecs, collators, samplers
from .environments import registry as environments
from .evaluate import aggregators, assessments, measurements
from .models import registry as models
from .tasks import registry as tasks
from .teachers import registry as teachers
from .train import stages

__all__ = [
    "aggregators",
    "assessments",
    "codecs",
    "collators",
    "environments",
    "measurements",
    "models",
    "samplers",
    "stages",
    "tasks",
    "teachers",
]
