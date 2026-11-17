"""A small training setup for tests: additions on the digit pad, learned by an apprentice."""

from typing import Any

from ssapling.data import Collections, GeneratedSampler, GeneratedSettings
from ssapling.evaluate import assessments
from tests.fixtures.environments import DigitPadEnv
from tests.fixtures.models import Apprentice
from tests.fixtures.tasks import AdditionTask

MONITORING = {
    "collections": ["validation"],
    "aggregations": {"accuracy": {"measurement": "score", "aggregator": "mean"}},
}


def addition(model: Any = None) -> dict[str, Any]:
    """A run's components, built in Python: the model, the digit pad, a sampler, and more.

    The model is an ``Apprentice``, right once it has had three updates,
    unless another is given. Four validation problems are reserved, and
    ``monitoring`` assesses them, statically.
    """
    task = AdditionTask()
    sampler = GeneratedSampler(GeneratedSettings(vary="digits", values=[1, 2]), source=task)
    collections = Collections(task)
    collections.build("validation", sampler=sampler, size=4, seed=1)
    return {
        "model": model or Apprentice(),
        "environment": DigitPadEnv(),
        "sampler": sampler,
        "collections": collections,
        "monitoring": assessments.create(
            "standard", MONITORING, collections=collections, task=task
        ),
    }
