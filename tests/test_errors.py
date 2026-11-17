"""Every SSAPling error survives pickling, so it can travel from a worker process to the caller."""

import pickle

import pytest

from ssapling.core import (
    ConfigError,
    DuplicateRegistrationError,
    MissingDependencyError,
    Problem,
    SettingsDeclarationError,
    SSAPlingError,
    UnknownComponentError,
)
from ssapling.data import (
    HeldOutError,
    SourceExhaustedError,
    StoreError,
    UnknownCollectionError,
    UnknownDocumentError,
    UnknownRecordError,
)
from ssapling.environments import EnvironmentCommunicationError, UnsupportedCapabilityError
from ssapling.interaction import ComponentProcessError
from ssapling.models import DescriptionMismatch, DescriptionMismatchError
from ssapling.tasks import MissingTeachingResourceError, UnknownInstanceError
from ssapling.teachers import MissingContextError, MissingResourcesError, UnsupportedOperationError
from ssapling.train import ProcedureStuckError

ERRORS = [
    ConfigError("settings for X", [Problem("a.b", "must be >= 0")]),
    SettingsDeclarationError("XSettings", [Problem("a", "unsupported setting type")]),
    UnknownComponentError("teacher", "stepwize", ["stepwise"], location="run.yaml: teacher.name"),
    DuplicateRegistrationError("teacher", "stepwise", int, str),
    MissingDependencyError("install ssapling[yaml]"),
    UnknownInstanceError("p9", task="puzzles"),
    MissingTeachingResourceError("procedure", ["example"]),
    UnsupportedOperationError("Tutor", "explain", ["review_action"]),
    MissingContextError("Tutor", "review_action", ["state"], ["instance"]),
    MissingResourcesError("Tutor", ["procedure"], ["example"]),
    DescriptionMismatchError(
        "Model", [DescriptionMismatch("action", "digit", "letter")], participant="p"
    ),
    EnvironmentCommunicationError("ArcEnv", "step", "reset by peer", may_have_executed=True),
    UnsupportedCapabilityError("NimEnv", "seeded_reset"),
    SourceExhaustedError("UniformSampler", ["validation"], 1000, building="test"),
    UnknownCollectionError("valid", ["validation"]),
    HeldOutError("test", "training"),
    UnknownRecordError("ep-1", "runs/a"),
    UnknownDocumentError("results/assess-1", "runs/a"),
    StoreError("runs/a/records/ep-1.jsonl, line 3: not valid JSON"),
    ComponentProcessError("DigitPadEnv", "exited (exit code 3)"),
    ProcedureStuckError({"episodes": 3, "steps": 0, "updates": 1, "stages": 4}),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda error: type(error).__name__)
def test_an_error_survives_pickling_with_its_message_and_attributes(error):
    error.add_note("While doing something.")
    back = pickle.loads(pickle.dumps(error))
    assert type(back) is type(error)
    assert str(back) == str(error)
    assert vars(back) == vars(error)  # attributes and notes


def test_every_ssapling_error_is_covered():
    """The imports above load every area, so every error class exists here."""

    def all_subclasses(cls: type) -> set[type]:
        return {sub for child in cls.__subclasses__() for sub in {child, *all_subclasses(child)}}

    assert all_subclasses(SSAPlingError) == {type(error) for error in ERRORS}
