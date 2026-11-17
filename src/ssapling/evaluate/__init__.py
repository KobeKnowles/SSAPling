"""SSAPling evaluate.

Responsibility: User-defined assessments, measurement collection, and
aggregation.

Not its responsibility: Learning and choosing training instances.

May import: everything except train. Enforced by tests/test_layering.py.
"""

from .aggregation import AggregationSettings, Summary, aggregate, aggregators
from .assessment import (
    Assessment,
    AssessmentResults,
    AssessmentSettings,
    StandardAssessment,
    assessments,
    load_results,
    save_results,
)
from .measurement import InstanceResult, MeasuredValue, Trial, ValueStatus, measurements

__all__ = [
    "AggregationSettings",
    "Assessment",
    "AssessmentResults",
    "AssessmentSettings",
    "InstanceResult",
    "MeasuredValue",
    "StandardAssessment",
    "Summary",
    "Trial",
    "ValueStatus",
    "aggregate",
    "aggregators",
    "assessments",
    "load_results",
    "measurements",
    "save_results",
]
