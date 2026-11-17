"""SSAPling core.

Responsibility: Component registration, configuration (including YAML files),
construction, and extension.

Not its responsibility: Subsystem contracts, seeds, run management, hashing.

May import: nothing. Enforced by tests/test_layering.py.
"""

from ._execution import Execution
from .errors import (
    ConfigError,
    DuplicateRegistrationError,
    MissingDependencyError,
    Problem,
    SettingsDeclarationError,
    SSAPlingError,
    UnknownComponentError,
)
from .loading import Config, load_config, load_named
from .registry import CatalogueEntry, Registration, Registry
from .settings import (
    check_settings_class,
    setting,
    settings_schema,
    settings_to_dict,
    validate_settings,
)
from .spec import ComponentSpec

__all__ = [
    "CatalogueEntry",
    "ComponentSpec",
    "Config",
    "ConfigError",
    "DuplicateRegistrationError",
    "Execution",
    "MissingDependencyError",
    "Problem",
    "Registration",
    "Registry",
    "SSAPlingError",
    "SettingsDeclarationError",
    "UnknownComponentError",
    "check_settings_class",
    "load_config",
    "load_named",
    "setting",
    "settings_schema",
    "settings_to_dict",
    "validate_settings",
]
