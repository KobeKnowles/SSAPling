"""SSAPling: infrastructure for System-Specification Alignment Problem projects.

SSAPling provides contracts, registries, and machinery. It contains no concrete
environments, tasks, teachers, models, or learning methods; those live in
downstream projects. Each package area is described in its own ``__init__``.
"""

from .core.errors import SSAPlingError

__version__ = "0.1.0"

__all__ = ["SSAPlingError", "__version__"]
