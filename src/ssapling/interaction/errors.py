"""Errors specific to interaction."""

from ..core.errors import SSAPlingError


class ComponentProcessError(SSAPlingError, RuntimeError):
    """The process hosting a component cannot do what was asked of it.

    Raised when the process has exited (it crashed, or the ``in_process``
    block has ended), or when a result could not be sent back from it. An
    error the component raises itself is not wrapped: it arrives as itself.

    Attributes:
        component: The component's class name.
        problem: What went wrong.
    """

    def __init__(self, component: str, problem: str) -> None:
        self.component = component
        self.problem = problem
        super().__init__(f"The process hosting {component} {problem}.")
