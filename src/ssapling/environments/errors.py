"""Errors specific to environments."""

from ..core.errors import SSAPlingError


class EnvironmentCommunicationError(SSAPlingError, ConnectionError):
    """A remote environment could not be reached, so the outcome of a call is unknown.

    This is not the end of an episode: an episode ends only through the
    termination and truncation signals of a ``StepResult``. A remote adapter
    raises this error when it loses its connection, and the runner raises it
    when an environment does not answer in time. Neither retries the call,
    because a step that may already have executed must not be submitted
    twice.

    Attributes:
        environment: The environment's class name.
        operation: The call that failed, such as ``"step"``.
        may_have_executed: Whether the call may have taken effect anyway, as
            a step sent to a service that did not answer may have.
        reason: What went wrong.
    """

    def __init__(
        self, environment: str, operation: str, reason: str, *, may_have_executed: bool
    ) -> None:
        self.environment = environment
        self.operation = operation
        self.reason = reason
        self.may_have_executed = may_have_executed
        effect = (
            "it may have taken effect, so it was not retried"
            if may_have_executed
            else "it did not take effect"
        )
        super().__init__(f"{environment}.{operation} failed to communicate: {reason}; {effect}.")


class UnsupportedCapabilityError(SSAPlingError, LookupError):
    """An optional capability was used with an environment that does not support it.

    Attributes:
        environment: The environment's class name.
        capability: The capability that was needed, such as ``"seeded_reset"``.
    """

    def __init__(self, environment: str, capability: str) -> None:
        self.environment = environment
        self.capability = capability
        super().__init__(
            f"Environment {environment} does not support {capability!r}; "
            f"check environments.supports(env, {capability!r}) before using it."
        )
