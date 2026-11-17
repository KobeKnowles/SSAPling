"""The runner: builds a run, runs its procedure stage by stage until it stops, and resumes it."""

import logging
import os
import re
from collections.abc import Mapping
from typing import Any, cast

from ..core import (
    ConfigError,
    Problem,
    UnknownComponentError,
    load_config,
    settings_to_dict,
    validate_settings,
)
from ..core._declarations import check_int, name_of, names
from ..core._execution import Execution
from ..core._text import join_path
from ..data import RecordStore, Store
from .components import build_components, check_components
from .context import RunContext
from .procedure import Condition, Controller, Decision, Entry, FixedController, Procedure
from .stages import stages

__all__ = ["Runner"]

_log = logging.getLogger("ssapling.train")
_CHECKPOINT = re.compile(r"(?:resume-\d+/)?checkpoints/(\d+)")
_REFUSALS = 3
"""Refused decisions in a row after which the run fails, with the last refusal."""


class Runner:
    """Runs a procedure: builds its components, then runs what the controller decides, checked.

    Example::

        runner = Runner(
            {
                "procedure": [
                    {"stage": "interact", "settings": {"episodes": 32}},
                    {"stage": "learn_operation", "every": {"episodes": 500},
                     "settings": {"operation": "consolidate"}},
                ],
                "stop": {"episodes": 100000},
            },
            {"model": model, "environment": env, "sampler": sampler},
            seed=7,
            output="runs/baseline",
        )
        context = await runner.run()

        runner = Runner.from_config("run.yaml", seed=7, output="runs/baseline")
        runner = Runner.resume("runs/baseline")     # from its last checkpoint

    Args:
        procedure: A ``Procedure``, or a mapping of its settings, such as
            one read from a file. Its ``components`` are built when the run
            starts. A mapping may leave out its ``procedure`` list when the
            controller offers one, as ``controller.procedure()``: every entry
            the controller can decide, which are then checked before the run.
        components: Components given as they are, keyed by role, such as a
            model built in Python, or a function. They are built before the
            procedure's own, which can use them.
        seed: The root seed: every seed in the run is derived from it.
        output: Where the run writes: a directory for a ``RecordStore``,
            or a ``Store``. It is the component ``"store"``. Records,
            results, checkpoints and the run's settings go there.
        controller: Decides what the run does next. By default, a
            ``FixedController`` following the procedure's list. If a run with
            a store ends in an error, the error's type and message are saved
            with its decisions, as ``"error"``.

    Raises:
        ConfigError: The procedure's settings, or its components', are
            invalid, listing every problem.
        ValueError: A component is given as ``"store"``.
    """

    def __init__(
        self,
        procedure: Procedure | Mapping[str, Any],
        components: Mapping[str, Any] | None = None,
        *,
        seed: int,
        output: str | os.PathLike[str] | Store | None = None,
        controller: Controller | None = None,
    ) -> None:
        offered = getattr(controller, "procedure", None)
        if isinstance(procedure, Mapping) and "procedure" not in procedure and callable(offered):
            procedure = {**procedure, "procedure": offered()}  # checked before the run, as given
        self.procedure = validate_settings(Procedure, procedure)
        self.components = dict(components or {})
        check_int(seed, "seed")
        self.seed = seed
        self.store = _store(output)
        self.controller = controller
        self._resuming = False
        if "store" in self.components:
            raise ValueError("the run's store is where it writes: give it as output=")
        problems = check_components(self.procedure.components, self._given())
        if problems:
            raise ConfigError("procedure", problems)

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str],
        *,
        seed: int,
        output: str | os.PathLike[str] | Store | None = None,
        components: Mapping[str, Any] | None = None,
        controller: Controller | None = None,
    ) -> "Runner":
        """A runner for the run file at ``path``: its components, procedure and stop condition.

        Raises:
            ConfigError: The file, or its settings, are invalid; the error
                names the file.
        """
        try:
            procedure = load_config(path)
            return cls(procedure, components, seed=seed, output=output, controller=controller)
        except ConfigError as error:
            raise ConfigError(f"run file {path}", error.problems) from None

    @classmethod
    def resume(
        cls,
        output: str | os.PathLike[str] | Store,
        *,
        components: Mapping[str, Any] | None = None,
        controller: Controller | None = None,
    ) -> "Runner":
        """A runner that continues the run written at ``output``, from its last checkpoint.

        The run is rebuilt from the settings it saved. Components it was
        given as they are, and a controller of its own, are given again.

        Raises:
            UnknownDocumentError: No run was written at ``output``.
            ValueError: A component or controller the run was given is
                missing.
        """
        store = _store(output)
        assert store is not None
        saved = store.load_document("run")
        missing = sorted(set(saved["given"]) - set(components or {}))
        if missing:
            raise ValueError(f"give the run its components again: {', '.join(map(repr, missing))}")
        if saved["controller"] is not None and controller is None:
            raise ValueError(f"give the run its controller again, a {saved['controller']}")
        runner = cls(
            saved["procedure"], components, seed=saved["seed"], output=store, controller=controller
        )
        runner._resuming = True
        return runner

    async def run(self) -> RunContext:
        """Run until the stop condition holds or the controller stops; return the run context.

        First the procedure's components are built, in order. Then, before
        any stage runs, every entry of the procedure is checked, and every
        problem is reported together: unknown stages, invalid settings,
        components a stage requires that the run does not have, a stage's
        own ``check``, and values a stage or a condition reads that no
        stage in the procedure publishes. A new run saves its settings in
        its store; a resumed one restores its last checkpoint. Then, until
        the stop condition holds, the controller decides, and its decision
        is checked the same way (now against the values published so far)
        and run. At the end, the components the run built are closed.

        Raises:
            ConfigError: The procedure, or a decision, cannot run.
            ProcedureStuckError: Nothing in a fixed procedure can run.
            ValueError: The store holds a run already, and this one is not
                resumed.
            Exception: A stage's own error, as itself.
        """
        components = self._given()
        try:
            seeds = build_components(self.procedure.components, components, seed=self.seed)
            controller = self.controller or FixedController(self.procedure)
            context = RunContext(components, seed=self.seed, controller=controller)
            self._check(context)
            if self._resuming:
                await self._restore(context)
            elif self.store is not None:
                self._save_run(seeds)
            ended: BaseException | None = None  # the error that ended the run, if one did
            try:
                await self._loop(context, controller)
            except BaseException as error:
                ended = error
                raise
            finally:  # how the run went, whatever ended it
                if self.store is not None:
                    document: dict[str, Any] = {"decisions": context.decisions}
                    if ended is not None:
                        document["error"] = {"type": type(ended).__name__, "message": str(ended)}
                    self.store.save_document(context.output_name("decisions"), document)
            return context
        finally:  # what the run built, it closes
            for role in reversed(self.procedure.components):
                close = getattr(components.get(role), "close", None)
                if callable(close):
                    await Execution().call(close)

    async def _loop(self, context: RunContext, controller: Controller) -> None:
        refused, last = 0, None  # decisions refused in a row, and why the last was
        while not self.procedure.stop.holds(context):
            decision: object = await Execution().call(controller.decide, context, last)
            if decision is None:
                _log.info("the controller stopped the run")
                return
            seen, entry = context.export_state(), None
            try:
                entry = validate_settings(Entry, cast(Decision, decision))  # checked here
                stage = self._decided(entry, context)
            except ConfigError as error:
                refused, last = refused + 1, error
                context.decisions.append(_logged(entry, seen, refused=str(error)))
                _log.info("decision refused: %s", error)
                if refused == _REFUSALS:
                    raise
                continue
            refused, last = 0, None
            context.decisions.append(_logged(entry, seen, refused=None))
            context.counters["stages"] += 1
            _log.info("stage %d: %s", context.counters["stages"], entry.stage)
            await Execution().call(stage.run, context)
        _log.info("the stop condition holds")

    def _decided(self, entry: Entry, context: RunContext) -> Any:
        """Build the stage a decision names, if it can run now.

        Raises:
            ConfigError: It cannot: as for an entry of the procedure, or
                because it reads a value that is not published yet.
        """
        stage, problems = self._prepare(entry, context, "")
        unread = _declared(stage, "reads") - set(context.values) - _declared(stage, "publishes")
        problems += [Problem("", f"reads {name!r}, which is not published") for name in unread]
        if problems:
            raise ConfigError(f"decision to run stage {entry.stage!r}", problems)
        return stage

    def _given(self) -> dict[str, Any]:
        """The components given as they are, with the run's store."""
        store = {} if self.store is None else {"store": self.store}
        return {**self.components, **store}

    def _save_run(self, seeds: Mapping[str, int]) -> None:
        """Save what rebuilds the run: its settings, its seeds, and what it was given."""
        assert self.store is not None
        settings = {
            "procedure": settings_to_dict(self.procedure),
            "seed": self.seed,
            "seeds": dict(seeds),
            "given": sorted(self.components),
            "controller": None if self.controller is None else name_of(self.controller),
        }
        try:
            self.store.save_document("run", settings)
        except ValueError as error:
            error.add_note("Resume that run with Runner.resume, or give this one a new output.")
            raise

    async def _restore(self, context: RunContext) -> None:
        """Continue from the last checkpoint, as a new session; or from the start, if none."""
        assert self.store is not None
        names = self.store.document_names()
        context.session = 1 + sum(name.startswith("resumes/") for name in names)
        checkpoint = _latest_checkpoint(names)
        self.store.save_document(f"resumes/{context.session}", {"checkpoint": checkpoint})
        if checkpoint is None:
            _log.info("resuming from the start: the run has no checkpoint")
            return
        saved = self.store.load_document(checkpoint)
        context.restore_state(saved["context"])
        context.decisions = list(saved["decisions"])
        for role, state in saved["components"].items():
            await _put_back(context.components[role], state, repr(role))
        if saved["controller"] is not None:
            await _put_back(context.controller, saved["controller"], "the controller")
        fresh = [role for role in context.components if role not in {*saved["components"], "store"}]
        _log.info("resumed from %s", checkpoint)
        if fresh:
            _log.warning(
                "These components do not export their state, so they start as built: %s",
                ", ".join(fresh),
            )

    def _check(self, context: RunContext) -> None:
        """Check every entry of the procedure before the run, reporting every problem."""
        problems: list[Problem] = []
        built = []
        for index, entry in enumerate(self.procedure.procedure):
            path = f"procedure[{index}]"
            stage, found = self._prepare(entry, context, path)
            problems += found
            built.append((path, entry, stage))
        published = {name for _, _, stage in built for name in _declared(stage, "publishes")}
        readers: list[tuple[str, object]] = []  # what reads values, and where it is
        for path, entry, stage in built:
            readers += [(path, stage), (f"{path}.until.value", entry.until)]
            readers += [(f"{path}.when.value", entry.when)]
        readers += [("stop.value", self.procedure.stop)]
        problems += [
            Problem(path, f"reads {name!r}, which no stage in the procedure publishes")
            for path, reader in readers
            for name in sorted(_reads(reader) - published)
        ]
        if problems:
            raise ConfigError("procedure", problems)

    def _prepare(self, entry: Entry, context: RunContext, path: str) -> tuple[Any, list[Problem]]:
        """Build ``entry``'s stage, and find what stops it running with the run's components.

        Returns the stage (``None`` if it cannot be built) and the problems.
        """
        settings_path = join_path(path, "settings")
        try:
            stage = stages.create(entry.stage, entry.settings)
        except UnknownComponentError as error:
            return None, [Problem(join_path(path, "stage"), str(error))]
        except ConfigError as error:
            return None, [
                Problem(join_path(settings_path, problem.path), problem.message)
                for problem in error.problems
            ]
        problems = []
        missing = [role for role in _declared(stage, "requires") if role not in context.components]
        if missing:
            have = ", ".join(map(repr, context.components)) or "none"
            problems.append(
                Problem(
                    path,
                    f"stage {entry.stage!r} requires {', '.join(map(repr, missing))}, which the "
                    f"run does not have (its components: {have})",
                )
            )
        check = getattr(stage, "check", None)
        if not missing and callable(check):
            try:
                check(context)
            except ValueError as error:
                problems.append(Problem(path, f"stage {entry.stage!r}: {error}"))
        return stage, problems


def _logged(entry: Entry | None, seen: Mapping[str, Any], *, refused: str | None) -> dict[str, Any]:
    """A decision as the run records it: the entry, what the controller saw, any refusal."""
    return {
        "stage": None if entry is None else entry.stage,
        "settings": None if entry is None else entry.settings,
        "seen": seen,
        "refused": refused,
    }


async def _put_back(owner: object, state: Any, what: str) -> None:
    """Give ``owner`` back the state it saved."""
    restore = getattr(owner, "restore_state", None)
    if not callable(restore):
        raise TypeError(f"{what} saved its state, but has no restore_state to take it back")
    await Execution().call(restore, state)


def _store(output: str | os.PathLike[str] | Store | None) -> Store | None:
    return RecordStore(output) if isinstance(output, str | os.PathLike) else output


def _latest_checkpoint(names: tuple[str, ...]) -> str | None:
    """The checkpoint taken at the latest stage.

    A resumed run continues from the latest checkpoint, so its own come at
    later stages: the latest stage is the latest checkpoint, whichever
    session took it.
    """
    found = [(int(match[1]), name) for name in names if (match := _CHECKPOINT.fullmatch(name))]
    return max(found)[1] if found else None


def _declared(stage: object, what: str) -> set[str]:
    """The names ``stage`` declares in ``what``: ``requires``, ``reads`` or ``publishes``."""
    if stage is None:
        return set()
    declared = getattr(stage, what, ())
    return set(names(declared, f"{name_of(stage)}.{what}", example="('model',)"))


def _reads(reader: object) -> set[str]:
    """The values a stage or a condition reads."""
    if isinstance(reader, Condition):
        return set() if reader.reads is None else {reader.reads}
    return _declared(reader, "reads")
