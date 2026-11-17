"""Toy environments for tests. They exercise the environment contract and are not part of SSAPling.

Each "native" environment has its own interface, as an existing library or
service would, and keeps a log of what it executed. Each adapter wraps one
and translates calls and results without changing its behaviour, so the
tests can compare native and adapted behaviour. Everything is a plain
module-level class, so the fixtures can be registered by name and rebuilt in
a worker process.
"""

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from ssapling import environments
from ssapling.environments import EnvironmentCommunicationError, ResetResult, StepResult
from ssapling.tasks import Instance
from ssapling.teachers import TeacherRequest, TeacherResponse

# Single agent: entering the answer to an addition, digit by digit


class NativeDigitPad:
    """Enter a + b one digit at a time, units first. A wrong digit ends the game.

    Native interface: ``reset(seed, problem) -> (observation, info)`` and
    ``step(digit) -> (observation, done, info)``. It has no rewards.
    """

    def __init__(self) -> None:
        self.a = self.b = 0
        self.entered = ""
        self.done = False
        self.log: list[str] = []

    def reset(self, seed: int | None = None, problem: tuple[int, int] | None = None) -> Any:
        rng = random.Random(seed)
        self.a, self.b = problem or (rng.randrange(10, 100), rng.randrange(10, 100))
        self.entered, self.done, self.log = "", False, []
        return self._observe(), {"digits": len(self._answer())}

    def step(self, digit: str) -> Any:
        self.log.append(digit)
        right = self._answer()[len(self.entered)]
        self.entered += digit
        self.done = digit != right or len(self.entered) == len(self._answer())
        return self._observe(), self.done, {"correct": digit == right}

    def _answer(self) -> str:
        return str(self.a + self.b)[::-1]

    def _observe(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "entered": self.entered}


class DigitPadEnv:
    """Adapts ``NativeDigitPad``: one agent, ``"student"``, with no rewards.

    It is set up from an instance's inputs (``options={"a": .., "b": ..}``),
    and offers every capability. It can own a teacher, which reviews each
    digit inside the transition, as a teacher placed in the environment does.
    """

    observation_descriptions: ClassVar[Mapping[str, str]] = {"student": "a string of digits"}
    action_descriptions: ClassVar[Mapping[str, str]] = {"student": "one digit"}

    def __init__(self, teacher: Any = None) -> None:
        self.native = NativeDigitPad()
        self.teacher = teacher
        self.closed = False
        self._requests = 0

    def reset(self, *, seed: int | None = None, options: Any = None) -> ResetResult:
        if self.teacher is not None:
            self.teacher.reset()  # the environment owns the teacher, so it resets it
        problem = (options["a"], options["b"]) if options else None
        observation, info = self.native.reset(seed, problem)
        return ResetResult(observations={"student": observation}, info=info)

    def step(self, actions: Mapping[str, Any]) -> StepResult:
        proposed = actions["student"]
        responses, executed = self._review(proposed)
        observation, done, info = self.native.step(executed)
        return StepResult(
            observations={"student": observation},
            terminated={"student": done},
            truncated=None,  # not reported by the native environment
            rewards=None,  # the native environment has no rewards
            info=info,
            executed_actions={"student": executed} if self.teacher else None,
            teacher_responses=responses,
        )

    def _review(self, proposed: str) -> tuple[tuple[TeacherResponse, ...] | None, str]:
        if self.teacher is None:
            return None, proposed
        self._requests += 1
        native = self.native
        request = TeacherRequest(
            id=f"pad-review-{self._requests}",
            operation="review_action",
            context={
                "instance": Instance(
                    id="digit-pad",
                    inputs={"a": native.a, "b": native.b},
                    reference=native.a + native.b,
                ),
                "state": native.entered,
            },
            proposals={"student": proposed},
            participant="student",
            step=len(native.entered),
        )
        response = self.teacher.respond(request)
        intervention = (
            response.contributions.get("intervention", {}) if response.status == "answered" else {}
        )
        return (response,), intervention.get("student", proposed)

    def render(self) -> str:
        n = self.native
        return f"{n.a} + {n.b} = {n.entered[::-1] or '?'}"

    def state(self) -> dict[str, Any]:
        return {"answer": self.native.a + self.native.b, **self.native._observe()}

    def snapshot(self) -> NativeDigitPad:
        return copy.deepcopy(self.native)

    def restore(self, saved: NativeDigitPad) -> None:
        self.native = copy.deepcopy(saved)

    def close(self) -> None:
        self.closed = True
        if self.teacher is not None and hasattr(self.teacher, "close"):
            self.teacher.close()


# Two players taking turns: Nim


class NativeNim:
    """Players alternate taking 1 to 3 stones; whoever takes the last stone wins.

    Native interface: ``current_player``, ``take(n)``, which returns the
    winner or ``None``, and ``scores()`` at the end. ``reset(seed)`` ignores
    the seed: every game starts from the same pile.
    """

    PLAYERS = ("player_1", "player_2")

    def __init__(self, stones: int = 5) -> None:
        self.start = stones
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.pile, self.turn = self.start, 0
        self.winner: str | None = None
        self.log: list[tuple[str, int]] = []

    @property
    def current_player(self) -> str:
        return self.PLAYERS[self.turn % 2]

    def take(self, n: int) -> str | None:
        if not 1 <= n <= min(3, self.pile):
            raise ValueError(f"cannot take {n} from {self.pile}")
        self.log.append((self.current_player, n))
        self.pile -= n
        if self.pile == 0:
            self.winner = self.current_player
        self.turn += 1
        return self.winner

    def scores(self) -> dict[str, int]:
        return {p: (1 if p == self.winner else -1) for p in self.PLAYERS}


class NimEnv:
    """Adapts ``NativeNim``: two agents taking turns, saying whose turn it is.

    Its ``reset`` takes a seed, but the native game ignores it, so the adapter
    switches seeded reset off rather than claim a guarantee it cannot keep.
    """

    agents = NativeNim.PLAYERS

    def __init__(self, stones: int = 5) -> None:
        self.native = NativeNim(stones)

    @property
    def next_agents(self) -> tuple[str, ...]:
        return () if self.native.winner else (self.native.current_player,)

    def supports(self, capability: str) -> bool:
        return capability != "seeded_reset"

    def reset(self, *, seed: int | None = None) -> ResetResult:
        self.native.reset(seed)
        return ResetResult(observations={self.native.current_player: self.native.pile})

    def step(self, actions: Mapping[str, Any]) -> StepResult:
        player = self.native.current_player
        if set(actions) != {player}:
            raise ValueError(f"it is {player}'s turn; got actions for {sorted(actions)}")
        winner = self.native.take(actions[player])
        if winner is None:
            return StepResult(
                observations={self.native.current_player: self.native.pile},
                rewards={p: 0 for p in self.agents},
            )
        done = dict.fromkeys(self.agents, True)
        return StepResult(
            observations=dict.fromkeys(self.agents, 0),  # each player's final observation
            rewards=self.native.scores(),
            terminated=done,
            truncated={p: False for p in self.agents},
        )


# Two players acting at once: matching pennies


class NativePennies:
    """Both players show heads or tails at once; ``player_1`` wins if they match.

    Native interface: ``play(choice_1, choice_2) -> (reward_1, reward_2)``. The
    game stops after a fixed number of rounds.
    """

    def __init__(self, rounds: int = 3) -> None:
        self.rounds = rounds
        self.played = 0
        self.log: list[tuple[str, str]] = []

    def play(self, choice_1: str, choice_2: str) -> tuple[int, int]:
        self.log.append((choice_1, choice_2))
        self.played += 1
        return (1, -1) if choice_1 == choice_2 else (-1, 1)


class PenniesEnv:
    """Adapts ``NativePennies``: both agents act every step, and a round limit truncates."""

    def __init__(self, rounds: int = 3) -> None:
        self.native = NativePennies(rounds)

    def reset(self) -> ResetResult:
        self.native = NativePennies(self.native.rounds)
        return ResetResult(observations={"player_1": None, "player_2": None})

    def step(self, actions: Mapping[str, Any]) -> StepResult:
        choice_1, choice_2 = actions["player_1"], actions["player_2"]
        reward_1, reward_2 = self.native.play(choice_1, choice_2)
        limit = self.native.played >= self.native.rounds
        return StepResult(
            observations={"player_1": choice_2, "player_2": choice_1},  # each sees the other
            rewards={"player_1": reward_1, "player_2": reward_2},
            terminated={"player_1": False, "player_2": False},
            truncated={"player_1": limit, "player_2": limit},
        )


# An interactive ARC-AGI-3-like game behind a remote service


class FakeArcService:
    """A stand-in for a remote game service with ARC-AGI-3's shapes.

    Each response holds a sequence of frames, the game state
    (``NOT_FINISHED``, ``WIN``, or ``GAME_OVER``), the levels completed, and
    the available actions. ``ACTION6`` takes coordinates: clicking the
    target cell completes a level, and three misses end the game.
    ``drop_next`` makes the next call fail after it has taken effect, as a
    lost connection can.
    """

    TARGETS = ((1, 0), (0, 1))  # one target cell per level

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.drop_next = False

    def reset(self, game_id: str) -> dict[str, Any]:
        self.calls.append(("RESET", game_id))
        self.level, self.misses, self.status = 0, 0, "NOT_FINISHED"
        return self._response([self._grid()])

    def act(self, action: str, data: Mapping[str, int] | None = None) -> dict[str, Any]:
        self.calls.append((action, dict(data or {})))
        frames = [self._grid()]
        if action == "ACTION6" and data is not None:
            if (data["x"], data["y"]) == self.TARGETS[self.level]:
                self.level += 1
                if self.level == len(self.TARGETS):
                    self.status = "WIN"
                else:
                    frames.append(self._grid())  # an animation frame, then the new level
            else:
                self.misses += 1
                if self.misses == 3:
                    self.status = "GAME_OVER"
        response = self._response(frames)
        if self.drop_next:
            self.drop_next = False
            raise ConnectionError("connection reset by peer")
        return response

    def _grid(self) -> list[list[int]]:
        return [[self.level, self.misses], [0, 0]]

    def _response(self, frames: list[list[list[int]]]) -> dict[str, Any]:
        return {
            "frame": frames,
            "state": self.status,
            "levels_completed": self.level,
            "available_actions": [] if self.status != "NOT_FINISHED" else ["ACTION1", "ACTION6"],
        }


class ArcEnv:
    """Adapts ``FakeArcService`` asynchronously, as a remote environment.

    It keeps the frame sequence, parameterised actions, available actions, and
    game status. Completing a level is progress, not the end of the episode;
    only ``WIN`` or ``GAME_OVER`` ends it. It invents no rewards: benchmark
    scoring is downstream. A lost connection is reported as such, never as
    the end of the episode, and the action is not retried.
    """

    action_descriptions: ClassVar[Mapping[str, list[str]]] = {"agent": ["ACTION1", "ACTION6"]}

    def __init__(self, service: FakeArcService | None = None, game_id: str = "game-a") -> None:
        self.service = service or FakeArcService()
        self.game_id = game_id

    async def reset(self, *, options: Any = None) -> ResetResult:
        game_id = (options or {}).get("game_id", self.game_id)
        response = self.service.reset(game_id)
        return ResetResult(
            observations={"agent": self._observe(response)},
            info={"state": response["state"]},
            action_restrictions={"agent": response["available_actions"]},
        )

    async def step(self, actions: Mapping[str, Any]) -> StepResult:
        action = dict(actions["agent"])
        name = action.pop("action")
        try:
            response = self.service.act(name, action or None)
        except ConnectionError as error:
            raise EnvironmentCommunicationError(
                "ArcEnv", "step", str(error), may_have_executed=True
            ) from error
        finished = response["state"] in ("WIN", "GAME_OVER")
        return StepResult(
            observations={"agent": self._observe(response)},
            terminated={"agent": finished},
            info={"state": response["state"], "levels_completed": response["levels_completed"]},
            action_restrictions={"agent": response["available_actions"]},
        )

    def _observe(self, response: Mapping[str, Any]) -> dict[str, Any]:
        return {"frames": response["frame"], "levels_completed": response["levels_completed"]}


# One agent against a clock


class Ticker:
    """Shows one agent the tick number, and takes any action, or none.

    An agent left out of a step gets the environment's own default action,
    ``"wait"``. The episode is truncated after ``ticks`` steps.
    """

    def __init__(self, ticks: int = 3) -> None:
        self.ticks = ticks
        self.received: list[Any] = []

    def reset(self) -> ResetResult:
        self.tick, self.received = 0, []
        return ResetResult(observations={"agent": "t0"})

    def step(self, actions: Mapping[str, Any]) -> StepResult:
        self.received.append(actions.get("agent", "wait"))
        self.tick += 1
        return StepResult(
            observations={"agent": f"t{self.tick}"}, truncated={"agent": self.tick >= self.ticks}
        )


# Wrappers a downstream project might write


class AsyncProxy:
    """Makes any synchronous environment asynchronous, as a remote proxy to it would be."""

    def __init__(self, env: Any) -> None:
        self.env = env

    async def reset(self, **arguments: Any) -> ResetResult:
        result: ResetResult = self.env.reset(**arguments)
        return result

    async def step(self, actions: Mapping[str, Any]) -> StepResult:
        result: StepResult = self.env.step(actions)
        return result

    def supports(self, capability: str) -> bool:
        # A wrapper asks about the environment it wraps; never about itself.
        return environments.supports(self.env, capability)

    def __getattr__(self, name: str) -> Any:
        if name == "env" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class LaneBatch:
    """A vectorising wrapper: independent copies of an environment, stepped together as lanes.

    A lane that ends waits to be reset; it is never reset automatically, so
    its final observation stays distinct from the next episode's first.
    """

    def __init__(self, make: Callable[[], Any], lanes: int) -> None:
        self.lanes = lanes
        self.envs = [make() for _ in range(lanes)]
        self.ended: set[int] = set(range(lanes))

    def reset_lanes(
        self, lanes: Sequence[int], *, seeds: Mapping[int, int] | None = None
    ) -> dict[int, ResetResult]:
        results = {}
        for lane in lanes:
            seed = (seeds or {}).get(lane)
            results[lane] = self.envs[lane].reset(**({"seed": seed} if seed is not None else {}))
            self.ended.discard(lane)
        return results

    def step_lanes(self, actions: Mapping[int, Mapping[str, Any]]) -> dict[int, StepResult]:
        results = {}
        for lane, lane_actions in actions.items():
            if lane in self.ended:
                raise ValueError(f"lane {lane} has ended; reset it first")
            results[lane] = result = self.envs[lane].step(lane_actions)
            if result.episode_over:
                self.ended.add(lane)
        return results
