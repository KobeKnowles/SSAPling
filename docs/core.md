# Core

`ssapling.core` finds an implementation by name, checks its settings, and
creates it. Downstream packages take part through the same public interfaces.
Core knows nothing about what teachers, tasks, or models *do*. It only
handles naming, settings, and construction.

| Part | What it does |
| --- | --- |
| [Registry](#registry) | Maps names to implementations, one registry per kind of component. |
| [Settings](#settings) | Declared settings: defaults, type checks, validation, JSON Schema export. |
| [Loading files](#loading-configuration-files) | YAML or JSON files into plain mappings; `{name, settings}` component entries. |
| [Construction](#construction) | `create(name, settings, **live)`: validate, then build. |
| [Catalogue](#catalogue) | Every registered component with its description and settings schema. |
| [Execution helper](#execution-helper) | One way to call sync and async methods, for downstream code too. |

The whole path, from a file to a working component:

```python
from ssapling.core import load_config

register_components()                          # downstream: fill the registries
config = load_config("run.yaml")               # file -> plain mappings
spec = config.component("teacher")             # one {name, settings} entry
teacher = teachers.create_from(spec, task=task)
# = StepwiseTeacher(config=<validated StepwiseTeacherSettings>, task=task)
```

## Registry

A **registry** maps names to implementations (classes or functions) of one kind
of component. Each kind has its own registry: teachers, samplers, stages,
measurements, and so on. A registry stores the implementation itself, never a
constructed object. Building objects is the job of [construction](#construction).

```python
from ssapling.core import Registry

teachers = Registry("teacher")          # "teacher" is used in error messages

teachers.register("stepwise", StepwiseTeacher)
teachers.get("stepwise")                # -> StepwiseTeacher (the class)
teachers.names()                        # -> ("stepwise",)
"stepwise" in teachers                  # -> True
```

You will rarely create a `Registry` yourself. Each area provides its own
registries, such as `ssapling.teachers.registry`, and `ssapling.registries`
collects them in one place.

### How a downstream package registers its components

Registration is explicit. A downstream package has one function that
registers everything, and the application calls it at start-up:

```python
# ssap_arithmetic/components.py
def register_components():
    teachers.register("stepwise", StepwiseTeacher)
    models.register("tiny", TinyModel)
```

SSAPling never imports downstream packages and has no automatic plugin
discovery.

### Rules

| Situation | Result |
| --- | --- |
| `get` a name that is not registered | `UnknownComponentError`, with close matches and the list of registered names. |
| `register` a name already taken by a **different** implementation | `DuplicateRegistrationError`. The original stays registered. |
| `register` a taken name with the same implementation but a **different settings class** | `DuplicateRegistrationError`, so a real conflict is never hidden. |
| Either of the above with `replace=True` | The new registration replaces the old one. |
| `register` the **same** implementation and settings class under the same name again | Nothing happens, so `register_components()` can safely run twice. The first registration, including its description, is kept. |
| One implementation under several names | Allowed (aliases). |
| A name that is not a string, is empty, or has surrounding whitespace | `TypeError` / `ValueError`. |

### Settings classes in the registry

A registry also records which settings class each implementation accepts
(see [Settings](#settings)). By default it reads the implementation's
`Settings` attribute; `settings=` overrides it, which is also how a function
gets one. The settings class is checked when it is registered, so a mistake in
a downstream settings declaration shows up at start-up.

```python
class StepwiseTeacher:
    Settings = StepwiseTeacherSettings

teachers.register("stepwise", StepwiseTeacher)
teachers.entry("stepwise")   # -> Registration(name="stepwise",
                             #      implementation=StepwiseTeacher,
                             #      settings=StepwiseTeacherSettings,
                             #      description="Reviews each step ...")
```

The **description** is the first paragraph of the implementation's docstring,
unless `description=` is passed. The [catalogue](#catalogue) shows it.

### Error messages

A misspelled name:

```text
UnknownComponentError: Unknown teacher 'stepwize'. Did you mean 'stepwise'?
Registered teacher names: 'scripted' and 'stepwise'.
```

A name looked up before anything was registered. This is usually a missing
`register_components()` call:

```text
UnknownComponentError: Unknown teacher 'stepwise': no teacher has been registered.
Components from a downstream package are available only after that package
registers them, for example by calling its register_components().
```

A clash:

```text
DuplicateRegistrationError: A teacher named 'stepwise' is already registered
(ssap_arithmetic.teachers.StepwiseTeacher). Registering ssap_arc.teachers.Other
under the same name would replace it; pass replace=True if that is intended.
```

When an implementation has a settings class, the message names it too.

The errors also carry their details as attributes, so code can use them
without parsing the message: `kind`, `name`, `available`, `suggestions` and
`location` on `UnknownComponentError`, and `kind`, `name`, `existing` and
`new` on `DuplicateRegistrationError`. `location` says where the name was
written, such as `run.yaml: teacher.name`; `create_from` and `validate_from`
set it.

### Errors in general

Every SSAPling error derives from `ssapling.SSAPlingError`, so
`except SSAPlingError` catches them all. Each also derives from the closest
built-in exception: `UnknownComponentError` is a `LookupError`, and
`DuplicateRegistrationError` is a `ValueError`.

Every SSAPling error can be **pickled**, with its message, attributes and
notes, so an error raised in a worker process reaches the caller as itself.

### Typing

`Registry` is generic in what it holds, so a type checker knows what `get`
returns:

```python
teachers: Registry[type[Teacher]] = Registry("teacher")
teacher_class = teachers.get("stepwise")   # type: type[Teacher]
```

## Settings

A component declares the settings it accepts as a standard-library
**dataclass**. Each field is one setting: the annotation is its type, the
default is its default, and `setting()` adds a description and constraints.

```python
from dataclasses import dataclass
from ssapling.core import setting

@dataclass(frozen=True)
class StepwiseTeacherSettings:
    """Settings for a teacher that reviews each step."""

    explain: bool = setting(False, description="Explain each correction.")
    max_interventions: int = setting(10, ge=0, description="Interventions per episode.")
```

Prefer `frozen=True`, so that settings cannot change after they are checked.
Plain fields (`explain: bool = False`) work too; `setting()` is needed only for
a description or constraints.

### Checking supplied values

`validate_settings` turns supplied values into a checked instance. The values
come from Python or from a [file](#loading-configuration-files), and they get the same checks
either way:

```python
validate_settings(StepwiseTeacherSettings, {"max_interventions": 5})
# -> StepwiseTeacherSettings(explain=False, max_interventions=5)

validate_settings(StepwiseTeacherSettings, {"explain": "yes", "max_interventions": -1, "colour": 1})
```

```text
ConfigError: Invalid settings for StepwiseTeacherSettings:
  - colour: unknown setting (known: explain, max_interventions)
  - explain: expected bool, got str 'yes'
  - max_interventions: must be >= 0, got -1
```

What is checked, in order:

1. **Unknown settings**: names the class does not declare, with a "did you
   mean" suggestion when one is close.
2. **Missing required settings**: fields with no default.
3. **Types**: no silent conversion. `"3"` is not an int, `1` is not a bool, and
   `True` is not an int. The only conversions are harmless ones: an int is
   accepted for a float, and a list for a tuple (YAML has no tuples).
4. **Constraints** declared with `setting()`.
5. **The class's own `validate()` method**, if it has one, for rules involving
   several settings. It runs only once 1–4 pass, and it raises `ValueError`
   to report a problem:

   ```python
   def validate(self) -> None:
       if self.mode == "weighted" and not self.weights:
           raise ValueError("mode 'weighted' needs at least one entry in weights")
   ```

Every problem from steps 1–4 is collected and reported together, each with its
path (`schedule.warmup`, `sizes[1]`, `weights.x`). Omitted settings take their
default. An existing instance can be passed instead of a mapping, and it is
checked again.

### Supported types

| Annotation | Accepts | JSON Schema |
| --- | --- | --- |
| `bool`, `int`, `float`, `str` | That type (an int also for `float`) | `boolean`, `integer`, `number`, `string` |
| `None` | `None` | `null` |
| `Literal["a", "b"]` | Exactly those values (str, int, bool, or None) | `enum` |
| `int \| None`, unions | Any of the options | `anyOf` |
| `list[T]` | A list or tuple of `T` | `array` |
| `tuple[T, ...]`, `tuple[A, B]` | Any length of `T`; exactly an `A` then a `B` | `array` (`prefixItems` for fixed) |
| `dict[str, T]` | A mapping with string keys | `object` |
| Another settings dataclass | A mapping of its settings (nested) | Its own object schema |
| `Any` | Any **plain** value: None, bool, int, float, str, and lists and string-keyed dicts of these | `{}` |

Anything else (`set`, `object`, your own classes, bare `list`) is rejected when
the class is checked. Settings hold plain values that can be saved and
reloaded. They never hold live objects such as an environment or a model,
which are passed separately when a component is [created](#construction).

### Constraints

| `setting(...)` | Applies to | Meaning |
| --- | --- | --- |
| `ge=`, `gt=`, `le=`, `lt=` | numbers | `>=`, `>`, `<=`, `<` |
| `min_length=`, `max_length=` | str, list, tuple, dict | Length bounds |
| `pattern=` | str | A regular expression the whole string must match |

Constraints apply to the setting's own value, not to the items inside it, and
are skipped when the value is `None`. For anything more, use `validate()`.

### Checking the declaration

`check_settings_class(cls)` checks the class itself and reports every problem
in a `SettingsDeclarationError`:

- unsupported types;
- constraints that don't fit their type (`ge=` on a string);
- invalid patterns;
- defaults that break their own type or constraints.

Registries run this check when a component is registered.

### Converting back, and JSON Schema

`settings_to_dict(instance)` gives plain values (nested settings become dicts,
and tuples become lists) that can be written as JSON or YAML.
`validate_settings` turns them back into an equal instance. Train uses this
to save the exact settings of a run.

`settings_schema(cls)` describes the class as JSON Schema (draft 2020-12):
types, descriptions, defaults, constraints, required settings, and no unknown
settings. The [catalogue](#catalogue) uses it, so a training run's
controller can be offered the stages and their settings (see
[Train: controllers](train.md#controllers)). The schema cannot express
`validate()` rules, so a value that matches the schema can still fail
`validate_settings`.

### Errors

| Error | When | Fix it in |
| --- | --- | --- |
| `ConfigError` (also a `ValueError`) | Supplied values are invalid. `.problems` lists them. | The values (the YAML file, the Python call) |
| `SettingsDeclarationError` (also a `TypeError`) | A settings class can't be checked. `.problems` lists why. | The settings class |

Each problem is a `Problem(path, message)`.

## Loading configuration files

`load_config(path)` reads a YAML (`.yaml`, `.yml`) or JSON (`.json`) file into
plain Python values. The top level must be a mapping. The result is a
`Config`: an ordinary `dict` that also remembers its file, so later errors can
name it.

```yaml
# run.yaml
teacher:
  name: stepwise
  settings:
    explain: true
    max_interventions: 5
```

```python
config = load_config("run.yaml")
config["teacher"]["settings"]["explain"]   # -> True: plain dicts all the way down
config.source                              # -> "run.yaml"
```

YAML needs the optional extra, `pip install "ssapling[yaml]"`. Without it,
loading a YAML file raises `MissingDependencyError`, which names the extra to
install. JSON needs nothing extra.

### What loading checks

| Problem | Result |
| --- | --- |
| Syntax error | `ConfigError` with the line and column. |
| The same key twice in one mapping | `ConfigError`: for YAML, naming both lines; for JSON, naming the key. YAML and JSON parsers normally keep the last value silently. |
| Top level is not a mapping, or the file is empty | `ConfigError`. |
| Unsupported suffix | `ConfigError`. |
| YAML tags that build Python objects (`!!python/...`) | `ConfigError`. Only plain data is read. |

**Exponent numbers.** PyYAML follows YAML 1.1, which reads `1e-4` and
`1.0e5` as *strings*, a common surprise with learning rates. SSAPling reads
them as floats, as JSON and YAML 1.2 do. Quote a value (`"1e-4"`) to keep a
string.

**Merge keys** (`<<: *defaults`) still work. Keys written after a merge
override the merged ones, and that is not counted as a duplicate.

What a whole file contains (which components, sweeps, overrides, includes,
presets) is up to the downstream project. Core reads files and single
component entries.

### Naming code

A file can name code as `MODULE:NAME`, such as a task class or a factory.
`load_named(spec, what)` imports the module and returns its attribute. It
searches the current directory too (putting it on `sys.path` if it isn't
there), so a command run from a project's root finds the project's own
modules. `what` says what is named, for the errors:

```python
factory = load_named("experiments.paper.run:make_run", "run function factory")
load_named("make_run", "run function factory")
# ValueError: give the run function factory as MODULE:NAME, such as package.module:Name: 'make_run'
```

### Component entries

A component entry is SSAPling's format for "this registered component, with
these settings": a mapping with `name` and, optionally, `settings`.
`ComponentSpec` holds one:

```python
spec = config.component("teacher")          # or a dotted path: "components.teacher"
spec.name, spec.settings                    # -> "stepwise", {"explain": True, ...}
spec.source, spec.path                      # -> "run.yaml", "teacher"

ComponentSpec("tiny", {"layers": 2})        # built directly in Python
spec.to_mapping()                           # -> {"name": ..., "settings": ...}, ready to save
```

Reading an entry checks its layout only: a non-empty `name`, `settings` (if
present) a mapping, and no other keys. The settings themselves are checked
when the component is validated or created. Layout errors name the file and
the path:

```text
ConfigError: Invalid component entry in run.yaml:
  - teacher.nmae: unknown key in a component entry; did you mean 'name'? (allowed: name, settings)
  - teacher.name: missing component name
```

## Construction

`create` validates the settings, then calls the implementation with them as
`config=`, plus the **live objects** passed as keyword arguments:

```python
teacher = teachers.create("stepwise", {"explain": True}, task=task)
# is the same as
teacher = StepwiseTeacher(config=StepwiseTeacherSettings(explain=True), task=task)
```

- **Settings** are plain values, checked by [Settings](#settings). They can
  be a mapping, an instance of the settings class, or omitted for all
  defaults.
- **Live objects** are things like a task, an environment, or a model: objects
  that are not settings. Which ones a component needs is part of its area's
  contract (for example "a teacher receives the task"). Core passes them on
  unchecked.
- A component with **no settings class** is called with the live objects
  only. Giving it settings is an error, though an empty mapping is fine.
- `config` is reserved for the settings, so it cannot be a live object.
- If the implementation itself raises, the error is passed on with a note
  naming the component: `While creating teacher 'stepwise' (pkg.StepwiseTeacher).`
- An implementation can be a class or a function that returns the component.
  A type checker knows what `create` returns: `Registry[type[StepwiseTeacher]]`
  gives a `StepwiseTeacher`.

`validate(name, settings)` does the checking without creating anything. It
returns the validated settings instance, or `None` for a component without
settings. Use it to check a whole configuration before building any of it.

### From a file

`create_from(spec, **live)` and `validate_from(spec)` do the same for a
component entry. Errors then name the file and the full path to the bad
value:

```text
ConfigError: Invalid settings for teacher 'stepwise' in run.yaml:
  - components.teacher.settings.max_interventions: must be >= 0, got -1

UnknownComponentError: run.yaml: teacher.name: Unknown teacher 'stepwize'.
Did you mean 'stepwise'? Registered teacher names: 'stepwise'.
```

Construction never decides *which* components to create or how an experiment
runs. That is the caller's job, and [Train's](train.md).

## Catalogue

`registry.catalogue()` describes every registered component as plain,
JSON-ready data, sorted by name:

```python
teachers.catalogue()
# [{"name": "stepwise",
#   "description": "Reviews each step of the student's work.",
#   "settings": {...JSON Schema of StepwiseTeacherSettings...}}]
```

A component without settings gets the schema of an empty object, which
accepts only `{}`. The catalogue is for documentation, and for offering
components to a procedure controller, for example as tool descriptions for a
language model (see [Train: controllers](train.md#controllers)).

## Execution helper

The teacher runner, the environment runner, and the interaction loop all
need to call component methods that might be
synchronous (`def`) or asynchronous (`async def`). They share one helper,
`ssapling.core.Execution`. Downstream code may use it too, for example a stage
that calls a model method that might be `async def`. Users of the runners
choose a placement through their `placement=` argument, or the episode loop's
`threads` setting.

```python
await Execution().call(env.step, actions)                 # sync: runs in the loop's thread
await Execution("thread").call(env.step, actions)         # sync: runs in a worker thread
await Execution("thread", single_thread_pool).call(...)   # always the same worker thread
await Execution().call(teacher.respond, request)          # async def: simply awaited
```

| Call | `"direct"` | `"thread"` |
| --- | --- | --- |
| `def` method | Runs in the event loop's thread. Blocks the loop until it returns, and a timeout cannot interrupt it. | Runs in a worker thread. The loop keeps going, and a timeout can stop *waiting* (the call itself still finishes in its thread). |
| `async def` method | Awaited in the loop. | The same: awaited in the loop. |

Some components must always be called from the same thread (a GUI, some
simulators, some GPU contexts). Pass a `ThreadPoolExecutor(max_workers=1)` for
those. The default thread pool may use a different thread for each call.

Timeouts are handled here, once for every caller:

```python
await Execution("thread").call_with_timeout(5.0, env.step, actions)
```

When the timeout passes first, it raises the private `DeadlineExpiredError`,
never `TimeoutError`, so a `TimeoutError` the component raises itself is not
mistaken for it. Each caller then decides what an expired deadline means: the
teacher runner records a `cancelled` response, and the environment runner
raises `EnvironmentCommunicationError`. `check_timeout` rejects a timeout
that is not a positive number, or one that could never interrupt the call (a
`def` method run directly). A caller that turns the component's failures into
results, as the teacher runner does, calls it first, so a mistake in the
timeout is still reported as the caller's.

Running a component in another *process* needs a long-lived worker that keeps
the component's state between calls. That is `ssapling.interaction.in_process`
(see [Interaction](interaction.md#running-a-component-in-another-process)),
whose stand-in is called like any other component. Process pools are rejected
here, because they would lose the state.

## Declaration helpers (internal)

Every area lets a component declare what it offers in the same few ways, and `ssapling.core._declarations` reads those
declarations once for all of them, with the same rules and errors. Users
never import it.

| Helper | Used for |
| --- | --- |
| `has_feature` | A feature declared by having methods, which the component's own `supports` may switch off but never on: task features, environment capabilities. |
| `names` | A declared list of names, where a single string is rejected: teacher operations and requirements, model learning operations and signals, environment participants. |
| `takes` | Whether a method accepts a keyword: a model's `key`, an environment's `seed` and `options`. |
| `keyed` | A mapping with string keys, copied when a result is made: teacher requests and responses, environment results. |
| `name_of` | A component's class name, for error messages. |
| `check_identifier` | An id or name that must be a non-empty string: instance, record, request and collection ids, provenance names. |
| `check_int` | An integer that is not a bool, with an optional minimum: seeds, step indexes, positions, sizes, batch sizes. |

`ssapling.core._plain.plain_copy` checks and copies **plain values**
(`None`, booleans, numbers, strings, lists, and mappings with string keys),
turning tuples into lists and reporting every part that is not plain with its
path. Settings use it for `Any`, provenance for its settings and
parameters, and the run context for published values, so all of them accept
exactly the same values.
