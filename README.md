# SSAPling

SSAPling is a Python library for the System–Specification Alignment Problem.

It provides general infrastructure, interfaces, and scaffolding for downstream
projects, such as one that teaches arithmetic or one on ARC. It defines how
environments, tasks, teachers, and models plug together, and it runs what is
common to every project: building components from settings, running episodes,
storing records, training procedures, and assessments. Concrete environments,
tasks, teachers, models, and learning methods live in downstream projects.

> **Status:** alpha. Every area is built and tested. Before 1.0, a minor
> release may change the public interface, and every such change is listed in
> the [changelog](https://github.com/KobeKnowles/SSAPling/blob/main/CHANGELOG.md).

## Installation

Requires Python 3.12 or newer. The base install has no third-party dependencies.

```bash
pip install ssapling
pip install "ssapling[yaml]"    # adds YAML configuration files
```

To work on SSAPling and a downstream project together, install it from a clone
in editable mode, so changes to SSAPling show up at once:

```bash
uv add --editable ../SSAPling   # or: pip install -e "../SSAPling[yaml]"
```

## Quick start

[`examples/arithmetic/`](https://github.com/KobeKnowles/SSAPling/blob/main/examples/arithmetic/)
is a complete small project: a learner that knows no arithmetic learns addition
from a tutor, and is assessed on problems it never trained on. From a clone:

```bash
uv sync
uv run python examples/arithmetic/run.py
```

[Using SSAPling](https://github.com/KobeKnowles/SSAPling/blob/main/docs/usage.md)
walks through it part by part.

## Documentation

- [Overview](https://github.com/KobeKnowles/SSAPling/blob/main/docs/overview.md):
  the package areas, their import rules, the packaging commitments, and a
  glossary.
- [Using SSAPling](https://github.com/KobeKnowles/SSAPling/blob/main/docs/usage.md):
  a whole project, from components to an assessed training run.
- [Extending SSAPling](https://github.com/KobeKnowles/SSAPling/blob/main/docs/extending.md):
  what each kind of component needs, and how to change SSAPling's defaults.

The area guides:

- [Core](https://github.com/KobeKnowles/SSAPling/blob/main/docs/core.md):
  registries, settings, configuration files, and building components.
- [Tasks](https://github.com/KobeKnowles/SSAPling/blob/main/docs/tasks.md):
  instances, generated and enumerated tasks, scoring, partial checks, and
  teaching resources.
- [Teachers](https://github.com/KobeKnowles/SSAPling/blob/main/docs/teachers.md):
  requests and responses, the teacher contract, the runner, and checking a
  teacher's requirements.
- [Models](https://github.com/KobeKnowles/SSAPling/blob/main/docs/models.md):
  acting (act-only or buffered), participants and lanes, state, learning
  operations, learning signals, and descriptions.
- [Environments](https://github.com/KobeKnowles/SSAPling/blob/main/docs/environments.md):
  reset and step results, turn order, capabilities, the runner, and how to
  write an adapter.
- [Data](https://github.com/KobeKnowles/SSAPling/blob/main/docs/data.md):
  records, sources, samplers, collections that never share a problem,
  provenance, storage, codecs, and batching.
- [Interaction](https://github.com/KobeKnowles/SSAPling/blob/main/docs/interaction.md):
  the episode loop, timing, lanes, static solving, and running a component in
  another process.
- [Evaluate](https://github.com/KobeKnowles/SSAPling/blob/main/docs/evaluate.md):
  assessments, measurements, aggregation, and results.
- [Train](https://github.com/KobeKnowles/SSAPling/blob/main/docs/train.md):
  procedures, stages, run files, checkpoints and resuming, and controllers.

[Development](https://github.com/KobeKnowles/SSAPling/blob/main/docs/development.md)
covers working on SSAPling itself.

## License

Apache License 2.0. See
[LICENSE](https://github.com/KobeKnowles/SSAPling/blob/main/LICENSE).
