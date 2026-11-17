# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Before 1.0, a minor release may
change the public interface; every such change is listed here.

## [Unreleased]

## [0.1.0] - 2026-11-17

The first release. Every area is built, tested and documented.

### Added

- **Core** (`ssapling.core`): registries that map names to implementations,
  with "did you mean" errors; settings declared as dataclasses, checked with
  every problem reported at once, and exported as JSON Schema; YAML (with the
  `yaml` extra) and JSON configuration files, with errors that give the file,
  line and path; building components from settings; a catalogue of every
  registered component; `Execution`, which calls sync and async methods alike;
  `load_named` for `MODULE:NAME` references.
- **Tasks** (`ssapling.tasks`): `Instance`, with inputs, reference,
  descriptors and an equivalence key; `Score`, which is never invented;
  generated and enumerated tasks; partial checks and teaching resources.
- **Teachers** (`ssapling.teachers`): requests and responses, with
  contributions by kind and agent; the teacher contract; `TeacherRunner`, for
  sync and async teachers, in place or in a thread, with timeouts.
- **Models** (`ssapling.models`): act-only and buffered models, and
  `as_buffered`; participants and lanes; state; learning operations and the
  learning signals a model declares; descriptions, with `Refusal` for a model
  that refuses one and says why; diagnostics.
- **Environments** (`ssapling.environments`): reset and step results, keyed by
  agent, with termination and truncation kept apart; turn order;
  capabilities; `EnvironmentRunner`.
- **Data** (`ssapling.data`): records of every attempt, step by step;
  `Instances`, a source of instances made in code; samplers (`uniform`, `stratified`, `generated`); collections that never
  share a problem, with held-out collections served only to assessments;
  provenance; `RecordStore`, a directory of JSON-lines files, optionally
  gzip-compressed, that never overwrites; codecs; batching, optionally in
  worker processes.
- **Interaction** (`ssapling.interaction`): `run_episode`, the episode loop,
  with an optional teacher that observes or intervenes, timing
  (`turn_based`, `deadline`, `real_time`) and lanes; `solve`, for static
  solving; `in_process`, which runs a component in its own process.
- **Evaluate** (`ssapling.evaluate`): the `standard` assessment;
  measurements; aggregation with `group_by`, `where` and equal or pooled
  weighting, where a missing value is never counted as zero; saved and
  reloaded results.
- **Train** (`ssapling.train`): procedures of stages with `once`, `every`,
  `until` and `when` triggers and a stop condition; the built-in stages
  `interact`, `learn_from_data`, `learn_operation`, `update_sampler`, `assess`
  and `checkpoint`; run files that describe a whole run; a run checked in full
  before it starts; seeds derived by name from one root seed (`derive_seed`);
  checkpoints and `Runner.resume`; controllers that decide as the run goes,
  with every decision logged and replayable (`ReplayController`).
- `ssapling.registries`: every area's registry in one place.
- Documentation: an overview with a glossary, a usage guide, an extension
  guide, and one guide per area.
- `examples/arithmetic/`: a complete small project, in which a learner learns
  addition from a tutor.

[Unreleased]: https://github.com/KobeKnowles/SSAPling/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KobeKnowles/SSAPling/releases/tag/v0.1.0
