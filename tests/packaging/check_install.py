"""Check an installed ssapling: every area imports using only the standard library.

Run as a script in a fresh interpreter. The packaging tests run it against the
development install; CI runs it against the built wheel installed into a clean
environment with no extras. It is standalone and imports nothing from tests/.
"""

import importlib
import importlib.resources
import importlib.util
import pkgutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    before = set(sys.modules)

    import ssapling

    modules = list(pkgutil.iter_modules(ssapling.__path__))
    areas = [info.name for info in modules if info.ispkg]
    for info in modules:
        importlib.import_module(f"ssapling.{info.name}")

    allowed_roots = set(sys.stdlib_module_names) | {"ssapling"}
    loaded = set(sys.modules) - before
    non_stdlib = sorted(name for name in loaded if name.partition(".")[0] not in allowed_roots)

    problems = []
    if non_stdlib:
        problems.append(f"importing ssapling loaded non-stdlib modules: {non_stdlib}")
    if not importlib.resources.files("ssapling").joinpath("py.typed").is_file():
        problems.append("py.typed is missing from the installed package")
    if importlib.util.find_spec("yaml") is None:
        problems += check_yaml_needs_the_extra()

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"OK: ssapling {ssapling.__version__}, areas: {', '.join(areas)}")
    return 0


def check_yaml_needs_the_extra() -> list[str]:
    """Without PyYAML, loading a YAML file must fail with a message naming the extra."""
    from ssapling.core import MissingDependencyError, load_config

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.yaml"
        path.write_text("teacher: {name: stepwise}\n", encoding="utf-8")
        try:
            load_config(path)
        except MissingDependencyError as error:
            if "ssapling[yaml]" in str(error):
                return []
            return [f"the missing-PyYAML message does not name the extra: {error}"]
    return ["loading YAML without PyYAML did not raise MissingDependencyError"]


if __name__ == "__main__":
    sys.exit(main())
