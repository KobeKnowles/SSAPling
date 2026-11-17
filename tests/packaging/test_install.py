"""The packaging commitments in docs/overview.md: no required dependencies, stdlib-only imports."""

import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import ssapling

ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = Path(__file__).with_name("check_install.py")


def test_base_install_has_no_required_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == []


def test_yaml_is_an_optional_extra():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert set(project["optional-dependencies"]) == {"yaml"}


def test_every_area_imports_using_only_the_standard_library():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_version_is_exposed_and_matches_the_installed_package():
    assert ssapling.__version__ == version("ssapling")
