"""The examples run as their READMEs say, so they and the usage guide cannot go stale."""

import subprocess
import sys
from pathlib import Path

from ssapling.data import RecordStore
from ssapling.evaluate import load_results

ROOT = Path(__file__).resolve().parents[1]


def run_example(*arguments: str) -> str:
    finished = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "arithmetic" / "run.py"), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout


def test_the_arithmetic_example_learns_and_resumes_to_the_same_result(tmp_path):
    output = str(tmp_path / "run")
    first = run_example("--output", output)
    assert "validation accuracy: 1.0" in first

    store = RecordStore(output)
    final = load_results(store, "final")
    assert final.provenance["model"]["episodes"] > 0
    assert {"run", "decisions"} <= set(store.document_names())

    # Resumed from its last checkpoint, it ends exactly where it ended before.
    again = run_example("--output", output, "--resume")
    assert again == first
    assert load_results(store, "resume-1/final").aggregates == final.aggregates
