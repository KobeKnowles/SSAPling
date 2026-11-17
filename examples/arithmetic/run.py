"""Train the fact learner on addition, then assess it once on the held-out test problems.

Run from the repository root:

    uv run python examples/arithmetic/run.py                     # writes runs/arithmetic
    uv run python examples/arithmetic/run.py --output runs/again
    uv run python examples/arithmetic/run.py --resume            # continue from the last checkpoint
"""

import argparse
import asyncio
from pathlib import Path

from components import entered_number, register_components

from ssapling.evaluate import save_results
from ssapling.train import Runner

RUN_FILE = Path(__file__).with_name("run.yaml")


async def main(output: str, resume: bool) -> None:
    register_components()
    given = {"outcome": entered_number}  # built in Python, not from the run file
    if resume:
        runner = Runner.resume(output, components=given)
    else:
        runner = Runner.from_config(RUN_FILE, seed=7, output=output, components=given)
    context = await runner.run()
    print(f"trained for {context.counters['episodes']} episodes")
    print(f"validation accuracy: {context.read('monitoring.accuracy')}")

    final = context.components["final"]  # the held-out problems, assessed only now
    results = await final.run(
        context.components["model"],
        run_id=context.output_name("final"),  # "resume-1/final" if resumed: nothing is overwritten
        seed=7,
        model_info=dict(context.counters),
    )
    save_results(context.components["store"], results)
    print(f"test accuracy: {results.aggregates['accuracy'].value}")
    for digits, summary in sorted(results.aggregates["by_digits"].groups.items()):
        print(f"  {digits}-digit problems: {summary.value} ({summary.contributed} problems)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="runs/arithmetic", help="where the run is written")
    parser.add_argument("--resume", action="store_true", help="continue the run at --output")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.output, arguments.resume))
