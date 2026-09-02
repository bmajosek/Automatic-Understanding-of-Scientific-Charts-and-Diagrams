"""Run every supported experiment task with shared checkpoints and Gemini cooldown."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ("qa", "verification", "summarization", "table_extraction")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all chart-understanding tasks in one command."
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_TASKS,
        default=list(DEFAULT_TASKS),
        help="Tasks to run. For the review cohort, use qa verification table_extraction.",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--experiment-dir",
        default="results/experiments/all_tasks_1000",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gemini-cooldown-minutes", type=int, default=60)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--retry-gemini", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = ROOT / experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for task in args.tasks:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiments.py"),
            "--task",
            task,
            "--data-dir",
            args.data_dir,
            "--split",
            args.split,
            "--config",
            args.config,
            "--model-name",
            "all",
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
            "--experiment-dir",
            str(experiment_dir),
            "--device",
            args.device,
            "--gemini-cooldown-minutes",
            str(args.gemini_cooldown_minutes),
        ]
        if args.skip_gemini:
            command.append("--skip-gemini")
        if args.retry_gemini and task == args.tasks[0]:
            command.append("--retry-gemini")
        if args.retry_errors:
            command.append("--retry-errors")
        if args.verbose:
            command.append("--verbose")

        print("\n" + "#" * 88)
        print(f"Running task: {task}")
        print("#" * 88)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            failures.append((task, completed.returncode))

    print(f"\nAll-task experiment directory: {experiment_dir.resolve()}")
    if failures:
        details = ", ".join(f"{task} (exit {code})" for task, code in failures)
        raise SystemExit(f"Some tasks failed: {details}")
    print("All supported tasks finished or checkpointed successfully.")


if __name__ == "__main__":
    main()
