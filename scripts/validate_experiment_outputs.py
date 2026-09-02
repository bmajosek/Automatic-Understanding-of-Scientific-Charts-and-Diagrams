"""Validate saved prediction CSVs against a task target subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from output_validation import validate_prediction_outputs  # noqa: E402
from pipelines.tasks import TASKS, load_task_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qa", choices=sorted(TASKS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument(
        "--output-dir", default="results/experiments/existing_outputs_audit"
    )
    args = parser.parse_args()

    rows, _ = load_task_rows(
        args.task, ROOT / args.data_dir, args.split, args.limit,
    )
    summary = validate_prediction_outputs(
        args.task,
        rows,
        ROOT / args.predictions_dir,
        ROOT / args.output_dir,
    )
    print(summary.to_string(index=False))
    print(f"\nValidation report: {(ROOT / args.output_dir / 'STATUS.md').resolve()}")


if __name__ == "__main__":
    main()
