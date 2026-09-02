"""Resume Gemini-based QA, verification, and summarization experiments.

The script preserves existing prediction CSV files.  It asks the ordinary
experiment runner to fill only missing records and to retry cached failures.
Run it again after a quota reset if Gemini pauses the experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.common import is_prediction_failure, safe_str  # noqa: E402
from pipelines.runner import (  # noqa: E402
    clear_gemini_quota_marker,
    gemini_quota_blocked,
)
from pipelines.tasks import TASK_SPEC, load_task_rows  # noqa: E402


GEMINI_MODELS = (
    "gemini_end_to_end",
    "deplot_table_gemini_pipeline",
    "ocr_gemini_reasoning_pipeline",
)


@dataclass(frozen=True)
class PredictionStatus:
    successful: int
    failed: int
    missing: int


@dataclass(frozen=True)
class TaskTarget:
    task: str
    ids: list[str]
    id_column: str
    output_file: str

    @property
    def size(self) -> int:
        return len(self.ids)


def _project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _task_target(
    task: str,
    data_dir: Path,
    split: str,
    requested_size: int | None,
) -> TaskTarget:
    rows, output_file = load_task_rows(task, data_dir, split, requested_size)
    if requested_size is not None and len(rows) != requested_size:
        raise SystemExit(
            f"The selected split contains {len(rows)} {task} rows, not the "
            f"requested {requested_size}."
        )

    id_column = TASK_SPEC[task]["id_col"]
    ids = rows[id_column].map(safe_str).tolist()
    if any(not record_id for record_id in ids):
        raise SystemExit(f"The common {task} cohort contains an empty {id_column}.")
    if len(set(ids)) != len(ids):
        raise SystemExit(
            f"The common {task} cohort contains duplicate {id_column} values."
        )
    return TaskTarget(task, ids, id_column, output_file)


def prediction_status(
    prediction_path: Path,
    target_ids: list[str],
    id_column: str = "question_id",
) -> PredictionStatus:
    """Count the latest successful, failed, and missing target predictions."""
    target = set(target_ids)
    if not prediction_path.exists():
        return PredictionStatus(successful=0, failed=0, missing=len(target))

    frame = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
    required = {id_column, "error_type"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        joined = ", ".join(sorted(missing_columns))
        raise SystemExit(f"{prediction_path} is missing required columns: {joined}")

    frame[id_column] = frame[id_column].map(safe_str)
    frame = frame[frame[id_column].isin(target)]
    frame = frame.drop_duplicates(subset=id_column, keep="last")
    failures = frame["error_type"].map(is_prediction_failure)
    failed = int(failures.sum())
    successful = int((~failures).sum())
    return PredictionStatus(
        successful=successful,
        failed=failed,
        missing=len(target) - len(frame),
    )


def _print_statuses(
    predictions_dir: Path,
    target: TaskTarget,
) -> dict[str, PredictionStatus]:
    statuses: dict[str, PredictionStatus] = {}
    print(f"\nGemini {target.task} status on the common target ({target.size} rows):")
    for model_name in GEMINI_MODELS:
        status = prediction_status(
            predictions_dir / model_name / target.output_file,
            target.ids,
            target.id_column,
        )
        statuses[model_name] = status
        print(
            f"  {model_name}: successful={status.successful}/{target.size}, "
            f"failed={status.failed}, missing={status.missing}"
        )
    return statuses


def _print_all_statuses(
    predictions_dir: Path,
    targets: list[TaskTarget],
) -> dict[str, dict[str, PredictionStatus]]:
    return {
        target.task: _print_statuses(predictions_dir, target)
        for target in targets
    }


def _selected_pairs(
    statuses: dict[str, dict[str, PredictionStatus]],
    targets: list[TaskTarget],
    only_below: int | None,
) -> set[tuple[str, str]]:
    """Select task/model pairs once, before any predictions are resumed."""
    pairs = {
        (target.task, model_name)
        for target in targets
        for model_name in GEMINI_MODELS
    }
    if only_below is None:
        return pairs
    return {
        pair
        for pair in pairs
        if statuses[pair[0]][pair[1]].successful < only_below
    }


def _runner_command(
    task: str,
    model_name: str,
    data_dir: Path,
    raw_data_dir: Path,
    experiment_dir: Path,
    split: str,
    target: int,
    cooldown_minutes: int,
    device: str,
    verbose: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--task",
        task,
        "--model-name",
        model_name,
        "--data-dir",
        str(data_dir),
        "--raw-data-dir",
        str(raw_data_dir),
        "--experiment-dir",
        str(experiment_dir),
        "--split",
        split,
        "--limit",
        str(target),
        "--batch-size",
        "0",
        "--retry-errors",
        "--gemini-cooldown-minutes",
        str(cooldown_minutes),
        "--device",
        device,
    ]
    if verbose:
        command.append("--verbose")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume Gemini end-to-end, DePlot+Gemini, and OCR+Gemini for "
            "QA, verification, and summarization on task-specific cohorts."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default="results/experiments/review_test_stratified_1000",
    )
    parser.add_argument("--data-dir", default="data/processed_review")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "qa-verification", "qa", "verification", "summarization"],
        help=(
            "Run every Gemini task by default, QA and verification together, "
            "or one selected task."
        ),
    )
    parser.add_argument("--qa-target", type=int, default=500)
    parser.add_argument(
        "--verification-target",
        type=int,
        default=0,
        help="Verification rows to use (0 = the entire split, currently 450).",
    )
    parser.add_argument(
        "--summarization-target",
        type=int,
        default=0,
        help="Summarization rows to use (0 = the entire split, currently 819).",
    )
    parser.add_argument(
        "--retry-gemini",
        action="store_true",
        help="Clear the saved Gemini quota cooldown before resuming.",
    )
    parser.add_argument("--gemini-cooldown-minutes", type=int, default=60)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-passes",
        type=int,
        default=2,
        help="Maximum resume attempts per approach for non-quota transient errors.",
    )
    parser.add_argument(
        "--only-below",
        type=int,
        default=None,
        metavar="N",
        help=(
            "At startup, run only task/model pairs with fewer than N successful "
            "predictions. The selected pairs still resume towards their full "
            "task target; other checkpoints are left untouched."
        ),
    )
    args = parser.parse_args()

    if args.qa_target <= 0:
        raise SystemExit("--qa-target must be positive.")
    if args.verification_target < 0:
        raise SystemExit("--verification-target cannot be negative.")
    if args.summarization_target < 0:
        raise SystemExit("--summarization-target cannot be negative.")
    if args.max_passes <= 0:
        raise SystemExit("--max-passes must be positive.")
    if args.only_below is not None and args.only_below <= 0:
        raise SystemExit("--only-below must be positive.")

    data_dir = _project_path(args.data_dir)
    raw_data_dir = _project_path(args.raw_data_dir)
    experiment_dir = _project_path(args.experiment_dir)
    predictions_dir = experiment_dir / "predictions"
    if args.task == "all":
        requested_tasks = ["qa", "verification", "summarization"]
    elif args.task == "qa-verification":
        requested_tasks = ["qa", "verification"]
    else:
        requested_tasks = [args.task]
    requested_sizes = {
        "qa": args.qa_target,
        "verification": args.verification_target or None,
        "summarization": args.summarization_target or None,
    }
    targets = [
        _task_target(task, data_dir, args.split, requested_sizes[task])
        for task in requested_tasks
    ]

    if args.retry_gemini:
        clear_gemini_quota_marker(predictions_dir)

    statuses = _print_all_statuses(predictions_dir, targets)
    selected_pairs = _selected_pairs(statuses, targets, args.only_below)
    if args.only_below is not None:
        print(
            f"\nSelected Gemini task/model pairs below "
            f"{args.only_below} successful predictions:"
        )
        for target in targets:
            for model_name in GEMINI_MODELS:
                pair = (target.task, model_name)
                status = statuses[target.task][model_name]
                decision = "RUN" if pair in selected_pairs else "SKIP"
                print(
                    f"  [{decision}] {target.task}/{model_name}: "
                    f"successful={status.successful}/{target.size}"
                )

    if not selected_pairs:
        print("\nNo requested Gemini task/model pair is below the threshold.")
        return

    if all(
        statuses[target.task][model_name].successful == target.size
        for target in targets
        for model_name in GEMINI_MODELS
        if (target.task, model_name) in selected_pairs
    ):
        print("\nAll selected Gemini experiments already have complete cohorts.")
        return

    if gemini_quota_blocked(predictions_dir):
        raise SystemExit(
            "\nGemini cooldown is active. After the API quota resets, rerun this "
            "script with --retry-gemini. Existing results will be preserved."
        )

    for target in targets:
        for model_name in GEMINI_MODELS:
            if (target.task, model_name) not in selected_pairs:
                continue
            status = statuses[target.task][model_name]
            if status.successful == target.size:
                continue

            for pass_number in range(1, args.max_passes + 1):
                before = status.successful
                print(
                    f"\nResuming {target.task}/{model_name} "
                    f"(pass {pass_number}/{args.max_passes}; "
                    f"need {target.size - before} successful predictions)..."
                )
                result = subprocess.run(
                    _runner_command(
                        task=target.task,
                        model_name=model_name,
                        data_dir=data_dir,
                        raw_data_dir=raw_data_dir,
                        experiment_dir=experiment_dir,
                        split=args.split,
                        target=target.size,
                        cooldown_minutes=args.gemini_cooldown_minutes,
                        device=args.device,
                        verbose=args.verbose,
                    ),
                    cwd=ROOT,
                    check=False,
                )
                if result.returncode:
                    raise SystemExit(
                        f"The experiment runner stopped for {target.task}/"
                        f"{model_name} with exit code {result.returncode}. Saved "
                        "predictions were not removed."
                    )

                status = prediction_status(
                    predictions_dir / model_name / target.output_file,
                    target.ids,
                    target.id_column,
                )
                statuses[target.task][model_name] = status
                print(
                    f"Current {target.task}/{model_name}: "
                    f"successful={status.successful}/{target.size}, "
                    f"failed={status.failed}, missing={status.missing}"
                )

                if status.successful == target.size:
                    break
                if gemini_quota_blocked(predictions_dir):
                    _print_all_statuses(predictions_dir, targets)
                    raise SystemExit(
                        "\nGemini quota paused the run. Rerun later with "
                        "--retry-gemini; the script will continue from these "
                        "checkpoints."
                    )
                if status.successful <= before:
                    print(
                        "No additional successful prediction was produced in this "
                        "pass; stopping repeated retries for this approach."
                    )
                    break

    final_statuses = _print_all_statuses(predictions_dir, targets)
    incomplete = [
        (target.task, model_name)
        for target in targets
        for model_name in GEMINI_MODELS
        if (target.task, model_name) in selected_pairs
        if final_statuses[target.task][model_name].successful != target.size
    ]
    if incomplete:
        raise SystemExit(
            "\nThe run ended before every requested task and approach was complete. "
            "Inspect failed rows, then rerun the same command."
        )

    print(
        "\nDone: every selected Gemini task/model pair is complete for its "
        "task-specific cohort."
    )


if __name__ == "__main__":
    main()
