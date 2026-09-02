"""
Run chart understanding pipelines.

Examples:
  python scripts/run_experiments.py --task qa --model-name chartocr_reasoning_pipeline --limit 10
  python scripts/run_experiments.py --task verification --model-name gemini_end_to_end --limit 5
  python scripts/run_experiments.py --task qa --model-name all --limit 10
  python scripts/run_experiments.py --task qa --full-1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

sys.path.insert(0, str(ROOT / "src"))

from config_loader import load_project_config  # noqa: E402
from output_validation import validate_prediction_outputs  # noqa: E402
from reproducibility import (  # noqa: E402
    environment_snapshot,
    prediction_hashes,
    prompt_manifest,
    sample_composition,
    sha256_file,
    stable_content_fingerprint,
    stable_id_fingerprint,
)
from pipelines.runner import (  # noqa: E402
    CHART_QA_HF_MODELS,
    IMPLEMENTED_MODELS,
    clear_gemini_quota_marker,
    gemini_quota_blocked,
    model_uses_gemini,
    run_pipeline,
)
from pipelines.tasks import MODEL_TASKS, TASKS, TASK_SPEC, load_task_rows  # noqa: E402
from pipelines.tesseract_config import resolve_tesseract  # noqa: E402


TASK_CONTENT_COLUMNS = {
    "qa": ("chart_id", "question", "answer", "answer_type"),
    "verification": (
        "chart_id", "claim", "question", "claimed_answer", "answer_type", "label",
    ),
    "summarization": ("chart_id", "summary"),
    "table_extraction": ("chart_id", "image_path"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chart experiment pipeline")
    parser.add_argument("--dataset", default="chartqa")
    parser.add_argument("--task", default="qa", choices=list(TASKS))
    parser.add_argument(
        "--model-name",
        default="all",
        help="Model name or 'all' to run all implemented models",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument(
        "--experiment-dir",
        default="",
        help=(
            "Self-contained output folder. Predictions are written below "
            "<experiment-dir>/predictions and validation files beside them."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Target subset size (default: 1000). Re-runs resume inside this target.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Maximum new samples per model in this invocation (0 = entire target).",
    )
    parser.add_argument(
        "--full-1000", action="store_true",
        help=(
            "Run the first 1000 split samples for every selected model in this one "
            "invocation; completed IDs are still skipped."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors", action=argparse.BooleanOptionalAction, default=False,
        help="Retry cached rows whose error_type is non-empty (default: disabled).",
    )
    parser.add_argument(
        "--skip-gemini", action="store_true",
        help="Run only local/non-Gemini approaches.",
    )
    parser.add_argument(
        "--retry-gemini", action="store_true",
        help="Clear the active Gemini cooldown marker and try Gemini again.",
    )
    parser.add_argument(
        "--gemini-cooldown-minutes",
        type=int,
        default=60,
        help="Pause all Gemini approaches for this many minutes after exhausted retries.",
    )
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tesseract-cmd", default="")
    parser.add_argument("--tessdata-prefix", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--deplot-max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.full_1000:
        args.limit = 1000
        args.batch_size = 1000

    experiment_dir: Path | None = None
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.is_absolute():
            experiment_dir = ROOT / experiment_dir
    elif args.full_1000:
        experiment_dir = ROOT / "results" / "experiments" / f"{args.task}_1000"

    if experiment_dir is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        args.predictions_dir = str((experiment_dir / "predictions").resolve())
    else:
        predictions_path = Path(args.predictions_dir)
        if not predictions_path.is_absolute():
            predictions_path = ROOT / predictions_path
        args.predictions_dir = str(predictions_path.resolve())

    if args.tesseract_cmd:
        os.environ["TESSERACT_CMD"] = args.tesseract_cmd
    if args.tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = args.tessdata_prefix

    if args.model_name != "all" and args.model_name not in IMPLEMENTED_MODELS:
        raise SystemExit(
            f"Unknown model. Choose from: {sorted(IMPLEMENTED_MODELS)} or 'all'"
        )

    model_names = (
        sorted(IMPLEMENTED_MODELS)
        if args.model_name == "all"
        else [args.model_name]
    )
    model_names = [
        name for name in model_names if args.task in MODEL_TASKS.get(name, set())
    ]
    if args.skip_gemini:
        model_names = [
            name for name in model_names
            if not model_uses_gemini(name, args.task)
        ]

    predictions_dir = Path(args.predictions_dir)
    if args.retry_gemini:
        clear_gemini_quota_marker(predictions_dir)

    try:
        settings = resolve_tesseract(
            os.environ.get("TESSERACT_CMD"),
            os.environ.get("TESSDATA_PREFIX"),
        )
        print(f"Tesseract: {settings.cmd}")
        print(f"Tessdata:  {settings.tessdata_dir}")
    except RuntimeError as exc:
        ocr_models = {
            "classical_cv_ocr_pipeline",
            "chartocr_reasoning_pipeline",
            "ocr_gemini_reasoning_pipeline",
        }

        selected_ocr_models = set(model_names) & ocr_models

        if selected_ocr_models:
            raise SystemExit(str(exc)) from exc

        print(f"Note: {exc}")

    data_dir = ROOT / args.data_dir
    base_cfg = load_project_config(ROOT / args.config, project_root=ROOT)

    rows, out_file = load_task_rows(
        args.task,
        data_dir,
        args.split,
        args.limit or None,
    )

    charts_path = data_dir / "charts.csv"
    if not charts_path.exists():
        raise FileNotFoundError(f"Missing {charts_path}")

    import pandas as pd

    charts = pd.read_csv(charts_path, dtype=str)

    print(f"dataset={args.dataset} task={args.task}")
    print(
        f"target_rows={len(rows)} split={args.split} "
        f"limit={args.limit or 'all'} batch_size={args.batch_size or 'all'}"
    )
    print(f"models={model_names}")
    print(f"predictions_dir={predictions_dir.resolve()}")
    if experiment_dir is not None:
        print(f"experiment_dir={experiment_dir.resolve()}")
    if args.full_1000:
        print("mode=full-1000 (one invocation; resumable checkpoints enabled)")

    started_at = datetime.now(timezone.utc)
    skipped_models: list[dict] = []
    completed_models: list[str] = []
    failed_models: list[dict] = []

    for model_name in model_names:
        if (
            model_uses_gemini(model_name, args.task)
            and gemini_quota_blocked(predictions_dir)
        ):
            reason = "Gemini quota cooldown is still active"
            print(f"\nSkipping {model_name}: {reason}.")
            skipped_models.append({"model_name": model_name, "reason": reason})
            continue

        print("\n" + "=" * 80)
        print(f"Running model: {model_name}")
        print("=" * 80)

        args.model_name = model_name

        try:
            cfg = load_project_config(
                ROOT / args.config,
                project_root=ROOT,
                model_name=model_name,
            )

            pred_path = run_pipeline(
                model_name,
                args.task,
                rows,
                charts,
                cfg,
                args,
            )
            print(f"Saved: {pred_path}")
            completed_models.append(model_name)
        except Exception as exc:
            failure = {
                "model_name": model_name,
                "error_type": type(exc).__name__,
                "message": str(exc)[:2000],
            }
            failed_models.append(failure)
            print(
                f"Model failed but the task will continue: {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    validation_dir = (
        experiment_dir / "validation" / args.task
        if experiment_dir is not None
        else ROOT / "results" / "experiments" / f"{args.task}_latest_validation"
    )
    validation = validate_prediction_outputs(
        args.task,
        rows,
        predictions_dir,
        validation_dir,
        model_names=model_names,
    )
    status_path = validation_dir / "STATUS.md"

    review_manifest = data_dir / "review_dataset_manifest.json"
    review_metadata = {}
    if review_manifest.exists():
        try:
            review_metadata = json.loads(review_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            review_metadata = {}

    manifest = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "split": args.split,
        "target_samples": len(rows),
        "models_requested": model_names,
        "models_completed": completed_models,
        "models_skipped": skipped_models,
        "models_failed": failed_models,
        "predictions_dir": str(predictions_dir.resolve()),
        "validation_summary": str((validation_dir / "validation_summary.csv").resolve()),
        "validation_errors": str((validation_dir / "validation_errors.csv").resolve()),
        "status_report": str(status_path.resolve()),
        "gemini_quota_paused": gemini_quota_blocked(predictions_dir),
        "dataset": args.dataset,
        "data_dir": str(data_dir.resolve()),
        "selection_method": review_metadata.get(
            "sampling_strategy",
            "filter by split, preserve file order, then take the first --limit rows",
        ),
        "target_id_column": TASK_SPEC[args.task]["id_col"],
        "target_id_sha256": stable_id_fingerprint(
            rows, TASK_SPEC[args.task]["id_col"],
        ),
        "target_content_sha256": stable_content_fingerprint(
            rows,
            TASK_SPEC[args.task]["id_col"],
            TASK_CONTENT_COLUMNS[args.task],
        ),
        "sample_composition": sample_composition(rows),
        "random_seed": review_metadata.get("seed"),
        "numerical_tolerance": float(base_cfg.evaluation.get("numerical_tolerance", 0.05)),
        "gemini": {
            "model": base_cfg.gemini.get("model", ""),
            "temperature": base_cfg.gemini.get("temperature", ""),
            "max_tokens": base_cfg.gemini.get("max_tokens", ""),
        },
        "prompts": prompt_manifest(base_cfg.prompts),
        "model_checkpoints": {
            name: (
                CHART_QA_HF_MODELS[name]
                if name in CHART_QA_HF_MODELS
                else {"model_id": "google/deplot"}
                if name in {"deplot_table_gemini_pipeline", "table_symbolic_reasoner_pipeline"}
                else {"model_id": base_cfg.gemini.get("model", "")}
                if model_uses_gemini(name, args.task)
                else {"model_id": "local heuristic"}
            )
            for name in model_names
        },
        "environment": environment_snapshot(ROOT, args.device),
        "prediction_csv_sha256": prediction_hashes(predictions_dir, model_names),
    }
    config_path = ROOT / args.config
    if config_path.exists():
        manifest["config_sha256"] = sha256_file(config_path)
    if review_manifest.exists():
        manifest["review_dataset_manifest"] = str(review_manifest.resolve())
        manifest["review_dataset_manifest_sha256"] = sha256_file(review_manifest)
    manifest_path = validation_dir / "experiment_manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_manifest.replace(manifest_path)
    failures_path = validation_dir / "model_failures.json"
    tmp_failures = failures_path.with_suffix(".json.tmp")
    tmp_failures.write_text(
        json.dumps(failed_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_failures.replace(failures_path)

    print("\nDone.")
    print("\nOutput validation:")
    print(validation.to_string(index=False))
    print(f"\nSTATUS: {status_path.resolve()}")
    print(f"MANIFEST: {manifest_path.resolve()}")
    print(f"MODEL FAILURES: {failures_path.resolve()}")
    print("Evaluate: python -m src.main evaluate --config config/config.yaml")
    if failed_models:
        raise SystemExit(
            f"{len(failed_models)} model(s) failed; other approaches were still completed."
        )


if __name__ == "__main__":
    main()
