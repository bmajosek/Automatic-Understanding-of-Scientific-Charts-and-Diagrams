"""Validate experiment prediction coverage and write human-readable audit files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from pipelines.common import is_prediction_failure, safe_str
from pipelines.gemini_client import is_gemini_quota_error
from pipelines.tasks import TASK_SPEC


def validate_prediction_outputs(
    task: str,
    target_rows: pd.DataFrame,
    predictions_dir: Path,
    output_dir: Path,
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    spec = TASK_SPEC[task]
    id_col = spec["id_col"]
    pred_file = spec["output_file"]
    target_ids = set(target_rows[id_col].map(safe_str))
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        model_names = sorted(
            p.name
            for p in predictions_dir.iterdir()
            if p.is_dir() and (p / pred_file).exists()
        ) if predictions_dir.exists() else []

    summaries: list[dict] = []
    error_details: list[dict] = []

    for model_name in model_names:
        path = predictions_dir / model_name / pred_file
        if not path.exists():
            summaries.append({
                "model_name": model_name,
                "prediction_file": str(path.resolve()),
                "status": "MISSING_FILE",
                "target_samples": len(target_ids),
                "unique_target_ids": 0,
                "missing_samples": len(target_ids),
                "duplicate_rows": 0,
                "successful_samples": 0,
                "failure_samples": 0,
                "quota_failure_samples": 0,
                "informational_samples": 0,
                "coverage_percent": 0.0,
                "success_percent_of_target": 0.0,
            })
            continue

        try:
            frame = pd.read_csv(path, dtype=str).fillna("")
        except Exception as exc:
            summaries.append({
                "model_name": model_name,
                "prediction_file": str(path.resolve()),
                "status": f"UNREADABLE:{type(exc).__name__}",
                "target_samples": len(target_ids),
                "unique_target_ids": 0,
                "missing_samples": len(target_ids),
                "duplicate_rows": 0,
                "successful_samples": 0,
                "failure_samples": 0,
                "quota_failure_samples": 0,
                "informational_samples": 0,
                "coverage_percent": 0.0,
                "success_percent_of_target": 0.0,
            })
            continue

        if id_col not in frame.columns:
            matching = pd.DataFrame(columns=frame.columns)
        else:
            matching = frame[frame[id_col].map(safe_str).isin(target_ids)].copy()

        if task == "table_extraction" and {
            id_col, "series", "category",
        }.issubset(matching.columns):
            duplicate_rows = int(
                matching.duplicated([id_col, "series", "category"]).sum()
            )
        else:
            duplicate_rows = (
                int(matching[id_col].duplicated().sum()) if id_col in matching else 0
            )
        unique_ids = int(matching[id_col].nunique()) if id_col in matching else 0
        errors = matching.get("error_type", pd.Series("", index=matching.index)).map(safe_str)
        failure_mask = errors.map(is_prediction_failure)
        quota_mask = errors.map(is_gemini_quota_error)
        informational_mask = errors.ne("") & ~failure_mask
        if task == "table_extraction" and id_col in matching:
            successful = int(matching.loc[~failure_mask, id_col].nunique())
            failures = int(matching.loc[failure_mask, id_col].nunique())
            quota_failures = int(matching.loc[quota_mask, id_col].nunique())
            informational = int(
                matching.loc[informational_mask, id_col].nunique()
            )
        else:
            successful = int((~failure_mask).sum()) - duplicate_rows
            failures = int(failure_mask.sum())
            quota_failures = int(quota_mask.sum())
            informational = int(informational_mask.sum())
        missing = max(0, len(target_ids) - unique_ids)

        if missing:
            status = "INCOMPLETE"
        elif failures:
            status = "COMPLETE_WITH_FAILURES"
        elif duplicate_rows:
            status = "COMPLETE_WITH_DUPLICATES"
        else:
            status = "COMPLETE"

        summaries.append({
            "model_name": model_name,
            "prediction_file": str(path.resolve()),
            "status": status,
            "target_samples": len(target_ids),
            "unique_target_ids": unique_ids,
            "missing_samples": missing,
            "duplicate_rows": duplicate_rows,
            "successful_samples": successful,
            "failure_samples": failures,
            "quota_failure_samples": quota_failures,
            "informational_samples": informational,
            "coverage_percent": round(100 * unique_ids / max(1, len(target_ids)), 2),
            "success_percent_of_target": round(
                100 * successful / max(1, len(target_ids)), 2
            ),
        })

        for _, row in matching[failure_mask].iterrows():
            error_details.append({
                "model_name": model_name,
                id_col: safe_str(row.get(id_col)),
                "error_type": safe_str(row.get("error_type")),
                "prediction_file": str(path.resolve()),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    pd.DataFrame(
        error_details,
        columns=["model_name", id_col, "error_type", "prediction_file"],
    ).to_csv(output_dir / "validation_errors.csv", index=False)

    lines = [
        f"# Experiment output validation: {task}",
        "",
        f"- Target samples: {len(target_ids)}",
        f"- Predictions directory: `{predictions_dir.resolve()}`",
        "- Machine-readable summary: `validation_summary.csv`",
        "- Failed-row details: `validation_errors.csv`",
        "",
        "| Model | Status | Coverage | Successful | Failures | Quota failures | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model_name']} | {row['status']} | "
            f"{row['coverage_percent']:.2f}% | {row['successful_samples']} | "
            f"{row['failure_samples']} | {row['quota_failure_samples']} | "
            f"{row['missing_samples']} |"
        )
    (output_dir / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
