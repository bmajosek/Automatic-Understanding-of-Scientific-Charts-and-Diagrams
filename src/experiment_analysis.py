"""Coverage-aware analysis for resumable experiments with unequal sample sizes."""

from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import numpy as np

from metrics import AnswerEvaluation, ConfusionMatrix, NumericalMetrics
from model_registry import model_family
from pipelines.common import is_prediction_failure, safe_str
from pipelines.runner import model_uses_gemini
from pipelines.tasks import MODEL_TASKS, TASK_SPEC, load_task_rows


MODEL_LABELS = {
    "classical_cv_ocr_pipeline": "Classical OCR",
    "chartocr_reasoning_pipeline": "ChartOCR-style heuristics",
    "ocr_gemini_reasoning_pipeline": "OCR + Gemini",
    "deplot_table_gemini_pipeline": "DePlot + Gemini",
    "pix2struct_ocr_free_pipeline": "Pix2Struct ChartQA",
    "matcha_chartqa_ocr_free_pipeline": "MatCha ChartQA",
    "matcha_plotqa_transfer_pipeline": "MatCha PlotQA-adapted",
    "table_symbolic_reasoner_pipeline": "DePlot + symbolic",
    "gemini_end_to_end": "Gemini end-to-end",
    "constant_answer_baseline": "Oracle-type constant baseline",
    "train_prior_baseline": "Oracle-type training prior",
    "random_train_prior_baseline": "Oracle-type random prior",
    "constant_supported_baseline": "Always-supported baseline",
    "oracle_upper_bound": "Oracle upper bound",
}

ANALYSIS_BASELINE_TASKS = {
    "constant_answer_baseline": {"qa"},
    "train_prior_baseline": {"qa"},
    "random_train_prior_baseline": {"qa"},
    "constant_supported_baseline": {"verification"},
    "oracle_upper_bound": {"qa"},
}

TABLE_EXTRACTION_COLUMNS = [
    "model_name",
    "model_label",
    "target_charts",
    "predicted_charts",
    "chart_coverage",
    "ground_truth_cells",
    "predicted_cells",
    "distinct_predicted_cells",
    "duplicate_predicted_rows",
    "matched_cells",
    "numeric_matched_cells",
    "cell_coverage",
    "exact_match_rate",
    "tolerance_accuracy",
    "end_to_end_accuracy",
    "mae",
]


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " "))


def _normalise_table_key(value: object) -> str:
    """Normalise table labels in the same way as the review audit."""
    return re.sub(r"[^a-z0-9]+", " ", safe_str(value).lower()).strip()


def _model_supports_task(model_name: str, task: str) -> bool:
    return task in (
        MODEL_TASKS.get(model_name, set())
        | ANALYSIS_BASELINE_TASKS.get(model_name, set())
    )


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


def verification_cluster_bootstrap(
    rows: pd.DataFrame,
    label_order: list[str],
    cluster_column: str,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Bootstrap verification metrics by linked claim group, not by claim row."""
    if rows.empty or cluster_column not in rows.columns:
        return {
            "interval_method": "not_available",
            "cluster_column": cluster_column,
            "cluster_count": 0,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "accuracy_ci95_low": np.nan,
            "accuracy_ci95_high": np.nan,
            "macro_f1_ci95_low": np.nan,
            "macro_f1_ci95_high": np.nan,
        }

    label_index = {label: index for index, label in enumerate(label_order)}
    group_matrices: list[np.ndarray] = []
    for _, group in rows.groupby(cluster_column, sort=True, dropna=False):
        matrix = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
        for actual, predicted in zip(group["label"], group["pred_label"]):
            actual_index = label_index.get(safe_str(actual).lower())
            predicted_index = label_index.get(safe_str(predicted).lower())
            if actual_index is not None and predicted_index is not None:
                matrix[actual_index, predicted_index] += 1
        if matrix.sum():
            group_matrices.append(matrix)

    if not group_matrices:
        return {
            "interval_method": "not_available",
            "cluster_column": cluster_column,
            "cluster_count": 0,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "accuracy_ci95_low": np.nan,
            "accuracy_ci95_high": np.nan,
            "macro_f1_ci95_low": np.nan,
            "macro_f1_ci95_high": np.nan,
        }

    matrices = np.stack(group_matrices)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(matrices),
        size=(samples, len(matrices)),
    )
    sampled = matrices[sampled_indices].sum(axis=1)
    totals = sampled.sum(axis=(1, 2))
    accuracy = np.trace(sampled, axis1=1, axis2=2) / totals

    true_positives = np.diagonal(sampled, axis1=1, axis2=2)
    false_positives = sampled.sum(axis=1) - true_positives
    false_negatives = sampled.sum(axis=2) - true_positives
    denominators = 2 * true_positives + false_positives + false_negatives
    class_f1 = np.divide(
        2 * true_positives,
        denominators,
        out=np.zeros_like(true_positives, dtype=float),
        where=denominators > 0,
    )
    macro_f1 = class_f1.mean(axis=1)
    accuracy_low, accuracy_high = np.quantile(accuracy, [0.025, 0.975])
    macro_low, macro_high = np.quantile(macro_f1, [0.025, 0.975])
    return {
        "interval_method": "cluster_bootstrap",
        "cluster_column": cluster_column,
        "cluster_count": len(matrices),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "accuracy_ci95_low": float(accuracy_low),
        "accuracy_ci95_high": float(accuracy_high),
        "macro_f1_ci95_low": float(macro_low),
        "macro_f1_ci95_high": float(macro_high),
    }


def _prediction_models(predictions_dir: Path) -> list[str]:
    if not predictions_dir.exists():
        return []
    return sorted(
        path.name
        for path in predictions_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )


def _load_prediction(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str).fillna("")
    if id_col not in frame.columns:
        return pd.DataFrame()
    return frame.drop_duplicates(subset=[id_col], keep="first")


def _failure_category(error_type: object) -> str:
    error = safe_str(error_type).lower()
    if not error or not is_prediction_failure(error):
        return "none"
    if "quota" in error or "resource_exhausted" in error or "429" in error:
        return "Gemini quota"
    if "gemini" in error or "api" in error or "timeout" in error:
        return "API/runtime"
    if "image" in error or "ocr" in error or "tesseract" in error:
        return "image/OCR"
    if "parse" in error or "empty" in error or "unsupported" in error:
        return "parse/unsupported"
    return "model runtime"


def _task_coverage(
    data_dir: Path,
    predictions_dir: Path,
    split: str,
    limit: int,
    models: Iterable[str],
    gemini_qa_limit: int | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    for task, spec in TASK_SPEC.items():
        target, output_file = load_task_rows(task, data_dir, split, limit)
        id_col = spec["id_col"]
        for model_name in models:
            model_target = (
                target.head(gemini_qa_limit)
                if task == "qa"
                and gemini_qa_limit
                and model_uses_gemini(model_name, task)
                else target
            )
            target_ids = set(model_target[id_col].map(safe_str))
            applicable = _model_supports_task(model_name, task)
            pred = _load_prediction(
                predictions_dir / model_name / output_file,
                id_col,
            )
            if pred.empty:
                observed_ids: set[str] = set()
                valid_ids: set[str] = set()
                failure_ids: set[str] = set()
            else:
                matching = pred[pred[id_col].map(safe_str).isin(target_ids)].copy()
                observed_ids = set(matching[id_col].map(safe_str))
                failures = matching["error_type"].map(is_prediction_failure)
                valid_ids = set(matching.loc[~failures, id_col].map(safe_str))
                failure_ids = set(matching.loc[failures, id_col].map(safe_str))
            target_n = len(target_ids)
            records.append({
                "task": task,
                "model_name": model_name,
                "model_label": model_label(model_name),
                "model_family": model_family(model_name),
                "uses_gemini": model_uses_gemini(model_name, task),
                "applicable": applicable,
                "target_samples": target_n,
                "attempted_samples": len(observed_ids),
                "valid_samples": len(valid_ids),
                "failure_samples": len(failure_ids),
                "missing_samples": max(0, target_n - len(observed_ids)),
                "coverage": len(valid_ids) / max(1, target_n),
            })
    return pd.DataFrame(records)


def _qa_analysis(
    data_dir: Path,
    predictions_dir: Path,
    split: str,
    limit: int,
    models: Iterable[str],
    numerical_tolerance: float,
    gemini_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target, output_file = load_task_rows("qa", data_dir, split, limit)
    id_col = TASK_SPEC["qa"]["id_col"]
    summaries: list[dict] = []
    breakdowns: list[dict] = []
    errors: list[dict] = []

    for model_name in models:
        if not _model_supports_task(model_name, "qa"):
            continue
        model_target = (
            target.head(gemini_limit)
            if gemini_limit and model_uses_gemini(model_name, "qa")
            else target
        )
        pred = _load_prediction(predictions_dir / model_name / output_file, id_col)
        joined = model_target.merge(
            pred[[id_col, "pred_answer", "error_type"]] if not pred.empty
            else pd.DataFrame(columns=[id_col, "pred_answer", "error_type"]),
            on=id_col,
            how="left",
            indicator=True,
        )
        joined["observed"] = joined["_merge"].eq("both")
        joined["failure"] = (
            joined["observed"]
            & joined["error_type"].fillna("").map(is_prediction_failure)
        )
        joined["valid"] = joined["observed"] & ~joined["failure"]
        joined["correct"] = False
        for index, row in joined[joined["valid"]].iterrows():
            joined.at[index, "correct"] = AnswerEvaluation.evaluate_answer(
                row.get("answer", ""),
                row.get("pred_answer", ""),
                row.get("answer_type", "text"),
                numerical_tolerance,
            )

        target_n = len(joined)
        attempted = int(joined["observed"].sum())
        valid = int(joined["valid"].sum())
        failures = int(joined["failure"].sum())
        correct = int(joined["correct"].sum())
        lower, upper = wilson_interval(correct, valid)
        summaries.append({
            "model_name": model_name,
            "model_label": model_label(model_name),
            "model_family": model_family(model_name),
            "uses_gemini": model_uses_gemini(model_name, "qa"),
            "target_samples": target_n,
            "attempted_samples": attempted,
            "valid_samples": valid,
            "failure_samples": failures,
            "missing_samples": max(0, target_n - attempted),
            "correct_samples": correct,
            "incorrect_samples": max(0, valid - correct),
            "coverage": valid / max(1, target_n),
            "accuracy": correct / max(1, valid),
            "ci95_low": lower,
            "ci95_high": upper,
        })

        for dimension in ("task", "operation", "answer_type"):
            if dimension not in joined.columns:
                continue
            for group, target_frame in joined.groupby(dimension, dropna=False):
                frame = target_frame[target_frame["valid"]]
                n = len(frame)
                group_target = len(target_frame)
                group_correct = int(frame["correct"].sum())
                low, high = wilson_interval(group_correct, n)
                breakdowns.append({
                    "model_name": model_name,
                    "model_label": model_label(model_name),
                    "uses_gemini": model_uses_gemini(model_name, "qa"),
                    "dimension": dimension,
                    "group": safe_str(group) or "unknown",
                    "samples": n,
                    "target_samples": group_target,
                    "coverage": n / max(1, group_target),
                    "correct": group_correct,
                    "accuracy": group_correct / n if n else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                })

        for category, count in (
            joined.loc[joined["failure"], "error_type"]
            .map(_failure_category)
            .value_counts()
            .items()
        ):
            errors.append({
                "task": "qa",
                "model_name": model_name,
                "model_label": model_label(model_name),
                "error_category": category,
                "count": int(count),
            })

    return pd.DataFrame(summaries), pd.DataFrame(breakdowns), pd.DataFrame(errors)


def _qa_execution_summary(qa_summary: pd.DataFrame) -> pd.DataFrame:
    """Separate experiment completion from runtime success and model quality."""
    columns = [
        "model_name",
        "model_label",
        "uses_gemini",
        "experimental_completion",
        "execution_success",
        "answer_accuracy",
        "correct_answer_yield",
    ]
    records: list[dict] = []
    for _, row in qa_summary.iterrows():
        target = max(1, int(row["target_samples"]))
        attempted = int(row["attempted_samples"])
        valid = int(row["valid_samples"])
        records.append({
            "model_name": row["model_name"],
            "model_label": row["model_label"],
            "uses_gemini": bool(row["uses_gemini"]),
            "experimental_completion": attempted / target,
            "execution_success": valid / max(1, attempted),
            "answer_accuracy": float(row["accuracy"]),
            "correct_answer_yield": int(row["correct_samples"]) / target,
        })
    return pd.DataFrame(records, columns=columns)


def _alignment_stage_analysis(
    qa_summary: pd.DataFrame,
    qa_breakdown: pd.DataFrame,
    verification_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return no stage profile until direct intermediate measures exist.

    Earlier versions plotted four overlapping subsets of final QA accuracy and
    called them alignment levels.  That visualisation is methodologically
    invalid: it neither observes intermediate artefacts nor tests propagation.
    Direct OCR-label coverage, table-cell accuracy, predicted-vs-reference-table
    reasoning, and balanced verification are produced by the review audit
    workflow instead.
    """
    columns = [
        "model_name",
        "model_label",
        "uses_gemini",
        "alignment_stage",
        "stage_order",
        "proxy_definition",
        "score",
        "evaluated_samples",
        "target_samples",
        "coverage",
        "display_status",
    ]
    return pd.DataFrame(columns=columns)


def _verification_analysis(
    data_dir: Path,
    predictions_dir: Path,
    split: str,
    limit: int,
    models: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target, output_file = load_task_rows("verification", data_dir, split, limit)
    exclusions_path = predictions_dir.parent / "analysis_exclusions.csv"
    exclusions: dict[tuple[str, str], str] = {}
    if exclusions_path.exists():
        exclusion_rows = pd.read_csv(exclusions_path, dtype=str).fillna("")
        required_columns = {"task", "model_name", "reason"}
        if required_columns.issubset(exclusion_rows.columns):
            exclusions = {
                (safe_str(row["task"]).lower(), safe_str(row["model_name"])): safe_str(row["reason"])
                for _, row in exclusion_rows.iterrows()
                if safe_str(row["task"]) and safe_str(row["model_name"])
            }
    id_col = TASK_SPEC["verification"]["id_col"]
    summaries: list[dict] = []
    confusions: list[dict] = []
    label_order = ["supported", "contradicted", "unverifiable"]
    target = target.copy()
    target["label"] = target["label"].fillna("unverifiable").astype(str).str.lower()
    label_counts = target["label"].value_counts()
    ground_truth_classes = int((label_counts > 0).sum())
    for model_name in models:
        if not _model_supports_task(model_name, "verification"):
            continue
        pred = _load_prediction(predictions_dir / model_name / output_file, id_col)
        joined = target.merge(
            pred[[id_col, "pred_label", "error_type"]] if not pred.empty
            else pd.DataFrame(columns=[id_col, "pred_label", "error_type"]),
            on=id_col,
            how="left",
            indicator=True,
        )
        joined["observed"] = joined["_merge"].eq("both")
        joined["failure"] = (
            joined["observed"]
            & joined["error_type"].fillna("").map(is_prediction_failure)
        )
        valid_rows = joined[joined["observed"] & ~joined["failure"]].copy()
        valid_rows["correct"] = valid_rows["label"] == valid_rows["pred_label"]
        valid = len(valid_rows)
        correct = int(valid_rows["correct"].sum())
        cluster_column = (
            "paired_claim_group"
            if "paired_claim_group" in valid_rows.columns
            else "chart_id"
        )
        uncertainty = verification_cluster_bootstrap(
            valid_rows,
            label_order,
            cluster_column,
        )
        if valid:
            cm = ConfusionMatrix.compute(
                valid_rows["label"].tolist(),
                valid_rows["pred_label"].tolist(),
                label_order=label_order,
            )
            macro_f1 = float(ConfusionMatrix.macro_f1(cm))
            for actual in cm.index:
                for predicted in cm.columns:
                    confusions.append({
                        "model_name": model_name,
                        "model_label": model_label(model_name),
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(cm.loc[actual, predicted]),
                    })
        else:
            macro_f1 = np.nan
        attempted = int(joined["observed"].sum())
        exclusion_reason = exclusions.get(("verification", model_name), "")
        if exclusion_reason:
            evaluation_status = "excluded"
        elif ground_truth_classes < 2:
            evaluation_status = "invalid_single_class"
        elif valid == 0:
            evaluation_status = "insufficient_coverage"
        else:
            evaluation_status = "reported"
        summaries.append({
            "model_name": model_name,
            "model_label": model_label(model_name),
            "model_family": model_family(model_name),
            "uses_gemini": model_uses_gemini(model_name, "verification"),
            "target_samples": len(joined),
            "attempted_samples": attempted,
            "valid_samples": valid,
            "failure_samples": int(joined["failure"].sum()),
            "missing_samples": max(0, len(joined) - attempted),
            "correct_samples": correct,
            "accuracy": correct / valid if valid else np.nan,
            "macro_f1": macro_f1,
            "supported_target": int(label_counts.get("supported", 0)),
            "contradicted_target": int(label_counts.get("contradicted", 0)),
            "unverifiable_target": int(label_counts.get("unverifiable", 0)),
            "ground_truth_class_count": ground_truth_classes,
            "evaluation_status": evaluation_status,
            "exclusion_reason": exclusion_reason,
            "coverage": valid / max(1, len(joined)),
            "ci95_low": uncertainty["accuracy_ci95_low"],
            "ci95_high": uncertainty["accuracy_ci95_high"],
            "macro_f1_ci95_low": uncertainty["macro_f1_ci95_low"],
            "macro_f1_ci95_high": uncertainty["macro_f1_ci95_high"],
            "interval_method": uncertainty["interval_method"],
            "cluster_column": uncertainty["cluster_column"],
            "cluster_count": uncertainty["cluster_count"],
            "bootstrap_samples": uncertainty["bootstrap_samples"],
            "bootstrap_seed": uncertainty["bootstrap_seed"],
        })
    return pd.DataFrame(summaries), pd.DataFrame(confusions)


def _token_f1(reference: object, prediction: object) -> float:
    import re
    ref = re.findall(r"\w+", safe_str(reference).lower())
    pred = re.findall(r"\w+", safe_str(prediction).lower())
    if not ref or not pred:
        return 0.0
    remaining: dict[str, int] = {}
    for token in ref:
        remaining[token] = remaining.get(token, 0) + 1
    overlap = 0
    for token in pred:
        if remaining.get(token, 0) > 0:
            overlap += 1
            remaining[token] -= 1
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _summarization_analysis(
    data_dir: Path,
    predictions_dir: Path,
    split: str,
    limit: int,
    models: Iterable[str],
) -> pd.DataFrame:
    target, output_file = load_task_rows("summarization", data_dir, split, limit)
    id_col = TASK_SPEC["summarization"]["id_col"]
    records: list[dict] = []
    for model_name in models:
        if "summarization" not in MODEL_TASKS.get(model_name, set()):
            continue
        pred = _load_prediction(predictions_dir / model_name / output_file, id_col)
        joined = target.merge(
            pred[[id_col, "pred_summary", "error_type"]] if not pred.empty
            else pd.DataFrame(columns=[id_col, "pred_summary", "error_type"]),
            on=id_col,
            how="left",
            indicator=True,
        )
        joined["observed"] = joined["_merge"].eq("both")
        joined["failure"] = (
            joined["observed"]
            & joined["error_type"].fillna("").map(is_prediction_failure)
        )
        valid = joined[joined["observed"] & ~joined["failure"]].copy()
        scores = np.array([
            _token_f1(row.get("summary", ""), row.get("pred_summary", ""))
            for _, row in valid.iterrows()
        ], dtype=float)
        mean = float(scores.mean()) if len(scores) else 0.0
        half = (
            1.96 * float(scores.std(ddof=1)) / sqrt(len(scores))
            if len(scores) > 1 else 0.0
        )
        attempted = int(joined["observed"].sum())
        records.append({
            "model_name": model_name,
            "model_label": model_label(model_name),
            "uses_gemini": model_uses_gemini(model_name, "summarization"),
            "target_samples": len(joined),
            "attempted_samples": attempted,
            "valid_samples": len(valid),
            "failure_samples": int(joined["failure"].sum()),
            "missing_samples": max(0, len(joined) - attempted),
            "coverage": len(valid) / max(1, len(joined)),
            "mean_token_f1": mean,
            "ci95_low": max(0.0, mean - half),
            "ci95_high": min(1.0, mean + half),
        })
    return pd.DataFrame(records)


def _table_extraction_analysis(
    data_dir: Path,
    predictions_dir: Path,
    split: str,
    limit: int,
    models: Iterable[str],
    numerical_tolerance: float,
) -> pd.DataFrame:
    charts, output_file = load_task_rows("table_extraction", data_dir, split, limit)
    target_ids = set(charts["chart_id"].map(safe_str))
    gt_path = data_dir / "tables_gt.csv"
    if not gt_path.exists():
        return pd.DataFrame(columns=TABLE_EXTRACTION_COLUMNS)
    try:
        gt = pd.read_csv(gt_path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=TABLE_EXTRACTION_COLUMNS)
    if not {"chart_id", "series", "category", "value"}.issubset(gt.columns):
        return pd.DataFrame(columns=TABLE_EXTRACTION_COLUMNS)
    gt = gt[gt["chart_id"].map(safe_str).isin(target_ids)].copy()
    keys = ["chart_id", "series", "category"]
    gt["_chart_id"] = gt["chart_id"].map(lambda value: safe_str(value).strip())
    gt["_series"] = gt["series"].map(_normalise_table_key)
    gt["_category"] = gt["category"].map(_normalise_table_key)
    normalized_keys = [f"_{key}" for key in keys]
    gt = gt.drop_duplicates(subset=normalized_keys, keep="first")
    records: list[dict] = []
    for model_name in models:
        if "table_extraction" not in MODEL_TASKS.get(model_name, set()):
            continue
        pred_path = predictions_dir / model_name / output_file
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path, dtype=str).fillna("")
        pred = pred[pred["chart_id"].map(safe_str).isin(target_ids)].copy()
        pred["_chart_id"] = pred["chart_id"].map(lambda value: safe_str(value).strip())
        pred["_series"] = pred["series"].map(_normalise_table_key)
        pred["_category"] = pred["category"].map(_normalise_table_key)
        predicted_cells = len(pred)
        duplicate_predicted_rows = int(pred.duplicated(normalized_keys).sum())
        pred = pred.drop_duplicates(subset=normalized_keys, keep="first")
        merged = gt.merge(
            pred[normalized_keys + ["pred_value", "error_type"]],
            on=normalized_keys,
            how="left",
            indicator=True,
        )
        matched = merged[merged["_merge"].eq("both")].copy()
        pairs: list[tuple[float, float]] = []
        tolerance_correct = 0
        exact = 0
        for _, row in matched.iterrows():
            gt_value = NumericalMetrics.parse_number(safe_str(row.get("value")))
            pred_value = NumericalMetrics.parse_number(safe_str(row.get("pred_value")))
            if (
                gt_value is None
                or pred_value is None
                or not isfinite(gt_value)
                or not isfinite(pred_value)
            ):
                continue
            pairs.append((gt_value, pred_value))
            if abs(gt_value - pred_value) <= 1e-12:
                exact += 1
            if AnswerEvaluation.evaluate_answer(
                str(gt_value), str(pred_value), "numeric", numerical_tolerance,
            ):
                tolerance_correct += 1
        gt_values = [pair[0] for pair in pairs]
        pred_values = [pair[1] for pair in pairs]
        numeric_n = len(pairs)
        predicted_chart_n = pred["chart_id"].map(safe_str).nunique()
        records.append({
            "model_name": model_name,
            "model_label": model_label(model_name),
            "target_charts": len(target_ids),
            "predicted_charts": int(predicted_chart_n),
            "chart_coverage": predicted_chart_n / max(1, len(target_ids)),
            "ground_truth_cells": len(gt),
            "predicted_cells": predicted_cells,
            "distinct_predicted_cells": len(pred),
            "duplicate_predicted_rows": duplicate_predicted_rows,
            "matched_cells": len(matched),
            "numeric_matched_cells": numeric_n,
            "cell_coverage": len(matched) / max(1, len(gt)),
            "exact_match_rate": exact / max(1, len(matched)),
            "tolerance_accuracy": tolerance_correct / max(1, len(matched)),
            "end_to_end_accuracy": tolerance_correct / max(1, len(gt)),
            "mae": NumericalMetrics.mae(gt_values, pred_values) if numeric_n else float("nan"),
        })
    return pd.DataFrame(records, columns=TABLE_EXTRACTION_COLUMNS)


def _dataset_scope(data_dir: Path, active_datasets: Iterable[str] | None) -> pd.DataFrame:
    questions_path = data_dir / "questions.csv"
    counts: dict[str, int] = {}
    if questions_path.exists():
        questions = pd.read_csv(questions_path, dtype=str).fillna("")
        if "source_dataset" in questions.columns:
            counts = {
                safe_str(name).lower(): int(count)
                for name, count in questions["source_dataset"].value_counts().items()
            }
    active = [safe_str(name).lower() for name in (active_datasets or []) if safe_str(name)]
    names = list(dict.fromkeys(active + sorted(counts)))
    return pd.DataFrame([
        {
            "dataset": name,
            "configured_active": name in active,
            "processed_questions": counts.get(name, 0),
            "available_for_evaluation": counts.get(name, 0) > 0,
        }
        for name in names
    ])


def build_experiment_analysis(
    data_dir: Path,
    predictions_dir: Path,
    split: str = "test",
    limit: int = 1000,
    numerical_tolerance: float = 0.05,
    active_datasets: Iterable[str] | None = None,
    gemini_qa_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build machine-readable summaries without treating missing rows as model errors."""
    data_dir = Path(data_dir)
    predictions_dir = Path(predictions_dir)
    models = _prediction_models(predictions_dir)
    coverage = _task_coverage(
        data_dir,
        predictions_dir,
        split,
        limit,
        models,
        gemini_qa_limit=gemini_qa_limit,
    )
    qa_summary, qa_breakdown, errors = _qa_analysis(
        data_dir,
        predictions_dir,
        split,
        limit,
        models,
        numerical_tolerance,
        gemini_limit=gemini_qa_limit,
    )
    verification, confusion = _verification_analysis(
        data_dir,
        predictions_dir,
        split,
        limit,
        models,
    )
    summarization = _summarization_analysis(
        data_dir, predictions_dir, split, limit, models,
    )
    table_extraction = _table_extraction_analysis(
        data_dir,
        predictions_dir,
        split,
        limit,
        models,
        numerical_tolerance,
    )
    qa_execution = _qa_execution_summary(qa_summary)
    alignment_stages = _alignment_stage_analysis(
        qa_summary,
        qa_breakdown,
        verification,
    )
    return {
        "coverage_by_task": coverage,
        "qa_model_summary": qa_summary,
        "qa_execution_summary": qa_execution,
        "qa_breakdown": qa_breakdown,
        "alignment_stage_summary": alignment_stages,
        "verification_model_summary": verification,
        "verification_confusion": confusion,
        "summarization_model_summary": summarization,
        "table_extraction_model_summary": table_extraction,
        "dataset_scope": _dataset_scope(data_dir, active_datasets),
        "error_taxonomy": errors,
    }


def save_analysis_csvs(
    analysis: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, frame in analysis.items():
        path = output_dir / f"{name}.csv"
        serializable = frame if len(frame.columns) else pd.DataFrame(columns=["status"])
        serializable.to_csv(path, index=False)
        paths.append(path)
    return paths
