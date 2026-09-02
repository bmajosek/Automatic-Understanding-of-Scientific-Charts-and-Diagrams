"""Generate supervisor-review diagnostics from saved CSV predictions.

This command does not run any model or consume Gemini quota.  It checks the
verification label distribution, sample provenance, OCR/ChartOCR identity,
paired Gemini performance, trivial baselines, and (when reference and predicted
tables exist) reasoning on predicted versus reference tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import AnswerEvaluation  # noqa: E402
from pipelines.common import is_prediction_failure, safe_str  # noqa: E402
from pipelines.table_reasoner import symbolic_answer  # noqa: E402
from pipelines.tasks import load_task_rows  # noqa: E402


GEMINI_MODELS = {
    "gemini_end_to_end": "Gemini image",
    "deplot_table_gemini_pipeline": "Gemini + DePlot table",
    "ocr_gemini_reasoning_pipeline": "Gemini + OCR text",
}

LOCAL_PAIRED_MODELS = {
    "matcha_chartqa_ocr_free_pipeline": "MatCha ChartQA",
    "pix2struct_ocr_free_pipeline": "Pix2Struct ChartQA",
    "matcha_plotqa_transfer_pipeline": "MatCha PlotQA-adapted",
}


def _format_percentage(value: float, decimals: int = 0) -> str:
    """Format percentages for the English-language thesis figures."""
    number = f"{100 * float(value):.{decimals}f}"
    return f"{number}%"


def _percentage_axis(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda value, _: _format_percentage(value, decimals))


def _save_percentage_bars(
    labels: list[str], values: list[float], title: str, ylabel: str, path: Path,
) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color="#2F6B8A", width=0.62)
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.18))
    ax.yaxis.set_major_formatter(_percentage_axis())
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(
        bars,
        labels=[_format_percentage(value, 1) for value in values],
        padding=4,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _create_audit_figures(
    output_dir: Path,
    label_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    paired_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    propagation_summary: pd.DataFrame,
    perceptual_summary: pd.DataFrame,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    specs: list[tuple[list[str], list[float], str, str, str]] = []
    if not label_frame.empty:
        specs.append((
            label_frame["label"].str.capitalize().tolist(),
            label_frame["share"].astype(float).tolist(),
            "Verification ground-truth composition",
            "Share of verification claims",
            "fig_verification_label_composition.png",
        ))
    if not sample_frame.empty:
        shares = sample_frame["count"].astype(float) / sample_frame["count"].sum()
        specs.append((
            sample_frame["group"].str.capitalize().tolist(), shares.tolist(),
            "ChartQA test-cohort composition", "Share of QA questions",
            "fig_qa_origin_composition.png",
        ))
    if not paired_summary.empty:
        paired = paired_summary[paired_summary["comparison"].eq("paired_mcnemar")].copy()
        if not paired.empty:
            labels = [
                f"{row.model_a} - {row.model_b}"
                for row in paired.itertuples()
            ]
            values = paired["accuracy_difference"].astype(float).to_numpy()
            low = paired["chart_cluster_bootstrap_difference_ci95_low"].astype(float).to_numpy()
            high = paired["chart_cluster_bootstrap_difference_ci95_high"].astype(float).to_numpy()
            y = np.arange(len(paired))
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            ax.errorbar(
                values, y, xerr=[values - low, high - values], fmt="o",
                color="#2F6B8A", ecolor="#2F6B8A", capsize=4,
                linewidth=1.4, markersize=6,
            )
            for index, value in enumerate(values):
                ax.text(value + 0.012, index, _format_percentage(value, 1), va="center")
            ax.axvline(0, color="#555555", linewidth=1)
            ax.set_yticks(y, labels)
            ax.xaxis.set_major_formatter(_percentage_axis())
            ax.set_xlabel("Paired accuracy difference with 95% chart-cluster bootstrap CI")
            ax.set_title("Pairwise differences among Gemini pipelines on 500 shared questions")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            target = figure_dir / "fig_paired_gemini_accuracy.png"
            fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            created.append(str(target.resolve()))
    if not cell_summary.empty:
        row = cell_summary.iloc[0]
        specs.append((
            ["Key coverage", "Value accuracy\non matched keys", "End-to-end\ncell accuracy"],
            [float(row["cell_key_coverage"]), float(row["value_accuracy_on_matched_keys"]),
             float(row["end_to_end_cell_accuracy"])],
            "DePlot alignment with reference table cells", "Score",
            "fig_table_cell_alignment.png",
        ))
    if not propagation_summary.empty:
        row = propagation_summary.iloc[0]
        matrix = np.array([
            [int(row["both_correct"]), int(row["reference_only_correct"])],
            [int(row["deplot_only_correct"]), int(row["both_wrong"])],
        ])
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#1A1A1A",
                    fontsize=13,
                )
        ax.set_xticks([0, 1], ["DePlot correct", "DePlot wrong"])
        ax.set_yticks([0, 1], ["Reference correct", "Reference wrong"])
        ax.set_title("Paired symbolic-reasoner outcomes by table source")
        fig.colorbar(image, ax=ax, label="Questions")
        fig.tight_layout()
        target = figure_dir / "fig_reasoning_propagation.png"
        fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(str(target.resolve()))
    if not perceptual_summary.empty:
        ordered = perceptual_summary.set_index("label_type").reindex(
            ["category", "series", "all"],
        ).dropna(subset=["label_recall"])
        specs.append((
            [str(label).capitalize() for label in ordered.index],
            ordered["label_recall"].astype(float).tolist(),
            "OCR recovery of reference chart labels", "Reference-label recall",
            "fig_perceptual_label_recall.png",
        ))
    for labels, values, title, ylabel, filename in specs:
        if values and all(math.isfinite(value) for value in values):
            _save_percentage_bars(labels, values, title, ylabel, figure_dir / filename)
            created.append(str((figure_dir / filename).resolve()))
    return created


def _origin(question_id: object) -> str:
    match = re.search(r"_(human|augmented)_", safe_str(question_id), re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def _load_predictions(path: Path, id_column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str).fillna("")
    if id_column not in frame.columns:
        return pd.DataFrame()
    return frame.drop_duplicates(id_column, keep="first")


def _exact_binomial_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if not total:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(discordant_a, discordant_b), total, 0.5).pvalue)
    except Exception:
        tail = sum(math.comb(total, k) for k in range(0, min(discordant_a, discordant_b) + 1))
        return min(1.0, 2.0 * tail / (2**total))


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original comparison order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _paired_gemini(
    questions: pd.DataFrame,
    predictions_dir: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = questions.copy()
    model_columns = []
    for model_name in GEMINI_MODELS:
        pred = _load_predictions(predictions_dir / model_name / "qa_pred.csv", "question_id")
        if pred.empty:
            return pd.DataFrame(), pd.DataFrame()
        pred = pred[["question_id", "pred_answer", "error_type"]].rename(columns={
            "pred_answer": f"pred_{model_name}",
            "error_type": f"error_{model_name}",
        })
        merged = merged.merge(pred, on="question_id", how="inner")
        model_columns.append(model_name)
    for model_name in model_columns:
        error_column = f"error_{model_name}"
        merged = merged[~merged[error_column].map(is_prediction_failure)].copy()
        merged[f"correct_{model_name}"] = [
            AnswerEvaluation.evaluate_answer(
                row.answer,
                getattr(row, f"pred_{model_name}"),
                row.answer_type,
                tolerance,
            )
            for row in merged.itertuples()
        ]

    bootstrap_samples = 10_000
    bootstrap_seed = 42
    correctness_columns = [f"correct_{name}" for name in model_columns]
    distributions = _chart_cluster_bootstrap(
        merged,
        correctness_columns,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    ) if not merged.empty else {}

    summary_rows = []
    for model_name, label in GEMINI_MODELS.items():
        correct = int(merged[f"correct_{model_name}"].sum()) if not merged.empty else 0
        distribution = distributions.get(f"correct_{model_name}")
        if distribution is not None:
            accuracy_low, accuracy_high = np.quantile(distribution, [0.025, 0.975])
        else:
            accuracy_low, accuracy_high = float("nan"), float("nan")
        summary_rows.append({
            "comparison": "model",
            "model_a": label,
            "model_b": "",
            "paired_samples": len(merged),
            "accuracy_a": correct / len(merged) if len(merged) else float("nan"),
            "accuracy_b": float("nan"),
            "a_correct_b_wrong": "",
            "a_wrong_b_correct": "",
            "mcnemar_exact_p": "",
            "mcnemar_holm_p": "",
            "accuracy_difference": "",
            "chart_cluster_bootstrap_accuracy_ci95_low": float(accuracy_low),
            "chart_cluster_bootstrap_accuracy_ci95_high": float(accuracy_high),
            "chart_cluster_bootstrap_difference_ci95_low": "",
            "chart_cluster_bootstrap_difference_ci95_high": "",
            "distinct_charts": int(merged["chart_id"].nunique()) if len(merged) else 0,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        })
    names = list(GEMINI_MODELS)
    paired_rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            left_correct = merged[f"correct_{left}"]
            right_correct = merged[f"correct_{right}"]
            left_only = int((left_correct & ~right_correct).sum())
            right_only = int((~left_correct & right_correct).sum())
            difference_distribution = (
                distributions[f"correct_{left}"] - distributions[f"correct_{right}"]
                if distributions else np.array([], dtype=float)
            )
            if len(difference_distribution):
                difference_low, difference_high = np.quantile(
                    difference_distribution, [0.025, 0.975],
                )
            else:
                difference_low, difference_high = float("nan"), float("nan")
            paired_rows.append({
                "comparison": "paired_mcnemar",
                "model_a": GEMINI_MODELS[left],
                "model_b": GEMINI_MODELS[right],
                "paired_samples": len(merged),
                "accuracy_a": float(left_correct.mean()) if len(merged) else float("nan"),
                "accuracy_b": float(right_correct.mean()) if len(merged) else float("nan"),
                "a_correct_b_wrong": left_only,
                "a_wrong_b_correct": right_only,
                "mcnemar_exact_p": _exact_binomial_p(left_only, right_only),
                "accuracy_difference": (
                    float(left_correct.mean() - right_correct.mean())
                    if len(merged) else float("nan")
                ),
                "chart_cluster_bootstrap_accuracy_ci95_low": "",
                "chart_cluster_bootstrap_accuracy_ci95_high": "",
                "chart_cluster_bootstrap_difference_ci95_low": float(difference_low),
                "chart_cluster_bootstrap_difference_ci95_high": float(difference_high),
                "distinct_charts": int(merged["chart_id"].nunique()) if len(merged) else 0,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            })
    adjusted = _holm_adjust([float(row["mcnemar_exact_p"]) for row in paired_rows])
    for row, adjusted_p in zip(paired_rows, adjusted):
        row["mcnemar_holm_p"] = adjusted_p
    summary_rows.extend(paired_rows)
    return merged, pd.DataFrame(summary_rows)


def _qa_model_correctness(
    questions: pd.DataFrame,
    predictions_dir: Path,
    model_name: str,
    tolerance: float,
) -> pd.DataFrame:
    """Return valid question-level correctness for one saved QA prediction file."""
    pred = _load_predictions(
        predictions_dir / model_name / "qa_pred.csv", "question_id",
    )
    if pred.empty:
        return pd.DataFrame()
    columns = [
        "question_id", "chart_id", "answer", "answer_type", "question_origin",
    ]
    merged = questions[columns].merge(
        pred[["question_id", "pred_answer", "error_type"]],
        on="question_id",
        how="inner",
    )
    merged = merged[~merged["error_type"].map(is_prediction_failure)].copy()
    merged["correct"] = [
        AnswerEvaluation.evaluate_answer(
            row.answer,
            row.pred_answer,
            row.answer_type,
            tolerance,
        )
        for row in merged.itertuples()
    ]
    merged["model_name"] = model_name
    merged["model_label"] = LOCAL_PAIRED_MODELS[model_name]
    return merged


def _chart_cluster_bootstrap(
    frame: pd.DataFrame,
    value_columns: list[str],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Resample charts, retaining every question attached to a sampled chart."""
    if frame.empty:
        return {}
    aggregations = {column: (column, "sum") for column in value_columns}
    grouped = frame.groupby("chart_id", sort=True).agg(
        question_count=("question_id", "size"),
        **aggregations,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(grouped),
        size=(samples, len(grouped)),
    )
    denominators = grouped["question_count"].to_numpy()[indices].sum(axis=1)
    return {
        column: grouped[column].to_numpy()[indices].sum(axis=1) / denominators
        for column in value_columns
    }


def _local_paired_statistics(
    questions: pd.DataFrame,
    predictions_dir: Path,
    tolerance: float,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyse full-coverage local models by origin and paired question ID."""
    model_frames = {
        model_name: _qa_model_correctness(
            questions, predictions_dir, model_name, tolerance,
        )
        for model_name in LOCAL_PAIRED_MODELS
    }

    origin_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for model_name, frame in model_frames.items():
        if frame.empty:
            continue
        for origin, group in frame.groupby("question_origin", sort=True):
            origin_rows.append({
                "model_name": model_name,
                "model_label": LOCAL_PAIRED_MODELS[model_name],
                "question_origin": origin,
                "samples": len(group),
                "correct": int(group["correct"].sum()),
                "accuracy": float(group["correct"].mean()),
            })
        distribution = _chart_cluster_bootstrap(
            frame,
            ["correct"],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )["correct"]
        low, high = np.quantile(distribution, [0.025, 0.975])
        bootstrap_rows.append({
            "model_name": model_name,
            "model_label": LOCAL_PAIRED_MODELS[model_name],
            "valid_questions": len(frame),
            "distinct_charts": int(frame["chart_id"].nunique()),
            "accuracy": float(frame["correct"].mean()),
            "chart_cluster_bootstrap_ci95_low": float(low),
            "chart_cluster_bootstrap_ci95_high": float(high),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        })

    left_name = "matcha_chartqa_ocr_free_pipeline"
    right_name = "pix2struct_ocr_free_pipeline"
    left = model_frames[left_name]
    right = model_frames[right_name]
    if left.empty or right.empty:
        return (
            pd.DataFrame(origin_rows),
            pd.DataFrame(bootstrap_rows),
            pd.DataFrame(),
        )
    paired = left[["question_id", "chart_id", "correct"]].merge(
        right[["question_id", "chart_id", "correct"]],
        on=["question_id", "chart_id"],
        suffixes=("_a", "_b"),
    )
    left_only = int((paired["correct_a"] & ~paired["correct_b"]).sum())
    right_only = int((~paired["correct_a"] & paired["correct_b"]).sum())
    distributions = _chart_cluster_bootstrap(
        paired,
        ["correct_a", "correct_b"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    differences = distributions["correct_a"] - distributions["correct_b"]
    difference_low, difference_high = np.quantile(differences, [0.025, 0.975])
    paired_summary = pd.DataFrame([{
        "model_a": LOCAL_PAIRED_MODELS[left_name],
        "model_b": LOCAL_PAIRED_MODELS[right_name],
        "paired_questions": len(paired),
        "distinct_charts": int(paired["chart_id"].nunique()),
        "accuracy_a": float(paired["correct_a"].mean()),
        "accuracy_b": float(paired["correct_b"].mean()),
        "a_correct_b_wrong": left_only,
        "a_wrong_b_correct": right_only,
        "mcnemar_exact_p": _exact_binomial_p(left_only, right_only),
        "accuracy_difference": float(
            paired["correct_a"].mean() - paired["correct_b"].mean()
        ),
        "chart_cluster_bootstrap_difference_ci95_low": float(difference_low),
        "chart_cluster_bootstrap_difference_ci95_high": float(difference_high),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }])
    return (
        pd.DataFrame(origin_rows),
        pd.DataFrame(bootstrap_rows),
        paired_summary,
    )


def _ocr_diff(predictions_dir: Path) -> tuple[pd.DataFrame, dict]:
    classical = _load_predictions(
        predictions_dir / "classical_cv_ocr_pipeline" / "qa_pred.csv", "question_id",
    )
    chartocr = _load_predictions(
        predictions_dir / "chartocr_reasoning_pipeline" / "qa_pred.csv", "question_id",
    )
    if classical.empty or chartocr.empty:
        return pd.DataFrame(), {"status": "missing_predictions"}
    joined = classical.merge(
        chartocr,
        on=["question_id", "chart_id"],
        suffixes=("_classical", "_chartocr"),
    )
    joined["answer_identical"] = (
        joined["pred_answer_classical"].astype(str)
        == joined["pred_answer_chartocr"].astype(str)
    )
    different = joined[~joined["answer_identical"]].copy()
    summary = {
        "common_ids": len(joined),
        "identical_answers": int(joined["answer_identical"].sum()),
        "different_answers": len(different),
        "identical_fraction": float(joined["answer_identical"].mean()) if len(joined) else 0.0,
        "diagnosis": (
            "QA predictions are identical; the current ChartOCR structural output "
            "does not affect the answer path."
            if len(different) == 0
            else "Some QA predictions differ; inspect the exported rows."
        ),
    }
    return different, summary


def _long_table_text(frame: pd.DataFrame, value_column: str) -> str:
    lines = []
    for row in frame.itertuples():
        lines.append(
            f"{safe_str(getattr(row, 'series', ''))} | "
            f"{safe_str(getattr(row, 'category', ''))} | "
            f"{safe_str(getattr(row, value_column, ''))}"
        )
    return "\n".join(lines)


def _normalise_table_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", safe_str(value).lower()).strip()


def _table_cell_alignment(
    data_dir: Path,
    predictions_dir: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare DePlot cells with ChartQA's distributed reference tables.

    Keys are matched by normalised ``(chart, series, category)`` labels.  The
    summary deliberately reports both key coverage and end-to-end cell
    accuracy, so missing cells cannot disappear from the denominator.
    """
    gt_path = data_dir / "tables_gt.csv"
    pred_path = predictions_dir / "deplot_table_gemini_pipeline" / "tables_pred.csv"
    if not gt_path.exists() or not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        gt = pd.read_csv(gt_path, dtype=str).fillna("")
        pred = pd.read_csv(pred_path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), pd.DataFrame()
    required_gt = {"chart_id", "series", "category", "value"}
    required_pred = {"chart_id", "series", "category", "pred_value"}
    if gt.empty or pred.empty or not required_gt.issubset(gt) or not required_pred.issubset(pred):
        return pd.DataFrame(), pd.DataFrame()

    def keyed(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        result = frame.copy()
        result["series_key"] = result["series"].map(_normalise_table_key)
        result["category_key"] = result["category"].map(_normalise_table_key)
        result = result.drop_duplicates(
            ["chart_id", "series_key", "category_key"], keep="first",
        )
        return result[["chart_id", "series_key", "category_key", value_column]]

    reference = keyed(gt, "value").rename(columns={"value": "reference_value"})
    prediction = keyed(pred, "pred_value").rename(columns={"pred_value": "predicted_value"})
    details = reference.merge(
        prediction,
        on=["chart_id", "series_key", "category_key"],
        how="left",
        indicator=True,
    )
    details["key_matched"] = details["_merge"].eq("both")
    details["predicted_value"] = details["predicted_value"].fillna("")
    details["value_correct"] = [
        bool(matched) and AnswerEvaluation.evaluate_answer(
            expected, observed, "numeric", tolerance,
        )
        for expected, observed, matched in zip(
            details["reference_value"], details["predicted_value"], details["key_matched"],
        )
    ]
    details = details.drop(columns="_merge")
    matched = int(details["key_matched"].sum())
    correct = int(details["value_correct"].sum())
    reference_cells = len(details)
    summary = pd.DataFrame([{
        "reference_charts": int(reference["chart_id"].nunique()),
        "predicted_charts": int(prediction["chart_id"].nunique()),
        "reference_cells": reference_cells,
        "predicted_cells": len(prediction),
        "matched_cell_keys": matched,
        "correct_cell_values": correct,
        "cell_key_coverage": matched / reference_cells if reference_cells else float("nan"),
        "value_accuracy_on_matched_keys": correct / matched if matched else float("nan"),
        "end_to_end_cell_accuracy": correct / reference_cells if reference_cells else float("nan"),
        "numerical_tolerance": tolerance,
    }])
    return details, summary


def _reasoning_propagation(
    questions: pd.DataFrame,
    data_dir: Path,
    predictions_dir: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_path = data_dir / "tables_gt.csv"
    pred_path = predictions_dir / "deplot_table_gemini_pipeline" / "tables_pred.csv"
    if not gt_path.exists() or not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        gt = pd.read_csv(gt_path, dtype=str).fillna("")
        pred = pd.read_csv(pred_path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), pd.DataFrame()
    if gt.empty or pred.empty:
        return pd.DataFrame(), pd.DataFrame()

    gt_text = {
        chart_id: _long_table_text(frame, "value")
        for chart_id, frame in gt.groupby("chart_id")
    }
    pred_text = {
        chart_id: _long_table_text(frame, "pred_value")
        for chart_id, frame in pred.groupby("chart_id")
    }
    rows = []
    for row in questions.itertuples():
        chart_id = safe_str(row.chart_id)
        if chart_id not in gt_text:
            continue
        gt_answer, gt_error = symbolic_answer(row.question, gt_text[chart_id])
        predicted_answer, predicted_error = symbolic_answer(
            row.question, pred_text.get(chart_id, ""),
        )
        rows.append({
            "question_id": row.question_id,
            "chart_id": chart_id,
            "reference_answer": row.answer,
            "reasoner_on_reference_table": gt_answer,
            "reasoner_on_predicted_table": predicted_answer,
            "reference_table_error": gt_error,
            "predicted_table_error": predicted_error,
            "reference_table_correct": AnswerEvaluation.evaluate_answer(
                row.answer, gt_answer, row.answer_type, tolerance,
            ),
            "predicted_table_correct": AnswerEvaluation.evaluate_answer(
                row.answer, predicted_answer, row.answer_type, tolerance,
            ),
            "predicted_table_available": chart_id in pred_text,
        })
    details = pd.DataFrame(rows)
    if details.empty:
        return details, pd.DataFrame()
    available = details[details["predicted_table_available"]]
    both_correct = int(
        (available["reference_table_correct"] & available["predicted_table_correct"]).sum()
    )
    reference_only = int(
        (available["reference_table_correct"] & ~available["predicted_table_correct"]).sum()
    )
    deplot_only = int(
        (~available["reference_table_correct"] & available["predicted_table_correct"]).sum()
    )
    both_wrong = int(
        (~available["reference_table_correct"] & ~available["predicted_table_correct"]).sum()
    )
    summary = pd.DataFrame([{
        "questions_with_reference_table": len(details),
        "questions_with_both_tables": len(available),
        "reasoner_accuracy_reference_tables": float(details["reference_table_correct"].mean()),
        "reasoner_accuracy_predicted_tables": (
            float(available["predicted_table_correct"].mean()) if len(available) else float("nan")
        ),
        "observed_propagation_gap": (
            float(details["reference_table_correct"].mean())
            - float(available["predicted_table_correct"].mean())
            if len(available) else float("nan")
        ),
        "both_correct": both_correct,
        "reference_only_correct": reference_only,
        "deplot_only_correct": deplot_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": _exact_binomial_p(reference_only, deplot_only),
    }])
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--experiment-dir", default="results/experiments/all_tasks_1000",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    experiment_dir = (ROOT / args.experiment_dir).resolve()
    predictions_dir = experiment_dir / "predictions"
    output_dir = (
        (ROOT / args.output_dir).resolve()
        if args.output_dir
        else experiment_dir / "review_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    questions, _ = load_task_rows("qa", data_dir, args.split, args.limit)
    questions["question_origin"] = questions.apply(
        lambda row: safe_str(row.get("question_origin")) or _origin(row.get("question_id")),
        axis=1,
    )
    claims, _ = load_task_rows("verification", data_dir, args.split, args.limit)
    label_counts = claims["label"].fillna("missing").str.lower().value_counts()
    supported_baseline = (
        float(label_counts.get("supported", 0)) / len(claims) if len(claims) else float("nan")
    )

    paired_rows, paired_summary = _paired_gemini(questions, predictions_dir, args.tolerance)
    origin_summary, cluster_bootstrap_summary, local_paired_summary = (
        _local_paired_statistics(questions, predictions_dir, args.tolerance)
    )
    ocr_differences, ocr_summary = _ocr_diff(predictions_dir)
    propagation_rows, propagation_summary = _reasoning_propagation(
        questions, data_dir, predictions_dir, args.tolerance,
    )
    cell_rows, cell_summary = _table_cell_alignment(
        data_dir, predictions_dir, args.tolerance,
    )
    perceptual_path = experiment_dir / "alignment_audit" / "perceptual_label_coverage.csv"
    perceptual_summary = (
        pd.read_csv(perceptual_path)
        if perceptual_path.exists()
        else pd.DataFrame()
    )

    label_frame = pd.DataFrame([
        {"label": label, "count": int(label_counts.get(label, 0)), "share": float(label_counts.get(label, 0)) / max(1, len(claims))}
        for label in ("supported", "contradicted", "unverifiable")
    ])
    sample_frame = pd.DataFrame([
        {"dimension": "question_origin", "group": key, "count": int(value)}
        for key, value in questions["question_origin"].value_counts().items()
    ])

    outputs = {
        "verification_label_distribution.csv": label_frame,
        "qa_sample_composition.csv": sample_frame,
        "paired_gemini_questions.csv": paired_rows,
        "paired_gemini_summary.csv": paired_summary,
        "qa_by_question_origin.csv": origin_summary,
        "qa_chart_cluster_bootstrap.csv": cluster_bootstrap_summary,
        "paired_local_model_summary.csv": local_paired_summary,
        "ocr_chartocr_differences.csv": ocr_differences,
        "reasoning_propagation_rows.csv": propagation_rows,
        "reasoning_propagation_summary.csv": propagation_summary,
        "table_cell_alignment_rows.csv": cell_rows,
        "table_cell_alignment_summary.csv": cell_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    figures = _create_audit_figures(
        output_dir,
        label_frame,
        sample_frame,
        paired_summary,
        cell_summary,
        propagation_summary,
        perceptual_summary,
    )

    audit = {
        "qa_rows": len(questions),
        "qa_split": args.split,
        "qa_origin_counts": questions["question_origin"].value_counts().to_dict(),
        "verification_rows": len(claims),
        "verification_label_counts": label_counts.to_dict(),
        "constant_supported_accuracy": supported_baseline,
        "verification_is_single_class": int((label_counts > 0).sum()) < 2,
        "ocr_chartocr": ocr_summary,
        "paired_gemini_common_valid_ids": len(paired_rows),
        "paired_local_model_comparison_available": not local_paired_summary.empty,
        "chart_cluster_bootstrap_available": not cluster_bootstrap_summary.empty,
        "qa_origin_results_available": not origin_summary.empty,
        "reasoning_propagation_available": not propagation_summary.empty,
        "table_cell_alignment_available": not cell_summary.empty,
        "perceptual_alignment_available": not perceptual_summary.empty,
        "figures": figures,
        "summary_reference_notes": (
            pd.read_csv(data_dir / "summaries.csv", dtype=str)
            .get("notes", pd.Series(dtype=str)).value_counts().to_dict()
            if (data_dir / "summaries.csv").exists() else {}
        ),
    }
    (output_dir / "review_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Review audit: {output_dir}")


if __name__ == "__main__":
    main()
