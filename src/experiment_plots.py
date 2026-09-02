"""Publication-ready plots for mixed-coverage chart-understanding experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LOCAL_COLOR = "#2E86AB"
GEMINI_COLOR = "#A23B72"
CORRECT_COLOR = "#3A7D44"
WRONG_COLOR = "#D9A441"
FAILURE_COLOR = "#C44E52"
MISSING_COLOR = "#D9DDE1"
NO_UNVERIFIABLE_CEILING = 2 / 3

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def _format_decimal(value: float, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}"


def _format_percentage(value: float, decimals: int = 0) -> str:
    """Format percentages for the English-language thesis figures."""
    number = _format_decimal(100 * float(value), decimals)
    return f"{number}%"


def _percentage_axis(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda value, _: _format_percentage(value, decimals))


def _format_count(value: int) -> str:
    """Group thousands with a narrow no-break space instead of a comma."""
    return f"{int(value):,}".replace(",", "\u202f")


def _save(fig, path: Path) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [path, pdf_path]


def _model_colors(frame: pd.DataFrame) -> list[str]:
    return [
        GEMINI_COLOR if bool(value) else LOCAL_COLOR
        for value in frame["uses_gemini"]
    ]


def coverage_matrix(coverage: pd.DataFrame, path: Path) -> list[Path]:
    data = coverage[coverage["applicable"]].copy()
    if data.empty:
        return []
    task_order = ["qa", "verification", "summarization", "table_extraction"]
    tasks = [task for task in task_order if task in set(data["task"])]
    model_order = (
        data.groupby(["model_name", "model_label"], as_index=False)["uses_gemini"]
        .max()
        .sort_values(["uses_gemini", "model_label"])
    )
    models = model_order["model_name"].tolist()
    labels = model_order["model_label"].tolist()
    matrix = np.full((len(models), len(tasks)), np.nan)
    annotations = [["" for _ in tasks] for _ in models]
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            row = data[(data["model_name"] == model) & (data["task"] == task)]
            if row.empty:
                continue
            item = row.iloc[0]
            matrix[i, j] = float(item["coverage"])
            annotations[i][j] = _format_percentage(float(item["coverage"]))
    fig, ax = plt.subplots(figsize=(9.8, max(4.8, 0.62 * len(models) + 1.7)))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)), [t.replace("_", " ") for t in tasks])
    ax.set_yticks(range(len(models)), labels)
    ax.set_title("Experiment coverage by model and task")
    for i in range(len(models)):
        for j in range(len(tasks)):
            if not annotations[i][j]:
                continue
            value = matrix[i, j]
            ax.text(
                j, i, annotations[i][j],
                ha="center", va="center",
                color="white" if value >= 0.58 else "#1A1A1A",
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Valid coverage of the target subset")
    colorbar.ax.yaxis.set_major_formatter(_percentage_axis())
    ax.text(
        0, -0.16,
        "Percent of the target subset with a valid prediction. Blank cells are not applicable.",
        transform=ax.transAxes, ha="left", va="top", color="#555555",
    )
    return _save(fig, path)


def qa_outcome_composition(summary: pd.DataFrame, path: Path) -> list[Path]:
    data = summary[
        summary["attempted_samples"].gt(0)
        & ~summary["model_name"].eq("oracle_upper_bound")
    ].copy()
    if data.empty:
        return []
    data = data.sort_values(["uses_gemini", "accuracy", "model_label"])
    target = data["target_samples"].clip(lower=1).to_numpy(float)
    segments = {
        "Correct": data["correct_samples"].to_numpy(float) / target,
        "Incorrect": data["incorrect_samples"].to_numpy(float) / target,
        "Runtime/API failure": data["failure_samples"].to_numpy(float) / target,
        "Not yet evaluated": data["missing_samples"].to_numpy(float) / target,
    }
    colors = [CORRECT_COLOR, WRONG_COLOR, FAILURE_COLOR, MISSING_COLOR]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11.2, max(5.0, 0.62 * len(data) + 2.0)))
    left = np.zeros(len(data))
    for (label, values), color in zip(segments.items(), colors):
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(y, data["model_label"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(_percentage_axis())
    ax.set_xlabel("Share of the target subset")
    ax.set_title("QA outcomes: performance and experimental completeness")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    for segment_values, segment_left in zip(segments.values(), np.cumsum(
        [np.zeros(len(data))] + list(segments.values())[:-1], axis=0
    )):
        for i, value in enumerate(segment_values):
            if value >= 0.04:
                ax.text(
                    segment_left[i] + value / 2,
                    i,
                    _format_percentage(value),
                    ha="center",
                    va="center",
                    fontsize=8.2,
                )
    ax.set_xlim(0, 1)
    return _save(fig, path)


def qa_accuracy_forest(summary: pd.DataFrame, path: Path) -> list[Path]:
    data = summary[
        summary["valid_samples"].gt(0)
        & ~summary["model_name"].eq("oracle_upper_bound")
    ].copy()
    if data.empty:
        return []
    local = data[~data["uses_gemini"].astype(bool)].sort_values("accuracy")
    gemini = data[data["uses_gemini"].astype(bool)].sort_values("accuracy")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, max(5.0, 0.58 * len(local) + 1.8)),
        sharex=True, gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    for ax, block, title in (
        (axes[0], local, "Balanced 1,000-question local cohort"),
        (axes[1], gemini, "Common 500-question Gemini cohort"),
    ):
        y = np.arange(len(block))
        for i, (_, row) in enumerate(block.iterrows()):
            color = GEMINI_COLOR if bool(row["uses_gemini"]) else LOCAL_COLOR
            marker = "D" if bool(row["uses_gemini"]) else "o"
            ax.errorbar(
                row["accuracy"], i,
                xerr=[[row["accuracy"] - row["ci95_low"]],
                      [row["ci95_high"] - row["accuracy"]]],
                fmt=marker, color=color, capsize=4, linewidth=1.5, markersize=6,
            )
            label_x = row["ci95_high"] + 0.018
            label_ha = "left"
            if label_x > 0.94:
                label_x = row["ci95_low"] - 0.018
                label_ha = "right"
            ax.text(
                label_x, i, _format_percentage(row["accuracy"], 1),
                va="center", ha=label_ha, fontsize=8.2,
            )
        ax.set_yticks(y, block["model_label"].tolist())
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(_percentage_axis())
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.2)
    fig.supxlabel("Accuracy with descriptive 95% Wilson interval")
    fig.suptitle("QA results for two predefined evaluation cohorts", y=1.01)
    axes[0].text(
        0, -0.13,
        "Percentages are comparable within a panel; the panels use different question sets.",
        transform=axes[0].transAxes, ha="left", va="top", color="#555555",
        fontsize=8.5,
    )
    return _save(fig, path)


def qa_accuracy_vs_coverage(summary: pd.DataFrame, path: Path) -> list[Path]:
    data = summary[summary["valid_samples"] > 0].copy()
    if data.empty:
        return []
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    grouped = (
        data.groupby(["coverage", "accuracy", "uses_gemini"], as_index=False)
        .agg(model_label=("model_label", lambda values: " / ".join(values)))
    )
    for _, row in grouped.iterrows():
        color = GEMINI_COLOR if bool(row["uses_gemini"]) else LOCAL_COLOR
        marker = "D" if bool(row["uses_gemini"]) else "o"
        ax.scatter(
            row["coverage"], row["accuracy"],
            s=65, color=color, marker=marker, edgecolor="white", linewidth=0.7,
        )
        ax.annotate(
            row["model_label"],
            (row["coverage"], row["accuracy"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8.5,
        )
    ax.axvline(0.10, color="#777777", linewidth=1, linestyle="--", alpha=0.65)
    ax.axvline(1.00, color="#777777", linewidth=1, linestyle=":", alpha=0.65)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(_percentage_axis())
    ax.yaxis.set_major_formatter(_percentage_axis())
    ax.set_xlabel("Valid coverage of the target subset")
    ax.set_ylabel("Accuracy among valid predictions")
    ax.set_title("QA accuracy–coverage trade-off")
    ax.grid(alpha=0.2)
    return _save(fig, path)


def qa_evaluation_status(summary: pd.DataFrame, path: Path) -> list[Path]:
    """Distinguish free-tier completion from execution success and model quality."""
    if summary.empty:
        return []
    data = summary.sort_values(["uses_gemini", "answer_accuracy", "model_label"]).copy()
    metrics = [
        ("experimental_completion", "Experiment completed"),
        ("execution_success", "Successful execution"),
        ("answer_accuracy", "Correct among valid"),
    ]
    y = np.arange(len(data))
    fig, axes = plt.subplots(
        1, 3,
        figsize=(12.2, max(5.2, 0.58 * len(data) + 2.0)),
        sharey=True,
    )
    for ax, (column, title) in zip(axes, metrics):
        for i, (_, row) in enumerate(data.iterrows()):
            color = GEMINI_COLOR if bool(row["uses_gemini"]) else LOCAL_COLOR
            marker = "D" if bool(row["uses_gemini"]) else "o"
            value = float(row[column])
            ax.scatter(value, i, color=color, marker=marker, s=42, zorder=3)
            if value > 0.88:
                label_x, label_ha = value - 0.025, "right"
            else:
                label_x, label_ha = value + 0.025, "left"
            ax.text(
                label_x, i, _format_percentage(value),
                va="center", ha=label_ha, fontsize=8.1,
            )
        ax.set_xlim(-0.02, 1.06)
        ax.xaxis.set_major_formatter(_percentage_axis())
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, data["model_label"])
    fig.suptitle("QA evaluation status: coverage is not API reliability", y=1.01)
    fig.text(
        0.5, -0.01,
        "Completion = attempted / target. Execution success = valid / attempted. "
        "Free-tier limits reduce completion; they do not by themselves measure model quality.",
        ha="center", va="top", color="#555555", fontsize=9,
    )
    return _save(fig, path)


def alignment_stage_profile(summary: pd.DataFrame, path: Path) -> list[Path]:
    """Show operational proxies for the four alignment levels in the thesis."""
    if summary.empty:
        return []
    model_order = (
        summary[["model_name", "model_label", "uses_gemini"]]
        .drop_duplicates()
        .sort_values(["uses_gemini", "model_label"])
    )
    stage_order = (
        summary[["alignment_stage", "stage_order"]]
        .drop_duplicates()
        .sort_values("stage_order")
    )
    models = model_order["model_name"].tolist()
    labels = model_order["model_label"].tolist()
    stages = stage_order["alignment_stage"].tolist()
    matrix = np.full((len(models), len(stages)), np.nan)
    annotations = [["" for _ in stages] for _ in models]
    for i, model in enumerate(models):
        for j, stage in enumerate(stages):
            rows = summary[
                summary["model_name"].eq(model)
                & summary["alignment_stage"].eq(stage)
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if row["display_status"] != "reported" or pd.isna(row["score"]):
                annotations[i][j] = "insufficient"
                continue
            score = float(row["score"])
            coverage = float(row["coverage"])
            matrix[i, j] = score
            annotations[i][j] = (
                f"{_format_percentage(score)}{'*' if coverage < 0.995 else ''}"
            )
    fig, ax = plt.subplots(figsize=(9.6, max(5.2, 0.60 * len(models) + 2.0)))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(stages)), [f"{stage}\nalignment" for stage in stages])
    ax.set_yticks(range(len(models)), labels)
    ax.set_title("Performance profile across the four alignment levels")
    for i in range(len(models)):
        for j in range(len(stages)):
            annotation = annotations[i][j]
            if not annotation:
                continue
            value = matrix[i, j]
            ax.text(
                j, i, annotation,
                ha="center", va="center",
                color=(
                    "white"
                    if not np.isnan(value) and value >= 0.58
                    else "#555555"
                ),
                fontsize=8.4,
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Proxy accuracy")
    colorbar.ax.yaxis.set_major_formatter(_percentage_axis())
    ax.text(
        0, -0.13,
        "* Incomplete coverage. “Insufficient” means fewer than 20 valid examples. "
        "These are task-based proxies, not direct measurements of internal model stages.",
        transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=8.7,
    )
    return _save(fig, path)


def qa_breakdown_heatmap(
    breakdown: pd.DataFrame,
    dimension: str,
    path: Path,
) -> list[Path]:
    data = breakdown[
        breakdown["dimension"].eq(dimension)
        & ~breakdown["model_name"].eq("oracle_upper_bound")
        & ~breakdown["model_name"].isin({
            "constant_answer_baseline",
            "train_prior_baseline",
            "random_train_prior_baseline",
        })
    ].copy()
    if data.empty:
        return []
    model_meta = (
        data[["model_name", "model_label", "uses_gemini"]]
        .drop_duplicates()
        .sort_values(["uses_gemini", "model_label"])
    )
    models = model_meta["model_name"].tolist()
    labels = model_meta["model_label"].tolist()
    groups = (
        data.groupby("group")["samples"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    matrix = np.full((len(models), len(groups)), np.nan)
    samples = np.zeros((len(models), len(groups)), dtype=int)
    coverage = np.zeros((len(models), len(groups)), dtype=float)
    for i, model in enumerate(models):
        for j, group in enumerate(groups):
            row = data[(data["model_name"] == model) & (data["group"] == group)]
            if row.empty:
                continue
            samples[i, j] = int(row.iloc[0]["samples"])
            matrix[i, j] = (
                float(row.iloc[0]["accuracy"])
                if samples[i, j] >= 20
                else np.nan
            )
            coverage[i, j] = float(row.iloc[0].get("coverage", 1.0))
    width = max(9.8, 1.18 * len(groups) + 4.0)
    fig, ax = plt.subplots(figsize=(width, max(5.0, 0.62 * len(models) + 2.0)))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(groups)), [g.replace("_", " ") for g in groups], rotation=32, ha="right")
    ax.set_yticks(range(len(models)), labels)
    title_dimension = dimension.replace("_", " ")
    ax.set_title(f"QA accuracy by {title_dimension}")
    for i in range(len(models)):
        for j in range(len(groups)):
            if 0 < samples[i, j] < 20:
                ax.text(
                    j, i, f"n={samples[i, j]}",
                    ha="center", va="center", color="#555555", fontsize=8.2,
                )
                continue
            if np.isnan(matrix[i, j]):
                continue
            ax.text(
                j, i,
                f"{_format_percentage(matrix[i, j])}"
                f"{'*' if coverage[i, j] < 0.995 else ''}",
                ha="center", va="center",
                color="white" if matrix[i, j] >= 0.58 else "#1A1A1A",
                fontsize=8.2,
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy")
    colorbar.ax.yaxis.set_major_formatter(_percentage_axis())
    ax.text(
        0, -0.16,
        "Cells with fewer than 20 valid examples show n instead of a percentage "
        "and are not colour-coded.",
        transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=8.5,
    )
    return _save(fig, path)


def verification_forest(summary: pd.DataFrame, path: Path) -> list[Path]:
    if summary.empty or "valid_samples" not in summary.columns:
        return []
    data = summary[summary["valid_samples"] > 0].copy()
    if "evaluation_status" in data.columns:
        data = data[data["evaluation_status"].eq("reported")]
    if data.empty:
        return []
    data = data.sort_values(["accuracy", "model_label"])
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11.2, max(5.0, 0.68 * len(data) + 1.8)))
    for i, (_, row) in enumerate(data.iterrows()):
        if int(row["valid_samples"]) < 20:
            ax.text(
                0.02, i, "insufficient evaluated coverage",
                va="center", color="#666666", fontsize=8.5,
            )
            continue
        ax.errorbar(
            row["accuracy"], i - 0.10,
            xerr=[[row["accuracy"] - row["ci95_low"]],
                  [row["ci95_high"] - row["accuracy"]]],
            fmt="o", color=LOCAL_COLOR, capsize=4, linewidth=1.4,
        )
        f1_low = row.get("macro_f1_ci95_low", np.nan)
        f1_high = row.get("macro_f1_ci95_high", np.nan)
        if pd.notna(f1_low) and pd.notna(f1_high):
            ax.errorbar(
                row["macro_f1"], i + 0.10,
                xerr=[[row["macro_f1"] - f1_low], [f1_high - row["macro_f1"]]],
                fmt="s", color="#555555", capsize=4, linewidth=1.4,
            )
        else:
            ax.scatter(row["macro_f1"], i + 0.10, marker="s", color="#555555", s=34)
        accuracy_x = min(0.73, float(row["ci95_high"]) + 0.007)
        accuracy_ha = "right" if accuracy_x > 0.92 else "left"
        ax.text(
            accuracy_x, i - 0.10, _format_percentage(row["accuracy"]),
            va="center", ha=accuracy_ha, fontsize=8.0,
        )
        f1_x = min(0.73, float(row["macro_f1"]) + 0.007)
        f1_ha = "right" if f1_x > 0.92 else "left"
        ax.text(
            f1_x, i + 0.10, _format_percentage(row["macro_f1"]),
            va="center", ha=f1_ha, fontsize=8.0,
        )
    display_labels = []
    for row in data.itertuples():
        if float(row.coverage) >= 0.995:
            display_labels.append(row.model_label)
            continue
        coverage_text = (
            _format_percentage(row.coverage, 1)
            if float(row.coverage) < 0.01
            else _format_percentage(row.coverage)
        )
        display_labels.append(f"{row.model_label} · {coverage_text} covered")
    ax.set_yticks(y, display_labels)
    ax.axvline(
        NO_UNVERIFIABLE_CEILING, color="#A33A3A", linestyle="--", linewidth=1.2,
        label="No-unverifiable ceiling (66.7%)",
    )
    ax.annotate(
        "66.7% ceiling without\nan unverifiable decision",
        xy=(NO_UNVERIFIABLE_CEILING, 0.98), xycoords=("data", "axes fraction"),
        xytext=(-8, -4), textcoords="offset points", ha="right", va="top",
        color="#A33A3A", fontsize=9,
    )
    ax.set_xlim(-0.01, 0.75)
    ax.xaxis.set_major_formatter(_percentage_axis())
    ax.set_xlabel("Score (%)")
    ax.set_title("QA-to-verification adapter diagnostic on 450 constructed claims")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#A33A3A", linestyle="--", label="No-unverifiable ceiling (66.7%)"),
            Line2D([0], [0], marker="o", color=LOCAL_COLOR, label="Accuracy (95% cluster-bootstrap CI)"),
            Line2D([0], [0], marker="s", color="#555555", label="Macro-F1 (95% cluster-bootstrap CI)"),
        ], loc="lower right", frameon=False,
    )
    return _save(fig, path)


def sample_size_precision(summary: pd.DataFrame, path: Path) -> list[Path]:
    n = np.arange(20, 1001)
    z = 1.96
    p = 0.5
    denominator = 1 + z * z / n
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(n, half * 100, color=LOCAL_COLOR, linewidth=2)
    gemini = summary[
        summary.get("uses_gemini", pd.Series(False, index=summary.index)).astype(bool)
        & summary.get("valid_samples", pd.Series(0, index=summary.index)).gt(0)
    ]
    local = summary[
        ~summary.get("uses_gemini", pd.Series(False, index=summary.index)).astype(bool)
        & summary.get("valid_samples", pd.Series(0, index=summary.index)).gt(0)
    ]
    gemini_sample = int(gemini["valid_samples"].min()) if not gemini.empty else 100
    local_sample = int(local["valid_samples"].max()) if not local.empty else 1000
    points = (
        (max(20, min(1000, gemini_sample)), GEMINI_COLOR, "Smallest Gemini run"),
        (max(20, min(1000, local_sample)), LOCAL_COLOR, "Largest local run"),
    )
    for sample, color, label in points:
        value = float(half[sample - 20] * 100)
        ax.scatter(sample, value, color=color, s=55, zorder=3)
        ax.annotate(
            f"{label}: n={_format_count(sample)}, "
            f"+/-{_format_decimal(value)} pp",
            (sample, value), xytext=(8, 8), textcoords="offset points",
        )
    ax.set_xlabel("Evaluated samples")
    ax.set_ylabel("Worst-case 95% Wilson interval half-width (percentage points)")
    ax.set_title("Sample size and statistical precision")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _format_count(round(value)))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _format_decimal(value))
    )
    ax.set_xlim(20, 1030)
    ax.set_ylim(0, max(half * 100) * 1.08)
    ax.grid(alpha=0.2)
    return _save(fig, path)


def model_component_matrix(path: Path) -> list[Path]:
    rows = [
        ("Classical OCR", [1, 0, 0, 0, 0, 0]),
        ("ChartOCR-style heuristics", [1, 1, 0, 0, 0, 0]),
        ("DePlot + symbolic", [0, 0, 1, 1, 0, 0]),
        ("Pix2Struct ChartQA", [0, 0, 0, 0, 1, 0]),
        ("MatCha ChartQA", [0, 0, 0, 0, 1, 0]),
        ("MatCha PlotQA-adapted", [0, 0, 0, 0, 1, 0]),
        ("OCR + Gemini", [1, 0, 0, 0, 0, 1]),
        ("DePlot + Gemini", [0, 0, 1, 0, 0, 1]),
        ("Gemini end-to-end", [0, 0, 0, 0, 1, 1]),
    ]
    columns = ["OCR", "Chart structure", "Plot-to-table", "Symbolic reasoning", "Vision-language", "Cloud LLM"]
    values = np.array([values for _, values in rows])
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=28, ha="right")
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.set_title("Component-level comparison of implemented approaches")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j, i, "●" if values[i, j] else "–",
                ha="center", va="center",
                color="white" if values[i, j] else "#777777",
                fontsize=13,
            )
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    return _save(fig, path)


def summarization_forest(summary: pd.DataFrame, path: Path) -> list[Path]:
    if summary.empty or "valid_samples" not in summary.columns:
        return []
    data = summary[summary["valid_samples"] > 0].copy()
    if data.empty:
        return []
    data = data.sort_values(["mean_token_f1", "model_label"])
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.68 * len(data) + 1.8)))
    for i, (_, row) in enumerate(data.iterrows()):
        color = GEMINI_COLOR if bool(row["uses_gemini"]) else LOCAL_COLOR
        marker = "D" if bool(row["uses_gemini"]) else "o"
        ax.errorbar(
            row["mean_token_f1"], i,
            xerr=[[row["mean_token_f1"] - row["ci95_low"]],
                  [row["ci95_high"] - row["mean_token_f1"]]],
            fmt=marker, color=color, capsize=4, linewidth=1.4,
        )
        ax.text(
            min(0.985, row["ci95_high"] + 0.015), i,
            _format_percentage(row["mean_token_f1"], 1),
            va="center", fontsize=8.4,
        )
    ax.set_yticks(range(len(data)), data["model_label"])
    visible_upper = min(1.0, max(0.35, float(data["ci95_high"].max()) + 0.25))
    ax.set_xlim(0, visible_upper)
    ax.xaxis.set_major_formatter(_percentage_axis())
    ax.set_xlabel("Mean token-overlap F1 with approximate 95% interval")
    ax.set_title("Chart summarization quality and coverage")
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, path)


def table_extraction_quality(summary: pd.DataFrame, path: Path) -> list[Path]:
    if summary.empty:
        return []
    data = summary.copy()
    metrics = [
        ("Chart coverage", "chart_coverage"),
        ("Reference-key coverage", "cell_coverage"),
        ("Value accuracy on matched keys", "tolerance_accuracy"),
        ("End-to-end cell accuracy", "end_to_end_accuracy"),
    ]
    fig, axes = plt.subplots(
        len(data), 1,
        figsize=(9.8, max(3.8, 1.3 * len(data) + 2.0)),
        squeeze=False,
    )
    for index, (_, row) in enumerate(data.iterrows()):
        ax = axes[index, 0]
        values = [float(row[column]) for _, column in metrics]
        bars = ax.barh(
            range(len(metrics)), values,
            color=[LOCAL_COLOR, "#5B9BD5", CORRECT_COLOR, WRONG_COLOR],
        )
        ax.set_yticks(range(len(metrics)), [label for label, _ in metrics])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(_percentage_axis())
        ax.set_title(row["model_label"])
        ax.grid(axis="x", alpha=0.2)
        ax.bar_label(
            bars,
            labels=[_format_percentage(value, 1) for value in values],
            padding=4,
        )
    fig.suptitle("Chart-to-table extraction coverage and numeric quality", y=1.01)
    return _save(fig, path)


def dataset_scope(scope: pd.DataFrame, path: Path) -> list[Path]:
    if scope.empty:
        return []
    data = scope.sort_values(["processed_questions", "dataset"])
    dataset_labels = {
        "chartqa": "ChartQA",
        "chartqapro": "ChartQAPro",
        "scigraphqa": "SciGraphQA",
        "polychartqa": "PolyChartQA",
        "chartx": "ChartX",
    }
    data["dataset_label"] = data["dataset"].map(dataset_labels).fillna(data["dataset"])
    total = max(1, int(data["processed_questions"].sum()))
    data["corpus_share"] = data["processed_questions"].astype(float) / total
    fig, ax = plt.subplots(figsize=(9.2, max(4.2, 0.62 * len(data) + 1.5)))
    colors = [
        LOCAL_COLOR if value > 0 else MISSING_COLOR
        for value in data["processed_questions"]
    ]
    bars = ax.barh(data["dataset_label"], data["corpus_share"], color=colors)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(_percentage_axis())
    ax.set_xlabel("Share of the processed evaluation corpus")
    ax.set_title("Current evaluation data are concentrated in one dataset")
    ax.grid(axis="x", alpha=0.2)
    labels = [
        _format_percentage(value) if value else "not prepared"
        for value in data["corpus_share"]
    ]
    ax.bar_label(bars, labels=labels, padding=4)
    return _save(fig, path)


def generate_experiment_figures(
    analysis: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    paths: list[Path] = []
    specs = [
        (coverage_matrix, (analysis["coverage_by_task"], output_dir / "fig_current_coverage_matrix.png")),
        (qa_outcome_composition, (analysis["qa_model_summary"], output_dir / "fig_current_qa_outcomes.png")),
        (qa_accuracy_forest, (analysis["qa_model_summary"], output_dir / "fig_current_qa_accuracy_ci.png")),
        (qa_accuracy_vs_coverage, (analysis["qa_model_summary"], output_dir / "fig_current_qa_accuracy_vs_coverage.png")),
        (qa_evaluation_status, (analysis["qa_execution_summary"], output_dir / "fig_qa_evaluation_status.png")),
        (alignment_stage_profile, (analysis["alignment_stage_summary"], output_dir / "fig_alignment_stage_profile.png")),
        (verification_forest, (analysis["verification_model_summary"], output_dir / "fig_current_verification_performance.png")),
        (summarization_forest, (analysis["summarization_model_summary"], output_dir / "fig_current_summarization_quality.png")),
        (table_extraction_quality, (analysis["table_extraction_model_summary"], output_dir / "fig_current_table_extraction_quality.png")),
        (dataset_scope, (analysis["dataset_scope"], output_dir / "fig_dataset_scope_gap.png")),
        (sample_size_precision, (analysis["qa_model_summary"], output_dir / "fig_sample_size_precision.png")),
        (model_component_matrix, (output_dir / "fig_model_component_matrix.png",)),
    ]
    for function, arguments in specs:
        paths.extend(function(*arguments))
    for dimension in ("task", "operation", "answer_type"):
        paths.extend(qa_breakdown_heatmap(
            analysis["qa_breakdown"],
            dimension,
            output_dir / f"fig_current_qa_by_{dimension}.png",
        ))
    return paths
