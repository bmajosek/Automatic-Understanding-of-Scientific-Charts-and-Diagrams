"""Publication-style figures for MSc thesis (Experiments + Results chapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    from .model_registry import BASELINES, model_family
except ImportError:  # Support direct execution with ``src`` on sys.path.
    from model_registry import BASELINES, model_family

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    _ensure(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _color(name: str) -> str:
    fam = model_family(name)
    return {
        "explicit": "#2E86AB",
        "implicit": "#A23B72",
        "baseline": "#95A5A6",
        "debug": "#F39C12",
    }.get(fam, "#7F8C8D")


def _label(value: object) -> str:
    return str(value).replace("_", " ").strip()


def _count_distribution(
    frame: pd.DataFrame,
    column: str,
    output_path: Path,
    title: str,
    axis_label: str,
) -> None:
    if frame.empty or column not in frame.columns:
        return
    counts = frame[column].fillna("unknown").replace("", "unknown").value_counts()
    if counts.empty:
        return
    counts = counts.sort_values()
    height = max(3.8, 0.48 * len(counts) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, height))
    bars = ax.barh(
        [_label(v) for v in counts.index],
        counts.values,
        color="#2E86AB",
    )
    ax.set_xlabel(axis_label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    ax.bar_label(
        bars,
        labels=[f"{value:,} ({value / counts.sum():.1%})" for value in counts.values],
        padding=4,
    )
    ax.set_xlim(0, max(counts.values) * 1.24)
    _save(fig, output_path)


def question_task_distribution(questions: pd.DataFrame, output_path: Path) -> None:
    _count_distribution(
        questions,
        "task",
        output_path,
        "Question distribution by reasoning task",
        "Number of questions",
    )


def question_operation_distribution(questions: pd.DataFrame, output_path: Path) -> None:
    _count_distribution(
        questions,
        "operation",
        output_path,
        "Question distribution by operation",
        "Number of questions",
    )


def answer_type_distribution(questions: pd.DataFrame, output_path: Path) -> None:
    _count_distribution(
        questions,
        "answer_type",
        output_path,
        "Question distribution by answer type",
        "Number of questions",
    )


def dataset_task_matrix(questions: pd.DataFrame, output_path: Path) -> None:
    required = {"source_dataset", "task"}
    if questions.empty or not required.issubset(questions.columns):
        return
    matrix = pd.crosstab(questions["source_dataset"], questions["task"])
    if matrix.empty or len(matrix.index) < 2:
        return
    shares = matrix.div(matrix.sum(axis=1), axis=0)
    fig, ax = plt.subplots(
        figsize=(max(8, len(matrix.columns) * 1.25), max(3.8, len(matrix.index) * 0.65))
    )
    image = ax.imshow(shares.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), [_label(v) for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [_label(v) for v in matrix.index])
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Task composition within each dataset")
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = shares.iloc[i, j]
            color = "white" if value > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="Share of dataset questions")
    _save(fig, output_path)


def qa_accuracy_with_confidence_intervals(
    qa_results: List[Dict], output_path: Path,
) -> None:
    rows = [
        row for row in qa_results
        if row.get("evaluated_questions", 0) > 0
        and row.get("model_name") != "oracle_upper_bound"
        and row.get("model_name") not in BASELINES
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row.get("accuracy", 0))
    names = [_label(row["model_name"]) for row in rows]
    accuracy = np.array([float(row.get("accuracy", 0)) for row in rows])
    sample_sizes = np.array([int(row.get("evaluated_questions", 0)) for row in rows])
    z = 1.96
    denominator = 1 + z**2 / sample_sizes
    center = (accuracy + z**2 / (2 * sample_sizes)) / denominator
    half = (
        z
        * np.sqrt(
            accuracy * (1 - accuracy) / sample_sizes
            + z**2 / (4 * sample_sizes**2)
        )
        / denominator
    )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(rows) * 0.68)))
    for index, row in enumerate(rows):
        ax.errorbar(
            accuracy[index],
            y[index],
            xerr=np.array([[
                accuracy[index] - (center[index] - half[index]),
                (center[index] + half[index]) - accuracy[index],
            ]]).T,
            fmt="o",
            color=_color(row["model_name"]),
            capsize=4,
            linewidth=1.5,
        )
        ax.text(
            min(1.01, center[index] + half[index] + 0.015),
            y[index],
            f"{accuracy[index]:.3f}  n={sample_sizes[index]:,}",
            va="center",
        )
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy with 95% Wilson confidence interval")
    ax.set_title("Chart QA accuracy and statistical uncertainty")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, output_path)


def prediction_validity_by_model(
    qa_results: List[Dict], output_path: Path,
) -> None:
    rows = [
        row for row in qa_results
        if int(row.get("prediction_rows", 0) or 0) > 0
        and row.get("model_name") != "oracle_upper_bound"
        and row.get("model_name") not in BASELINES
    ]
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("valid_prediction_rows", row.get("evaluated_questions", 0)) or 0)
            / max(1, int(row.get("prediction_rows", 0) or 0))
        ),
    )
    names = [_label(row["model_name"]) for row in rows]
    attempted = np.array([max(1, int(row.get("prediction_rows", 0) or 0)) for row in rows])
    valid = np.array([
        int(row.get("valid_prediction_rows", row.get("evaluated_questions", 0)) or 0)
        for row in rows
    ])
    failed = np.array([int(row.get("failed_prediction_rows", 0) or 0) for row in rows])
    valid_share = valid / attempted
    failed_share = failed / attempted
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(rows) * 0.68)))
    valid_bars = ax.barh(y, valid_share, color="#2E86AB", label="Valid prediction")
    failed_bars = ax.barh(
        y, failed_share, left=valid_share, color="#C44E52", label="Runtime/API failure"
    )
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of attempted predictions")
    ax.set_title("Prediction validity and API reliability", pad=48)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False)
    for bars, values in ((valid_bars, valid), (failed_bars, failed)):
        labels = [f"{value:,}" if value else "" for value in values]
        ax.bar_label(bars, labels=labels, label_type="center", color="white")
    _save(fig, output_path)


def filter_qa_rows(qa_results: List[Dict], pilot_only: bool = True) -> List[Dict]:
    out = [r for r in qa_results if r.get("evaluated_questions", 0) > 0]
    if pilot_only:
        return [r for r in out if r.get("is_pilot_run")]
    return [r for r in out if not r.get("is_pilot_run")]


def filter_pipelines(qa_results: List[Dict]) -> List[Dict]:
    return [
        r for r in qa_results
        if r.get("evaluated_questions", 0) > 0
        and r.get("model_name") not in BASELINES
        and not r.get("is_pilot_run") is False or r.get("is_pilot_run")
    ]


def qa_accuracy_bar(
    qa_results: List[Dict],
    output_path: Path,
    title: str = "Chart QA accuracy (pilot subset)",
    pilot_only: bool = True,
) -> None:
    rows = filter_qa_rows(qa_results, pilot_only=pilot_only)
    rows = [r for r in rows if r.get("model_name") not in ("oracle_upper_bound",)]
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x.get("accuracy", 0), reverse=True)
    names = [r["model_name"] for r in rows]
    accs = [r.get("accuracy", 0) for r in rows]
    ns = [r.get("evaluated_questions", 0) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [_color(n) for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}",
                ha="center", va="bottom", fontsize=8)
    _save(fig, output_path)


def qa_task_heatmap(qa_results: List[Dict], output_path: Path) -> None:
    rows = [r for r in qa_results if r.get("evaluated_questions", 0) > 0 and r.get("is_pilot_run")]
    rows = [r for r in rows if r.get("model_name") != "oracle_upper_bound"]
    if not rows:
        return
    tasks = sorted({t for r in rows for t in r.get("accuracy_by_task", {})})
    rows = sorted(rows, key=lambda row: row["model_name"])
    models = [r["model_name"] for r in rows]
    mat = np.zeros((len(models), len(tasks)))
    sample_mat = np.zeros((len(models), len(tasks)), dtype=int)
    for i, r in enumerate(rows):
        for j, t in enumerate(tasks):
            mat[i, j] = r.get("accuracy_by_task", {}).get(t, np.nan)
            sample_mat[i, j] = int(r.get("samples_by_task", {}).get(t, 0) or 0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([_label(task) for task in tasks], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_label(model) for model in models])
    ax.set_title("QA accuracy by task (pilot runs)")
    for i in range(len(models)):
        for j in range(len(tasks)):
            value = mat[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}\n(n={sample_mat[i, j]:,})",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#1A1A1A",
                )
    fig.colorbar(im, ax=ax, label="Accuracy")
    _save(fig, output_path)


def explicit_vs_implicit_bar(qa_results: List[Dict], output_path: Path) -> None:
    groups = {"Explicit pipelines": [], "Implicit (Gemini e2e)": [], "Baselines": []}
    for r in qa_results:
        if r.get("evaluated_questions", 0) == 0:
            continue
        name = r["model_name"]
        if name == "oracle_upper_bound":
            continue
        fam = model_family(name)
        if fam == "explicit" and r.get("is_pilot_run"):
            groups["Explicit pipelines"].append(r.get("accuracy", 0))
        elif fam == "implicit":
            groups["Implicit (Gemini e2e)"].append(r.get("accuracy", 0))
        elif fam == "baseline" and not r.get("is_pilot_run"):
            groups["Baselines"].append(r.get("accuracy", 0))
    labels, means = [], []
    for label, vals in groups.items():
        if vals:
            labels.append(label)
            means.append(float(np.mean(vals)))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, color=["#2E86AB", "#A23B72", "#95A5A6"][: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Method family comparison")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


def datasets_overview(charts: pd.DataFrame, questions: pd.DataFrame, output_path: Path) -> None:
    if charts.empty:
        return
    src = charts.groupby("source").size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    src.plot(kind="bar", ax=axes[0], color="#2E86AB")
    axes[0].set_title("Charts per dataset")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for container in axes[0].containers:
        axes[0].bar_label(container, labels=[f"{int(v):,}" for v in src.values], padding=3)
    if not questions.empty and "source_dataset" in questions.columns:
        qsrc = questions.groupby("source_dataset").size().sort_values(ascending=False)
        qsrc.plot(kind="bar", ax=axes[1], color="#E67E22")
        axes[1].set_title("Questions per dataset")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=30)
        for container in axes[1].containers:
            axes[1].bar_label(container, labels=[f"{int(v):,}" for v in qsrc.values], padding=3)
    fig.tight_layout()
    _save(fig, output_path)


def verification_label_distribution(claims: pd.DataFrame, output_path: Path) -> None:
    if claims.empty or "label" not in claims.columns:
        return
    counts = claims["label"].fillna("missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#2E86AB")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set_ylabel("Number of claims")
    ax.set_xlabel("Ground-truth label")
    ax.set_title("Verification ground-truth label distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3)
    ax.text(
        0.99, 0.91, f"Total n={int(counts.sum()):,}; classes={len(counts)}",
        transform=ax.transAxes, ha="right", va="top",
    )
    _save(fig, output_path)


def verification_performance(
    verification_results: List[Dict], output_path: Path,
) -> None:
    rows = [r for r in verification_results if r.get("evaluated_claims", 0) > 0]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r.get("accuracy", 0), reverse=True)
    names = [r["model_name"] for r in rows]
    accuracy = [r.get("accuracy", 0) for r in rows]
    macro_f1 = [r.get("macro_f1", 0) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(rows) * 0.65)))
    height = 0.36
    acc_bars = ax.barh(y - height / 2, accuracy, height, label="Accuracy", color="#2E86AB")
    f1_bars = ax.barh(y + height / 2, macro_f1, height, label="Macro-F1", color="#A23B72")
    ax.set_yticks(y, labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score (0–1)")
    ax.set_title("Claim verification performance", pad=34)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False)
    ax.bar_label(acc_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)
    for i, row in enumerate(rows):
        ax.text(1.075, i, f"n={row['evaluated_claims']:,}", ha="right", va="center", fontsize=8)
    _save(fig, output_path)


def generate_all_thesis_figures(
    charts: pd.DataFrame,
    questions: pd.DataFrame,
    qa_results: List[Dict],
    figures_dir: Path,
    claims: pd.DataFrame | None = None,
    verification_results: List[Dict] | None = None,
) -> List[Path]:
    """Write thesis-ready PNGs under results/figures/thesis/."""
    out_dir = figures_dir / "thesis"
    paths = []
    specs = [
        (lambda: datasets_overview(charts, questions, out_dir / "fig_datasets_overview.png"),
         out_dir / "fig_datasets_overview.png"),
        (lambda: question_task_distribution(
            questions, out_dir / "fig_question_task_distribution.png"),
         out_dir / "fig_question_task_distribution.png"),
        (lambda: question_operation_distribution(
            questions, out_dir / "fig_question_operation_distribution.png"),
         out_dir / "fig_question_operation_distribution.png"),
        (lambda: answer_type_distribution(
            questions, out_dir / "fig_answer_type_distribution.png"),
         out_dir / "fig_answer_type_distribution.png"),
        (lambda: dataset_task_matrix(
            questions, out_dir / "fig_dataset_task_matrix.png"),
         out_dir / "fig_dataset_task_matrix.png"),
        (lambda: qa_accuracy_bar(qa_results, out_dir / "fig_qa_pilot_accuracy.png", pilot_only=True),
         out_dir / "fig_qa_pilot_accuracy.png"),
        (lambda: qa_accuracy_with_confidence_intervals(
            qa_results, out_dir / "fig_qa_accuracy_confidence_intervals.png"),
         out_dir / "fig_qa_accuracy_confidence_intervals.png"),
        (lambda: prediction_validity_by_model(
            qa_results, out_dir / "fig_prediction_validity.png"),
         out_dir / "fig_prediction_validity.png"),
        (lambda: qa_accuracy_bar(
            qa_results, out_dir / "fig_qa_full_baselines.png",
            title="Non-visual baselines (full test set)", pilot_only=False),
         out_dir / "fig_qa_full_baselines.png"),
        (lambda: qa_task_heatmap(qa_results, out_dir / "fig_qa_task_heatmap.png"),
         out_dir / "fig_qa_task_heatmap.png"),
        (lambda: explicit_vs_implicit_bar(qa_results, out_dir / "fig_explicit_vs_implicit.png"),
         out_dir / "fig_explicit_vs_implicit.png"),
        (lambda: verification_label_distribution(
            claims if claims is not None else pd.DataFrame(),
            out_dir / "fig_verification_label_distribution.png"),
         out_dir / "fig_verification_label_distribution.png"),
        (lambda: verification_performance(
            verification_results or [], out_dir / "fig_verification_performance.png"),
         out_dir / "fig_verification_performance.png"),
    ]
    for fn, p in specs:
        fn()
        if p.exists():
            paths.append(p)
    return paths
