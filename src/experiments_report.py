"""Experiments chapter draft (datasets, setup, protocol)."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

try:
    from .dataset_catalog import list_planned, list_downloadable, load_catalog
except ImportError:  # Support direct execution with ``src`` on sys.path.
    from dataset_catalog import list_planned, list_downloadable, load_catalog


def generate_experiments_chapter(
    charts: pd.DataFrame,
    questions: pd.DataFrame,
    models: pd.DataFrame,
    processed_dir: Path,
) -> str:
    report = "# Experiments\n\n"
    report += (
        "This chapter describes the chart understanding benchmarks, model families "
        "(explicit pipelines vs implicit Gemini), and the evaluation protocol used "
        "in the thesis.\n\n"
    )

    report += "## Benchmark datasets\n\n"
    report += (
        "Datasets are registered in `config/datasets.yaml`. "
        "Implemented sets are downloaded with `python -m src.main download-data` "
        "and converted with `prepare-data` into a unified schema under `data/processed/`.\n\n"
    )

    manifest_path = processed_dir / "datasets_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        report += "| Dataset | Charts | Questions | Claims |\n"
        report += "|---------|--------|-----------|--------|\n"
        for row in manifest:
            report += (
                f"| {row.get('dataset', '')} | {row.get('charts', 0)} | "
                f"{row.get('questions', 0)} | {row.get('claims', 0)} |\n"
            )
        report += "\n"
    elif not charts.empty and "source" in charts.columns:
        report += "### Charts in processed corpus\n\n"
        for src, n in charts.groupby("source").size().items():
            report += f"- **{src}**: {n} charts\n"
        report += "\n"

    report += "### Auto-downloadable (Hugging Face)\n\n"
    for name in list_downloadable():
        entry = load_catalog().get(name, {})
        report += f"- **{name}**: `{entry.get('hf_repo', entry.get('source_url', ''))}`\n"
    report += "\n"

    report += "### Planned (manual setup)\n\n"
    for name in list_planned():
        entry = load_catalog().get(name, {})
        instr = entry.get("instructions", entry.get("source_url", "TBD"))
        report += f"- **{name}**: {instr}\n"
    report += "\n"

    report += "## Tasks\n\n"
    report += (
        "| Task | Ground truth | Predictions | Metric |\n"
        "|------|--------------|-------------|--------|\n"
        "| Chart QA | `questions.csv` | `qa_pred.csv` | Accuracy (typed) |\n"
        "| Verification | `claims.csv` | `claims_pred.csv` | Accuracy / macro-F1 |\n"
        "| Summarization | `charts.csv` | `summaries_pred.csv` | Qualitative / BLEU (future) |\n"
        "| Table extraction | `tables_gt.csv` | `tables_pred.csv` | Cell MAE / tolerance |\n\n"
    )

    if not questions.empty:
        report += "## ChartQA question taxonomy (processed corpus)\n\n"
        if "task" in questions.columns:
            for task, n in questions["task"].value_counts().items():
                report += f"- {task}: {n}\n"
        report += "\n"

    report += "## Model families\n\n"
    report += (
        "1. **Explicit pipelines** — OCR (Tesseract), chart-structure heuristics, "
        "DePlot table extraction, optional Gemini reasoning on text/table context only.\n"
        "2. **Implicit** — `gemini_end_to_end`: image + question, no OCR/table intermediate.\n"
        "3. **Baselines** — constant and train-prior (no image); oracle is debug-only.\n\n"
    )

    if not models.empty:
        report += "### Registered models\n\n"
        report += "| Model | Family | Status |\n"
        report += "|-------|--------|--------|\n"
        for _, row in models.iterrows():
            notes = str(row.get("notes", ""))
            status = "planned" if "PLANNED" in notes.upper() else "implemented"
            report += f"| {row.get('model_name')} | {row.get('model_family')} | {status} |\n"
        report += "\n"

    report += "## Protocol\n\n"
    report += (
        "1. `python scripts/generate_baselines.py` — non-visual baselines.\n"
        "2. `python scripts/run_experiments.py --task qa --model-name <model> --split test --limit N` "
        "for reproducible pilot runs.\n"
        "3. `python -m src.main evaluate --config config/config.yaml` — metrics, thesis figures, reports.\n\n"
        "**Pilot vs full evaluation:** Pipeline runs with small `--limit` are reported separately "
        "(subset accuracy, n shown in tables). Baselines without `--limit` cover the full split. "
        "Oracle upper bound is excluded from comparative analysis.\n\n"
    )

    report += "## Thesis figures\n\n"
    report += (
        "Generated under `results/figures/thesis/`:\n"
        "- `fig_datasets_overview.png` — dataset scale\n"
        "- `fig_qa_pilot_accuracy.png` — pipeline comparison (pilot n)\n"
        "- `fig_qa_task_heatmap.png` — accuracy by question task\n"
        "- `fig_explicit_vs_implicit.png` — method families\n\n"
        "- `fig_question_task_distribution.png` — reasoning-task composition\n"
        "- `fig_question_operation_distribution.png` — operation composition\n"
        "- `fig_answer_type_distribution.png` — answer-type composition\n"
        "- `fig_qa_accuracy_confidence_intervals.png` — accuracy with 95% intervals\n"
        "- `fig_prediction_validity.png` — valid predictions vs runtime/API failures\n\n"
    )

    return report
