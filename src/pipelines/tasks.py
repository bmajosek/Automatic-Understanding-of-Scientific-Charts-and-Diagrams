"""Task definitions for QA, verification, summarization, and table extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


TASK_SPEC = {
    "qa": {
        "input_file": "questions.csv",
        "output_file": "qa_pred.csv",
        "id_col": "question_id",
        "columns": ["question_id", "chart_id", "pred_answer", "error_type", "notes"],
    },
    "verification": {
        "input_file": "claims.csv",
        "output_file": "claims_pred.csv",
        "id_col": "claim_id",
        "columns": [
            "claim_id", "chart_id", "pred_label", "raw_output",
            "error_type", "notes",
        ],
    },
    "summarization": {
        "input_file": "summaries.csv",
        "output_file": "summaries_pred.csv",
        "id_col": "chart_id",
        "columns": ["chart_id", "pred_summary", "error_type", "notes"],
    },
    "table_extraction": {
        "input_file": "charts.csv",
        "output_file": "tables_pred.csv",
        "id_col": "chart_id",
        "columns": ["chart_id", "series", "category", "pred_value", "error_type", "notes"],
    },
}

TASKS = set(TASK_SPEC.keys())

MODEL_TASKS = {
    "classical_cv_ocr_pipeline": {"qa", "verification", "summarization"},
    "chartocr_reasoning_pipeline": {"qa", "verification", "summarization"},
    "ocr_gemini_reasoning_pipeline": {"qa", "verification", "summarization"},
    "deplot_table_gemini_pipeline": {"qa", "verification", "summarization", "table_extraction"},
    "pix2struct_ocr_free_pipeline": {"qa", "verification", "summarization"},
    "matcha_chartqa_ocr_free_pipeline": {"qa", "verification"},
    "matcha_plotqa_transfer_pipeline": {"qa", "verification"},
    "table_symbolic_reasoner_pipeline": {"qa", "verification", "summarization"},
    "gemini_end_to_end": {"qa", "verification", "summarization"},
}


def load_task_rows(
    task: str,
    data_dir: Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, str]:
    if task not in TASK_SPEC:
        raise ValueError(f"Unknown task {task!r}. Choose from {sorted(TASKS)}")

    spec = TASK_SPEC[task]
    path = Path(data_dir) / spec["input_file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing task input file: {path}")

    rows = pd.read_csv(path, dtype=str)
    if split and "split" in rows.columns:
        rows = rows[rows["split"].fillna("").str.lower() == split.lower()]

    if limit and limit > 0:
        rows = rows.head(limit)

    return rows.reset_index(drop=True), spec["output_file"]
