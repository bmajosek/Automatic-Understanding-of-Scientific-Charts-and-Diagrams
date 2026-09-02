"""Merge per-dataset staging CSVs into unified processed files.

This version also guarantees that QA-only datasets can be used for the
verification and summarization tasks by generating synthetic task targets from
questions.csv when claims.csv or summaries.csv are empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import pandas as pd


STAGING_SUBDIR = "_staging"

FILES = ("charts", "questions", "tables_gt", "claims", "components_gt", "summaries")


CLAIMS_COLUMNS = [
    "claim_id",
    "chart_id",
    "split",
    "claim",
    "label",
    "notes",
    "source_dataset",
]

SUMMARIES_COLUMNS = [
    "chart_id",
    "split",
    "summary",
    "source_dataset",
    "notes",
]


QUESTION_COLUMNS = [
    "question_id",
    "chart_id",
    "split",
    "task",
    "operation",
    "question",
    "answer",
    "answer_type",
    "paraphrase_group",
    "dialogue_id",
    "turn_id",
    "source_dataset",
    "language",
]


def staging_dir(processed_dir: Path, dataset_name: str) -> Path:
    return processed_dir / STAGING_SUBDIR / dataset_name


def save_staging(output_dfs: Dict[str, pd.DataFrame], processed_dir: Path, dataset_name: str) -> None:
    """Save one converted dataset under data/processed/_staging/<dataset>."""
    out = staging_dir(processed_dir, dataset_name)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in output_dfs.items():
        if df is not None and not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)


def merge_staging_into_processed(processed_dir: Path) -> Dict[str, int]:
    """Concatenate staging datasets into data/processed/*.csv.

    If the source datasets only provide QA rows, this function creates:
      - claims.csv: supported factual claims derived from question/answer pairs
      - summaries.csv: pseudo summaries grouped by chart_id

    This keeps verification and summarization runnable without requiring a
    separate ChartCheck/Chart-to-Text dataset.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    root = processed_dir / STAGING_SUBDIR
    counts: Dict[str, int] = {}

    if not root.exists():
        return counts

    merged_frames: Dict[str, pd.DataFrame] = {}

    for key in FILES:
        frames: List[pd.DataFrame] = []
        for ds_path in sorted(root.iterdir()):
            if not ds_path.is_dir():
                continue

            fp = ds_path / f"{key}.csv"
            if fp.exists():
                try:
                    df = pd.read_csv(fp, dtype=str).fillna("")
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame()

                if not df.empty:
                    frames.append(df)

        if frames:
            merged_frames[key] = pd.concat(frames, ignore_index=True).fillna("")
        else:
            merged_frames[key] = pd.DataFrame()

    questions = _ensure_columns(merged_frames.get("questions", pd.DataFrame()), QUESTION_COLUMNS)

    claims = _ensure_columns(merged_frames.get("claims", pd.DataFrame()), CLAIMS_COLUMNS)
    if claims.empty and not questions.empty:
        claims = _claims_from_questions(questions)

    summaries = _ensure_columns(merged_frames.get("summaries", pd.DataFrame()), SUMMARIES_COLUMNS)
    if summaries.empty and not questions.empty:
        summaries = _summaries_from_questions(questions)

    merged_frames["claims"] = claims
    merged_frames["summaries"] = summaries

    for key in FILES:
        df = merged_frames.get(key, pd.DataFrame())
        if key == "claims":
            df = _ensure_columns(df, CLAIMS_COLUMNS)
        elif key == "summaries":
            df = _ensure_columns(df, SUMMARIES_COLUMNS)
        elif key == "questions":
            df = _ensure_columns(df, QUESTION_COLUMNS)

        out_path = processed_dir / f"{key}.csv"
        df.to_csv(out_path, index=False)
        counts[key] = len(df)

    _write_manifest(root, processed_dir)
    return counts


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = df.copy().fillna("")
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[columns]


def _claims_from_questions(questions: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for i, row in questions.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        chart_id = str(row.get("chart_id", "")).strip()

        if not question or not answer or not chart_id:
            continue

        qid = str(row.get("question_id", "")).strip() or f"row_{i:06d}"
        split = str(row.get("split", "test")).strip() or "test"
        source_dataset = str(row.get("source_dataset", "")).strip()

        rows.append({
            "claim_id": f"claim_{qid}",
            "chart_id": chart_id,
            "split": split,
            "claim": f"For the chart question '{question}', the correct answer is '{answer}'.",
            "label": "supported",
            "notes": "synthetic_from_qa",
            "source_dataset": source_dataset,
        })

    return pd.DataFrame(rows, columns=CLAIMS_COLUMNS)


def _summaries_from_questions(questions: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    required = {"chart_id", "question", "answer"}
    if not required.issubset(set(questions.columns)):
        return pd.DataFrame(columns=SUMMARIES_COLUMNS)

    for chart_id, group in questions.groupby("chart_id", dropna=True):
        chart_id = str(chart_id).strip()
        if not chart_id:
            continue

        split = str(group["split"].iloc[0]).strip() if "split" in group.columns else "test"
        source_dataset = (
            str(group["source_dataset"].iloc[0]).strip()
            if "source_dataset" in group.columns
            else ""
        )

        facts: List[str] = []
        for _, row in group.head(5).iterrows():
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if question and answer:
                facts.append(f"{question} Answer: {answer}.")

        if not facts:
            continue

        rows.append({
            "chart_id": chart_id,
            "split": split or "test",
            "summary": " ".join(facts),
            "source_dataset": source_dataset,
            "notes": "synthetic_from_qa",
        })

    return pd.DataFrame(rows, columns=SUMMARIES_COLUMNS)


def _write_manifest(root: Path, processed_dir: Path) -> None:
    manifest = []

    for ds_path in sorted(root.iterdir()):
        if ds_path.is_dir():
            manifest.append({
                "dataset": ds_path.name,
                "charts": _count_csv(ds_path / "charts.csv"),
                "questions": _count_csv(ds_path / "questions.csv"),
                "claims": _count_csv(ds_path / "claims.csv"),
                "summaries": _count_csv(ds_path / "summaries.csv"),
            })

    manifest_path = processed_dir / "datasets_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path, dtype=str))
    except pd.errors.EmptyDataError:
        return 0
