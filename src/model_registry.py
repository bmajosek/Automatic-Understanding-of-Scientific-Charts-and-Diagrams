"""Which models to include in thesis evaluation tables."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
import pandas as pd

EXCLUDE_FROM_SCOREBOARD = {
    "oracle_upper_bound",
    "ocr_llm_reasoning_pipeline",  # legacy folder name
}

EXPLICIT_PIPELINES = {
    "classical_cv_ocr_pipeline",
    "chartocr_reasoning_pipeline",
    "ocr_gemini_reasoning_pipeline",
    "deplot_table_gemini_pipeline",
    "pix2struct_ocr_free_pipeline",
    "matcha_chartqa_ocr_free_pipeline",
    "matcha_plotqa_transfer_pipeline",
    "table_symbolic_reasoner_pipeline",
}

IMPLICIT_MODELS = {"gemini_end_to_end"}

BASELINES = {
    "constant_answer_baseline",
    "train_prior_baseline",
    "random_train_prior_baseline",
    "constant_supported_baseline",
}


def load_models_csv(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "models.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def implemented_model_names(data_dir: Path) -> Set[str]:
    df = load_models_csv(data_dir)
    names: Set[str] = set()
    if df.empty:
        return names
    for _, row in df.iterrows():
        name = str(row.get("model_name", "")).strip()
        notes = str(row.get("notes", "")).upper()
        if not name:
            continue
        if "PLANNED" in notes:
            continue
        if "IMPLEMENTED" in notes or "DEBUG" in notes or "baseline" in notes.lower():
            names.add(name)
    return names


def scoreboard_models(data_dir: Path, prediction_dirs: List[str]) -> List[str]:
    """Models that may appear in thesis scoreboards (predictions on disk, not debug-only)."""
    allowed = implemented_model_names(data_dir)
    out = []
    for name in sorted(prediction_dirs):
        if name in EXCLUDE_FROM_SCOREBOARD:
            continue
        if allowed and name not in allowed:
            continue
        out.append(name)
    return out


def model_family(name: str) -> str:
    if name in EXPLICIT_PIPELINES:
        return "explicit"
    if name in IMPLICIT_MODELS:
        return "implicit"
    if name in BASELINES:
        return "baseline"
    if name == "oracle_upper_bound":
        return "debug"
    return "other"
