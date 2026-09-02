"""Shared utilities for prediction pipelines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "chart", "did", "do",
    "does", "for", "from", "give", "graph", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "plot", "show", "shows", "than", "that", "the",
    "there", "this", "to", "value", "what", "when", "where", "which", "who",
    "why", "with", "x", "y",
}

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

INFORMATIONAL_ERROR_PREFIXES = (
    "ocr_heuristic",
    "chartocr_heuristic",
    "ocr_heuristic_verification",
    "chartocr_heuristic_verification",
    "table_symbolic_summary",
    "chartocr_summary",
    "ocr_summary",
    "constant_baseline",
    "constant_supported_baseline",
    "question_only_prior",
    "question_only_random_prior",
    "oracle_debug",
)


@dataclass
class Prediction:
    question_id: str
    chart_id: str
    pred_answer: str
    error_type: str = ""
    notes: str = ""


def safe_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def is_prediction_failure(error_type: object) -> bool:
    """Return True only for actual failures, not method/provenance annotations."""
    error = safe_str(error_type).strip()
    if not error:
        return False
    return not error.startswith(INFORMATIONAL_ERROR_PREFIXES)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def extract_final_answer(raw: str) -> str:
    """Keep only a concise final answer from model output."""
    text = normalize_text(raw)
    if not text:
        return "unknown"
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("final answer:"):
            return normalize_text(line.split(":", 1)[-1])
    return text[:200]


def question_keywords(question: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*", question.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def extract_number_tokens(text: str) -> List[str]:
    return NUMBER_RE.findall(safe_str(text))


def parse_number(token: str) -> Optional[float]:
    token = safe_str(token).strip().replace("$", "").replace(",", "")
    is_percent = token.endswith("%")
    token = token.rstrip("%")
    try:
        value = float(token)
        return value / 100.0 if is_percent else value
    except Exception:
        return None


def split_ocr_lines(text: str) -> List[str]:
    lines = []
    for line in safe_str(text).splitlines():
        line = normalize_text(line)
        if line:
            lines.append(line)
    return lines


def score_line_for_question(line: str, question: str) -> int:
    keys = question_keywords(question)
    if not keys:
        return 0
    line_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*", line.lower()))
    return len(keys & line_words)


def choose_numeric_answer(question: str, context: str) -> str:
    lines = split_ocr_lines(context)
    candidates: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        nums = extract_number_tokens(line)
        if not nums:
            continue
        score = score_line_for_question(line, question)
        for num in nums:
            candidates.append((score, -idx, num))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    nums = extract_number_tokens(context)
    return nums[0] if nums else "unknown"


def choose_boolean_answer(question: str, context: str) -> str:
    q = question.lower()
    yes_clues = {"yes", "true", "supported"}
    no_clues = {"no", "false", "not", "contradicted"}
    text_words = set(re.findall(r"[a-zA-Z]+", context.lower()))
    if yes_clues & text_words and not (no_clues & text_words):
        return "yes"
    if no_clues & text_words and not (yes_clues & text_words):
        return "no"
    if q.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ")):
        return "unknown"
    return "unknown"


def choose_text_answer(question: str, context: str) -> str:
    lines = split_ocr_lines(context)
    if not lines:
        return "unknown"
    scored = []
    for idx, line in enumerate(lines):
        score = score_line_for_question(line, question)
        clean = re.sub(r"[^A-Za-z0-9 .,_/%$:+-]", "", line).strip()
        if clean and len(clean) <= 80:
            scored.append((score, -idx, clean))
    if scored:
        scored.sort(reverse=True)
        return scored[0][2] or "unknown"
    return "unknown"


def answer_from_context(question: str, answer_type: str, context: str) -> str:
    answer_type = safe_str(answer_type).lower().strip()
    if answer_type == "numeric":
        return choose_numeric_answer(question, context)
    if answer_type == "boolean":
        return choose_boolean_answer(question, context)
    return choose_text_answer(question, context)


def load_pil_image(image_path: Path):
    from PIL import Image, ImageOps

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess_for_ocr(image):
    from PIL import ImageOps

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    w, h = gray.size
    scale = 2 if max(w, h) < 1800 else 1
    if scale > 1:
        gray = gray.resize((w * scale, h * scale))
    return gray


def run_tesseract_ocr(
    image_path: Path,
    tesseract_cmd: Optional[str] = None,
    tessdata_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    try:
        import pytesseract
        from .tesseract_config import apply_pytesseract, resolve_tesseract
    except ImportError as exc:
        if "pytesseract" in str(exc):
            return "", "missing_pytesseract"
        raise

    try:
        settings = resolve_tesseract(tesseract_cmd, tessdata_prefix)
        apply_pytesseract(settings)
        img = load_pil_image(image_path)
        img = preprocess_for_ocr(img)
        text = pytesseract.image_to_string(img, config=settings.ocr_config)
        return text, ""
    except Exception as exc:
        return "", f"ocr_error:{type(exc).__name__}:{exc}"


def build_image_lookup(charts: pd.DataFrame, raw_data_dir: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not charts.empty and "chart_id" in charts.columns:
        for _, row in charts.iterrows():
            chart_id = safe_str(row.get("chart_id"))
            image_path = safe_str(row.get("image_path"))
            if chart_id and image_path:
                lookup[chart_id] = image_path
    return lookup


def resolve_image_path(
    chart_id: str,
    image_lookup: Dict[str, str],
    project_root: Path,
    data_dir: Path,
    raw_data_dir: Path,
) -> Optional[Path]:
    raw = image_lookup.get(chart_id, "")
    candidates: List[Path] = []
    if raw:
        p = Path(raw)
        candidates.extend([p, project_root / p, data_dir / p, raw_data_dir / p])
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidates.append(raw_data_dir / f"{chart_id}{ext}")
        candidates.append(raw_data_dir / "chartqa" / f"{chart_id}{ext}")

    for c in candidates:
        if c.exists():
            return c

    if raw:
        stem = Path(raw).stem
        name = Path(raw).name
        if raw_data_dir.exists():
            for p in raw_data_dir.rglob(name):
                if p.is_file():
                    return p
            for p in raw_data_dir.rglob(f"{stem}.*"):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    return p

    if raw_data_dir.exists():
        for ext in ("png", "jpg", "jpeg", "webp"):
            matches = list(raw_data_dir.rglob(f"{chart_id}.{ext}"))
            if matches:
                return matches[0]
    return None


def filter_questions(questions: pd.DataFrame, split: str, limit: Optional[int]) -> pd.DataFrame:
    out = questions.copy()
    if split and split.lower() != "all" and "split" in out.columns:
        out = out[out["split"].astype(str).str.lower() == split.lower()]
    if limit is not None and limit > 0:
        out = out.head(limit)
    return out.reset_index(drop=True)


def load_existing_predictions(path: Path) -> Dict[str, Prediction]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    out: Dict[str, Prediction] = {}
    for _, row in df.iterrows():
        qid = safe_str(row.get("question_id"))
        if not qid:
            continue
        out[qid] = Prediction(
            question_id=qid,
            chart_id=safe_str(row.get("chart_id")),
            pred_answer=safe_str(row.get("pred_answer")),
            error_type=safe_str(row.get("error_type")),
            notes=safe_str(row.get("notes")),
        )
    return out


def write_predictions(predictions: List[Prediction], predictions_dir: Path, model_name: str) -> Path:
    model_dir = predictions_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "qa_pred.csv"
    df = pd.DataFrame([p.__dict__ for p in predictions])
    df.to_csv(path, index=False)
    return path


def write_task_csv(rows: List[dict], path: Path, columns: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df[columns].to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def load_existing_rows(path: Path, id_col: str) -> Dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        rid = safe_str(row.get(id_col))
        if rid:
            out[rid] = {k: safe_str(row.get(k)) for k in row.index}
    return out


def append_intermediate(model_dir: Path, record: dict, mode: str = "a") -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "intermediates.jsonl"
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_intermediates_file(model_dir: Path, force: bool) -> None:
    path = model_dir / "intermediates.jsonl"
    if force and path.exists():
        path.unlink()
    elif not force and path.exists():
        return
    if force or not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
