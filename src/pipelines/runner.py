"""Run pipelines for qa, verification, summarization, and table extraction."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config_loader import ProjectConfig
from .chart_structure import detect_chart_structure
from .common import (
    answer_from_context,
    build_image_lookup,
    extract_number_tokens,
    init_intermediates_file,
    is_prediction_failure,
    load_existing_rows,
    resolve_image_path,
    run_tesseract_ocr,
    safe_str,
    write_task_csv,
)
from .gemini_client import (
    gemini_text,
    gemini_vision,
    is_gemini_quota_error,
    parse_gemini_answer,
    parse_verification_label,
)
from .hf_models import HFImageTextModel
from .table_reasoner import parse_deplot_cells, symbolic_answer
from .tasks import MODEL_TASKS, TASK_SPEC
from .tesseract_config import resolve_tesseract
from metrics import AnswerEvaluation


IMPLEMENTED_MODELS = set(MODEL_TASKS.keys())
CHART_QA_HF_MODELS = {
    "pix2struct_ocr_free_pipeline": {
        "model_id": "google/pix2struct-chartqa-base",
        "processor_id": None,
    },
    "matcha_chartqa_ocr_free_pipeline": {
        "model_id": "google/matcha-chartqa",
        "processor_id": None,
    },
    "matcha_plotqa_transfer_pipeline": {
        "model_id": "google/matcha-plotqa-v2",
        # The PlotQA v2 repository contains fine-tuned weights but no processor
        # metadata. MatCha ChartQA uses the same Pix2Struct/MatCha processor and
        # is already required by the in-domain baseline.
        "processor_id": "google/matcha-chartqa",
    },
}
GEMINI_MODELS = {
    "deplot_table_gemini_pipeline",
    "gemini_end_to_end",
    "ocr_gemini_reasoning_pipeline",
}


def model_uses_gemini(model_name: str, task: str) -> bool:
    """Whether this model/task combination makes Gemini API requests."""
    if task == "table_extraction" and model_name == "deplot_table_gemini_pipeline":
        return False
    return model_name in GEMINI_MODELS

SYNTHETIC_QA_CLAIM_RE = re.compile(
    r"(?:For (?:the chart )?question|For the question)\s+['\"](?P<question>.+?)['\"]"
    r"\s*,?\s*(?:the correct answer is|the answer shown by the chart is)\s+"
    r"['\"](?P<answer>.*?)['\"]\.?$",
    re.IGNORECASE,
)


UNKNOWN_ANSWERS = {
    "", "unknown", "unanswerable", "not shown", "not available", "cannot determine",
    "insufficient information", "n/a", "none",
}


def _qa_claim_fields(row: pd.Series) -> tuple[str, str, str]:
    """Return question, claimed answer and answer type from a claim row."""
    question = safe_str(row.get("question")).strip()
    claimed_answer = safe_str(row.get("claimed_answer")).strip()
    answer_type = safe_str(row.get("answer_type")).strip().lower()
    if question and claimed_answer:
        if not answer_type:
            answer_type = (
                "numeric"
                if re.fullmatch(r"[-+]?[$€£]?\d[\d,]*(?:\.\d+)?%?", claimed_answer)
                else "text"
            )
        return question, claimed_answer, answer_type

    claim = safe_str(row.get("claim")).strip()
    match = SYNTHETIC_QA_CLAIM_RE.match(claim)
    if not match:
        return "", "", ""
    claimed_answer = match.group("answer")
    answer_type = (
        "numeric"
        if re.fullmatch(r"[-+]?[$€£]?\d[\d,]*(?:\.\d+)?%?", claimed_answer.strip())
        else "text"
    )
    return match.group("question"), claimed_answer, answer_type


def _verification_label_from_answer(
    claimed_answer: str,
    predicted_answer: str,
    answer_type: str,
    numerical_tolerance: float,
) -> str:
    normalized = safe_str(predicted_answer).strip().lower().rstrip(".")
    if normalized in UNKNOWN_ANSWERS or normalized.startswith("cannot determine"):
        return "unverifiable"
    matches = AnswerEvaluation.evaluate_answer(
        claimed_answer,
        predicted_answer,
        answer_type=answer_type or "text",
        numerical_tolerance=numerical_tolerance,
    )
    return "supported" if matches else "contradicted"


def _pix2struct_verification(
    claim: str,
    image_path: Path,
    model,
    *,
    question: str = "",
    claimed_answer: str = "",
    answer_type: str = "",
    numerical_tolerance: float = 0.05,
) -> tuple[str, str, str]:
    """Verify QA-derived claims through a chart-grounded answer comparison."""
    if not question or not claimed_answer:
        parsed = _qa_claim_fields(pd.Series({"claim": claim}))
        question, claimed_answer, parsed_type = parsed
        answer_type = answer_type or parsed_type
    if not question or not claimed_answer:
        raw = model.generate(image_path, f"Is this statement true? {claim}")
        text = safe_str(raw).strip().lower()
        if text in {"yes", "true", "supported"}:
            return "supported", safe_str(raw), ""
        if text in {"no", "false", "contradicted"}:
            return "contradicted", safe_str(raw), ""
        return "unverifiable", safe_str(raw), "unsupported_claim_format"

    raw = model.generate(image_path, question)
    label = _verification_label_from_answer(
        claimed_answer,
        raw,
        answer_type,
        numerical_tolerance,
    )
    return label, safe_str(raw), ""


def _ocr(image_path: Path, args) -> tuple[str, str]:
    cmd = getattr(args, "tesseract_cmd", None) or os.environ.get("TESSERACT_CMD")
    prefix = os.environ.get("TESSDATA_PREFIX")
    return run_tesseract_ocr(image_path, cmd, prefix)


def _gemini_cfg(cfg: ProjectConfig) -> tuple[str, float, int]:
    return (
        cfg.gemini.get("model", "gemini-2.5-flash-lite"),
        float(cfg.gemini.get("temperature", 0)),
        int(cfg.gemini.get("max_tokens", 256)),
    )


def _gemini_quota_marker_path(predictions_dir: Path) -> Path:
    return predictions_dir / "_gemini_cooldown.json"


def gemini_quota_blocked(predictions_dir: Path) -> bool:
    path = _gemini_quota_marker_path(Path(predictions_dir))
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        blocked_until = datetime.fromisoformat(marker["blocked_until_utc"])
        return datetime.now(timezone.utc) < blocked_until
    except Exception:
        return False


def clear_gemini_quota_marker(predictions_dir: Path) -> None:
    path = _gemini_quota_marker_path(Path(predictions_dir))
    if path.exists():
        path.unlink()


def _write_gemini_quota_marker(
    predictions_dir: Path,
    model_name: str,
    task: str,
    row_id: str,
    error: str,
    cooldown_minutes: int,
) -> Path:
    path = _gemini_quota_marker_path(predictions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    blocked_until = now + timedelta(minutes=max(1, cooldown_minutes))
    payload = {
        "timestamp_utc": now.isoformat(),
        "blocked_until_utc": blocked_until.isoformat(),
        "cooldown_minutes": max(1, cooldown_minutes),
        "model_name": model_name,
        "task": task,
        "row_id": row_id,
        "reason": "Gemini free-tier quota exhausted after configured retries.",
        "error": error[:1000],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path



def _deplot_cache_path(args) -> Path:
    cache_dir = Path(args.predictions_dir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "deplot_tables.json"


def _load_deplot_disk_cache(args) -> dict:
    path = _deplot_cache_path(args)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_deplot_disk_cache(args, tables: dict) -> None:
    path = _deplot_cache_path(args)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _model_uses_deplot(model_name: str, task: str) -> bool:
    return (
        model_name in ("deplot_table_gemini_pipeline", "table_symbolic_reasoner_pipeline")
        or task == "table_extraction"
    )


def _precompute_deplot_tables(rows: pd.DataFrame, cfg: ProjectConfig, args, image_lookup: Dict[str, str], hf_cache: Dict[str, Any]) -> None:
    """Run DePlot once per unique chart, before row-level Gemini calls.

    This avoids repeatedly resolving images and makes the cache reusable across tasks/runs.
    HF generation is intentionally kept sequential because most image-text HF pipelines are
    not thread-safe and GPU memory contention usually makes parallel generation slower.
    """
    disk_tables = _load_deplot_disk_cache(args)
    hf_cache["tables"] = disk_tables
    project_root = cfg.project_root
    data_dir = Path(args.data_dir)
    raw_data_dir = Path(args.raw_data_dir)

    chart_ids = []
    seen = set()
    for chart_id in rows["chart_id"].map(safe_str):
        if chart_id and chart_id not in seen:
            seen.add(chart_id)
            chart_ids.append(chart_id)

    changed = False
    for i, chart_id in enumerate(chart_ids, start=1):
        if chart_id in disk_tables:
            continue
        image_path = resolve_image_path(chart_id, image_lookup, project_root, data_dir, raw_data_dir)
        if not image_path:
            continue
        text, _err = _deplot_table(chart_id, image_path, hf_cache, args)
        if text:
            disk_tables[chart_id] = text
            changed = True
        if changed and i % 10 == 0:
            _save_deplot_disk_cache(args, disk_tables)
        if getattr(args, "verbose", False) and i % 10 == 0:
            print(f"  deplot cached {i}/{len(chart_ids)} unique charts")
    if changed:
        _save_deplot_disk_cache(args, disk_tables)

def _deplot_table(chart_id: str, image_path: Path, hf_cache: dict, args) -> tuple[str, str]:
    if "tables" not in hf_cache:
        hf_cache["tables"] = _load_deplot_disk_cache(args)
    if chart_id in hf_cache.get("tables", {}):
        return hf_cache["tables"][chart_id], ""
    if "deplot_model" not in hf_cache:
        hf_cache["deplot_model"] = HFImageTextModel(
            "google/deplot",
            device=args.device,
            max_new_tokens=int(args.deplot_max_new_tokens),
        )
    try:
        text = hf_cache["deplot_model"].generate(
            image_path, "Generate underlying data table of the figure below:",
        )
        hf_cache.setdefault("tables", {})[chart_id] = text
        _save_deplot_disk_cache(args, hf_cache["tables"])
        return text, ""
    except Exception as exc:
        return "", f"deplot_error:{type(exc).__name__}"


def _deplot_to_table_rows(chart_id: str, table_text: str, err: str) -> List[dict]:
    rows = [
        {
            "chart_id": chart_id,
            "series": series,
            "category": category,
            "pred_value": value,
            "error_type": err,
            "notes": "parsed DePlot linearized table",
        }
        for series, category, value in parse_deplot_cells(table_text)
    ]
    if not rows:
        rows.append({
            "chart_id": chart_id, "series": "deplot", "category": "table",
            "pred_value": (table_text or "")[:500], "error_type": err or "empty_table",
            "notes": "",
        })
    return rows


def _error_row(task: str, row: pd.Series, error_type: str, notes: str) -> List[dict]:
    chart_id = safe_str(row.get("chart_id"))
    if task == "qa":
        return [{
            "question_id": safe_str(row.get("question_id")),
            "chart_id": chart_id, "pred_answer": "unknown",
            "error_type": error_type, "notes": notes,
        }]
    if task == "verification":
        return [{
            "claim_id": safe_str(row.get("claim_id")),
            "chart_id": chart_id, "pred_label": "unverifiable",
            "error_type": error_type, "notes": notes,
        }]
    if task == "summarization":
        return [{
            "chart_id": chart_id, "pred_summary": "",
            "error_type": error_type, "notes": notes,
        }]
    if task == "table_extraction":
        return [{
            "chart_id": chart_id, "series": "", "category": "",
            "pred_value": "", "error_type": error_type, "notes": notes,
        }]
    return []


def _process_row(
    task: str,
    row: pd.Series,
    model_name: str,
    cfg: ProjectConfig,
    args,
    image_lookup: Dict[str, str],
    hf_cache: Dict[str, Any],
) -> List[dict]:
    chart_id = safe_str(row.get("chart_id"))
    project_root = cfg.project_root
    data_dir = Path(args.data_dir)
    raw_data_dir = Path(args.raw_data_dir)
    image_path = resolve_image_path(chart_id, image_lookup, project_root, data_dir, raw_data_dir)
    if not image_path:
        return _error_row(task, row, "missing_image", "Could not resolve image path.")

    model, temp, max_tok = _gemini_cfg(cfg)

    if task == "qa":
        qid = safe_str(row.get("question_id"))
        question = safe_str(row.get("question"))
        answer_type = safe_str(row.get("answer_type"))

        if model_name == "classical_cv_ocr_pipeline":
            ocr_text, err = _ocr(image_path, args)
            ans = answer_from_context(question, answer_type, ocr_text) if ocr_text else "unknown"
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": err, "notes": f"OCR heuristic; {image_path}"}]

        if model_name == "chartocr_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            structure = detect_chart_structure(image_path)
            ctx = ocr_text + "\n" + str(structure.get("possible_axis_labels", []))
            ans = answer_from_context(question, answer_type, ctx) if ocr_text else "unknown"
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": err, "notes": f"ChartOCR heuristic; {image_path}"}]

        if model_name == "ocr_gemini_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            prompt = cfg.prompts.get("qa_ocr_context", "").format(
                question=question, ocr_text=ocr_text,
                numeric_tokens=", ".join(extract_number_tokens(ocr_text)),
                text_tokens=", ".join(ocr_text.split()[:80]),
            )
            raw, gerr = gemini_text(prompt, cfg.gemini_api_key or "", model, temp, max_tok)
            ans, perr = parse_gemini_answer(raw, gerr)
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": perr or err, "notes": "OCR+Gemini text-only"}]

        if model_name == "deplot_table_gemini_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            prompt = cfg.prompts.get("qa_table_context", "").format(
                question=question, table_text=table_text,
            )
            raw, gerr = gemini_text(prompt, cfg.gemini_api_key or "", model, temp, max_tok)
            ans, perr = parse_gemini_answer(raw, gerr)
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": perr or derr, "notes": "DePlot+Gemini"}]

        if model_name in CHART_QA_HF_MODELS:
            cache_key = f"chart_qa_model:{model_name}"
            checkpoint = CHART_QA_HF_MODELS[model_name]
            if cache_key not in hf_cache:
                hf_cache[cache_key] = HFImageTextModel(
                    checkpoint["model_id"],
                    device=args.device, max_new_tokens=int(args.max_new_tokens),
                    processor_id=checkpoint["processor_id"],
                )
            try:
                ans = hf_cache[cache_key].generate(image_path, question)
                err = ""
            except Exception as exc:
                ans, err = "unknown", f"hf_error:{type(exc).__name__}"
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": safe_str(ans),
                     "error_type": err,
                     "notes": f"Local chart QA checkpoint: {checkpoint['model_id']}"}]

        if model_name == "table_symbolic_reasoner_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            ans, serr = symbolic_answer(question, table_text)
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": serr or derr, "notes": "Symbolic reasoner"}]

        if model_name == "gemini_end_to_end":
            prompt = cfg.prompts.get("qa_end_to_end", "{question}").format(question=question)
            raw, gerr = gemini_vision(image_path, prompt, cfg.gemini_api_key or "", model, temp, max_tok)
            ans, perr = parse_gemini_answer(raw, gerr)
            return [{"question_id": qid, "chart_id": chart_id, "pred_answer": ans,
                     "error_type": perr, "notes": "Gemini vision"}]

    if task == "verification":
        cid = safe_str(row.get("claim_id"))
        claim = safe_str(row.get("claim"))
        question, claimed_answer, answer_type = _qa_claim_fields(row)
        numerical_tolerance = float(cfg.evaluation.get("numerical_tolerance", 0.05))

        if model_name == "gemini_end_to_end":
            prompt = cfg.prompts.get("verification", "").format(
                claim=claim,
                context="(see image)",
            )
            raw, gerr = gemini_vision(
                image_path,
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            label, perr = parse_verification_label(raw, gerr)
            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": raw,
                "error_type": perr,
                "notes": "Gemini vision",
            }]

        if model_name == "deplot_table_gemini_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            prompt = cfg.prompts.get("verification", "").format(
                claim=claim,
                context=table_text,
            )
            raw, gerr = gemini_text(
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            label, perr = parse_verification_label(raw, gerr)
            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": raw,
                "error_type": perr or derr,
                "notes": "DePlot table + Gemini",
            }]

        if model_name == "ocr_gemini_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            prompt = cfg.prompts.get("verification", "").format(
                claim=claim,
                context=ocr_text,
            )
            raw, gerr = gemini_text(
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            label, perr = parse_verification_label(raw, gerr)
            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": raw,
                "error_type": perr or err,
                "notes": "OCR + Gemini",
            }]

        if model_name in CHART_QA_HF_MODELS:
            cache_key = f"chart_qa_model:{model_name}"
            checkpoint = CHART_QA_HF_MODELS[model_name]
            if cache_key not in hf_cache:
                hf_cache[cache_key] = HFImageTextModel(
                    checkpoint["model_id"],
                    device=args.device,
                    max_new_tokens=int(args.max_new_tokens),
                    processor_id=checkpoint["processor_id"],
                )
            try:
                label, raw, err = _pix2struct_verification(
                    claim,
                    image_path,
                    hf_cache[cache_key],
                    question=question,
                    claimed_answer=claimed_answer,
                    answer_type=answer_type,
                    numerical_tolerance=numerical_tolerance,
                )
            except Exception as exc:
                label = "unverifiable"
                raw = ""
                err = f"hf_error:{type(exc).__name__}"

            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": raw,
                "error_type": err,
                "notes": (
                    f"{checkpoint['model_id']} answer compared with claimed answer"
                ),
            }]

        if model_name == "table_symbolic_reasoner_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            if question and claimed_answer:
                predicted_answer, reasoner_error = symbolic_answer(question, table_text)
                label = _verification_label_from_answer(
                    claimed_answer,
                    predicted_answer,
                    answer_type,
                    numerical_tolerance,
                )
                err = derr
            else:
                predicted_answer = ""
                label = "unverifiable"
                reasoner_error = "unsupported_claim_format"
                err = derr

            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": predicted_answer,
                "error_type": err,
                "notes": f"Table symbolic verification; {reasoner_error}".rstrip("; "),
            }]

        if model_name == "chartocr_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            structure = detect_chart_structure(image_path)
            context = ocr_text + "\n" + str(structure.get("possible_axis_labels", []))
            predicted_answer = (
                answer_from_context(question, answer_type, context)
                if question and claimed_answer and ocr_text
                else "unknown"
            )
            label = _verification_label_from_answer(
                claimed_answer,
                predicted_answer,
                answer_type,
                numerical_tolerance,
            ) if claimed_answer else "unverifiable"

            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": predicted_answer,
                "error_type": err,
                "notes": "ChartOCR heuristic verification",
            }]

        if model_name == "classical_cv_ocr_pipeline":
            ocr_text, err = _ocr(image_path, args)
            predicted_answer = (
                answer_from_context(question, answer_type, ocr_text)
                if question and claimed_answer and ocr_text
                else "unknown"
            )
            label = _verification_label_from_answer(
                claimed_answer,
                predicted_answer,
                answer_type,
                numerical_tolerance,
            ) if claimed_answer else "unverifiable"

            return [{
                "claim_id": cid,
                "chart_id": chart_id,
                "pred_label": label,
                "raw_output": predicted_answer,
                "error_type": err,
                "notes": "Classical OCR heuristic verification",
            }]

    if task == "summarization":
        if model_name == "gemini_end_to_end":
            prompt = cfg.prompts.get("summarization", "").format(context="(see image)")
            raw, gerr = gemini_vision(
                image_path,
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            summary, perr = parse_gemini_answer(raw, gerr)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": perr,
                "notes": "Gemini vision summary",
            }]

        if model_name == "deplot_table_gemini_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            prompt = cfg.prompts.get("summarization", "").format(context=table_text)
            raw, gerr = gemini_text(
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            summary, perr = parse_gemini_answer(raw, gerr)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": perr or derr,
                "notes": "DePlot table + Gemini summary",
            }]

        if model_name == "ocr_gemini_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            prompt = cfg.prompts.get("summarization", "").format(context=ocr_text)
            raw, gerr = gemini_text(
                prompt,
                cfg.gemini_api_key or "",
                model,
                temp,
                max_tok,
            )
            summary, perr = parse_gemini_answer(raw, gerr)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": perr or err,
                "notes": "OCR + Gemini summary",
            }]

        if model_name == "pix2struct_ocr_free_pipeline":
            if "pix2struct" not in hf_cache:
                hf_cache["pix2struct"] = HFImageTextModel(
                    "google/pix2struct-chartqa-base",
                    device=args.device,
                    max_new_tokens=int(args.max_new_tokens),
                )
            try:
                summary = hf_cache["pix2struct"].generate(
                    image_path,
                    "Summarize the chart in one concise paragraph.",
                )
                err = ""
            except Exception as exc:
                summary = ""
                err = f"hf_error:{type(exc).__name__}"

            return [{
                "chart_id": chart_id,
                "pred_summary": safe_str(summary),
                "error_type": err,
                "notes": "Pix2Struct summary",
            }]

        if model_name == "table_symbolic_reasoner_pipeline":
            table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
            summary = normalize_summary(table_text)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": derr or "table_symbolic_summary",
                "notes": "Table symbolic summary",
            }]

        if model_name == "chartocr_reasoning_pipeline":
            ocr_text, err = _ocr(image_path, args)
            structure = detect_chart_structure(image_path)
            context = ocr_text + "\n" + str(structure.get("possible_axis_labels", []))
            summary = normalize_summary(context)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": err,
                "notes": "ChartOCR summary",
            }]

        if model_name == "classical_cv_ocr_pipeline":
            ocr_text, err = _ocr(image_path, args)
            summary = normalize_summary(ocr_text)
            return [{
                "chart_id": chart_id,
                "pred_summary": summary,
                "error_type": err,
                "notes": "Classical OCR summary",
            }]

    if task == "table_extraction":
        if model_name not in ("deplot_table_gemini_pipeline", "gemini_end_to_end"):
            return _error_row(task, row, "unsupported_model_for_task",
                              f"{model_name} does not support table_extraction")
        table_text, derr = _deplot_table(chart_id, image_path, hf_cache, args)
        return _deplot_to_table_rows(chart_id, table_text, derr)

    raise ValueError(f"Unhandled task={task} model={model_name}")


def normalize_summary(ocr_text: str, max_len: int = 400) -> str:
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    text = " ".join(lines)
    return text[:max_len] if text else ""


def run_pipeline(
    model_name: str,
    task: str,
    rows: pd.DataFrame,
    charts: pd.DataFrame,
    cfg: ProjectConfig,
    args,
) -> Path:
    if model_name not in IMPLEMENTED_MODELS:
        raise ValueError(f"Unknown model {model_name!r}")
    if task not in MODEL_TASKS.get(model_name, set()):
        raise ValueError(f"Model {model_name!r} does not support task {task!r}")

    cfg.validate_for_model(model_name)

    # Configure Tesseract once at start (validates eng.traineddata).
    if model_name in ("classical_cv_ocr_pipeline", "chartocr_reasoning_pipeline", "ocr_gemini_reasoning_pipeline"):
        resolve_tesseract(
            getattr(args, "tesseract_cmd", None) or None,
            os.environ.get("TESSDATA_PREFIX"),
        )

    spec = TASK_SPEC[task]
    id_col = spec["id_col"]
    out_file = spec["output_file"]
    columns = spec["columns"]

    predictions_dir = Path(args.predictions_dir)
    model_dir = predictions_dir / model_name
    pred_path = model_dir / out_file

    existing: Dict[str, List[dict]] = {}
    if not args.force and pred_path.exists():
        for rid, rowdict in load_existing_rows(pred_path, id_col).items():
            existing[rid] = [rowdict]

    if args.save_intermediates:
        init_intermediates_file(model_dir, force=args.force)

    image_lookup = build_image_lookup(charts, Path(args.raw_data_dir))
    hf_cache: Dict[str, Any] = {}
    if _model_uses_deplot(model_name, task):
        _precompute_deplot_tables(rows, cfg, args, image_lookup, hf_cache)

    total = len(rows)
    batch_size = max(0, int(getattr(args, "batch_size", 0) or 0))
    retry_errors = bool(getattr(args, "retry_errors", False))
    processed_new = 0
    stop_reason = ""

    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        rid = safe_str(row.get(id_col))
        if not args.force and rid in existing:
            cached_error = safe_str(existing[rid][0].get("error_type"))
            if not (retry_errors and is_prediction_failure(cached_error)):
                continue
        if batch_size and processed_new >= batch_size:
            break
        try:
            new_rows = _process_row(task, row, model_name, cfg, args, image_lookup, hf_cache)
        except Exception as exc:
            new_rows = _error_row(task, row, f"pipeline_error:{type(exc).__name__}", str(exc))
        existing[rid] = new_rows
        processed_new += 1

        quota_error = next(
            (
                safe_str(item.get("error_type"))
                for item in new_rows
                if is_gemini_quota_error(item.get("error_type"))
            ),
            "",
        )
        if model_uses_gemini(model_name, task) and quota_error:
            marker = _write_gemini_quota_marker(
                predictions_dir,
                model_name,
                task,
                rid,
                quota_error,
                int(getattr(args, "gemini_cooldown_minutes", 60)),
            )
            stop_reason = f"gemini_quota_paused:{marker}"
            print(
                "Gemini quota exhausted. Saved the current checkpoint and paused "
                f"Gemini approaches for {int(getattr(args, 'gemini_cooldown_minutes', 60))} "
                f"minutes. Marker: {marker.resolve()}"
            )
            _flush(existing, rows, id_col, pred_path, columns)
            break

        if args.verbose and idx % 10 == 0:
            print(f"  processed {idx}/{total}")

        if idx % 25 == 0:
            _flush(existing, rows, id_col, pred_path, columns)

    _flush(existing, rows, id_col, pred_path, columns)
    _write_progress_csv(
        model_dir, task, rows, existing, processed_new, stop_reason=stop_reason,
    )
    return pred_path


def _flush(existing: Dict[str, List[dict]], rows: pd.DataFrame, id_col: str, path: Path, columns: List[str]) -> None:
    ordered: List[dict] = []
    for _, row in rows.iterrows():
        rid = safe_str(row.get(id_col))
        if rid in existing:
            ordered.extend(existing[rid])
    write_task_csv(ordered, path, columns)


def _write_progress_csv(
    model_dir: Path,
    task: str,
    rows: pd.DataFrame,
    existing: Dict[str, List[dict]],
    processed_new: int,
    stop_reason: str = "",
) -> None:
    spec = TASK_SPEC[task]
    target_ids = {safe_str(row.get(spec["id_col"])) for _, row in rows.iterrows()}
    records = [
        values[0] for rid, values in existing.items()
        if rid in target_ids and values
    ]
    errors = sum(is_prediction_failure(record.get("error_type")) for record in records)
    informational = sum(
        bool(safe_str(record.get("error_type")))
        and not is_prediction_failure(record.get("error_type"))
        for record in records
    )
    completed = len(records)
    progress_path = model_dir / "run_progress.csv"
    new_row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "target_samples": len(rows),
        "completed_samples": completed,
        "successful_samples": completed - errors,
        "error_samples": errors,
        "informational_samples": informational,
        "remaining_samples": max(0, len(rows) - completed),
        "new_samples_this_run": processed_new,
        "status": "PAUSED_GEMINI_QUOTA" if stop_reason else (
            "COMPLETE" if completed >= len(rows) else "IN_PROGRESS"
        ),
        "stop_reason": stop_reason,
    }])
    if progress_path.exists():
        try:
            history = pd.read_csv(progress_path)
            new_row = pd.concat([history, new_row], ignore_index=True)
        except Exception:
            pass
    tmp = progress_path.with_suffix(".csv.tmp")
    new_row.to_csv(tmp, index=False)
    tmp.replace(progress_path)
