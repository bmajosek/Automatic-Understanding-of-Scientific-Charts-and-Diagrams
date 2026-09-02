"""Generic Hugging Face chart-QA dataset converter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from dataset_catalog import get_entry, huggingface_repo_id, normalize_name
    from dataset_converters.base import BaseConverter
except ImportError:
    from ..dataset_catalog import get_entry, huggingface_repo_id, normalize_name
    from .base import BaseConverter

QUESTION_KEYS = ("question", "query", "q", "question_string", "instruction")
ANSWER_KEYS = ("answer", "label", "answers", "ans", "response", "output")
IMAGE_KEYS = ("image", "img", "chart", "figure", "screenshot")
SPLIT_KEYS = ("split", "data_split")


class HuggingFaceQAConverter(BaseConverter):
    """Convert HF datasets with image + question + answer fields."""

    def __init__(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        dataset_name: str,
        repo_id: Optional[str] = None,
        max_examples: Optional[int] = None,
    ):
        super().__init__(raw_data_dir, output_dir)
        self._dataset_name = normalize_name(dataset_name)
        self._repo_id = repo_id
        self._max_examples = max_examples

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _repo(self) -> str:
        if self._repo_id:
            return self._repo_id
        entry = get_entry(self._dataset_name)
        if entry:
            rid = huggingface_repo_id(entry.get("source_url"))
            if rid:
                return rid
        raise ValueError(f"No Hugging Face repo for {self._dataset_name}")

    def check_raw_data(self) -> bool:
        local = self.raw_data_dir / self._dataset_name
        if local.exists() and any(local.rglob("*")):
            return True
        try:
            from datasets import load_dataset  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_records(self) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets: pip install datasets") from exc

        repo = self._repo()
        local = self.raw_data_dir / self._dataset_name
        records: List[Dict[str, Any]] = []

        try:
            if local.exists():
                ds = load_dataset(str(local), trust_remote_code=True)
            else:
                ds = load_dataset(repo, trust_remote_code=True)
        except Exception:
            ds = load_dataset(repo, trust_remote_code=True)

        if hasattr(ds, "items"):
            for split_name, split_ds in ds.items():
                for row in split_ds:
                    rec = dict(row)
                    rec["_split"] = split_name
                    records.append(rec)
                    if self._max_examples and len(records) >= self._max_examples:
                        return records
        else:
            for row in ds:
                records.append(dict(row))
                if self._max_examples and len(records) >= self._max_examples:
                    break
        return records

    @staticmethod
    def _pick(record: Dict[str, Any], keys: tuple) -> Optional[Any]:
        for k in keys:
            if k in record and record[k] is not None:
                return record[k]
        return None

    def _save_image(self, image_obj: Any, chart_id: str, split: str) -> str:
        img_dir = self.raw_data_dir / self._dataset_name / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        out_path = img_dir / f"{chart_id}.png"
        if hasattr(image_obj, "save"):
            image_obj.save(out_path)
            return str(Path("data/raw") / self._dataset_name / "images" / split / f"{chart_id}.png")
        return ""

    def convert(self) -> Dict[str, pd.DataFrame]:
        records = self._load_records()
        if not records:
            raise ValueError(f"No records loaded for {self._dataset_name}")

        question_rows: List[Dict[str, Any]] = []
        chart_rows: Dict[str, Dict[str, Any]] = {}
        claim_rows: List[Dict[str, Any]] = []
        summary_by_chart: Dict[str, Dict[str, Any]] = {}

        for idx, record in enumerate(records):
            question = self._pick(record, QUESTION_KEYS)
            answer = self._pick(record, ANSWER_KEYS)
            if question is None or answer is None:
                continue

            split = str(record.get("_split") or self._pick(record, SPLIT_KEYS) or "test").lower()
            if split in ("validation", "valid"):
                split = "val"

            chart_id = f"{self._dataset_name}_{split}_{idx:06d}"
            image_obj = self._pick(record, IMAGE_KEYS)
            image_path = ""
            if image_obj is not None:
                try:
                    image_path = self._save_image(image_obj, chart_id, split)
                except Exception:
                    image_path = ""

            chart_rows[chart_id] = {
                "chart_id": chart_id,
                "image_path": image_path,
                "chart_type": str(record.get("chart_type", record.get("type", "unknown"))),
                "source": self._dataset_name,
                "split": split,
                "width": None,
                "height": None,
                "notes": f"hf={self._repo()}",
            }

            qstr = str(question).strip()
            astr = str(answer).strip()
            question_rows.append({
                "question_id": f"{self._dataset_name}_{split}_{idx:06d}",
                "chart_id": chart_id,
                "split": split,
                "task": self._infer_task(qstr),
                "operation": "other",
                "question": qstr,
                "answer": astr,
                "answer_type": self._infer_answer_type(astr),
                "paraphrase_group": None,
                "dialogue_id": record.get("dialogue_id"),
                "turn_id": record.get("turn_id"),
                "source_dataset": self._dataset_name,
                "language": record.get("language", "en"),
            })

            # Synthetic verification claim derived from QA ground truth.
            # This gives the project a non-QA factual verification task even for
            # datasets that only provide question/answer supervision.
            claim_rows.append({
                "claim_id": f"{self._dataset_name}_qa_claim_{idx:06d}",
                "chart_id": chart_id,
                "split": split,
                "claim": f"For the question '{qstr}', the answer shown by the chart is '{astr}'.",
                "label": "supported",
                "notes": "synthetic_from_qa",
                "source_dataset": self._dataset_name,
            })

            # Pseudo-summary derived from the available QA annotations for this chart.
            # It is not a human-written caption, but it lets summarization pipelines
            # and evaluation run on QA-only datasets.
            slot = summary_by_chart.setdefault(chart_id, {
                "chart_id": chart_id,
                "split": split,
                "summary_parts": [],
                "source_dataset": self._dataset_name,
                "notes": "synthetic_from_qa",
            })
            if len(slot["summary_parts"]) < 5:
                slot["summary_parts"].append(f"{qstr} Answer: {astr}")

            claim = record.get("claim") or record.get("statement")
            label = record.get("label_claim") or record.get("verdict")
            if claim and label and str(label).lower() in ("supported", "contradicted", "unverifiable", "true", "false"):
                claim_rows.append({
                    "claim_id": f"{self._dataset_name}_claim_{idx:06d}",
                    "chart_id": chart_id,
                    "split": split,
                    "claim": str(claim),
                    "label": self._norm_label(str(label)),
                    "notes": "",
                    "source_dataset": self._dataset_name,
                })

        charts_df = pd.DataFrame(list(chart_rows.values()))
        questions_df = pd.DataFrame(question_rows)
        claims_df = pd.DataFrame(claim_rows) if claim_rows else pd.DataFrame()
        summaries_df = pd.DataFrame([
            {
                "chart_id": chart_id,
                "split": meta["split"],
                "summary": " ".join(meta["summary_parts"]),
                "source_dataset": meta["source_dataset"],
                "notes": meta["notes"],
            }
            for chart_id, meta in summary_by_chart.items()
            if meta.get("summary_parts")
        ])

        if charts_df.empty or questions_df.empty:
            raise ValueError(f"Empty conversion for {self._dataset_name}")

        return {
            "charts": charts_df,
            "questions": questions_df,
            "tables_gt": pd.DataFrame(),
            "claims": claims_df,
            "components_gt": pd.DataFrame(),
            "summaries": summaries_df,
        }

    @staticmethod
    def _infer_task(question: str) -> str:
        q = question.lower()
        if any(w in q for w in ("sum", "total", "average", "mean")):
            return "arithmetic"
        if any(w in q for w in ("higher", "lower", "more", "less", "compare")):
            return "comparison"
        if "trend" in q:
            return "trend"
        if any(w in q for w in ("why", "how", "explain")):
            return "reasoning"
        return "value_retrieval"

    @staticmethod
    def _infer_answer_type(answer: str) -> str:
        a = answer.strip().lower()
        if a in ("yes", "no", "true", "false"):
            return "boolean"
        if re.match(r"^[-+]?\d", a.replace(",", "")):
            return "numeric"
        return "categorical"

    @staticmethod
    def _norm_label(label: str) -> str:
        l = label.lower().strip()
        if l in ("true", "yes", "supported"):
            return "supported"
        if l in ("false", "no", "contradicted"):
            return "contradicted"
        return "unverifiable"
