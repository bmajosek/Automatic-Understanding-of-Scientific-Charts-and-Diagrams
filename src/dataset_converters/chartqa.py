"""ChartQA dataset converter.

Converts the downloaded Hugging Face ChartQA repository layout, including:

    data/raw/chartqa/ChartQA Dataset/ChartQA Dataset/{train,val,test}/...

into the project CSV format used by the evaluator:

    data/processed/charts.csv
    data/processed/questions.csv

The official ChartQA archive also contains one CSV table per chart.  These
tables are converted to the project's long-form ``tables_gt.csv`` schema so
that chart-to-table quality can be measured directly rather than inferred
from downstream QA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    from .base import BaseConverter
except ImportError:  # allows running the file directly during debugging
    from base import BaseConverter


class ChartQAConverter(BaseConverter):
    """Convert ChartQA raw files into the standard evaluation CSV schema."""

    @property
    def dataset_name(self) -> str:
        return "chartqa"

    def _candidate_roots(self) -> List[Path]:
        """Return likely ChartQA roots, ordered from most to least likely."""
        roots: List[Path] = []

        # main.py passes raw_data_dir as data/raw, so include data/raw/chartqa first
        if (self.raw_data_dir / "chartqa").exists():
            roots.append(self.raw_data_dir / "chartqa")

        # also allow passing data/raw/chartqa directly
        if self.raw_data_dir.exists():
            roots.append(self.raw_data_dir)

        # Hugging Face zip extraction commonly nests "ChartQA Dataset/ChartQA Dataset"
        expanded: List[Path] = []
        for root in roots:
            expanded.append(root)
            expanded.extend([p for p in root.rglob("*") if p.is_dir() and self._has_split_dirs(p)])

        # de-duplicate while preserving order
        seen = set()
        unique: List[Path] = []
        for root in expanded:
            key = root.resolve() if root.exists() else root
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    @staticmethod
    def _has_split_dirs(path: Path) -> bool:
        names = {p.name.lower() for p in path.iterdir()} if path.exists() and path.is_dir() else set()
        return bool(names & {"train", "val", "valid", "validation", "test"})

    def _dataset_root(self) -> Optional[Path]:
        """Find the directory that contains split folders and JSON annotations."""
        for root in self._candidate_roots():
            if not root.exists():
                continue
            if self._has_split_dirs(root) and list(root.rglob("*.json")):
                return root
        return None

    def check_raw_data(self) -> bool:
        root = self._dataset_root()
        return root is not None and any(root.rglob("*.json"))

    def convert(self) -> Dict[str, pd.DataFrame]:
        root = self._dataset_root()
        if root is None:
            raise FileNotFoundError(
                "Could not find ChartQA raw data. Expected a directory like "
                "data/raw/chartqa/ChartQA Dataset/ChartQA Dataset with train/val/test folders."
            )

        question_rows: List[Dict[str, Any]] = []
        chart_rows_by_id: Dict[str, Dict[str, Any]] = {}
        table_rows: List[Dict[str, Any]] = []

        json_files = sorted(p for p in root.rglob("*.json") if p.is_file())
        if not json_files:
            raise FileNotFoundError(f"No ChartQA JSON files found under {root}")

        for json_path in json_files:
            split = self._split_from_path(json_path, root)
            if split is None:
                # Ignore non-annotation metadata JSONs outside train/val/test.
                continue

            records = self._load_records(json_path)
            for idx, record in enumerate(records):
                image_name = self._get_first(
                    record,
                    [
                        "imgname",
                        "image",
                        "image_name",
                        "image_path",
                        "filename",
                        "file_name",
                        "img",
                    ],
                )
                question = self._get_first(record, ["query", "question", "question_string", "q"])
                answer = self._get_first(record, ["label", "answer", "answers", "ans"])

                if question is None or answer is None:
                    self.add_warning(
                        self.dataset_name,
                        f"{json_path.name}:{idx}",
                        "Missing question or answer field",
                        record,
                    )
                    continue

                image_path = self._resolve_image_path(root, json_path, image_name, split)
                chart_id = self._chart_id(image_name, image_path, json_path, idx)

                chart_rows_by_id.setdefault(
                    chart_id,
                    {
                        "chart_id": chart_id,
                        "image_path": self._path_for_csv(image_path) if image_path else str(image_name or ""),
                        "chart_type": self._infer_chart_type(record),
                        "source": "chartqa",
                        "split": split,
                        "width": None,
                        "height": None,
                        "notes": f"annotation_file={json_path.name}",
                    },
                )

                answer_str = self._stringify_answer(answer)
                question_str = str(question).strip()
                source_kind = str(record.get("human_or_machine", record.get("source", ""))).strip()
                if not source_kind:
                    stem = json_path.stem.lower()
                    if "augmented" in stem:
                        source_kind = "augmented"
                    elif "human" in stem:
                        source_kind = "human"

                question_rows.append(
                    {
                        "question_id": f"chartqa_{split}_{json_path.stem}_{idx:06d}",
                        "chart_id": chart_id,
                        "split": split,
                        "task": self._infer_task(question_str),
                        "operation": self._infer_operation(question_str),
                        "question": question_str,
                        "answer": answer_str,
                        "answer_type": self._infer_answer_type(answer_str),
                        "paraphrase_group": None,
                        "dialogue_id": None,
                        "turn_id": None,
                        "source_dataset": "chartqa",
                        "question_origin": source_kind or "unknown",
                        "language": "en",
                    }
                )

                # Some variants may include table-like data. Keep it if present.
                table_rows.extend(self._extract_table_rows(record, chart_id))

        # The distributed ChartQA archive stores reference tables beside the
        # images.  Earlier versions of this converter inspected only fields in
        # the QA JSON and therefore produced an empty tables_gt.csv.
        table_rows.extend(self._extract_distributed_tables(root, set(chart_rows_by_id)))

        charts_df = pd.DataFrame(
            list(chart_rows_by_id.values()),
            columns=["chart_id", "image_path", "chart_type", "source", "split", "width", "height", "notes"],
        )
        questions_df = pd.DataFrame(
            question_rows,
            columns=[
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
                "question_origin",
                "language",
            ],
        )
        tables_df = pd.DataFrame(table_rows, columns=["chart_id", "series", "category", "value"])

        if charts_df.empty or questions_df.empty:
            raise RuntimeError(
                f"ChartQA conversion produced charts={len(charts_df)} and questions={len(questions_df)}. "
                f"Check JSON schema under {root}."
            )

        output: Dict[str, pd.DataFrame] = {
            "charts": charts_df,
            "questions": questions_df,
        }
        if not tables_df.empty:
            output["tables_gt"] = tables_df

        if self.warnings:
            # Save warnings next to outputs for debugging.
            self.save_warnings(self.output_dir / "conversion_warnings.csv")

        return output

    @staticmethod
    def _load_records(json_path: Path) -> List[Dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]

        if isinstance(payload, dict):
            # Common wrappers used by dataset exports.
            for key in ["data", "annotations", "questions", "qas", "examples", "items"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]

            # Dict of records keyed by id.
            if all(isinstance(v, dict) for v in payload.values()):
                return [v for v in payload.values() if isinstance(v, dict)]

        return []

    @staticmethod
    def _get_first(record: Dict[str, Any], keys: Iterable[str]) -> Any:
        for key in keys:
            if key in record and record[key] not in (None, ""):
                return record[key]
        return None

    @staticmethod
    def _split_from_path(path: Path, root: Path) -> Optional[str]:
        parts = [p.lower() for p in path.relative_to(root).parts]
        for part in parts:
            if part in {"train", "training"}:
                return "train"
            if part in {"val", "valid", "validation", "dev"}:
                return "validation"
            if part == "test":
                return "test"
        return None

    @staticmethod
    def _path_for_csv(path: Path) -> str:
        try:
            return path.as_posix()
        except Exception:
            return str(path)

    @staticmethod
    def _chart_id(image_name: Any, image_path: Optional[Path], json_path: Path, idx: int) -> str:
        if image_path is not None:
            return image_path.stem
        if image_name:
            return Path(str(image_name)).stem
        return f"{json_path.stem}_{idx:06d}"

    @staticmethod
    def _stringify_answer(answer: Any) -> str:
        if isinstance(answer, list):
            if len(answer) == 1:
                return str(answer[0]).strip()
            return " | ".join(str(x).strip() for x in answer)
        if isinstance(answer, dict):
            for key in ["text", "answer", "label", "value"]:
                if key in answer:
                    return str(answer[key]).strip()
            return json.dumps(answer, ensure_ascii=False)
        return str(answer).strip()

    @staticmethod
    def _infer_answer_type(answer: str) -> str:
        s = str(answer).strip().lower()
        if s in {"yes", "no", "true", "false", "supported", "contradicted"}:
            return "boolean"
        numeric = s.replace(",", "").replace("%", "").replace("$", "")
        try:
            float(numeric)
            return "numeric"
        except ValueError:
            pass
        # Short labels such as "Q2" or "USA" are categorical; longer answers are text.
        return "categorical" if len(s.split()) <= 5 else "text"

    @staticmethod
    def _infer_operation(question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["greater", "higher", "lower", "less", "more", "compare", "difference between"]):
            return "compare"
        if any(w in q for w in ["sum", "total", "altogether"]):
            return "sum"
        if any(w in q for w in ["difference", "subtract", "increase", "decrease"]):
            return "difference"
        if any(w in q for w in ["maximum", "highest", "largest", "most"]):
            return "max"
        if any(w in q for w in ["minimum", "lowest", "smallest", "least"]):
            return "min"
        if any(w in q for w in ["average", "mean"]):
            return "average"
        if any(w in q for w in ["ratio", "percentage", "percent"]):
            return "ratio"
        if any(w in q for w in ["trend", "increase", "decrease", "rise", "fall"]):
            return "trend_detection"
        return "read_value"

    @classmethod
    def _infer_task(cls, question: str) -> str:
        op = cls._infer_operation(question)
        if op == "read_value":
            return "value_retrieval"
        if op == "compare":
            return "comparison"
        if op in {"sum", "average", "ratio", "difference"}:
            return "arithmetic"
        if op == "trend_detection":
            return "trend"
        if op in {"max", "min"}:
            return "reasoning"
        return "reasoning"

    @staticmethod
    def _infer_chart_type(record: Dict[str, Any]) -> str:
        for key in ["chart_type", "type", "plot_type"]:
            value = record.get(key)
            if value:
                return str(value).lower().replace(" ", "_")
        return "unknown"

    @staticmethod
    def _resolve_image_path(root: Path, json_path: Path, image_name: Any, split: str) -> Optional[Path]:
        if not image_name:
            return None

        image_name_str = str(image_name).replace("\\", "/").strip()
        image_candidate = Path(image_name_str)

        candidates: List[Path] = []
        if image_candidate.is_absolute():
            candidates.append(image_candidate)
        else:
            candidates.extend(
                [
                    root / image_candidate,
                    json_path.parent / image_candidate,
                    json_path.parent / "png" / image_candidate.name,
                    json_path.parent / "images" / image_candidate.name,
                    root / split / image_candidate,
                    root / split / "png" / image_candidate.name,
                    root / split / "images" / image_candidate.name,
                ]
            )
            if split == "validation":
                candidates.extend(
                    [
                        root / "val" / image_candidate,
                        root / "val" / "png" / image_candidate.name,
                        root / "val" / "images" / image_candidate.name,
                    ]
                )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fallback: search by basename or stem under root. This is slower but robust.
        basename = image_candidate.name
        if basename:
            matches = list(root.rglob(basename))
            if matches:
                return matches[0]

        stem = image_candidate.stem
        if stem:
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                matches = list(root.rglob(stem + ext))
                if matches:
                    return matches[0]

        return json_path.parent / image_candidate

    @staticmethod
    def _extract_table_rows(record: Dict[str, Any], chart_id: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        table = None
        for key in ["table", "data_table", "table_data"]:
            if key in record:
                table = record[key]
                break

        if isinstance(table, list):
            for i, row in enumerate(table):
                if isinstance(row, dict):
                    rows.append(
                        {
                            "chart_id": chart_id,
                            "series": str(row.get("series", row.get("name", ""))),
                            "category": str(row.get("category", row.get("x", row.get("label", i)))),
                            "value": str(row.get("value", row.get("y", ""))),
                        }
                    )
        return rows

    @classmethod
    def _extract_distributed_tables(
        cls,
        root: Path,
        known_chart_ids: set[str],
    ) -> List[Dict[str, str]]:
        """Convert ChartQA's wide per-chart CSV files to long-form cells."""
        rows: List[Dict[str, str]] = []
        for csv_path in sorted(root.rglob("tables/*.csv")):
            chart_id = csv_path.stem
            if known_chart_ids and chart_id not in known_chart_ids:
                continue
            try:
                table = pd.read_csv(csv_path, dtype=str).fillna("")
            except Exception:
                continue
            if table.empty or len(table.columns) < 2:
                continue
            category_column = table.columns[0]
            for _, record in table.iterrows():
                category = str(record.get(category_column, "")).strip()
                for series in table.columns[1:]:
                    value = str(record.get(series, "")).strip()
                    if not value:
                        continue
                    rows.append({
                        "chart_id": chart_id,
                        "series": str(series).strip(),
                        "category": category,
                        "value": value,
                    })
        return rows
