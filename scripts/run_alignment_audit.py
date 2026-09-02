"""Measure perceptual alignment from OCR artefacts and ChartQA table labels.

The run is resumable.  OCR text and ChartOCR-style structural detections are
saved per chart before aggregate label coverage is calculated.  This replaces
the former value-retrieval-accuracy proxy with a direct intermediate-output
measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.chart_structure import detect_chart_structure  # noqa: E402
from pipelines.common import (  # noqa: E402
    build_image_lookup,
    resolve_image_path,
    run_tesseract_ocr,
    safe_str,
)
from pipelines.tesseract_config import resolve_tesseract  # noqa: E402


def _normalise(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", safe_str(value).casefold()))


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _reference_labels(tables: pd.DataFrame, chart_ids: set[str]) -> pd.DataFrame:
    records = []
    subset = tables[tables["chart_id"].astype(str).isin(chart_ids)]
    for chart_id, frame in subset.groupby("chart_id"):
        for label_type, column in (("series", "series"), ("category", "category")):
            for label in frame[column].fillna("").astype(str).drop_duplicates():
                normalised = _normalise(label)
                if normalised:
                    records.append({
                        "chart_id": chart_id,
                        "label_type": label_type,
                        "reference_label": label,
                        "normalised_label": normalised,
                    })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed_review")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument(
        "--experiment-dir", default="results/experiments/review_test_stratified_1000",
    )
    parser.add_argument("--limit-charts", type=int, default=0)
    parser.add_argument("--tesseract-cmd", default="")
    parser.add_argument("--tessdata-prefix", default="")
    args = parser.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    raw_data_dir = (ROOT / args.raw_data_dir).resolve()
    experiment_dir = (ROOT / args.experiment_dir).resolve()
    output_dir = experiment_dir / "alignment_audit"
    artifacts_path = output_dir / "ocr_chart_artifacts.csv"

    questions = pd.read_csv(data_dir / "review_qa_cohort.csv", dtype=str).fillna("")
    charts = pd.read_csv(data_dir / "charts.csv", dtype=str).fillna("")
    tables = pd.read_csv(data_dir / "tables_gt.csv", dtype=str).fillna("")
    chart_ids = questions["chart_id"].astype(str).drop_duplicates().tolist()
    if args.limit_charts > 0:
        chart_ids = chart_ids[:args.limit_charts]

    settings = resolve_tesseract(
        args.tesseract_cmd or None,
        args.tessdata_prefix or None,
    )
    image_lookup = build_image_lookup(charts, raw_data_dir)
    if artifacts_path.exists():
        artifacts = pd.read_csv(artifacts_path, dtype=str).fillna("")
        existing = set(artifacts["chart_id"].astype(str))
        records = artifacts.to_dict("records")
    else:
        existing = set()
        records = []

    for index, chart_id in enumerate(chart_ids, start=1):
        if chart_id in existing:
            continue
        image_path = resolve_image_path(
            chart_id, image_lookup, ROOT, data_dir, raw_data_dir,
        )
        if image_path is None:
            records.append({
                "chart_id": chart_id,
                "image_path": "",
                "ocr_text": "",
                "ocr_error": "missing_image",
                "detected_lines": "[]",
                "detected_components": "[]",
                "structure_error": "missing_image",
            })
            continue
        ocr_text, ocr_error = run_tesseract_ocr(
            image_path, settings.cmd, str(settings.tessdata_dir),
        )
        structure = detect_chart_structure(image_path)
        records.append({
            "chart_id": chart_id,
            "image_path": str(image_path),
            "ocr_text": ocr_text,
            "ocr_error": ocr_error,
            "detected_lines": json.dumps(structure.get("detected_lines", [])),
            "detected_components": json.dumps(structure.get("detected_components", [])),
            "structure_error": structure.get("error", ""),
        })
        if index % 10 == 0:
            _write_atomic(pd.DataFrame(records), artifacts_path)
            print(f"Saved OCR artefacts: {index}/{len(chart_ids)}")
    artifacts = pd.DataFrame(records)
    _write_atomic(artifacts, artifacts_path)

    references = _reference_labels(tables, set(chart_ids))
    ocr_by_chart = {
        row.chart_id: _normalise(row.ocr_text)
        for row in artifacts.itertuples()
        if not safe_str(row.ocr_error)
    }
    references["ocr_available"] = references["chart_id"].isin(ocr_by_chart)
    references["label_found"] = references.apply(
        lambda row: (
            row["normalised_label"] in ocr_by_chart.get(row["chart_id"], "")
            if row["ocr_available"] else False
        ),
        axis=1,
    )
    _write_atomic(references, output_dir / "perceptual_label_matches.csv")
    summary = (
        references.groupby("label_type", as_index=False)
        .agg(
            reference_labels=("normalised_label", "size"),
            labels_found=("label_found", "sum"),
            labels_on_charts_with_ocr=("ocr_available", "sum"),
        )
    )
    summary["label_recall"] = summary["labels_found"] / summary["reference_labels"].clip(lower=1)
    overall = pd.DataFrame([{
        "label_type": "all",
        "reference_labels": len(references),
        "labels_found": int(references["label_found"].sum()),
        "labels_on_charts_with_ocr": int(references["ocr_available"].sum()),
        "unique_charts_with_ocr": int(artifacts["ocr_error"].eq("").sum()),
        "label_recall": float(references["label_found"].mean()) if len(references) else float("nan"),
    }])
    summary = pd.concat([summary, overall], ignore_index=True)
    _write_atomic(summary, output_dir / "perceptual_label_coverage.csv")

    print(f"OCR artefacts: {artifacts_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
