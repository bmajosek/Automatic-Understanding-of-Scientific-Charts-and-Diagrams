"""Prepare the deterministic evaluation cohort requested in the MSc review.

The script creates a separate processed-data directory.  It never modifies the
original ``data/processed`` files.  The QA cohort is sampled from the ChartQA
test split with equal human/augmented representation, and verification claims
are generated in three balanced classes.

Unverifiable claims are constructed by pairing a chart with a question from a
different chart.  They are exported to a manual-review queue because automatic
generation cannot prove that an unrelated concept is absent from every image.
The experiment should be reported as balanced only after that queue is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGIN_RE = re.compile(r"_(augmented|human)_", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin(row: pd.Series) -> str:
    explicit = str(row.get("question_origin", "")).strip().lower()
    if explicit in {"human", "augmented"}:
        return explicit
    match = ORIGIN_RE.search(str(row.get("question_id", "")))
    return match.group(1).lower() if match else "unknown"


def _stratified_test_cohort(
    questions: pd.DataFrame,
    split: str,
    limit: int,
    seed: int,
    human_fraction: float,
) -> pd.DataFrame:
    candidates = questions[
        questions["split"].fillna("").str.lower().eq(split.lower())
    ].copy()
    candidates["question_origin"] = candidates.apply(_origin, axis=1)
    human_target = round(limit * human_fraction)
    augmented_target = limit - human_target
    parts = []
    for origin, target in (("human", human_target), ("augmented", augmented_target)):
        group = candidates[candidates["question_origin"].eq(origin)]
        if len(group) < target:
            raise ValueError(
                f"Requested {target} {origin} questions, but only {len(group)} are available."
            )
        parts.append(group.sample(n=target, random_state=seed, replace=False))
    cohort = pd.concat(parts, ignore_index=True)
    # A second deterministic shuffle prevents one origin from occupying one
    # contiguous block while preserving exact stratum counts.
    return cohort.sample(frac=1, random_state=seed + 1).reset_index(drop=True)


def _parse_number(value: object) -> float | None:
    text = str(value).strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _contradicted_answer(row: pd.Series, alternatives: Iterable[str]) -> str:
    answer = str(row.get("answer", "")).strip()
    answer_type = str(row.get("answer_type", "")).strip().lower()
    if answer_type == "boolean":
        return "no" if answer.lower() in {"yes", "true"} else "yes"
    numeric = _parse_number(answer)
    if numeric is not None:
        # The change is safely outside the configured 5% relaxed tolerance,
        # including near zero.  Preserve an integer form where appropriate.
        delta = max(1.0, abs(numeric) * 0.20)
        changed = numeric + delta
        return str(int(round(changed))) if float(changed).is_integer() else f"{changed:.6g}"
    for candidate in alternatives:
        candidate = str(candidate).strip()
        if candidate and candidate.casefold() != answer.casefold():
            return candidate
    return f"not {answer}" if answer else "not reported"


def _claim_text(question: str, answer: str) -> str:
    return (
        f"For the chart question '{question}', "
        f"the answer shown by the chart is '{answer}'."
    )


def _balanced_claims(
    cohort: pd.DataFrame,
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cohort["chart_id"].nunique() < per_class:
        raise ValueError("Not enough distinct charts for the requested verification pairs.")
    base = (
        cohort.drop_duplicates("chart_id")
        .sample(n=per_class, random_state=seed + 2, replace=False)
        .reset_index(drop=True)
    )
    alternatives = cohort["answer"].dropna().astype(str).drop_duplicates().tolist()
    records: list[dict] = []
    review_records: list[dict] = []

    for index, row in base.iterrows():
        qid = str(row["question_id"])
        chart_id = str(row["chart_id"])
        question = str(row["question"])
        answer = str(row["answer"])
        common = {
            "chart_id": chart_id,
            "split": "test",
            "source_dataset": "chartqa",
            "source_question_id": qid,
            "question": question,
            "source_answer": answer,
            "answer_type": str(row.get("answer_type", "text")),
            "paired_claim_group": f"review_pair_{index:04d}",
        }

        supported = {
            **common,
            "claim_id": f"review_supported_{index:04d}",
            "claim": _claim_text(question, answer),
            "label": "supported",
            "claimed_answer": answer,
            "claim_transform": "identity",
            "notes": "review_balanced_from_qa",
            "requires_manual_review": "false",
        }
        records.append(supported)

        wrong = _contradicted_answer(row, alternatives)
        contradicted = {
            **common,
            "claim_id": f"review_contradicted_{index:04d}",
            "claim": _claim_text(question, wrong),
            "label": "contradicted",
            "claimed_answer": wrong,
            "claim_transform": "answer_changed_outside_tolerance",
            "notes": "review_balanced_from_qa",
            "requires_manual_review": "false",
        }
        records.append(contradicted)

        foreign = base.iloc[(index + 1) % len(base)]
        if str(foreign["chart_id"]) == chart_id:
            foreign = base.iloc[(index + 2) % len(base)]
        foreign_question = str(foreign["question"])
        foreign_answer = str(foreign["answer"])
        unverifiable = {
            **common,
            "claim_id": f"review_unverifiable_{index:04d}",
            "claim": _claim_text(foreign_question, foreign_answer),
            "label": "unverifiable",
            "claimed_answer": foreign_answer,
            "question": foreign_question,
            "answer_type": str(foreign.get("answer_type", "text")),
            "claim_transform": "question_from_different_chart",
            "notes": "manual_absence_check_required",
            "requires_manual_review": "true",
            "foreign_question_id": str(foreign["question_id"]),
            "foreign_chart_id": str(foreign["chart_id"]),
        }
        records.append(unverifiable)
        review_records.append(unverifiable)

    claims = pd.DataFrame(records).sample(frac=1, random_state=seed + 3).reset_index(drop=True)
    return claims, pd.DataFrame(review_records)


def _reference_tables(raw_root: Path, chart_ids: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(raw_root.rglob("tables/*.csv")):
        if path.stem not in chart_ids:
            continue
        try:
            table = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            continue
        if table.empty or len(table.columns) < 2:
            continue
        category_column = table.columns[0]
        for _, record in table.iterrows():
            category = str(record.get(category_column, "")).strip()
            for series in table.columns[1:]:
                value = str(record.get(series, "")).strip()
                if value:
                    rows.append({
                        "chart_id": path.stem,
                        "series": str(series).strip(),
                        "category": category,
                        "value": value,
                    })
    return pd.DataFrame(rows, columns=["chart_id", "series", "category", "value"])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", default="data/processed")
    parser.add_argument(
        "--raw-chartqa-root",
        default="data/raw/chartqa/ChartQA Dataset/ChartQA Dataset",
    )
    parser.add_argument("--output-dir", default="data/processed_review")
    parser.add_argument("--split", default="test")
    parser.add_argument("--qa-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-fraction", type=float, default=0.5)
    parser.add_argument("--verification-per-class", type=int, default=150)
    args = parser.parse_args()

    source = (ROOT / args.source_data_dir).resolve()
    raw_root = (ROOT / args.raw_chartqa_root).resolve()
    output = (ROOT / args.output_dir).resolve()
    if source == output:
        raise SystemExit("Refusing to overwrite the source processed-data directory.")
    output.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(source / "questions.csv", dtype=str).fillna("")
    cohort = _stratified_test_cohort(
        questions, args.split, args.qa_limit, args.seed, args.human_fraction,
    )
    # Keep training/validation rows after the selected test cohort so baselines
    # can estimate priors without using evaluation labels.
    non_test = questions[~questions["split"].str.lower().eq(args.split.lower())].copy()
    if "question_origin" not in non_test.columns:
        non_test["question_origin"] = non_test.apply(_origin, axis=1)
    questions_out = pd.concat([cohort, non_test], ignore_index=True, sort=False)

    claims, review_queue = _balanced_claims(
        cohort, args.verification_per_class, args.seed,
    )
    chart_ids = set(cohort["chart_id"].astype(str))
    charts = pd.read_csv(source / "charts.csv", dtype=str).fillna("")
    charts_out = charts[charts["chart_id"].astype(str).isin(chart_ids)].copy()
    tables = _reference_tables(raw_root, chart_ids)

    summaries_path = source / "summaries.csv"
    if summaries_path.exists():
        summaries = pd.read_csv(summaries_path, dtype=str).fillna("")
        summaries = summaries[summaries["chart_id"].astype(str).isin(chart_ids)].copy()
    else:
        summaries = pd.DataFrame(columns=["chart_id", "split", "summary", "source_dataset", "notes"])

    _write_csv(questions_out, output / "questions.csv")
    _write_csv(charts_out, output / "charts.csv")
    _write_csv(claims, output / "claims.csv")
    _write_csv(tables, output / "tables_gt.csv")
    _write_csv(summaries, output / "summaries.csv")
    _write_csv(pd.DataFrame(columns=["chart_id", "component_type"]), output / "components_gt.csv")
    _write_csv(cohort, output / "review_qa_cohort.csv")
    _write_csv(review_queue, output / "unverifiable_manual_review_queue.csv")
    if (source / "models.csv").exists():
        shutil.copy2(source / "models.csv", output / "models.csv")

    files = [
        output / "questions.csv",
        output / "charts.csv",
        output / "claims.csv",
        output / "tables_gt.csv",
        output / "summaries.csv",
        output / "review_qa_cohort.csv",
        output / "unverifiable_manual_review_queue.csv",
    ]
    manifest = {
        "purpose": "Supervisor-review evaluation cohort",
        "split": args.split,
        "sampling_strategy": "seeded stratified sample by ChartQA human/augmented origin",
        "seed": args.seed,
        "qa_target": len(cohort),
        "qa_unique_charts": int(cohort["chart_id"].nunique()),
        "qa_origin_counts": cohort["question_origin"].value_counts().to_dict(),
        "verification_label_counts": claims["label"].value_counts().to_dict(),
        "verification_generation": {
            "supported": "original ChartQA question and answer",
            "contradicted": "answer changed beyond the 5% tolerance",
            "unverifiable": "question from a different chart; manual absence check required",
        },
        "unverifiable_manual_review_required": len(review_queue),
        "table_reference_cells": len(tables),
        "summary_reference_source": "synthetic_from_qa; exploratory only, not Chart-to-Text",
        "files_sha256": {path.name: _sha256(path) for path in files},
    }
    manifest_path = output / "review_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Review data: {output}")
    print(f"QA cohort: {len(cohort)} ({manifest['qa_origin_counts']})")
    print(f"Verification claims: {len(claims)} ({manifest['verification_label_counts']})")
    print(f"Reference table cells: {len(tables)}")
    print(f"Manual unverifiable review queue: {output / 'unverifiable_manual_review_queue.csv'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
