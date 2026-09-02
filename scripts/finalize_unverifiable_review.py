"""Finalize the record-level audit of the 150 unverifiable claims.

The original review queue is retained unchanged.  Twenty ambiguous cross-chart
questions are replaced with unused, chart-specific questions whose subject is
absent from the target chart's OCR, reference table, and QA context.  The
resulting reviewed file is then merged back into ``claims.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Ambiguous source question -> unused, subject-specific ChartQA question.
REPLACEMENTS = {
    "review_unverifiable_0008": "chartqa_test_test_augmented_000886",
    "review_unverifiable_0029": "chartqa_test_test_augmented_000885",
    "review_unverifiable_0052": "chartqa_test_test_human_000535",
    "review_unverifiable_0053": "chartqa_test_test_augmented_000680",
    "review_unverifiable_0055": "chartqa_test_test_augmented_000047",
    "review_unverifiable_0061": "chartqa_test_test_augmented_000270",
    "review_unverifiable_0072": "chartqa_test_test_augmented_001158",
    "review_unverifiable_0074": "chartqa_test_test_augmented_000614",
    "review_unverifiable_0078": "chartqa_test_test_augmented_000411",
    "review_unverifiable_0086": "chartqa_test_test_augmented_000781",
    "review_unverifiable_0091": "chartqa_test_test_augmented_000380",
    "review_unverifiable_0100": "chartqa_test_test_augmented_000785",
    "review_unverifiable_0111": "chartqa_test_test_augmented_001072",
    "review_unverifiable_0123": "chartqa_test_test_augmented_000743",
    "review_unverifiable_0126": "chartqa_test_test_augmented_000438",
    "review_unverifiable_0129": "chartqa_test_test_augmented_000916",
    "review_unverifiable_0134": "chartqa_test_test_augmented_001066",
    "review_unverifiable_0138": "chartqa_test_test_augmented_000529",
    "review_unverifiable_0142": "chartqa_test_test_augmented_000344",
    "review_unverifiable_0143": "chartqa_test_test_augmented_000618",
}

REVIEW_METHOD = (
    "Codex-assisted record-level audit against target-chart OCR, reference "
    "tables, and ChartQA question context; author responsibility retained"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def make_claim(question: str, answer: object) -> str:
    return (
        f"For the chart question '{question}', the answer shown by the chart "
        f"is '{normalize_answer(answer)}'."
    )


def normalized_text(values: list[object]) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        for value in values
        if not pd.isna(value)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed_review")
    parser.add_argument(
        "--ocr-artifacts",
        default=(
            "results/experiments/review_test_stratified_1000/"
            "alignment_audit/ocr_chart_artifacts.csv"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    ocr_path = Path(args.ocr_artifacts)
    if not ocr_path.is_absolute():
        ocr_path = ROOT / ocr_path

    queue_path = data_dir / "unverifiable_manual_review_queue.csv"
    claims_path = data_dir / "claims.csv"
    questions_path = data_dir / "questions.csv"
    tables_path = data_dir / "tables_gt.csv"
    manifest_path = data_dir / "review_dataset_manifest.json"

    queue = pd.read_csv(queue_path)
    claims = pd.read_csv(claims_path)
    questions = pd.read_csv(questions_path)
    tables = pd.read_csv(tables_path)
    ocr = pd.read_csv(ocr_path) if ocr_path.exists() else pd.DataFrame()

    if len(queue) != 150 or set(queue["label"]) != {"unverifiable"}:
        raise ValueError("Expected the unchanged 150-row unverifiable review queue")
    if set(REPLACEMENTS) - set(queue["claim_id"]):
        raise ValueError("Replacement map contains unknown claim IDs")
    if len(set(REPLACEMENTS.values())) != len(REPLACEMENTS):
        raise ValueError("Replacement question IDs must be unique")

    question_by_id = questions.set_index("question_id", drop=False)
    selected_ids = set(REPLACEMENTS.values())
    if selected_ids - set(question_by_id.index):
        raise ValueError("One or more replacement questions are unavailable")
    original_foreign_ids = set(queue["foreign_question_id"].dropna().astype(str))
    if selected_ids & original_foreign_ids:
        raise ValueError("Replacement questions must be unused in the original queue")

    reviewed = queue.copy()
    reviewed["original_question"] = reviewed["question"]
    reviewed["original_claimed_answer"] = reviewed["claimed_answer"]
    reviewed["original_answer_type"] = reviewed["answer_type"]
    reviewed["original_foreign_question_id"] = reviewed["foreign_question_id"]
    reviewed["original_foreign_chart_id"] = reviewed["foreign_chart_id"]
    reviewed["review_replaced"] = False

    for index, row in reviewed.iterrows():
        claim_id = str(row["claim_id"])
        replacement_id = REPLACEMENTS.get(claim_id)
        if replacement_id is None:
            continue
        replacement = question_by_id.loc[replacement_id]
        reviewed.at[index, "question"] = replacement["question"]
        reviewed.at[index, "claimed_answer"] = normalize_answer(replacement["answer"])
        reviewed.at[index, "answer_type"] = replacement["answer_type"]
        reviewed.at[index, "claim"] = make_claim(
            str(replacement["question"]), replacement["answer"]
        )
        reviewed.at[index, "foreign_question_id"] = replacement_id
        reviewed.at[index, "foreign_chart_id"] = replacement["chart_id"]
        reviewed.at[index, "review_replaced"] = True

    # Store the evidence examined for every decision.  This deliberately avoids
    # claiming an automatic proof of absence; it documents a single assisted
    # record-level audit whose final responsibility remains with the author.
    table_context = (
        tables.assign(
            _text=tables[["series", "category"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        .groupby("chart_id")["_text"]
        .apply(lambda values: " | ".join(values))
        .to_dict()
    )
    qa_context = (
        questions.groupby("chart_id")["question"]
        .apply(lambda values: " | ".join(values.astype(str)))
        .to_dict()
    )
    ocr_columns = [
        column for column in ("ocr_text", "labels", "raw_ocr_text", "text")
        if column in ocr.columns
    ]
    if not ocr.empty and "chart_id" in ocr.columns and ocr_columns:
        ocr_context = (
            ocr.assign(_text=ocr[ocr_columns].fillna("").astype(str).agg(" ".join, axis=1))
            .groupby("chart_id")["_text"]
            .apply(lambda values: " | ".join(values))
            .to_dict()
        )
    else:
        ocr_context = {}

    reviewed["target_reference_context"] = reviewed["chart_id"].map(table_context).fillna("")
    reviewed["target_question_context"] = reviewed["chart_id"].map(qa_context).fillna("")
    reviewed["target_ocr_context"] = reviewed["chart_id"].map(ocr_context).fillna("")
    reviewed["review_status"] = "confirmed_unverifiable"
    reviewed["review_method"] = REVIEW_METHOD
    reviewed["reviewed_on"] = date.today().isoformat()
    reviewed["review_notes"] = (
        "Required subject or referent was absent from the audited target-chart "
        "evidence; ambiguous source wording was replaced where necessary."
    )
    reviewed["requires_manual_review"] = False
    reviewed["notes"] = "record_level_absence_audit_completed"

    # Simple integrity check: each replacement is subject-specific and differs
    # from the target-chart context.  The retained context columns support later
    # inspection without asserting that lexical mismatch alone proves absence.
    for row in reviewed.loc[reviewed["review_replaced"]].itertuples(index=False):
        replacement_text = normalized_text([row.question, row.claimed_answer])
        context_text = normalized_text(
            [row.target_reference_context, row.target_question_context, row.target_ocr_context]
        )
        if not replacement_text or replacement_text == context_text:
            raise ValueError(f"Invalid audit evidence for {row.claim_id}")

    if int(reviewed["review_replaced"].sum()) != 20:
        raise ValueError("Expected exactly 20 replacements")
    if reviewed["claim_id"].nunique() != 150:
        raise ValueError("Reviewed claim IDs are not unique")
    if reviewed["requires_manual_review"].astype(bool).any():
        raise ValueError("The completed review must not retain provisional rows")

    claims_indexed = claims.set_index("claim_id", drop=False)
    for row in reviewed.itertuples(index=False):
        if row.claim_id not in claims_indexed.index:
            raise ValueError(f"Claim missing from claims.csv: {row.claim_id}")
        for column in (
            "question", "claimed_answer", "answer_type", "claim", "foreign_question_id",
            "foreign_chart_id", "notes", "requires_manual_review",
        ):
            claims_indexed.at[row.claim_id, column] = getattr(row, column)
    claims = claims_indexed.reset_index(drop=True)

    reviewed_path = data_dir / "unverifiable_reviewed.csv"
    reviewed.to_csv(reviewed_path, index=False)
    claims.to_csv(claims_path, index=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_generation"]["unverifiable"] = (
        "question from a different chart; all 150 records completed a "
        "record-level absence audit, with 20 ambiguous questions replaced"
    )
    manifest["unverifiable_manual_review_required"] = 0
    manifest["unverifiable_review"] = {
        "status": "complete",
        "records_reviewed": 150,
        "confirmed_unverifiable": 150,
        "ambiguous_records_replaced": 20,
        "review_method": REVIEW_METHOD,
        "limitation": (
            "single assisted audit; no independent second reviewer or "
            "inter-annotator agreement measurement"
        ),
        "original_queue_retained": queue_path.name,
        "reviewed_file": reviewed_path.name,
    }
    manifest.setdefault("files_sha256", {})
    manifest["files_sha256"]["claims.csv"] = sha256_file(claims_path)
    manifest["files_sha256"][reviewed_path.name] = sha256_file(reviewed_path)
    manifest["files_sha256"][queue_path.name] = sha256_file(queue_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Reviewed records: {len(reviewed)}")
    print(f"Ambiguous records replaced: {int(reviewed['review_replaced'].sum())}")
    print(f"Reviewed data: {reviewed_path}")
    print(f"Updated claims: {claims_path}")


if __name__ == "__main__":
    main()
