"""
Generate ChartQA baseline predictions (no real chart understanding).

Usage:
    python scripts/generate_baselines.py --data-dir data/processed --predictions-dir predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL_ROWS = [
    {
        "model_name": "oracle_upper_bound",
        "model_family": "oracle",
        "approach": "debug_upper_bound",
        "temperature": "",
        "max_tokens": "",
        "prompt_version": "baseline_v1",
        "input_type": "image+ground_truth",
        "notes": "DEBUG ONLY: copies ground-truth answers. Not a real model.",
    },
    {
        "model_name": "constant_answer_baseline",
        "model_family": "baseline",
        "approach": "non_visual_baseline",
        "temperature": "",
        "max_tokens": "",
        "prompt_version": "baseline_v1",
        "input_type": "question_only",
        "notes": "Sanity baseline: 0 / yes / unknown by answer type.",
    },
    {
        "model_name": "train_prior_baseline",
        "model_family": "baseline",
        "approach": "question_only_prior",
        "temperature": "",
        "max_tokens": "",
        "prompt_version": "baseline_v1",
        "input_type": "question_only",
        "notes": "Sanity baseline: most frequent training answer prior.",
    },
    {
        "model_name": "random_train_prior_baseline",
        "model_family": "baseline",
        "approach": "non_visual_baseline",
        "temperature": "",
        "max_tokens": "",
        "prompt_version": "baseline_v2",
        "input_type": "question_only",
        "notes": "Seeded random answer drawn from the training distribution by answer type.",
    },
    {
        "model_name": "constant_supported_baseline",
        "model_family": "baseline",
        "approach": "non_visual_verification_baseline",
        "temperature": "",
        "max_tokens": "",
        "prompt_version": "baseline_v2",
        "input_type": "claim_only",
        "notes": "Always predicts supported; required for verification interpretation.",
    },
]


def _safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def constant_answer(answer_type: str) -> str:
    answer_type = _safe_str(answer_type).strip().lower()
    if answer_type == "numeric":
        return "0"
    if answer_type == "boolean":
        return "yes"
    return "unknown"


def most_frequent(series: pd.Series, default: str = "unknown") -> str:
    cleaned = series.dropna().astype(str)
    if cleaned.empty:
        return default
    return str(cleaned.value_counts().idxmax())


def build_train_prior_predictions(
    questions: pd.DataFrame,
    evaluation_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    train_df = questions[questions.get("split", "").astype(str).str.lower().eq("train")]
    if train_df.empty:
        train_df = questions
    global_prior = most_frequent(train_df["answer"], default="unknown")
    priors_by_type = {}
    if "answer_type" in train_df.columns:
        for answer_type, group in train_df.groupby(train_df["answer_type"].astype(str).str.lower()):
            priors_by_type[answer_type] = most_frequent(group["answer"], default=global_prior)
    rows = []
    for _, row in (evaluation_rows if evaluation_rows is not None else questions).iterrows():
        answer_type = _safe_str(row.get("answer_type", "")).strip().lower()
        pred = priors_by_type.get(answer_type, global_prior)
        rows.append({
            "question_id": row["question_id"],
            "chart_id": row["chart_id"],
            "pred_answer": pred,
            "error_type": "question_only_prior",
            "notes": "No image used.",
        })
    return pd.DataFrame(rows)


def build_random_train_prior_predictions(
    questions: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    import random

    train_df = questions[questions.get("split", "").astype(str).str.lower().eq("train")]
    if train_df.empty:
        raise ValueError("Random train-prior baseline requires training rows.")
    pools = {
        answer_type: group["answer"].dropna().astype(str).tolist()
        for answer_type, group in train_df.groupby(
            train_df["answer_type"].fillna("text").astype(str).str.lower()
        )
    }
    global_pool = train_df["answer"].dropna().astype(str).tolist()
    rng = random.Random(seed)
    rows = []
    for _, row in evaluation_rows.iterrows():
        answer_type = _safe_str(row.get("answer_type", "text")).strip().lower()
        pool = pools.get(answer_type) or global_pool
        rows.append({
            "question_id": row["question_id"],
            "chart_id": row["chart_id"],
            "pred_answer": rng.choice(pool) if pool else "unknown",
            "error_type": "question_only_random_prior",
            "notes": f"No image used; seed={seed}.",
        })
    return pd.DataFrame(rows)


def save_prediction(predictions_dir: Path, model_name: str, df: pd.DataFrame) -> None:
    model_dir = predictions_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_dir / "qa_pred.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument("--write-models-csv", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    questions_path = data_dir / "questions.csv"
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing {questions_path}")

    questions = pd.read_csv(questions_path, dtype=str)
    required = {"question_id", "chart_id", "answer"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    evaluation_rows = questions[
        questions["split"].fillna("").str.lower().eq(args.split.lower())
    ].head(args.limit).copy()
    oracle = evaluation_rows[["question_id", "chart_id", "answer"]].copy()
    oracle = oracle.rename(columns={"answer": "pred_answer"})
    oracle["error_type"] = "oracle_debug"
    oracle["notes"] = "DEBUG ONLY: copied ground-truth answer."
    save_prediction(predictions_dir, "oracle_upper_bound", oracle)

    constant_rows = []
    for _, row in evaluation_rows.iterrows():
        constant_rows.append({
            "question_id": row["question_id"],
            "chart_id": row["chart_id"],
            "pred_answer": constant_answer(row.get("answer_type", "")),
            "error_type": "constant_baseline",
            "notes": "No image used.",
        })
    save_prediction(predictions_dir, "constant_answer_baseline", pd.DataFrame(constant_rows))
    save_prediction(
        predictions_dir,
        "train_prior_baseline",
        build_train_prior_predictions(questions, evaluation_rows),
    )
    save_prediction(
        predictions_dir,
        "random_train_prior_baseline",
        build_random_train_prior_predictions(questions, evaluation_rows, args.seed),
    )

    claims_path = data_dir / "claims.csv"
    if claims_path.exists():
        claims = pd.read_csv(claims_path, dtype=str).fillna("")
        claims = claims[claims["split"].str.lower().eq(args.split.lower())].head(args.limit)
        supported = claims[["claim_id", "chart_id"]].copy()
        supported["pred_label"] = "supported"
        supported["raw_output"] = "supported"
        supported["error_type"] = "constant_supported_baseline"
        supported["notes"] = "No chart or claim content used."
        model_dir = predictions_dir / "constant_supported_baseline"
        model_dir.mkdir(parents=True, exist_ok=True)
        supported.to_csv(model_dir / "claims_pred.csv", index=False)

    if args.write_models_csv:
        models_path = data_dir / "models.csv"
        existing = pd.read_csv(models_path, dtype=str) if models_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, pd.DataFrame(MODEL_ROWS)], ignore_index=True, sort=False)
        combined = combined.drop_duplicates("model_name", keep="last")
        combined.to_csv(models_path, index=False)

    print("Created baselines:")
    for name in (
        "oracle_upper_bound", "constant_answer_baseline", "train_prior_baseline",
        "random_train_prior_baseline",
    ):
        print(f"  {predictions_dir / name / 'qa_pred.csv'}")
    if claims_path.exists():
        print(f"  {predictions_dir / 'constant_supported_baseline' / 'claims_pred.csv'}")


if __name__ == "__main__":
    main()
