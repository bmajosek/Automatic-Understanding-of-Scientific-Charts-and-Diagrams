from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_analysis import (
    _qa_analysis,
    _verification_analysis,
    verification_cluster_bootstrap,
)
from run_review_audit import _holm_adjust


def test_holm_adjustment_preserves_order_and_monotonicity() -> None:
    adjusted = _holm_adjust([0.04, 0.001, 0.02])

    assert adjusted == [0.04, 0.003, 0.04]


def test_verification_uncertainty_resamples_linked_claim_groups() -> None:
    rows = pd.DataFrame([
        {"paired_claim_group": "g1", "label": "supported", "pred_label": "supported"},
        {"paired_claim_group": "g1", "label": "contradicted", "pred_label": "contradicted"},
        {"paired_claim_group": "g1", "label": "unverifiable", "pred_label": "unverifiable"},
        {"paired_claim_group": "g2", "label": "supported", "pred_label": "contradicted"},
        {"paired_claim_group": "g2", "label": "contradicted", "pred_label": "supported"},
        {"paired_claim_group": "g2", "label": "unverifiable", "pred_label": "supported"},
    ])

    result = verification_cluster_bootstrap(
        rows,
        ["supported", "contradicted", "unverifiable"],
        "paired_claim_group",
        samples=500,
        seed=7,
    )

    assert result["interval_method"] == "cluster_bootstrap"
    assert result["cluster_count"] == 2
    assert 0.0 <= result["accuracy_ci95_low"] <= result["accuracy_ci95_high"] <= 1.0
    assert 0.0 <= result["macro_f1_ci95_low"] <= result["macro_f1_ci95_high"] <= 1.0


def test_qa_analysis_uses_the_planned_gemini_subcohort(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    predictions_dir = tmp_path / "predictions"
    model = "gemini_end_to_end"
    (predictions_dir / model).mkdir(parents=True)
    data_dir.mkdir()
    pd.DataFrame([
        {
            "question_id": f"q{index}",
            "chart_id": f"c{index}",
            "split": "test",
            "question": "Value?",
            "answer": "1",
            "answer_type": "numeric",
            "task": "retrieval",
            "operation": "read_value",
        }
        for index in range(6)
    ]).to_csv(data_dir / "questions.csv", index=False)
    pd.DataFrame([
        {
            "question_id": f"q{index}",
            "chart_id": f"c{index}",
            "pred_answer": "1",
            "error_type": "",
        }
        for index in range(3)
    ]).to_csv(predictions_dir / model / "qa_pred.csv", index=False)

    summary, _, _ = _qa_analysis(
        data_dir,
        predictions_dir,
        "test",
        6,
        [model],
        0.05,
        gemini_limit=3,
    )

    assert int(summary.iloc[0]["target_samples"]) == 3
    assert int(summary.iloc[0]["valid_samples"]) == 3
    assert float(summary.iloc[0]["coverage"]) == 1.0


def test_verification_without_valid_predictions_is_not_reported_as_zero(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    predictions_dir = tmp_path / "predictions"
    model = "gemini_end_to_end"
    data_dir.mkdir()
    (predictions_dir / model).mkdir(parents=True)
    pd.DataFrame([
        {
            "claim_id": "c1",
            "chart_id": "chart-1",
            "split": "test",
            "label": "supported",
            "paired_claim_group": "g1",
        },
        {
            "claim_id": "c2",
            "chart_id": "chart-1",
            "split": "test",
            "label": "contradicted",
            "paired_claim_group": "g1",
        },
        {
            "claim_id": "c3",
            "chart_id": "chart-1",
            "split": "test",
            "label": "unverifiable",
            "paired_claim_group": "g1",
        },
    ]).to_csv(data_dir / "claims.csv", index=False)
    pd.DataFrame([{
        "claim_id": "c1",
        "chart_id": "chart-1",
        "pred_label": "",
        "error_type": "gemini_error:quota",
    }]).to_csv(predictions_dir / model / "claims_pred.csv", index=False)

    summary, _ = _verification_analysis(
        data_dir,
        predictions_dir,
        "test",
        3,
        [model],
    )

    row = summary.iloc[0]
    assert int(row["valid_samples"]) == 0
    assert row["evaluation_status"] == "insufficient_coverage"
    assert pd.isna(row["accuracy"])
    assert pd.isna(row["macro_f1"])
    assert pd.isna(row["ci95_low"])
    assert pd.isna(row["macro_f1_ci95_low"])
