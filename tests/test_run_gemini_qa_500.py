from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_gemini_qa_500 import (
    GEMINI_MODELS,
    PredictionStatus,
    TaskTarget,
    _runner_command,
    _selected_pairs,
    prediction_status,
)


def test_prediction_status_uses_latest_target_rows(tmp_path: Path) -> None:
    path = tmp_path / "qa_pred.csv"
    pd.DataFrame(
        [
            {"question_id": "q1", "error_type": "gemini_error:timeout"},
            {"question_id": "q1", "error_type": ""},
            {"question_id": "q2", "error_type": "gemini_error:quota"},
            {"question_id": "outside", "error_type": ""},
        ]
    ).to_csv(path, index=False)

    status = prediction_status(path, ["q1", "q2", "q3"])

    assert status.successful == 1
    assert status.failed == 1
    assert status.missing == 1


def test_runner_command_uses_one_common_target_and_retries_errors() -> None:
    command = _runner_command(
        task="verification",
        model_name="gemini_end_to_end",
        data_dir=Path("data"),
        raw_data_dir=Path("raw"),
        experiment_dir=Path("experiment"),
        split="test",
        target=500,
        cooldown_minutes=60,
        device="cpu",
        verbose=False,
    )

    assert command[command.index("--limit") + 1] == "500"
    assert command[command.index("--task") + 1] == "verification"
    assert command[command.index("--batch-size") + 1] == "0"
    assert "--retry-errors" in command
    assert "--force" not in command


def test_selected_pairs_keeps_only_conditions_below_threshold() -> None:
    targets = [
        TaskTarget("qa", ["q1", "q2"], "question_id", "qa_pred.csv"),
        TaskTarget(
            "verification",
            ["c1", "c2"],
            "claim_id",
            "claims_pred.csv",
        ),
    ]
    statuses = {
        "qa": {
            model_name: PredictionStatus(successful=500, failed=0, missing=0)
            for model_name in GEMINI_MODELS
        },
        "verification": {
            "gemini_end_to_end": PredictionStatus(
                successful=396, failed=1, missing=53
            ),
            "deplot_table_gemini_pipeline": PredictionStatus(
                successful=91, failed=2, missing=357
            ),
            "ocr_gemini_reasoning_pipeline": PredictionStatus(
                successful=0, failed=1, missing=449
            ),
        },
    }

    selected = _selected_pairs(statuses, targets, only_below=200)

    assert selected == {
        ("verification", "deplot_table_gemini_pipeline"),
        ("verification", "ocr_gemini_reasoning_pipeline"),
    }


def test_selected_pairs_without_threshold_keeps_every_requested_pair() -> None:
    target = TaskTarget("qa", ["q1"], "question_id", "qa_pred.csv")
    statuses = {
        "qa": {
            model_name: PredictionStatus(successful=1, failed=0, missing=0)
            for model_name in GEMINI_MODELS
        }
    }

    selected = _selected_pairs(statuses, [target], only_below=None)

    assert selected == {("qa", model_name) for model_name in GEMINI_MODELS}

