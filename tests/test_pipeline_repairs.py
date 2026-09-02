from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipelines.runner import (
    CHART_QA_HF_MODELS,
    _deplot_to_table_rows,
    _pix2struct_verification,
    _verification_label_from_answer,
    _write_gemini_quota_marker,
    clear_gemini_quota_marker,
    gemini_quota_blocked,
    model_uses_gemini,
)
from pipelines.common import is_prediction_failure
from pipelines.gemini_client import is_gemini_quota_error
from pipelines.hf_models import resolve_hf_device
from pipelines.table_reasoner import parse_deplot_cells, symbolic_answer
from pipelines.tasks import MODEL_TASKS
from experiment_analysis import (
    _alignment_stage_analysis,
    _qa_execution_summary,
    _table_extraction_analysis,
)
from output_validation import validate_prediction_outputs
from reproducibility import stable_content_fingerprint
from run_review_audit import _local_paired_statistics

import pandas as pd


def test_content_fingerprint_changes_when_claim_text_changes() -> None:
    rows = pd.DataFrame(
        [
            {"claim_id": "b", "claim": "second", "label": "supported"},
            {"claim_id": "a", "claim": "first", "label": "unverifiable"},
        ]
    )
    reordered = rows.iloc[::-1].reset_index(drop=True)
    edited = rows.copy()
    edited.loc[edited["claim_id"] == "a", "claim"] = "changed"

    original_hash = stable_content_fingerprint(
        rows, "claim_id", ("claim", "label")
    )
    assert original_hash == stable_content_fingerprint(
        reordered, "claim_id", ("claim", "label")
    )
    assert original_hash != stable_content_fingerprint(
        edited, "claim_id", ("claim", "label")
    )


def test_content_fingerprint_ignores_repeated_id_in_content_columns() -> None:
    rows = pd.DataFrame([
        {"chart_id": "chart-2", "image_path": "b.png"},
        {"chart_id": "chart-1", "image_path": "a.png"},
    ])
    expected = stable_content_fingerprint(rows, "chart_id", ("image_path",))
    assert expected == stable_content_fingerprint(
        rows,
        "chart_id",
        ("chart_id", "image_path"),
    )


class StubModel:
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts = []

    def generate(self, image_path: Path, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_verification_answer_mapping_supports_all_three_classes() -> None:
    assert _verification_label_from_answer("47", "47", "numeric", 0.05) == "supported"
    assert _verification_label_from_answer("47", "60", "numeric", 0.05) == "contradicted"
    assert _verification_label_from_answer("47", "unknown", "numeric", 0.05) == "unverifiable"


def test_pix2struct_verifies_qa_derived_numeric_claim() -> None:
    model = StubModel("47")
    claim = (
        "For the chart question 'How many stores were operated?', "
        "the correct answer is '47'."
    )
    label, raw, error = _pix2struct_verification(claim, Path("unused.png"), model)
    assert (label, raw, error) == ("supported", "47", "")
    assert model.prompts == ["How many stores were operated?"]


def test_pix2struct_marks_wrong_qa_answer_contradicted() -> None:
    model = StubModel("60")
    claim = (
        "For the question 'How many stores were operated?', "
        "the answer shown by the chart is '47'."
    )
    label, raw, error = _pix2struct_verification(claim, Path("unused.png"), model)
    assert (label, raw, error) == ("contradicted", "60", "")


def test_pix2struct_uses_chartqa_numeric_tolerance() -> None:
    model = StubModel("104")
    claim = "For the chart question 'Estimate the value?', the correct answer is '100'."
    label, _, _ = _pix2struct_verification(claim, Path("unused.png"), model)
    assert label == "supported"


def test_informational_heuristic_tags_are_not_failures() -> None:
    assert not is_prediction_failure("chartocr_heuristic")
    assert not is_prediction_failure("ocr_heuristic")
    assert is_prediction_failure("gemini_error:ClientError:429 RESOURCE_EXHAUSTED")


def test_gemini_quota_detection() -> None:
    assert is_gemini_quota_error("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert not is_gemini_quota_error("ocr_error:RuntimeError")


def test_gemini_cooldown_marker() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        _write_gemini_quota_marker(
            tmp_path, "gemini_end_to_end", "qa", "q1", "429 quota", 60,
        )
        assert gemini_quota_blocked(tmp_path)
        clear_gemini_quota_marker(tmp_path)
        assert not gemini_quota_blocked(tmp_path)


def test_hf_device_falls_back_to_cpu_without_cuda() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    assert resolve_hf_device(FakeTorch, "auto") == "cpu"
    assert resolve_hf_device(FakeTorch, "cuda") == "cpu"
    assert resolve_hf_device(FakeTorch, "cpu") == "cpu"


def test_deplot_table_extraction_does_not_require_gemini() -> None:
    assert not model_uses_gemini(
        "deplot_table_gemini_pipeline", "table_extraction"
    )
    assert model_uses_gemini("deplot_table_gemini_pipeline", "qa")
    assert model_uses_gemini("gemini_end_to_end", "qa")


def test_deplot_parser_decodes_literal_row_tokens_and_preserves_commas() -> None:
    table = (
        "Country | Share of children who are wasted, 2010 <0x0A> "
        "Haiti | 6.12 <0x0A> Libya | 5.32"
    )
    assert parse_deplot_cells(table) == [
        ("Share of children who are wasted, 2010", "Haiti", "6.12"),
        ("Share of children who are wasted, 2010", "Libya", "5.32"),
    ]


def test_deplot_parser_expands_multiseries_rows_to_chartqa_cells() -> None:
    table = (
        "Year | Germany | Poland <0x0A> "
        "2020 | 0.41 | 0.32 <0x0A> 2019 | 0.40 | 0.34"
    )
    assert parse_deplot_cells(table) == [
        ("Germany", "2020", "0.41"),
        ("Poland", "2020", "0.32"),
        ("Germany", "2019", "0.40"),
        ("Poland", "2019", "0.34"),
    ]
    rows = _deplot_to_table_rows("chart-1", table, "")
    assert len(rows) == 4
    assert rows[0]["series"] == "Germany"
    assert rows[0]["category"] == "2020"
    assert rows[0]["pred_value"] == "0.41"


def test_symbolic_reasoner_uses_decoded_deplot_rows() -> None:
    table = "Characteristic | Sales <0x0A> Alpha | 10 <0x0A> Beta | 25"
    assert symbolic_answer("Which category has the highest sales?", table)[0] == "25.0"


def test_matcha_baselines_are_local_and_registered() -> None:
    for model_name in (
        "matcha_chartqa_ocr_free_pipeline",
        "matcha_plotqa_transfer_pipeline",
    ):
        assert {"qa", "verification"} <= MODEL_TASKS[model_name]
        assert not model_uses_gemini(model_name, "qa")
        assert model_name in CHART_QA_HF_MODELS
    assert (
        CHART_QA_HF_MODELS["matcha_plotqa_transfer_pipeline"]["processor_id"]
        == "google/matcha-chartqa"
    )


def test_execution_summary_separates_completion_from_runtime_success() -> None:
    summary = pd.DataFrame([{
        "model_name": "gemini_end_to_end",
        "model_label": "Gemini end-to-end",
        "uses_gemini": True,
        "target_samples": 1000,
        "attempted_samples": 100,
        "valid_samples": 99,
        "correct_samples": 75,
        "accuracy": 75 / 99,
    }])
    row = _qa_execution_summary(summary).iloc[0]
    assert row["experimental_completion"] == 0.1
    assert row["execution_success"] == 0.99
    assert row["answer_accuracy"] == 75 / 99


def test_table_validation_allows_multiple_distinct_cells_per_chart() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions"
        model_dir = predictions / "deplot_table_gemini_pipeline"
        model_dir.mkdir(parents=True)
        pd.DataFrame([
            {
                "chart_id": "chart-1", "series": "A", "category": "2020",
                "pred_value": "10", "error_type": "", "notes": "",
            },
            {
                "chart_id": "chart-1", "series": "B", "category": "2020",
                "pred_value": "20", "error_type": "", "notes": "",
            },
        ]).to_csv(model_dir / "tables_pred.csv", index=False)
        summary = validate_prediction_outputs(
            "table_extraction",
            pd.DataFrame([{"chart_id": "chart-1"}]),
            predictions,
            root / "validation",
            model_names=["deplot_table_gemini_pipeline"],
        ).iloc[0]
        assert summary["status"] == "COMPLETE"
        assert summary["duplicate_rows"] == 0
        assert summary["successful_samples"] == 1


def test_table_analysis_uses_distinct_normalised_keys_and_end_to_end_denominator() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        predictions = root / "predictions"
        model_dir = predictions / "deplot_table_gemini_pipeline"
        data_dir.mkdir()
        model_dir.mkdir(parents=True)
        pd.DataFrame([
            {"chart_id": "chart-1", "split": "test"},
        ]).to_csv(data_dir / "charts.csv", index=False)
        pd.DataFrame([
            {
                "chart_id": "chart-1", "series": "Revenue (%)",
                "category": "2020", "value": "100",
            },
            {
                "chart_id": "chart-1", "series": "Revenue",
                "category": "2021", "value": "200",
            },
        ]).to_csv(data_dir / "tables_gt.csv", index=False)
        pd.DataFrame([
            {
                "chart_id": "chart-1", "series": "Revenue %",
                "category": "2020", "pred_value": "105", "error_type": "",
            },
            {
                "chart_id": "chart-1", "series": "Revenue %",
                "category": "2020", "pred_value": "999", "error_type": "",
            },
            {
                "chart_id": "chart-1", "series": "Revenue",
                "category": "2021", "pred_value": "100", "error_type": "",
            },
        ]).to_csv(model_dir / "tables_pred.csv", index=False)

        row = _table_extraction_analysis(
            data_dir,
            predictions,
            split="test",
            limit=1000,
            models=["deplot_table_gemini_pipeline"],
            numerical_tolerance=0.05,
        ).iloc[0]
        assert row["ground_truth_cells"] == 2
        assert row["predicted_cells"] == 3
        assert row["distinct_predicted_cells"] == 2
        assert row["duplicate_predicted_rows"] == 1
        assert row["matched_cells"] == 2
        assert row["cell_coverage"] == 1.0
        assert row["tolerance_accuracy"] == 0.5
        assert row["end_to_end_accuracy"] == 0.5


def test_local_paired_statistics_use_shared_questions_and_chart_clusters() -> None:
    questions = pd.DataFrame([
        {
            "question_id": "q1", "chart_id": "c1", "answer": "10",
            "answer_type": "numeric", "question_origin": "human",
        },
        {
            "question_id": "q2", "chart_id": "c1", "answer": "20",
            "answer_type": "numeric", "question_origin": "augmented",
        },
        {
            "question_id": "q3", "chart_id": "c2", "answer": "yes",
            "answer_type": "boolean", "question_origin": "human",
        },
        {
            "question_id": "q4", "chart_id": "c2", "answer": "no",
            "answer_type": "boolean", "question_origin": "augmented",
        },
    ])
    answers = {
        "matcha_chartqa_ocr_free_pipeline": ["10", "20", "yes", "yes"],
        "pix2struct_ocr_free_pipeline": ["10", "0", "yes", "no"],
        "matcha_plotqa_transfer_pipeline": ["0", "0", "no", "yes"],
    }
    with tempfile.TemporaryDirectory() as directory:
        predictions = Path(directory)
        for model_name, predicted in answers.items():
            model_dir = predictions / model_name
            model_dir.mkdir(parents=True)
            pd.DataFrame({
                "question_id": questions["question_id"],
                "pred_answer": predicted,
                "error_type": [""] * 4,
            }).to_csv(model_dir / "qa_pred.csv", index=False)
        origins, bootstrap, paired = _local_paired_statistics(
            questions,
            predictions,
            tolerance=0.05,
            bootstrap_samples=200,
            bootstrap_seed=42,
        )
        assert len(origins) == 6
        assert len(bootstrap) == 3
        result = paired.iloc[0]
        assert result["paired_questions"] == 4
        assert result["distinct_charts"] == 2
        assert result["a_correct_b_wrong"] == 1
        assert result["a_wrong_b_correct"] == 1


def test_invalid_qa_subset_alignment_profile_is_suppressed() -> None:
    qa = pd.DataFrame([{
        "model_name": "test_model",
        "model_label": "Test model",
        "uses_gemini": False,
    }])
    breakdown = pd.DataFrame([
        {
            "model_name": "test_model",
            "dimension": "task",
            "group": "value_retrieval",
            "samples": 50,
            "target_samples": 50,
            "correct": 40,
        },
        {
            "model_name": "test_model",
            "dimension": "answer_type",
            "group": "numeric",
            "samples": 50,
            "target_samples": 50,
            "correct": 35,
        },
        {
            "model_name": "test_model",
            "dimension": "task",
            "group": "arithmetic",
            "samples": 30,
            "target_samples": 30,
            "correct": 15,
        },
    ])
    verification = pd.DataFrame([{
        "model_name": "test_model",
        "valid_samples": 2,
        "target_samples": 1000,
        "accuracy": 0.5,
        "coverage": 0.002,
    }])
    result = _alignment_stage_analysis(qa, breakdown, verification)
    assert result.empty
