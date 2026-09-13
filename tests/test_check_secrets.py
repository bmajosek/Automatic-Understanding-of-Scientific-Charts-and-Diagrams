from pathlib import Path

from scripts.check_secrets import scan_text


def test_empty_and_placeholder_credentials_are_allowed() -> None:
    text = "GEMINI_API_KEY=\nHF_TOKEN=your_hugging_face_token\n"

    assert scan_text(Path(".env.example"), text) == []


def test_known_token_shape_is_reported_without_storing_a_token_fixture() -> None:
    synthetic_value = "hf_" + ("A" * 40)

    findings = scan_text(Path("example.txt"), f"HF_TOKEN={synthetic_value}\n")

    assert [finding.rule for finding in findings] == ["HUGGING_FACE_TOKEN"]


def test_non_placeholder_secret_assignment_is_reported() -> None:
    value = "runtime" + "-only-value"

    findings = scan_text(Path("config.yaml"), f"SERVICE_PASSWORD: {value}\n")

    assert [finding.rule for finding in findings] == [
        "NON_PLACEHOLDER_SECRET_ASSIGNMENT"
    ]
