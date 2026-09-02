"""
Load project configuration, optional .env, and environment secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Models that require specific API keys when used in experiments.
GEMINI_MODELS = {
    "gemini_end_to_end",
    "ocr_gemini_reasoning_pipeline",
    "deplot_table_gemini_pipeline",
}
HF_MODELS = {
    "pix2struct_ocr_free_pipeline",
    "deplot_table_gemini_pipeline",
    "table_symbolic_reasoner_pipeline",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_dotenv_if_present(project_root: Optional[Path] = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = project_root or Path.cwd()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def get_env(name: str, required: bool = False) -> Optional[str]:
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set the value."
        )
    return value


def apply_tesseract_env(secrets: Dict[str, Any]) -> None:
    ocr_cfg = secrets.get("ocr", {})
    cmd_env = ocr_cfg.get("tesseract_cmd_env", "TESSERACT_CMD")
    data_env = ocr_cfg.get("tessdata_prefix_env", "TESSDATA_PREFIX")
    cmd = os.environ.get(cmd_env)
    prefix = os.environ.get(data_env)
    if cmd:
        os.environ["TESSERACT_CMD"] = cmd
    if prefix:
        os.environ["TESSDATA_PREFIX"] = prefix


class ProjectConfig:
    """Merged configuration for experiments and evaluation."""

    def __init__(
        self,
        config: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
        secrets: Optional[Dict[str, Any]] = None,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.prompts = prompts or {}
        self.secrets = secrets or {}
        self.project_root = project_root or Path.cwd()

        paths = config.get("paths", {})
        self.data_dir = Path(paths.get("data_dir", "data/processed"))
        self.raw_data_dir = Path(paths.get("raw_data_dir", "data/raw"))
        self.processed_data_dir = Path(paths.get("processed_data_dir", "data/processed"))
        self.predictions_dir = Path(paths.get("predictions_dir", "predictions"))
        self.results_dir = Path(paths.get("results_dir", "results"))
        self.gemini = dict(config.get("gemini", {}))
        self.evaluation = dict(config.get("evaluation", {}))

    @property
    def gemini_api_key(self) -> Optional[str]:
        env_name = self.secrets.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
        return os.environ.get(env_name)

    @property
    def hf_token(self) -> Optional[str]:
        env_name = self.secrets.get("huggingface", {}).get("token_env", "HF_TOKEN")
        return os.environ.get(env_name)

    def validate_for_model(self, model_name: str) -> None:
        """Validate secrets only for models that need them."""
        if model_name in GEMINI_MODELS:
            env_name = self.secrets.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
            get_env(env_name, required=True)
        if model_name in HF_MODELS:
            env_name = self.secrets.get("huggingface", {}).get("token_env", "HF_TOKEN")
            # HF token is optional for public models but recommended.
            if not os.environ.get(env_name):
                print(
                    f"Warning: {env_name} is not set. Public Hugging Face models may still download; "
                    "gated models will fail."
                )


def load_project_config(
    config_path: str | Path = "config/config.yaml",
    project_root: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> ProjectConfig:
    root = project_root or Path.cwd()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path

    load_dotenv_if_present(root)

    config = _load_yaml(config_path)
    config_dir = config_path.parent

    prompts_path = config_dir / "prompts.yaml"
    prompts = _load_yaml(prompts_path) if prompts_path.exists() else {}

    secrets_path = config_dir / "secrets.yaml"
    if not secrets_path.exists():
        secrets_path = config_dir / "secrets.example.yaml"
    secrets = _load_yaml(secrets_path) if secrets_path.exists() else {}

    apply_tesseract_env(secrets)

    proj = ProjectConfig(config, prompts, secrets, root)
    if model_name:
        proj.validate_for_model(model_name)
    return proj
