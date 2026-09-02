"""Dataset converters package entry point."""

from pathlib import Path
from typing import Optional
try:
    from dataset_catalog import get_entry, normalize_name, huggingface_repo_id
    from dataset_converters.base import BaseConverter
    from dataset_converters.chartqa import ChartQAConverter
    from dataset_converters.hf_qa import HuggingFaceQAConverter
    from dataset_converters.stub import StubConverter
except ImportError:
    from ..dataset_catalog import get_entry, normalize_name, huggingface_repo_id
    from .base import BaseConverter
    from .chartqa import ChartQAConverter
    from .hf_qa import HuggingFaceQAConverter
    from .stub import StubConverter

HF_CONVERTER_DATASETS = {
    "chartqapro",
    "scigraphqa",
    "polychartqa",
    "chartx",
}


def get_converter(
    dataset_name: str,
    raw_data_dir: Path,
    output_dir: Path,
    max_examples: Optional[int] = None,
) -> Optional[BaseConverter]:
    name = normalize_name(dataset_name)

    if name == "chartqa":
        return ChartQAConverter(raw_data_dir, output_dir)

    entry = get_entry(name)
    if entry and str(entry.get("status", "")).upper() == "IMPLEMENTED" and name in HF_CONVERTER_DATASETS:
        repo = entry.get("hf_repo") or huggingface_repo_id(entry.get("source_url"))
        return HuggingFaceQAConverter(
            raw_data_dir, output_dir, name, repo_id=repo, max_examples=max_examples,
        )

    if entry and str(entry.get("status", "")).upper() == "PLANNED":
        return StubConverter(
            raw_data_dir,
            output_dir,
            name,
            instructions=str(entry.get("instructions", entry.get("source_url", ""))),
        )

    return None


__all__ = [
    "BaseConverter",
    "ChartQAConverter",
    "HuggingFaceQAConverter",
    "StubConverter",
    "get_converter",
]
