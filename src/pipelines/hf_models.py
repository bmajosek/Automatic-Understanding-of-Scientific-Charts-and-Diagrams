"""Hugging Face Pix2Struct and DePlot helpers."""

from __future__ import annotations

from pathlib import Path

from .common import load_pil_image, normalize_text


def _from_pretrained_local_first(loader, model_id: str):
    """Use a complete Hugging Face cache without making a network HEAD request."""
    try:
        from huggingface_hub import snapshot_download

        cached_snapshot = snapshot_download(model_id, local_files_only=True)
        return loader.from_pretrained(cached_snapshot, local_files_only=True)
    except (ImportError, OSError, ValueError):
        return loader.from_pretrained(model_id)


def resolve_hf_device(torch_module, requested: str) -> str:
    """Resolve a usable device without crashing on CPU-only PyTorch builds."""
    requested = (requested or "auto").lower()
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        print(
            "Warning: CUDA was requested, but this PyTorch build has no CUDA "
            "support. Falling back to CPU."
        )
        return "cpu"
    return requested


class HFImageTextModel:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 64,
        processor_id: str | None = None,
    ):
        try:
            import torch
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Missing Hugging Face dependencies. Install: pip install transformers torch sentencepiece"
            ) from exc

        self.torch = torch
        # Some fine-tuned checkpoints publish model weights without duplicating
        # the base checkpoint's processor files (for example MatCha PlotQA v2).
        self.processor = _from_pretrained_local_first(
            Pix2StructProcessor, processor_id or model_id
        )
        self.model = _from_pretrained_local_first(
            Pix2StructForConditionalGeneration, model_id
        )
        self.device = resolve_hf_device(torch, device)
        self.max_new_tokens = max_new_tokens
        self.model.to(self.device)
        self.model.eval()

    def generate(self, image_path: Path, prompt: str) -> str:
        image = load_pil_image(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return normalize_text(self.processor.decode(output[0], skip_special_tokens=True))
