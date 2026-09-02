"""Reproducibility metadata for thesis experiments (no secrets are recorded)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_fingerprint(rows: pd.DataFrame, id_column: str) -> str:
    values = sorted(rows[id_column].fillna("").astype(str).tolist())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def stable_content_fingerprint(
    rows: pd.DataFrame,
    id_column: str,
    content_columns: Iterable[str],
) -> str:
    """Hash ordered record contents so resume metadata detects edited inputs."""
    columns = [
        id_column,
        *[
            column
            for column in content_columns
            if column != id_column and column in rows.columns
        ],
    ]
    ordered = rows[columns].copy()
    ordered = ordered.fillna("").astype(str).sort_values(id_column, kind="stable")
    payload = "\n".join(
        "\x1f".join(row)
        for row in ordered.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "git", "-c", f"safe.directory={root.as_posix()}",
                "rev-parse", "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        # Sandboxed Windows runs can reject Git commands because the checkout
        # owner SID differs from the execution SID.  Reading HEAD is safe and
        # still records the exact checked-out revision when refs are unpacked.
        try:
            head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref_path = root / ".git" / head.removeprefix("ref: ")
                if ref_path.exists():
                    return ref_path.read_text(encoding="utf-8").strip()
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    ref_name = head.removeprefix("ref: ")
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line and not line.startswith(("#", "^")):
                            revision, name = line.split(" ", 1)
                            if name == ref_name:
                                return revision
                return f"unresolved:{head}"
            return head
        except Exception:
            return "unavailable"


def _git_state(root: Path) -> dict[str, Any]:
    revision = _git_revision(root)
    try:
        completed = subprocess.run(
            [
                "git", "-c", f"safe.directory={root.as_posix()}",
                "status", "--porcelain=v1", "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = completed.stdout.replace("\r\n", "\n")
        return {
            "git_revision": revision,
            "git_dirty": bool(status.strip()),
            "git_status_sha256": hashlib.sha256(
                status.encode("utf-8")
            ).hexdigest(),
        }
    except Exception:
        return {
            "git_revision": revision,
            "git_dirty": "unavailable",
            "git_status_sha256": "unavailable",
        }


def environment_snapshot(root: Path, requested_device: str) -> dict[str, Any]:
    torch_info: dict[str, Any] = {
        "version": _version("torch"),
        "requested_device": requested_device,
        "cuda_available": False,
        "cuda_runtime": "unavailable",
        "gpu_names": [],
    }
    try:
        import torch

        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_runtime"] = str(torch.version.cuda or "none")
        if torch.cuda.is_available():
            torch_info["gpu_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        torch_info["inspection_error"] = f"{type(exc).__name__}: {exc}"

    git_state = _git_state(root)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        **git_state,
        "libraries": {
            name: _version(name)
            for name in (
                "pandas", "numpy", "scipy", "scikit-learn", "matplotlib",
                "Pillow", "pytesseract", "opencv-python", "torch",
                "transformers", "huggingface-hub", "google-genai",
            )
        },
        "torch": torch_info,
    }


def prediction_hashes(predictions_dir: Path, model_names: Iterable[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for model_name in model_names:
        model_dir = Path(predictions_dir) / model_name
        files = {}
        if model_dir.exists():
            for path in sorted(model_dir.glob("*.csv")):
                files[path.name] = sha256_file(path)
        result[model_name] = files
    return result


def prompt_manifest(prompts: dict[str, Any]) -> dict[str, dict[str, str]]:
    manifest = {}
    for name, value in sorted(prompts.items()):
        text = str(value)
        manifest[name] = {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return manifest


def sample_composition(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(len(rows))}
    for column in ("split", "question_origin", "label", "task", "answer_type"):
        if column in rows.columns:
            counts = rows[column].fillna("missing").astype(str).value_counts()
            result[f"{column}_counts"] = {
                str(label): int(count) for label, count in counts.items()
            }
    return result


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
