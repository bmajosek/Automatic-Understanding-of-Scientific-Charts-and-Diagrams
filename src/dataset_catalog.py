"""Load dataset registry from config/datasets.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_CATALOG: Optional[Dict[str, Dict[str, Any]]] = None


def _config_path() -> Path:
    return Path("config/datasets.yaml")


def load_catalog(reload: bool = False) -> Dict[str, Dict[str, Any]]:
    global _CATALOG
    if _CATALOG is not None and not reload:
        return _CATALOG
    path = _config_path()
    if not path.exists():
        _CATALOG = {}
        return _CATALOG
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _CATALOG = data.get("datasets", {})
    return _CATALOG


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("-", "_")


def get_entry(name: str) -> Optional[Dict[str, Any]]:
    cat = load_catalog()
    key = normalize_name(name)
    if key in cat:
        return cat[key]
    alt = name.lower().strip()
    return cat.get(alt)


def huggingface_repo_id(source_url: Optional[str]) -> Optional[str]:
    if not source_url:
        return None
    url = str(source_url).strip().rstrip("/")
    if "huggingface.co/datasets/" in url:
        part = url.split("huggingface.co/datasets/", 1)[1].split("?")[0].strip("/")
        pieces = [p for p in part.split("/") if p]
        if len(pieces) >= 2:
            return "/".join(pieces[:2])
    if "://" not in url and url.count("/") == 1:
        return url
    return None


def list_all() -> List[str]:
    return sorted(load_catalog().keys())


def list_implemented() -> List[str]:
    return sorted(
        k for k, v in load_catalog().items()
        if str(v.get("status", "")).upper() == "IMPLEMENTED"
    )


def list_downloadable() -> List[str]:
    out = []
    for name, meta in load_catalog().items():
        if meta.get("can_download_auto"):
            if huggingface_repo_id(meta.get("source_url")) or name == "chartqa":
                out.append(name)
    return sorted(out)


def list_planned() -> List[str]:
    return sorted(
        k for k, v in load_catalog().items()
        if str(v.get("status", "")).upper() == "PLANNED"
    )
