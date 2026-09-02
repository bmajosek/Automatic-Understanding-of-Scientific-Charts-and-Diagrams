"""Google Gemini client with rate limiting, retry, debug logs, and image resize."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .common import extract_final_answer, load_pil_image, normalize_text


_LAST_GEMINI_CALL = 0.0

GEMINI_WAIT_SECONDS = float(os.environ.get("GEMINI_WAIT_SECONDS", "4.5"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "2"))
GEMINI_QUOTA_SLEEP_BASE = float(os.environ.get("GEMINI_QUOTA_SLEEP_BASE", "10"))
GEMINI_QUOTA_SLEEP_MAX = float(os.environ.get("GEMINI_QUOTA_SLEEP_MAX", "60"))
GEMINI_IMAGE_MAX_SIZE = int(os.environ.get("GEMINI_IMAGE_MAX_SIZE", "1024"))


def _wait_for_rate_limit() -> None:
    global _LAST_GEMINI_CALL

    now = time.monotonic()
    elapsed = now - _LAST_GEMINI_CALL

    if elapsed < GEMINI_WAIT_SECONDS:
        time.sleep(GEMINI_WAIT_SECONDS - elapsed)

    _LAST_GEMINI_CALL = time.monotonic()


def _short_error(exc: Exception) -> str:
    msg = str(exc).replace("\n", " ").replace("\r", " ").strip()
    if len(msg) > 600:
        msg = msg[:600] + "..."
    return f"gemini_error:{type(exc).__name__}:{msg}"


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
        or "rate limit" in msg.lower()
    )


def is_gemini_quota_error(error_text: object) -> bool:
    text = str(error_text or "").lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota exceeded" in text
        or "rate limit" in text
    )


def _get_client(api_key: str):
    if not api_key or not api_key.strip():
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY in .env or config.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Install with: pip install google-genai"
        ) from exc

    return genai.Client(api_key=api_key.strip())


def _retry_delay_from_error(exc: Exception) -> float | None:
    import re

    msg = str(exc)
    match = re.search(r"Please retry in ([0-9.]+)s", msg)
    if match:
        return float(match.group(1)) + 2.0

    match = re.search(r"'retryDelay': '([0-9.]+)s'", msg)
    if match:
        return float(match.group(1)) + 2.0

    return None


def _generate_with_retry(
    client: Any,
    *,
    model: str,
    contents: Any,
    temperature: float,
    max_tokens: int,
):
    last_exc: Exception | None = None
    is_multimodal = isinstance(contents, list)
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            print(f"[Gemini] model={model} multimodal={is_multimodal}")
            _wait_for_rate_limit()

            return client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )

        except Exception as exc:
            last_exc = exc
            print(f"[Gemini error] {type(exc).__name__}: {str(exc)[:1000]}")

            if not _is_quota_error(exc):
                raise

            if attempt >= GEMINI_MAX_RETRIES:
                break

            retry_after = _retry_delay_from_error(exc)
            if retry_after is None:
                retry_after = min(
                    GEMINI_QUOTA_SLEEP_MAX,
                    GEMINI_QUOTA_SLEEP_BASE * attempt,
                )

            print(
                f"[Gemini quota] attempt {attempt}/{GEMINI_MAX_RETRIES}; "
                f"sleeping {retry_after:.1f}s"
            )
            time.sleep(retry_after)

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Gemini request failed without exception.")


def gemini_text(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[str, str]:
    try:
        client = _get_client(api_key)

        response = _generate_with_retry(
            client,
            model=model,
            contents=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw = normalize_text(getattr(response, "text", "") or "")
        return raw, ""

    except Exception as exc:
        if os.environ.get("GEMINI_RAISE_ERRORS", "").lower() in {"1", "true", "yes"}:
            raise
        return "", _short_error(exc)


def gemini_vision(
    image_path: Path,
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[str, str]:
    try:
        client = _get_client(api_key)

        image = load_pil_image(image_path)
        image.thumbnail((GEMINI_IMAGE_MAX_SIZE, GEMINI_IMAGE_MAX_SIZE))

        response = _generate_with_retry(
            client,
            model=model,
            contents=[prompt, image],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw = normalize_text(getattr(response, "text", "") or "")
        return raw, ""

    except Exception as exc:
        if os.environ.get("GEMINI_RAISE_ERRORS", "").lower() in {"1", "true", "yes"}:
            raise
        return "", _short_error(exc)


def parse_gemini_answer(raw: str, error_type: str) -> tuple[str, str]:
    if error_type:
        return "unknown", error_type
    if not raw:
        return "unknown", "gemini_empty_response"
    return extract_final_answer(raw), ""


def parse_verification_label(raw: str, error_type: str) -> tuple[str, str]:
    if error_type:
        return "unverifiable", error_type
    if not raw:
        return "unverifiable", "gemini_empty_response"

    text = raw.lower()

    for label in ("supported", "contradicted", "unverifiable"):
        if label in text:
            return label, ""

    try:
        import json
        import re

        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            label = str(data.get("label", "")).lower().strip()
            if label in ("supported", "contradicted", "unverifiable"):
                return label, ""

    except Exception:
        pass

    return "unverifiable", "verification_parse_failed"
