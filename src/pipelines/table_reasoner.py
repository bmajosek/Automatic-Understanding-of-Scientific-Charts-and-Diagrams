"""Deterministic symbolic reasoning over linearized table text."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .common import extract_number_tokens, parse_number, safe_str


DEPLOT_ROW_TOKEN_RE = re.compile(r"<0x0a>", re.IGNORECASE)
DEPLOT_TAB_TOKEN_RE = re.compile(r"<0x09>", re.IGNORECASE)


def normalise_deplot_table_text(table_text: str) -> str:
    """Decode the row and tab markers emitted by the DePlot checkpoint.

    Hugging Face's decoded DePlot output contains the literal token ``<0x0A>``
    between table rows.  Treating it as ordinary text collapses the complete
    table into one line, which in turn makes both cell alignment and symbolic
    reasoning invalid.
    """
    text = safe_str(table_text)
    text = DEPLOT_ROW_TOKEN_RE.sub("\n", text)
    return DEPLOT_TAB_TOKEN_RE.sub("\t", text)


def _split_table_line(line: str) -> List[str]:
    """Split a DePlot row without breaking labels that contain commas."""
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return []
    return [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", stripped)]


def parse_deplot_cells(table_text: str) -> List[Tuple[str, str, str]]:
    """Convert a linearized DePlot table to ``(series, category, value)`` cells.

    DePlot emits a header row followed by data rows.  The first header cell
    names the category axis, the remaining header cells are series names, and
    the first cell of each data row is the category.  Values without a matching
    header are ignored rather than assigned an invented key.
    """
    table_lines: List[List[str]] = []
    for line in normalise_deplot_table_text(table_text).splitlines():
        cells = _split_table_line(line)
        if not cells:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        table_lines.append(cells)

    if len(table_lines) < 2 or len(table_lines[0]) < 2:
        return []

    series_headers = table_lines[0][1:]
    parsed: List[Tuple[str, str, str]] = []
    for row in table_lines[1:]:
        if len(row) < 2:
            continue
        category = row[0].strip()
        if not category:
            continue
        for series, value in zip(series_headers, row[1:]):
            series = series.strip()
            value = value.strip()
            if series and value:
                parsed.append((series, category, value))
    return parsed


def _table_rows(table_text: str) -> List[str]:
    rows = []
    text = normalise_deplot_table_text(table_text)
    parsed_cells = (
        parse_deplot_cells(text)
        if DEPLOT_ROW_TOKEN_RE.search(safe_str(table_text))
        else []
    )
    if parsed_cells:
        return [f"{series} | {category} | {value}" for series, category, value in parsed_cells]
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(line)
    return rows


def _row_numbers(row: str) -> List[float]:
    nums = []
    for tok in extract_number_tokens(row):
        val = parse_number(tok)
        if val is not None:
            nums.append(val)
    return nums


def symbolic_answer(question: str, table_text: str) -> Tuple[str, str]:
    q = safe_str(question).lower()
    rows = _table_rows(table_text)
    if not rows:
        return "unknown", "symbolic_reasoning_failed"

    all_nums: List[float] = []
    for row in rows:
        all_nums.extend(_row_numbers(row))

    if not all_nums:
        return "unknown", "symbolic_reasoning_failed"

    if any(k in q for k in ("maximum", "max", "highest", "largest", "most")):
        return str(max(all_nums)), ""
    if any(k in q for k in ("minimum", "min", "lowest", "smallest", "least")):
        return str(min(all_nums)), ""
    if "average" in q or "mean" in q:
        return str(sum(all_nums) / len(all_nums)), ""
    if "sum" in q or "total" in q:
        return str(sum(all_nums)), ""
    if "difference" in q or "subtract" in q:
        if len(all_nums) >= 2:
            return str(abs(all_nums[0] - all_nums[1])), ""
    if "compare" in q or "greater" in q or "less" in q:
        if len(all_nums) >= 2:
            return "yes" if all_nums[0] > all_nums[1] else "no", ""

    # value lookup: return first number on best matching row
    keys = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*", q))
    best_score = -1
    best_val: Optional[float] = None
    for row in rows:
        row_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*", row.lower()))
        score = len(keys & row_words)
        nums = _row_numbers(row)
        if nums and score >= best_score:
            best_score = score
            best_val = nums[0]
    if best_val is not None:
        return str(best_val), ""

    if len(all_nums) == 1:
        return str(all_nums[0]), ""

    return "unknown", "symbolic_reasoning_failed"
