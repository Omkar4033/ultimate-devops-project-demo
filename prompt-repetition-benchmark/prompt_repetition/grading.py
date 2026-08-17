"""Answer extraction and grading for model responses."""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    text = text.strip().strip(".,!?:;'\"*` ")
    return " ".join(text.split()).lower()


def extract_answer(response: str) -> str:
    if not response:
        return ""
    match = re.search(r"answer is[:\s]*\**\"?'?\s*([^\n\".]+)", response, re.IGNORECASE)
    if match:
        candidate = match.group(1)
    else:
        lines = [line for line in response.strip().splitlines() if line.strip()]
        candidate = lines[-1] if lines else ""
    return _normalize(candidate)


def is_correct(response: str, expected: str) -> bool:
    extracted = extract_answer(response)
    expected_norm = _normalize(expected)
    if not extracted or not expected_norm:
        return False
    return extracted == expected_norm or expected_norm in extracted
