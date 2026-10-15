"""Answer parser for VLM outputs (REQ-058, p.5 Sec. IV-D b)).

Rule (D-082): use the last <answer>...</answer> block (case-insensitive tags); inside it every
integer, written as "12" or "Instance_12", is an instance ID, in order of appearance.
No tag -> [].
"""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ID_RE = re.compile(r"(?<![\d.])\d+(?![\d.])")


# REQ-058: extract instance IDs from the last <answer></answer> block
def parse_answer(text: str) -> list[int]:
    blocks = ANSWER_RE.findall(text or "")
    if not blocks:
        return []
    return [int(m) for m in ID_RE.findall(blocks[-1])]
