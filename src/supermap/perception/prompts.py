"""Predefined object prompt list (REQ-013, Sec. VII Limitation).

Symbol map (paper -> code):
    c (label set of L_j, Eq. 10) -> PromptList.phrases, class index 0..C-1
    C (number of classes)        -> PromptList.C
The detector text input follows the GroundingDINO convention "chair . table . plant .".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

SEPARATOR = " . "
TERMINATOR = " ."


# REQ-013: normalize a phrase (strip, lowercase, single spaces)
def _norm(p: object) -> str:
    return " ".join(str(p).strip().lower().split())


# REQ-013: fixed list of object class phrases from config; defines the label domain c
@dataclass(frozen=True)
class PromptList:
    """Ordered, lowercase, unique object phrases. Index i is class i (0-based, D-022)."""

    phrases: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...] = ()
    spans: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    vocab_class_ids: np.ndarray = field(init=False, repr=False, compare=False)

    # REQ-013: normalize (strip, lowercase) and validate the phrases and aliases
    def __post_init__(self) -> None:
        norm = tuple(_norm(p) for p in self.phrases)
        if len(norm) == 0:
            raise ValueError("prompt list must not be empty")
        raw_aliases = tuple(self.aliases) if self.aliases else tuple(() for _ in norm)
        if len(raw_aliases) != len(norm):
            raise ValueError("aliases must have one tuple per class phrase")
        alias = tuple(tuple(_norm(a) for a in al) for al in raw_aliases)
        vocab = norm + tuple(a for al in alias for a in al)
        for p in vocab:
            if not p:
                raise ValueError("prompt phrases must be non-empty")
            if "." in p:
                raise ValueError(f"prompt phrase {p!r} must not contain '.' (GroundingDINO separator)")
        if len(set(vocab)) != len(vocab):
            raise ValueError(f"prompt phrases and aliases must be unique, got {vocab}")
        ids = list(range(len(norm))) + [c for c, al in enumerate(alias) for _ in al]
        spans, pos = [], 0
        for p in vocab:
            spans.append((pos, pos + len(p)))
            pos += len(p) + len(SEPARATOR)
        object.__setattr__(self, "phrases", norm)
        object.__setattr__(self, "aliases", alias)
        object.__setattr__(self, "spans", tuple(spans))
        object.__setattr__(self, "vocab_class_ids", np.asarray(ids, dtype=np.int64))

    # REQ-013: build from a config list and optional {class phrase: [alias, ...]} mapping
    @classmethod
    def from_config(cls, prompts: Iterable[str], aliases: Mapping[str, object] | None = None) -> "PromptList":
        if isinstance(prompts, str):
            raise ValueError("prompts must be a list of strings, not a single string")
        phrases = tuple(_norm(p) for p in prompts)
        per_class: dict[str, tuple[str, ...]] = {p: () for p in phrases}
        for key, val in (aliases or {}).items():
            k = _norm(key)
            if k not in per_class:
                raise ValueError(f"alias key {key!r} is not a class phrase in {phrases}")
            vals = (val,) if isinstance(val, str) else tuple(val)
            per_class[k] = per_class[k] + tuple(vals)
        return cls(phrases, tuple(per_class[p] for p in phrases))

    # REQ-013: all detector phrases in text order (class phrases, then aliases class by class)
    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self.phrases + tuple(a for al in self.aliases for a in al)

    @property
    def C(self) -> int:
        return len(self.phrases)

    def __len__(self) -> int:
        return len(self.phrases)

    # REQ-013: detector text input, e.g. "chair . table . plant ."
    def text(self) -> str:
        return SEPARATOR.join(self.vocabulary) + TERMINATOR

    # REQ-013: map a label string to its class index
    def index(self, label: str) -> int:
        key = " ".join(str(label).strip().lower().split())
        try:
            return self.phrases.index(key)
        except ValueError:
            raise ValueError(f"label {label!r} is not in the prompt list {self.phrases}") from None

    # REQ-012, REQ-013: class index of each text token from tokenizer char offsets (-1 = not a phrase token)
    def token_class_ids(self, offsets: Sequence[tuple[int, int]]) -> np.ndarray:
        out = np.full(len(offsets), -1, dtype=np.int64)
        for k, (s, e) in enumerate(offsets):
            if e <= s:
                continue  # special tokens have empty offsets
            for v, (ps, pe) in enumerate(self.spans):
                if s >= ps and e <= pe:
                    out[k] = self.vocab_class_ids[v]
                    break
        return out


# REQ-013, REQ-015: convert label strings to class indices z_t in the prompt list
def labels_to_indices(labels: Sequence[str], prompts: PromptList | Sequence[str]) -> np.ndarray:
    pl = prompts if isinstance(prompts, PromptList) else PromptList.from_config(prompts)
    return np.asarray([pl.index(l) for l in labels], dtype=np.int64)
