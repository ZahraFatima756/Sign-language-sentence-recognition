"""Urdu text helpers.

Why this exists: the dataset folder names and the sentences were typed with
different Unicode characters (Arabic ي and ك versus Urdu ی and ک, and an extra
letter in the engineering spelling). Everything is normalised to Urdu code
points before matching, and output text is always the normalised Urdu form.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from config import SIGN_ALIASES

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff"), None)
# Arabic letters that look like Urdu letters -> the Urdu letter used in PSL text
_ARABIC_TO_URDU = str.maketrans({
    "\u064a": "\u06cc",  # Arabic yeh   -> Urdu yeh  (ی)
    "\u0649": "\u06cc",  # alef maksura -> Urdu yeh  (ی)
    "\u0643": "\u06a9",  # Arabic kaf   -> Urdu kaf  (ک)
    "\u0647": "\u06c1",  # Arabic heh   -> Urdu heh  (ہ)
})


def normalise(text: str) -> str:
    """Return the Urdu-normalised form of a string (safe for matching and display)."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = text.translate(_ARABIC_TO_URDU)
    return re.sub(r"\s+", " ", text).strip()


def skeleton(text: str) -> str:
    """Letters only, with ی and ئ removed, so spelling variants of one word compare equal."""
    return normalise(text).replace("\u06cc", "").replace("\u0626", "").replace(" ", "")


def canonical_sign(folder_name: str, known_signs: list[str]) -> str | None:
    """Map a dataset folder name to the sign it represents, or None if no sign matches."""
    if folder_name in SIGN_ALIASES:
        return normalise(SIGN_ALIASES[folder_name])
    target = normalise(folder_name)
    distinct = {normalise(s) for s in known_signs}  # the same sign can appear in many sentences
    if target in distinct:
        return target
    loose = {s for s in distinct if skeleton(s) == skeleton(target)}
    if len(loose) == 1:
        return loose.pop()
    return None


def load_sentences(path: Path) -> list[dict]:
    """Read sentences.csv. Columns: id, urdu, signs (signs separated by ' | ')."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            signs = [normalise(s) for s in row["signs"].split("|") if s.strip()]
            rows.append({
                "id": int(row["id"]),
                "urdu": normalise(row["urdu"]),
                "signs": signs,
            })
    return rows


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two sign sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_error_rate(refs: list[list[str]], hyps: list[list[str]]) -> float:
    """Corpus-level WER: total sign edits divided by total reference signs."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / total if total else 0.0


def match_sentence(predicted: list[str], sentences: list[dict]) -> tuple[dict | None, float]:
    """Return (best sentence, normalised distance). Distance 0 means an exact match."""
    if not sentences:
        return None, 1.0
    scored = []
    for s in sentences:
        d = edit_distance(predicted, s["signs"]) / max(len(predicted), len(s["signs"]), 1)
        scored.append((d, s))
    scored.sort(key=lambda t: (t[0], t[1]["id"]))
    best_d, best = scored[0]
    return best, best_d
