"""Stitch isolated word clips into continuous sentences.

Every random choice is written into a transcript record (clip ids, speed factors,
blend lengths, idle frames). render_transcript() rebuilds the exact frames from
that record, so the transcript file is the source of truth for the dataset.

Each stitched sentence has:
  * a random speed factor per word (time-warp), so the same sign does not always
    have the same duration;
  * 0-MAX_REST_FRAMES idle frames at the start and end (the signer rests);
  * 1-3 blended frames between words, so transitions are not hard cuts.
"""
from __future__ import annotations

import numpy as np

from config import BLEND_RANGE, MAX_REST_FRAMES, SPEED_RANGE


def sample_transcript(sentence: dict, split: str, pools: dict, rng: np.random.Generator) -> dict:
    """Draw one stitched variant of a sentence from the clip pools of one split."""
    clip_ids = [int(rng.choice(pools[sign])) for sign in sentence["signs"]]
    n_gaps = max(0, len(clip_ids) - 1)
    return {
        "sentence_id": sentence["id"],
        "urdu": sentence["urdu"],
        "signs": list(sentence["signs"]),
        "split": split,
        "clip_ids": clip_ids,
        "speeds": [round(float(s), 3) for s in rng.uniform(*SPEED_RANGE, size=len(clip_ids))],
        "blends": [int(b) for b in rng.integers(BLEND_RANGE[0], BLEND_RANGE[1] + 1, size=n_gaps)],
        "rest_start": int(rng.integers(0, MAX_REST_FRAMES + 1)),
        "rest_end": int(rng.integers(0, MAX_REST_FRAMES + 1)),
    }


def time_warp(frames: np.ndarray, speed: float) -> np.ndarray:
    """Resample a clip so it plays `speed` times faster (speed > 1 shortens it)."""
    n = len(frames)
    length = max(2, int(round(n / speed)))
    pos = np.linspace(0, n - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo).astype(np.float32)[:, None, None]
    out = (1 - w) * frames[lo].astype(np.float32) + w * frames[hi].astype(np.float32)
    return np.round(out).astype(np.uint8)


def render_transcript(t: dict, clip_frames) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Rebuild the stitched frames from a transcript.

    clip_frames: callable clip_id -> (T, S, S) uint8 array (trimmed clip).
    Returns (frames, word_spans, transition_spans); spans are [start, end) indices.
    """
    segments = [time_warp(clip_frames(cid), sp) for cid, sp in zip(t["clip_ids"], t["speeds"])]

    parts: list[np.ndarray] = []
    word_spans: list[tuple[int, int]] = []
    trans_spans: list[tuple[int, int]] = []
    pos = 0
    if t["rest_start"]:
        parts.append(np.repeat(segments[0][:1], t["rest_start"], axis=0))
        pos += t["rest_start"]
    for i, seg in enumerate(segments):
        word_spans.append((pos, pos + len(seg)))
        parts.append(seg)
        pos += len(seg)
        if i < len(segments) - 1:
            b = t["blends"][i]
            last = seg[-1].astype(np.float32)
            nxt = segments[i + 1][0].astype(np.float32)
            alphas = np.arange(1, b + 1, dtype=np.float32) / (b + 1)
            blend = np.stack([(1 - a) * last + a * nxt for a in alphas])
            parts.append(np.round(blend).astype(np.uint8))
            trans_spans.append((pos, pos + b))
            pos += b
    if t["rest_end"]:
        parts.append(np.repeat(segments[-1][-1:], t["rest_end"], axis=0))
    return np.concatenate(parts), word_spans, trans_spans


def transcript_key(t: dict) -> tuple:
    """Identity of a variant, used to avoid generating the same sentence twice."""
    return (tuple(t["clip_ids"]), tuple(t["speeds"]), tuple(t["blends"]),
            t["rest_start"], t["rest_end"])
