"""Data access for training and evaluation.

Stitched sentences are rendered from their transcript when they are loaded, so the
disk holds only the clip cache and the transcript files.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CACHE_DIR, FRAME_STRIDE, RUN_DIR
from psl.stitch import render_transcript


class ClipStore:
    """Read-only access to the trimmed clip cache built by 01_build_clips.py."""

    def __init__(self):
        self.frames = np.load(CACHE_DIR / "frames.npy", mmap_mode="r")
        self.clips = json.loads((CACHE_DIR / "clips.json").read_text(encoding="utf-8"))
        self.by_id = {c["clip_id"]: c for c in self.clips}

    def frames_of(self, cid: int) -> np.ndarray:
        c = self.by_id[cid]
        return np.asarray(self.frames[c["offset"]:c["offset"] + c["length"]])

    def ids_for(self, sign: str, split: str) -> list[int]:
        return [c["clip_id"] for c in self.clips if c["sign"] == sign and c["split"] == split]


def load_transcripts(split: str) -> list[dict]:
    path = RUN_DIR / "transcripts" / f"{split}.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_vocab(store: ClipStore, transcripts: list[dict]) -> list[str]:
    """Vocabulary = every sign that appears in a transcript, in a stable order."""
    return sorted({s for t in transcripts for s in t["signs"]})


class StitchedDataset(Dataset):
    def __init__(self, transcripts, store: ClipStore, vocab: list[str], train: bool, seed: int = 0):
        self.transcripts = transcripts
        self.store = store
        self.vocab = vocab
        self.index = {s: i + 1 for i, s in enumerate(vocab)}  # 0 is the CTC blank
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, i):
        t = self.transcripts[i]
        frames, _, _ = render_transcript(t, self.store.frames_of)
        # random phase when training, so the model sees both halves of the 10 fps stream
        offset = int(self.rng.integers(0, FRAME_STRIDE)) if self.train else 0
        frames = frames[offset::FRAME_STRIDE]
        x = torch.from_numpy(frames.astype(np.float32) / 255.0)
        y = torch.tensor([self.index[s] for s in t["signs"]], dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    x = torch.zeros(len(xs), int(lengths.max()), *xs[0].shape[1:])
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
    y = torch.cat(ys)
    target_lengths = torch.tensor([len(t) for t in ys], dtype=torch.long)
    return x, y, lengths, target_lengths


def make_clip_samples(store: ClipStore, split: str, vocab: list[str]) -> list[tuple[str, torch.Tensor]]:
    """Single isolated clips, used for the isolated-word accuracy test."""
    out = []
    for sign in vocab:
        for cid in store.ids_for(sign, split):
            frames = store.frames_of(cid)[::FRAME_STRIDE]
            out.append((sign, torch.from_numpy(frames.astype(np.float32) / 255.0)))
    return out
