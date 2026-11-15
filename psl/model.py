"""Sentence recogniser: frame CNN -> BiGRU -> CTC over sign tokens.

CTC means the network learns to output the sign sequence without needing word
boundaries. Index 0 is the CTC blank; sign i of the vocabulary is index i + 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONV_CHANNELS, GRU_HIDDEN, MODEL_SIZE


class SignSentenceNet(nn.Module):
    def __init__(self, n_signs: int, channels=CONV_CHANNELS, hidden=GRU_HIDDEN):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:  # each block halves the resolution
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.encoder = nn.Sequential(*layers)
        side = MODEL_SIZE
        for _ in channels:
            side = (side + 1) // 2
        self.frame_fc = nn.Sequential(nn.Flatten(), nn.Linear(c_in * side * side, hidden), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, n_signs + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, S, S) float in 0..1  ->  logits (B, T, n_signs + 1)."""
        b, t, s, _ = x.shape
        feats = self.frame_fc(self.encoder(x.reshape(b * t, 1, s, s)))
        out, _ = self.rnn(feats.reshape(b, t, -1))
        return self.head(out)


def greedy_decode(logits: torch.Tensor, lengths: list[int], vocab: list[str]) -> list[list[str]]:
    """Best-path CTC decoding: take the argmax per frame, merge repeats, drop blanks."""
    best = logits.argmax(-1).cpu().numpy()
    results = []
    for seq, n in zip(best, lengths):
        words, prev = [], 0
        for idx in seq[:n]:
            if idx != prev and idx != 0:
                words.append(vocab[idx - 1])
            prev = idx
        results.append(words)
    return results
