"""Inference helpers shared by evaluation, live camera and real-test scripts."""
from __future__ import annotations

import numpy as np
import torch

from config import MODEL_DIR, SENTENCE_MATCH_MAX
from psl.model import SignSentenceNet, greedy_decode
from psl.urdu import match_sentence


def load_model(path=MODEL_DIR / "best.pt"):
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run 03_train.py first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SignSentenceNet(len(ckpt["vocab"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"], ckpt


@torch.no_grad()
def predict_batch(model, sequences: list[torch.Tensor], vocab: list[str]) -> list[list[str]]:
    """Predict sign sequences for a list of (T, S, S) float tensors in 0..1."""
    lengths = [len(x) for x in sequences]
    x = torch.zeros(len(sequences), max(lengths), *sequences[0].shape[1:])
    for i, xi in enumerate(sequences):
        x[i, :len(xi)] = xi
    return greedy_decode(model(x), lengths, vocab)


def to_tensor(model_frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(model_frames.astype(np.float32) / 255.0)


def interpret(signs: list[str], sentences: list[dict]) -> dict:
    """Map a predicted sign list to the closest Urdu sentence and say how sure we are."""
    best, dist = match_sentence(signs, sentences)
    accepted = best is not None and dist <= SENTENCE_MATCH_MAX
    return {
        "signs": signs,
        "matched": accepted,
        "sentence_id": best["id"] if accepted else None,
        "urdu": best["urdu"] if accepted else None,
        "closest_urdu": best["urdu"] if best else None,
        "distance": round(dist, 3),
    }
