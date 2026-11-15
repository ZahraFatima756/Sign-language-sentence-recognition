"""Step 4 - evaluate on held-out data.

Two tests, both on recordings the model never saw in training:
  1. Stitched test sentences (400 variants built from test recordings):
     word error rate (WER) and exact sentence accuracy, per sentence.
  2. Isolated words: each test recording on its own, one word per clip.
     This is the "does one sign classify correctly" number.

Caveat: splits are by recording, not by signer. The dataset has no signer labels,
so these numbers can be optimistic for a new person. Use 07_evaluate_real.py with
real continuous signing for the honest number.
"""
import json
from collections import defaultdict

import torch

from config import REPORT_DIR, SENTENCES_CSV
from psl.data import ClipStore, StitchedDataset, load_transcripts, make_clip_samples
from psl.infer import load_model, predict_batch
from psl.urdu import load_sentences, word_error_rate


def main():
    model, vocab, ckpt = load_model()
    store = ClipStore()
    sentences = {s["id"]: s for s in load_sentences(SENTENCES_CSV)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. stitched test sentences
    test_tx = load_transcripts("test")
    ds = StitchedDataset(test_tx, store, vocab, train=False)
    refs, hyps = [], []
    for start in range(0, len(ds), 32):
        batch = [ds[i][0] for i in range(start, min(start + 32, len(ds)))]
        hyps += predict_batch(model, batch, vocab)
        refs += [test_tx[i]["signs"] for i in range(start, min(start + 32, len(ds)))]
    wer = word_error_rate(refs, hyps)
    exact = sum(r == h for r, h in zip(refs, hyps)) / len(refs)

    per_sent = defaultdict(lambda: [0, 0])
    for t, r, h in zip(test_tx, refs, hyps):
        per_sent[t["sentence_id"]][0] += int(r == h)
        per_sent[t["sentence_id"]][1] += 1

    # ---- 2. isolated words
    iso = make_clip_samples(store, "test", vocab)
    iso_pred = predict_batch(model, [x for _, x in iso], vocab)
    # Two numbers. The model was trained on whole sentences, so it tends to add signs
    # after a single word. "Recognised" asks whether the target sign appears in the
    # output at all; "exact" asks whether the output is exactly that one sign.
    per_sign = defaultdict(lambda: [0, 0, 0])  # recognised, exact, total
    for (sign, _), pred in zip(iso, iso_pred):
        per_sign[sign][0] += int(sign in pred)
        per_sign[sign][1] += int(pred == [sign])
        per_sign[sign][2] += 1
    iso_recognised = sum(v[0] for v in per_sign.values()) / max(len(iso), 1)
    iso_exact = sum(v[1] for v in per_sign.values()) / max(len(iso), 1)

    results = {
        "checkpoint_epoch": ckpt["epoch"],
        "stitched_test": {"variants": len(refs), "WER": round(wer, 4), "exact_sentence_accuracy": round(exact, 4),
                          "per_sentence": {str(k): {"correct": v[0], "total": v[1],
                                                    "urdu": sentences[k]["urdu"]} for k, v in sorted(per_sent.items())}},
        "isolated_words": {"clips": len(iso), "target_sign_recognised": round(iso_recognised, 4),
                           "exact_single_sign_output": round(iso_exact, 4),
                           "per_sign": {k: {"recognised": v[0], "exact": v[1], "total": v[2]}
                                        for k, v in sorted(per_sign.items())}},
    }
    (REPORT_DIR / "test_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# Test results (held-out recordings)", "",
             f"Checkpoint: epoch {ckpt['epoch']}", "",
             f"## Stitched test sentences ({len(refs)} variants)", "",
             f"- Word error rate: **{wer:.3f}**",
             f"- Exact sentence accuracy: **{exact:.3f}**", "",
             "| id | Urdu | correct / total |", "|---|---|---|"]
    for k, v in sorted(per_sent.items()):
        lines.append(f"| {k} | {sentences[k]['urdu']} | {v[0]} / {v[1]} |")
    lines += ["", f"## Isolated words ({len(iso)} single clips)", "",
              f"- Target sign recognised somewhere in the output: **{iso_recognised:.3f}**",
              f"- Output exactly that one sign: **{iso_exact:.3f}**", "",
              "| sign | recognised | exact | total |", "|---|---|---|---|"]
    for k, v in sorted(per_sign.items()):
        lines.append(f"| {k} | {v[0]} | {v[1]} | {v[2]} |")
    lines += ["", "Caveat: splits are by recording, not by signer. Stitched results include "
              "transitions the model has learned from synthetic data, so real continuous "
              "signing (07_evaluate_real.py) is the honest test."]
    (REPORT_DIR / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Stitched test: WER {wer:.3f}   exact sentence accuracy {exact:.3f}   ({len(refs)} variants)")
    print(f"Isolated words: target recognised {iso_recognised:.3f}   exact single-sign {iso_exact:.3f}   ({len(iso)} clips)")
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
