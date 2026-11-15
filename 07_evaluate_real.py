"""Step 7 - evaluate on REAL continuous signing (the honest test).

Reads real_test/manifest.csv (written by 06_record.py --sentence ...), runs each
recording through the same steps as the live camera, and reports:
  * word error rate and exact sentence accuracy for the sign sequences,
  * accuracy of the final Urdu sentence,
  * results per signer, so you can see whether a new person does worse.

Put 10-20 recordings from 2-3 people here before you claim any accuracy.
"""
import csv
from collections import defaultdict

from config import FRAME_STRIDE, REAL_DIR, REPORT_DIR, SENTENCES_CSV
from psl.features import read_clip_frames, trim_idle
from psl.infer import interpret, load_model, predict_batch, to_tensor
from psl.urdu import load_sentences, word_error_rate


def main():
    manifest = REAL_DIR / "manifest.csv"
    if not manifest.exists():
        raise SystemExit("No real recordings yet. Use: python 06_record.py --sentence <id> --count 10 --signer <name>")
    model, vocab, _ = load_model()
    sentences = load_sentences(SENTENCES_CSV)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))

    refs, hyps, sentence_ok, details = [], [], [], []
    for row in rows:
        frames = read_clip_frames(REAL_DIR / row["file"])
        kept, _ = trim_idle(frames)
        pred = predict_batch(model, [to_tensor(kept[::FRAME_STRIDE])], vocab)[0]
        result = interpret(pred, sentences)
        ref = [s.strip() for s in row["signs"].split("|")]
        target_id = int(row["sentence_id"])
        refs.append(ref)
        hyps.append(pred)
        ok = bool(result["matched"] and result["sentence_id"] == target_id)
        sentence_ok.append(ok)
        details.append({"file": row["file"], "signer": row["signer"], "target": row["urdu"],
                        "predicted_signs": " | ".join(pred), "predicted_urdu": result["urdu"] or "(no match)",
                        "correct": ok})

    wer = word_error_rate(refs, hyps)
    exact = sum(r == h for r, h in zip(refs, hyps)) / len(refs)
    sent_acc = sum(sentence_ok) / len(sentence_ok)

    by_signer = defaultdict(list)
    for d in details:
        by_signer[d["signer"]].append(d["correct"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Real continuous-signing test", "",
             f"Recordings: {len(rows)}",
             f"- Word error rate (sign level): **{wer:.3f}**",
             f"- Exact sign-sequence accuracy: **{exact:.3f}**",
             f"- Correct Urdu sentence: **{sent_acc:.3f}**", "",
             "| signer | recordings | correct sentence |", "|---|---|---|"]
    for signer, oks in sorted(by_signer.items()):
        lines.append(f"| {signer} | {len(oks)} | {sum(oks)} / {len(oks)} |")
    lines += ["", "| file | signer | target | predicted Urdu | correct |", "|---|---|---|---|---|"]
    for d in details:
        lines.append(f"| {d['file']} | {d['signer']} | {d['target']} | {d['predicted_urdu']} | {'yes' if d['correct'] else 'no'} |")
    (REPORT_DIR / "real_test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Recordings: {len(rows)}   WER {wer:.3f}   exact sign accuracy {exact:.3f}   "
          f"correct sentence {sent_acc:.3f}")
    print(f"Written to {REPORT_DIR / 'real_test_results.md'}")


if __name__ == "__main__":
    main()
