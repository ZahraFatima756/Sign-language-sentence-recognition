"""Step 2 - generate the stitched sentence dataset.

For every sentence whose signs all have recordings, this writes
VARIANTS_PER_SENTENCE[split] unique transcripts per split. Each transcript says
which recordings were used, the speed of each word, the transition blends and the
idle frames, plus the Urdu sentence and the frame boundaries of every word.

The frames themselves are rebuilt from the transcript when training (exactly the
same result every time). Use --export-videos to write preview .avi files and a
montage image so you can check the seams by eye.

Outputs:
  runs/transcripts/{train,val,test}.jsonl
  runs/transcripts/summary.json
  runs/examples/*.avi, runs/examples/*_seams.png   (only with --export-videos)
"""
import argparse
import json
from collections import Counter

import cv2
import numpy as np

from config import CACHE_DIR, EXAMPLE_DIR, RUN_DIR, SEED, SENTENCES_CSV, VARIANTS_PER_SENTENCE, WRITE_FPS
from psl.stitch import render_transcript, sample_transcript, transcript_key
from psl.urdu import load_sentences

TRANSCRIPT_DIR = RUN_DIR / "transcripts"


def load_cache():
    frames = np.load(CACHE_DIR / "frames.npy", mmap_mode="r")
    clips = json.loads((CACHE_DIR / "clips.json").read_text(encoding="utf-8"))
    return frames, clips


def make_clip_accessor(frames, clips):
    by_id = {c["clip_id"]: c for c in clips}

    def clip_frames(cid):
        c = by_id[cid]
        return np.asarray(frames[c["offset"]:c["offset"] + c["length"]])
    return clip_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-videos", type=int, default=0,
                    help="write preview videos for the first N train variants of each sentence")
    args = ap.parse_args()

    frames, clips = load_cache()
    sentences = load_sentences(SENTENCES_CSV)
    signs_available = {c["sign"] for c in clips}
    clip_frames = make_clip_accessor(frames, clips)

    pools = {split: {} for split in ("train", "val", "test")}
    for c in clips:
        pools[c["split"]].setdefault(c["sign"], []).append(c["clip_id"])

    ready, blocked = [], []
    for s in sentences:
        missing = [g for g in dict.fromkeys(s["signs"]) if g not in signs_available]
        (blocked if missing else ready).append((s, missing))

    rng = np.random.default_rng(SEED)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    samples_by_split = {}
    for split in ("train", "val", "test"):
        seen = set()
        out = []
        for s, _ in ready:
            if any(sign not in pools[split] for sign in s["signs"]):
                continue
            made = 0
            attempts = 0
            while made < VARIANTS_PER_SENTENCE[split] and attempts < VARIANTS_PER_SENTENCE[split] * 20:
                attempts += 1
                t = sample_transcript(s, split, pools[split], rng)
                key = transcript_key(t)
                if key in seen:
                    continue
                seen.add(key)
                rendered, wspans, tspans = render_transcript(t, clip_frames)
                t["n_frames"] = int(len(rendered))
                t["word_spans"] = wspans
                t["transition_spans"] = tspans
                out.append(t)
                made += 1
            counts[(split, s["id"])] = made
        samples_by_split[split] = out
        with open(TRANSCRIPT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for t in out:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # printed summary
    print("Stitched sentence dataset")
    print(f"  {'id':>3}  {'urdu':<42}{'train':>7}{'val':>6}{'test':>6}")
    for s, _ in ready:
        print(f"  {s['id']:>3}  {s['urdu']:<42}"
              f"{counts.get(('train', s['id']), 0):>7}{counts.get(('val', s['id']), 0):>6}"
              f"{counts.get(('test', s['id']), 0):>6}")
    print("\n  Blocked (need recordings of the missing signs before they can be built):")
    for s, missing in blocked:
        print(f"  {s['id']:>3}  {s['urdu']:<42} missing: {' | '.join(missing)}")

    totals = {sp: len(samples_by_split[sp]) for sp in samples_by_split}
    print(f"\nTotal variants: train={totals['train']}  val={totals['val']}  test={totals['test']}  "
          f"(all = {sum(totals.values())})")
    word_counter = Counter(sign for sp in samples_by_split.values() for t in sp for sign in t["signs"])
    print("Word occurrences in the stitched data:", dict(word_counter.most_common()))

    summary = {
        "ready_sentences": [s["id"] for s, _ in ready],
        "blocked_sentences": {s["id"]: missing for s, missing in blocked},
        "variants": totals,
        "per_sentence": {f"{sp}:{sid}": n for (sp, sid), n in counts.items()},
    }
    (TRANSCRIPT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.export_videos:
        export_examples(samples_by_split["train"], clip_frames, args.export_videos)
    print(f"\nTranscripts written to {TRANSCRIPT_DIR}")


def export_examples(train_samples, clip_frames, per_sentence):
    """Write preview videos and a seam montage for a few variants per sentence."""
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    taken = Counter()
    for t in train_samples:
        if taken[t["sentence_id"]] >= per_sentence:
            continue
        taken[t["sentence_id"]] += 1
        frames, wspans, tspans = render_transcript(t, clip_frames)
        name = f"sentence{t['sentence_id']:02d}_v{taken[t['sentence_id']]}"
        scale = 6
        h = frames.shape[1] * scale
        writer = cv2.VideoWriter(str(EXAMPLE_DIR / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                 WRITE_FPS, (h, h), isColor=False)
        for f in frames:
            writer.write(cv2.resize(f, (h, h), interpolation=cv2.INTER_NEAREST))
        writer.release()

        if taken[t["sentence_id"]] == 1:
            tiles = []
            for i, f in enumerate(frames[::max(1, len(frames) // 24)][:24]):
                tile = cv2.resize(f, (96, 96), interpolation=cv2.INTER_NEAREST)
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                tiles.append(tile)
            while len(tiles) < 24:
                tiles.append(np.zeros((96, 96, 3), np.uint8))
            rows = [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(3)]
            cv2.imwrite(str(EXAMPLE_DIR / f"{name}_seams.png"), np.vstack(rows))
    print(f"Preview videos and seam montages written to {EXAMPLE_DIR}")


if __name__ == "__main__":
    main()
