"""Step 1 - build the clip cache.

Reads every .avi under data/FinalDataset/<word>/, converts each frame to the
model input, trims idle frames, maps the folder to its sign, and splits the
recordings into train / val / test. A recording is never shared between splits,
which is what keeps the stitched sentences honest.

Outputs (in cache/):
  frames.npy   all trimmed frames, concatenated (N, MODEL_SIZE, MODEL_SIZE) uint8
  clips.json   one entry per recording: sign, split, offset, length, trim info
  signs.json   the sign list with clip counts per split
"""
import json
import random
import statistics
import time
from collections import defaultdict

import numpy as np

from config import CACHE_DIR, DATA_DIR, MOTION_THRESHOLD, SEED, SPLIT_FRACTIONS, SENTENCES_CSV
from psl.features import motion_profile, read_clip_frames, trim_idle
from psl.urdu import canonical_sign, load_sentences


def split_recordings(clips_by_sign, seed):
    """Assign each recording to train/val/test, stratified per sign."""
    rng = random.Random(seed)
    for sign, ids in clips_by_sign.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * SPLIT_FRACTIONS["test"]))
        n_val = max(1, round(n * SPLIT_FRACTIONS["val"]))
        for k, cid in enumerate(ids):
            if k < n_test:
                yield cid, "test"
            elif k < n_test + n_val:
                yield cid, "val"
            else:
                yield cid, "train"


def main():
    t0 = time.time()
    sentences = load_sentences(SENTENCES_CSV)
    sentence_signs = sorted({s for row in sentences for s in row["signs"]})
    folders = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f"No word folders found in {DATA_DIR}. See README step 0.")

    folder_to_sign = {}
    print("Folder -> sign mapping")
    for folder in folders:
        sign = canonical_sign(folder, sentence_signs)
        if sign is None:
            sign = folder.strip()
            print(f"  WARNING  '{folder}' matches no sign in sentences.csv; kept as its own sign")
        else:
            flag = "" if sign == folder else "   (spelling normalised)"
            print(f"  {folder!r:18} -> {sign!r}{flag}")
        folder_to_sign[folder] = sign

    chunks, clips = [], []
    motion_values = []
    for folder in folders:
        sign = folder_to_sign[folder]
        for avi in sorted((DATA_DIR / folder).glob("*.avi"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            raw = read_clip_frames(avi)
            if len(raw) == 0:
                print(f"  skip empty clip {avi}")
                continue
            motion_values.append(motion_profile(raw)[1:])
            kept, (start, end) = trim_idle(raw)
            clips.append({
                "clip_id": len(clips),
                "sign": sign,
                "folder": folder,
                "file": str(avi.relative_to(DATA_DIR)),
                "orig_len": int(len(raw)),
                "trim_start": int(start),
                "trim_end": int(end),
                "length": int(len(kept)),
                "offset": int(sum(len(c) for c in chunks)),
            })
            chunks.append(kept)

    # recording-level split, stratified by sign
    by_sign = defaultdict(list)
    for c in clips:
        by_sign[c["sign"]].append(c["clip_id"])
    for cid, split in split_recordings(by_sign, SEED):
        clips[cid]["split"] = split

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = np.concatenate(chunks) if chunks else np.zeros((0, 1, 1), np.uint8)
    np.save(CACHE_DIR / "frames.npy", frames)
    with open(CACHE_DIR / "clips.json", "w", encoding="utf-8") as fh:
        json.dump(clips, fh, ensure_ascii=False, indent=1)

    signs_summary = {}
    for sign in sorted({c["sign"] for c in clips}):
        mine = [c for c in clips if c["sign"] == sign]
        signs_summary[sign] = {
            "clips": len(mine),
            "train": sum(c["split"] == "train" for c in mine),
            "val": sum(c["split"] == "val" for c in mine),
            "test": sum(c["split"] == "test" for c in mine),
            "in_sentences": sign in sentence_signs,
        }
    with open(CACHE_DIR / "signs.json", "w", encoding="utf-8") as fh:
        json.dump(signs_summary, fh, ensure_ascii=False, indent=1)

    print("\nClip summary")
    print(f"  {'sign':<14}{'clips':>6}{'train':>7}{'val':>5}{'test':>6}   frames(orig->trimmed, median)")
    for sign, s in signs_summary.items():
        mine = [c for c in clips if c["sign"] == sign]
        print(f"  {sign:<14}{s['clips']:>6}{s['train']:>7}{s['val']:>5}{s['test']:>6}   "
              f"{statistics.median(c['orig_len'] for c in mine):>5.0f} -> "
              f"{statistics.median(c['length'] for c in mine):>3.0f}")
    total = len(clips)
    trimmed = sum(1 for c in clips if c["length"] < c["orig_len"])
    print(f"\nTotal recordings: {total}   frames stored: {len(frames)}   "
          f"recordings with idle frames trimmed: {trimmed} ({100 * trimmed / max(total, 1):.0f}%)")

    mv = np.concatenate(motion_values) if motion_values else np.zeros(1)
    pct = np.percentile(mv, [25, 50, 75, 90])
    print(f"Per-frame motion percentiles (25/50/75/90): {np.round(pct, 4).tolist()}   "
          f"MOTION_THRESHOLD={MOTION_THRESHOLD}")
    print(f"Done in {time.time() - t0:.1f}s  ->  {CACHE_DIR}")


if __name__ == "__main__":
    main()
