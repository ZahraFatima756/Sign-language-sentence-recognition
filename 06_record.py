"""Step 6 - record new clips with the laptop camera.

Two uses:

  Missing words (adds to the dataset, same format as the author's clips):
      python 06_record.py --word "ہیں" --count 40

  Real continuous signing for honest testing (one whole sentence per clip):
      python 06_record.py --sentence 5 --count 10 --signer alice

Controls: S start, R stop and save, Q quit. Each clip is saved only if a hand was
seen long enough. Frames are re-timed to 10 fps, the same timing as the dataset.
Real recordings go to real_test/ and are listed in real_test/manifest.csv.
"""
import argparse
import csv
import time

import cv2
import numpy as np
import mediapipe as mp

from config import DATA_DIR, LIVE_MIN_FRAMES, REAL_DIR, SENTENCES_CSV, WRITE_FPS, CAPTURE_SIZE
from psl.features import resample_by_time
from psl.urdu import load_sentences, normalise
from psl.vision import process_hands


def next_index(folder):
    folder.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem) for p in folder.glob("*.avi") if p.stem.isdigit()]
    return max(nums, default=0) + 1


def save_clip(path, frames, times):
    frames = resample_by_time(np.stack(frames), np.array(times), WRITE_FPS)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), WRITE_FPS,
                             (CAPTURE_SIZE, CAPTURE_SIZE), False)
    for f in frames:
        writer.write(f)
    writer.release()
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--word", help="sign name, exactly as in sentences.csv (adds a dataset clip)")
    g.add_argument("--sentence", type=int, help="sentence id from sentences.csv (real test clip)")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--signer", default="unknown", help="label for real recordings, e.g. a first name")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    sentences = load_sentences(SENTENCES_CSV)
    manifest = REAL_DIR / "manifest.csv"
    if args.word:
        word = normalise(args.word)
        known = {s for row in sentences for s in row["signs"]}
        if word not in known:
            raise SystemExit(f"'{word}' is not a sign in sentences.csv. Check the spelling.")
        folder = DATA_DIR / word
        label = word
    else:
        row = next((r for r in sentences if r["id"] == args.sentence), None)
        if row is None:
            raise SystemExit(f"No sentence with id {args.sentence} in sentences.csv")
        folder = REAL_DIR / f"sentence_{args.sentence:02d}"
        label = row["urdu"]
        print(f"Sign the whole sentence: {label}")
        print("Signs in order:", " | ".join(row["signs"]))

    capture = cv2.VideoCapture(args.camera)
    hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    saved, recording, frames, times = 0, False, [], []
    t0 = time.time()
    print(f"Recording {args.count} clip(s) for: {label}. S start, R stop, Q quit.")
    while saved < args.count:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed = process_hands(frame, results.multi_hand_landmarks, mp_drawing, mp_hands)
        if recording and processed is not None:
            frames.append(processed)
            times.append(time.time() - t0)
        view = frame.copy()
        cv2.putText(view, f"{'REC' if recording else 'ready'}  saved {saved}/{args.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow("Recorder", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not recording:
            recording, frames, times = True, [], []
        elif key == ord("r") and recording:
            recording = False
            if len(frames) < LIVE_MIN_FRAMES:
                print("  too short, not saved. Try again.")
                continue
            path = folder / f"{next_index(folder)}.avi"
            n = save_clip(path, frames, times)
            if args.sentence:
                new = not manifest.exists()
                manifest.parent.mkdir(parents=True, exist_ok=True)
                with open(manifest, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(["file", "sentence_id", "signer", "signs", "urdu"])
                    w.writerow([str(path.relative_to(REAL_DIR)), args.sentence, args.signer,
                                " | ".join(row["signs"]), row["urdu"]])
            saved += 1
            print(f"  saved {path} ({n} frames at {WRITE_FPS} fps)")
        elif key == ord("q"):
            break
    capture.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"Done. {saved} clip(s) saved.")


if __name__ == "__main__":
    main()
