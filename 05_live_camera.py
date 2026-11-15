"""Step 5 - live recognition from the laptop camera.

Controls (the same letters the author's app2_video.py uses):
  S  start recording a sentence (sign it in front of the camera)
  R  stop recording and predict (P also works)
  Q  quit

For each recording the script:
  1. processes every frame the way the dataset was made (vision.py),
  2. re-times the frames to 10 fps (features.resample_by_time),
  3. trims idle frames (features.trim_idle),
  4. predicts the sign sequence with the trained network,
  5. matches it to the closest sentence in sentences.csv and shows the Urdu text.

Results are appended to runs/live_log.csv.
"""
import argparse
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from config import FRAME_STRIDE, LIVE_MIN_FRAMES, REPORT_DIR, SENTENCES_CSV, WRITE_FPS
from psl.display import draw_urdu, urdu_available
from psl.features import resample_by_time, to_model_input, trim_idle
from psl.infer import interpret, load_model, predict_batch, to_tensor
from psl.urdu import load_sentences
from psl.vision import process_hands


def predict_recording(model, vocab, frames, times, sentences):
    frames = resample_by_time(np.stack(frames), np.array(times), WRITE_FPS)
    kept, _ = trim_idle(frames)
    seq = to_tensor(kept[::FRAME_STRIDE])
    signs = predict_batch(model, [seq], vocab)[0]
    return interpret(signs, sentences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--show-skeleton", action="store_true", help="show the 400x400 processed hand image")
    args = ap.parse_args()

    model, vocab, _ = load_model()
    sentences = load_sentences(SENTENCES_CSV)
    REPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR.parent / "live_log.csv"
    new_log = not log_path.exists()
    log_fh = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_fh)
    if new_log:
        log.writerow(["time", "matched", "sentence_id", "urdu", "closest_urdu", "distance", "signs"])

    capture = cv2.VideoCapture(args.camera)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    recording, frames, times = False, [], []
    last_result, status = None, "Press S to start a sign"
    t0 = time.time()
    print("Ready. S = start, R = stop and predict, Q = quit.")
    if not urdu_available():
        print("(Urdu text is printed here only. Set PSL_URDU_FONT to an Urdu .ttf to draw it on screen.)")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera not available.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        processed = process_hands(frame, results.multi_hand_landmarks, mp_drawing, mp_hands)
        if recording and processed is not None:
            frames.append(to_model_input(processed))
            times.append(time.time() - t0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not recording:
            recording, frames, times = True, [], []
            status = "Recording... press R to stop"
        elif key in (ord("r"), ord("p")) and recording:
            recording = False
            if len(frames) < LIVE_MIN_FRAMES:
                status = "Too short - no hand seen long enough. Try again."
            else:
                last_result = predict_recording(model, vocab, frames, times, sentences)
                r = last_result
                shown = r["urdu"] if r["matched"] else "(no confident match)"
                status = shown
                print(f"\nSigns:   {' | '.join(r['signs'])}")
                print(f"Urdu:    {shown}")
                if not r["matched"]:
                    print(f"Closest: {r['closest_urdu']}  (distance {r['distance']})")
                log.writerow([datetime.now().isoformat(timespec="seconds"), r["matched"], r["sentence_id"],
                              r["urdu"] or "", r["closest_urdu"] or "", r["distance"], " | ".join(r["signs"])])
                log_fh.flush()
        elif key == ord("q"):
            break

        view = frame.copy()
        lines = [" | ".join(last_result["signs"])] if last_result else []
        view = draw_urdu(view, status, lines)
        if not urdu_available():  # no Urdu font: keep an ASCII status on screen
            cv2.putText(view, "Urdu shown in terminal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("PSL recogniser", view)
        if args.show_skeleton and processed is not None:
            cv2.imshow("Processed hand (400x400)", processed)

    capture.release()
    hands.close()
    log_fh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
