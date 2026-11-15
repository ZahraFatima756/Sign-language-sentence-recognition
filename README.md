# PSL Sentence Recogniser

Recognises Pakistan Sign Language (PSL) sentences from a laptop camera and shows the
matching Urdu sentence. It is built on the SignSpeak final-year dataset.

Pipeline: webcam → MediaPipe hand skeleton → same image processing as the dataset →
neural network (CNN + BiGRU + CTC) → sign sequence → closest Urdu sentence in `sentences.csv`.

---

## 1. What the dataset really is (read this first)

The archive `FinalDataset.rar` is **not** a landmark dataset. It contains:

| Item                     | Value                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| Word folders             | 11 (see table in section 6)                                                               |
| Clips                    | 751 `.avi` files, 61 to 79 per word                                                       |
| Frame                    | 400 × 400 gray; a white MediaPipe **hand skeleton** on black, cropped and padded          |
| Frame rate / length      | 10 fps; 6 to 120 frames per clip (median 45)                                              |
| Signer or session labels | **None.** Splits are by recording, not by person                                          |
| Source                   | `app2_video.py` (included, copied to `reference/`) shows exactly how the frames were made |

Because the frames are skeleton images, the classifier works on images, and the live
camera reproduces the author's image pipeline step by step (see `psl/vision.py`).
This is what makes the live predictions comparable to the training data.

---

## 2. Setup

Use Python 3.10 to 3.12.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Training and tests need only `numpy`, `opencv-python` and `torch`. The camera scripts
also need `mediapipe==0.10.14` (the pinned version that still has the `mp.solutions`
API used by the author). Newer MediaPipe releases removed it.

**Step 0: put the dataset in place.** Extract `FinalDataset.rar` so the folder layout is:

```
psl_sentence_recognizer/
  data/FinalDataset/
    آپ/            1.avi 2.avi ...
    انجينيئرنگ/     ...
    ...            (11 word folders)
```

If you keep it elsewhere, set `PSL_DATA_DIR` to that folder.

---

## 3. Run it (in order)

```bash
python 01_build_clips.py          # ~1 min: decode clips, trim idle frames, split by recording
python 02_stitch_sentences.py     # seconds: build stitched sentence transcripts + counts
python 03_train.py                # the slow step (~5 min per epoch on one CPU core)
python 04_evaluate.py             # test scores on held-out recordings
python 05_live_camera.py          # live recognition from the laptop camera
```

Optional extra steps:

```bash
python 02_stitch_sentences.py --export-videos 1   # preview .avi + seam montage PNGs
python tests/test_pipeline.py                     # or: python -m unittest discover -s tests -v
```

### Live camera controls

| Key          | Action                                |
| ------------ | ------------------------------------- |
| `S`          | start recording one sentence          |
| `R` (or `P`) | stop, predict, show the Urdu sentence |
| `Q`          | quit                                  |

Each prediction is appended to `runs/live_log.csv`. Add `--show-skeleton` to see the
400 × 400 hand image the network actually receives.

**Urdu on screen:** set `PSL_URDU_FONT` to an Urdu font, for example
`NotoNastaliqUrdu-Regular.ttf` (free from Google Fonts). Without it the Urdu is
printed in the terminal, which is always correct.

---

## 4. Adding the missing words (needed for 10 of the 20 sentences)

Ten sentences need words the dataset does not have: **ہیں, یہ, ہے, کام, اور**. Those
sentences show as "blocked" in the output of step 2. To unblock them, record each word
the same way the dataset was recorded:

```bash
python 06_record.py --word "ہیں" --count 40
python 06_record.py --word "یہ"   --count 40
python 06_record.py --word "ہے"   --count 40
python 06_record.py --word "کام"  --count 40
python 06_record.py --word "اور"  --count 40
```

Controls: `S` start, `R` stop and save, `Q` quit. Then re-run steps 1 to 4. The new
words are picked up automatically.

---

## 5. Honest testing with real signing (do this before you report accuracy)

The stitched test set is built from the same signs, so its scores are optimistic.
The honest test is **continuous signing you record yourself**.

```bash
python 06_record.py --sentence 5 --count 10 --signer alice
python 06_record.py --sentence 5 --count 10 --signer bilal
python 07_evaluate_real.py
```

Aim for 10 to 20 recordings from 2 or 3 people. Each recording is one full sentence,
signed the way you would normally sign it. `07_evaluate_real.py` writes a per-signer
table to `runs/reports/real_test_results.md`.

---

## 6. The 11 signs in the dataset

| Folder (as in dataset) | Sign used in sentences.csv                                          |
| ---------------------- | ------------------------------------------------------------------- |
| ہیلو                   | ہیلو                                                                |
| آپ                     | آپ                                                                  |
| کیسے ہیں               | کیسے ہیں (one sign, two words)                                      |
| سب                     | سب                                                                  |
| میں                    | میں                                                                 |
| ہوں                    | ہوں                                                                 |
| نہیں                   | نہیں                                                                |
| طالب علم               | طالب علم (one sign, two words)                                      |
| کا                     | کا                                                                  |
| کمپیوٹر                | کمپیوٹر                                                             |
| انجينيئرنگ             | انجینئرنگ (the folder spelling is matched automatically, see below) |

**Urdu spelling.** The dataset folder `انجينيئرنگ` uses Arabic ي and an extra ی. The
sentences use the standard Urdu spelling `انجینئرنگ`. The code matches the two
automatically (`psl/urdu.py`), and the output always uses the standard Urdu form.
The live app shows only the text from `sentences.csv`, never dataset folder names.

---

## 7. Sentences

`sentences.csv` holds the 20 sentences with their sign sequence. Each sentence is one
row: `id, urdu, signs` (signs separated by `|`). Edit it to change the sentences or
the word order. The word order must match PSL, not Urdu, so check it with your
supervisor or a PSL signer.

Currently ready: 1, 2, 3, 4, 8, 11, 13, 17, 18, 20. Blocked until the missing words
are recorded: 5, 6, 7, 9, 10, 12, 14, 15, 16, 19.

---

## 8. How the stitched data is made

Each stitched variant is built by:

1. choosing one recording for each sign from the correct split (train, val or test),
2. time-warping each word by a random factor between 0.8× and 1.25×,
3. adding 1 to 3 blended frames between words, so transitions are not hard cuts,
4. adding 0 to 3 idle frames at the start and end, as a signer resting.

The random choices are saved in `runs/transcripts/*.jsonl` with each variant's Urdu
sentence and frame boundaries for every word. Frames are rebuilt from those records,
so the same transcript always gives the same video. Splits are made per recording
before any stitching, so no recording appears in more than one split.

Default size: 300 train, 40 validation and 40 test variants per ready sentence.
Change `VARIANTS_PER_SENTENCE` in `config.py` to make more or fewer.

---

## 9. Results

Model: best checkpoint, epoch 4 of a run that was stopped after epoch 6 (selected on
validation word error rate). Full tables: `runs/reports/test_results.md`.

| Test                              | Result                                                                 | Notes                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Stitched held-out sentences (400) | WER **1.6%**, exact sentence **94.8%**                                 | Optimistic: only 10 sentence patterns, recordings are held out but signers are not             |
| Isolated single signs (111 clips) | target sign appears in output **60%**, exact single-sign output **0%** | Model adds signs after a single word; `نہیں` (0/12) and `کا` (0/10) are never recognised alone |
| Real continuous signing           | **not yet measured**                                                   | Needs recordings: `06_record.py --sentence ...` then `07_evaluate_real.py`                     |

**Read this before quoting any number.** The stitched test has only ten sentence
patterns, all seen in training. A network can score well by recognising the whole
sentence from its shape, not each sign. The isolated-sign results support that
concern. The real-signing test in section 5 is the only test that says anything about
new signing.

Training was stopped after epoch 6 of 20 when the session ended. Validation scores
moved a lot between epochs (WER 1.3% at epoch 4, 3.6% at epoch 5), so the epoch-4
checkpoint should be treated as a single run, not a stable result.

---

## 10. Limits to know about

- **Stitched results are optimistic.** Stitching has no natural co-articulation. Real
  continuous signing is the honest test (section 5).
- **No signer labels.** Splits are by recording, so a new person may do worse. Record
  2 or 3 people to see.
- **Camera timing.** Clips play at 10 fps. Live frames are re-timed to 10 fps by
  their capture times (`psl/features.py: resample_by_time`). This assumes the
  author's clips were recorded in real time. If signs look too fast or slow in live
  use, that assumption is the first thing to check.
- **Camera match.** The live pipeline follows the author's code exactly, but the
  dataset was made on the author's camera and lighting. Check the first live
  `processed hand` image (`--show-skeleton`) against a dataset frame. If the
  background is much brighter or darker, the threshold in `config.py` may need
  adjusting, and then the dataset would need re-recording to match.
- **Sentence-level priors.** The network was trained on sentences, so with a single
  word it tends to add signs. `04_evaluate.py` reports both "target sign recognised"
  and "exact single-sign output" for this reason.
- **Sentences are matched, not generated.** A recognised sign sequence is matched to
  the closest sentence in `sentences.csv`. If nothing is close enough
  (`SENTENCE_MATCH_MAX` in `config.py`), the app says "no confident match" and shows
  the closest sentence.
- **MediaPipe version.** Use `mediapipe==0.10.14` for the camera scripts. Newer versions
  do not include `mp.solutions.hands`.

---

## 11. Files

```
config.py                  settings (capture constants copied from the author's code)
sentences.csv              the 20 sentences and their sign sequences
01_build_clips.py          step 1: decode clips, trim, split
02_stitch_sentences.py     step 2: stitched transcripts and dataset counts
03_train.py                step 3: train the CTC network
04_evaluate.py             step 4: evaluate on held-out recordings
05_live_camera.py          step 5: live laptop-camera recognition
06_record.py               record missing words or real test sentences
07_evaluate_real.py        evaluate on real continuous signing
psl/urdu.py                Urdu normalisation, folder matching, WER
psl/features.py            frame preprocessing shared by dataset and camera
psl/vision.py              camera pipeline copied from app2_video.py
psl/stitch.py              sentence stitching (transcripts → frames)
psl/data.py, psl/model.py  data loading, CNN + BiGRU + CTC network
psl/infer.py, psl/display.py  prediction helpers, Urdu on-screen text
tests/test_pipeline.py     offline unit tests
reference/                 the author's app2_video.py (for comparison)
runs/                      generated: transcripts, model, reports, examples
```

## 12. What has been tested

- Tested here: clip decoding and splitting on all 751 dataset clips, stitching and the
  seam montage, training (stopped after epoch 6), held-out evaluation, the 15 unit tests,
  Urdu shaping output, and the real-recording evaluator on a stand-in clip.
- **Not tested here:** the live camera script, the recorder, MediaPipe itself (the
  sandbox has no webcam and the pinned MediaPipe version was not installed), and a
  real Urdu font on screen. Run `05_live_camera.py --show-skeleton` once on your laptop
  before relying on it.

