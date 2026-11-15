"""Tests that run without a webcam or MediaPipe.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CAPTURE_SIZE, MODEL_SIZE, PAD_VALUE  # noqa: E402
from psl.features import resample_by_time, to_model_input, trim_idle  # noqa: E402
from psl.model import greedy_decode  # noqa: E402
from psl.stitch import render_transcript, time_warp  # noqa: E402
from psl.urdu import canonical_sign, edit_distance, load_sentences, normalise, word_error_rate  # noqa: E402
from psl.vision import crop_hand_frame  # noqa: E402


class UrduTests(unittest.TestCase):
    def test_arabic_letters_are_normalised_to_urdu(self):
        self.assertEqual(normalise("\u064a"), "\u06cc")  # Arabic yeh -> Urdu yeh
        self.assertEqual(normalise("\u0643"), "\u06a9")  # Arabic kaf -> Urdu kaf

    def test_dataset_folder_matches_sentence_sign(self):
        signs = [s for row in load_sentences(ROOT / "sentences.csv") for s in row["signs"]]
        folder = "انجينيئرنگ"  # spelling used by the dataset folder
        self.assertEqual(canonical_sign(folder, signs), normalise("انجینئرنگ"))

    def test_all_sentences_load_with_signs(self):
        rows = load_sentences(ROOT / "sentences.csv")
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(len(r["signs"]) >= 2 for r in rows))


class MetricTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "c"]), 1)
        self.assertEqual(edit_distance([], ["a"]), 1)

    def test_wer(self):
        self.assertAlmostEqual(word_error_rate([["a", "b"]], [["a", "x"]]), 0.5)


class FeatureTests(unittest.TestCase):
    def test_model_input_shape_and_empty_frame(self):
        blank = np.ones((CAPTURE_SIZE, CAPTURE_SIZE), np.uint8)
        out = to_model_input(blank)
        self.assertEqual(out.shape, (MODEL_SIZE, MODEL_SIZE))
        self.assertEqual(out.max(), 0)

    def test_model_input_keeps_skeleton(self):
        frame = np.ones((CAPTURE_SIZE, CAPTURE_SIZE), np.uint8)
        frame[100:200, 150:155] = 255
        out = to_model_input(frame)
        self.assertGreater(out.max(), 100)

    def test_trim_keeps_moving_part(self):
        still = np.zeros((10, MODEL_SIZE, MODEL_SIZE), np.uint8)
        moving = still.copy()
        for i in range(10):
            moving[i, i * 2:i * 2 + 20, 10:30] = 255  # large enough to pass the motion threshold
        frames = np.concatenate([still[:4], moving[:8], still[:4]])
        kept, (s, e) = trim_idle(frames)
        self.assertLess(len(kept), len(frames))
        self.assertGreaterEqual(len(kept), 8)

    def test_resample_by_time_length(self):
        frames = np.arange(5).reshape(5, 1, 1)
        times = np.array([0.0, 0.05, 0.1, 0.25, 0.3])  # irregular capture times
        out = resample_by_time(frames, times, fps=10)
        self.assertEqual(len(out), 4)  # 0.0 .. 0.3 at 10 fps


class CameraCropTests(unittest.TestCase):
    def test_crop_output_is_capture_size_with_author_padding(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[200:260, 300:340] = 255  # bright hand-sized area
        out = crop_hand_frame(frame, (300, 200, 340, 260))
        self.assertEqual(out.shape, (CAPTURE_SIZE, CAPTURE_SIZE))
        self.assertEqual(out[0, 0], PAD_VALUE)  # padded border uses the author's value

    def test_empty_bbox_returns_none(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertIsNone(crop_hand_frame(frame, (10, 10, 10, 10)))


class StitchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clips = {i: rng.integers(0, 256, size=(12 + i, MODEL_SIZE, MODEL_SIZE), dtype=np.uint8)
                      for i in range(3)}
        self.t = {"clip_ids": [0, 1, 2], "speeds": [1.0, 1.1, 0.9], "blends": [2, 1],
                  "rest_start": 2, "rest_end": 1}

    def test_render_is_deterministic(self):
        a, _, _ = render_transcript(self.t, self.clips.__getitem__)
        b, _, _ = render_transcript(self.t, self.clips.__getitem__)
        np.testing.assert_array_equal(a, b)

    def test_spans_cover_words_and_transitions(self):
        frames, words, trans = render_transcript(self.t, self.clips.__getitem__)
        self.assertEqual(len(words), 3)
        self.assertEqual(len(trans), 2)
        self.assertEqual(words[0][0], self.t["rest_start"])
        self.assertEqual(trans[0], (words[0][1], words[0][1] + 2))
        self.assertEqual(words[-1][1] + self.t["rest_end"], len(frames))

    def test_time_warp_changes_length(self):
        x = np.zeros((20, 2, 2), np.uint8)
        self.assertEqual(len(time_warp(x, 1.25)), 16)
        self.assertEqual(len(time_warp(x, 0.8)), 25)


class DecodeTests(unittest.TestCase):
    def test_greedy_ctc_collapses_repeats_and_blanks(self):
        vocab = ["a", "b"]
        # frames: a a blank b b blank  -> ['a', 'b']
        logits = torch.full((1, 6, 3), -5.0)
        for t, idx in enumerate([1, 1, 0, 2, 2, 0]):
            logits[0, t, idx] = 5.0
        self.assertEqual(greedy_decode(logits, [6], vocab), [["a", "b"]])


if __name__ == "__main__":
    unittest.main()
