"""Preprocessing shared by the dataset builder and the live camera.

The dataset frames are 400x400 gray images of a white MediaPipe skeleton on a
black background (value 1). This module turns one such frame into a small
binary-ish image for the neural network. The dataset and the live camera call
the SAME function, so both sides see identical inputs.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import (DILATE_KERNEL, INK_THRESHOLD, MIN_CLIP_FRAMES,
                    MODEL_SIZE, MOTION_THRESHOLD)


def to_model_input(gray400: np.ndarray) -> np.ndarray:
    """One saved 400x400 gray frame -> MODEL_SIZE x MODEL_SIZE uint8 image.

    Steps: binarise, thicken the skeleton lines, crop to the ink bounding box,
    pad to a square, shrink. An empty frame gives an all-zero image.
    """
    ink = np.where(gray400 > INK_THRESHOLD, 255, 0).astype(np.uint8)
    if DILATE_KERNEL > 1:
        ink = cv2.dilate(ink, np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8))
    ys, xs = np.nonzero(ink)
    out = np.zeros((MODEL_SIZE, MODEL_SIZE), np.uint8)
    if ys.size == 0:
        return out
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(crop.shape)
    square = np.zeros((side, side), np.uint8)
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return cv2.resize(square, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)


def read_clip_frames(path) -> np.ndarray:
    """Decode an .avi clip into an array of model-input frames (T, S, S) uint8."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(to_model_input(gray))
    cap.release()
    if not frames:
        return np.zeros((0, MODEL_SIZE, MODEL_SIZE), np.uint8)
    return np.stack(frames)


def motion_profile(frames: np.ndarray) -> np.ndarray:
    """Per-frame movement: mean absolute change from the previous frame, scaled to 0..1."""
    m = np.zeros(len(frames), np.float32)
    if len(frames) > 1:
        diff = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
        m[1:] = diff.mean(axis=(1, 2)) / 255.0
    return m


def trim_idle(frames: np.ndarray, threshold: float = MOTION_THRESHOLD) -> tuple[np.ndarray, tuple[int, int]]:
    """Remove still frames at the start and end of a clip.

    Returns the trimmed frames and the (start, end) indices kept from the original.
    A clip with no movement at all is returned untouched.
    """
    n = len(frames)
    if n <= MIN_CLIP_FRAMES:
        return frames, (0, n)
    active = np.nonzero(motion_profile(frames) > threshold)[0]
    if active.size == 0:
        return frames, (0, n)
    start = max(0, int(active[0]) - 1)
    end = min(n, int(active[-1]) + 2)
    if end - start < MIN_CLIP_FRAMES:
        return frames, (0, n)
    return frames[start:end], (start, end)


def resample_by_time(frames: np.ndarray, times: np.ndarray, fps: int) -> np.ndarray:
    """Re-time frames captured at irregular times onto a fixed fps grid.

    The dataset clips play at WRITE_FPS (10 fps). A webcam loop does not capture
    at exactly that rate, so frames are placed on a 10 fps timeline by holding the
    latest captured frame. Without this, signs would look faster or slower than
    the training data.
    """
    if len(frames) == 0:
        return frames
    n = max(1, int(round((times[-1] - times[0]) * fps)) + 1)
    targets = times[0] + np.arange(n) / fps
    idx = np.searchsorted(times, targets, side="right") - 1
    return frames[np.clip(idx, 0, len(frames) - 1)]
