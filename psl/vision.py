"""Camera-side hand processing.

This mirrors the author's app2_video.py step by step, so a live frame goes through
the same operations that produced the dataset:
  flip -> MediaPipe hands -> bounding box -> draw rectangle and landmarks ->
  mask to the bounding box -> threshold at 215 -> crop -> resize -> pad to 400x400.

The resize branches reproduce the author's code exactly, including the swapped
width and height in one branch. Changing them would make live frames differ from
the dataset.

Needs the MediaPipe 'solutions' API (mediapipe 0.10.14 or earlier, as in the author's
app). Dataset tools do not need MediaPipe.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from config import CAPTURE_SIZE, HAND_SIZE_MAX, PAD_VALUE, THRESH_VALUE


def process_hands(frame_bgr: np.ndarray, multi_hand_landmarks, mp_drawing, mp_hands):
    """Return the 400x400 gray hand image for this frame, or None if no hand is seen.

    frame_bgr must already be flipped horizontally, as in app2_video.py.
    Drawing on frame_bgr happens in place, exactly as the author's code does.
    """
    if not multi_hand_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for hand in multi_hand_landmarks:
        for lm in hand.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            x_min, x_max = min(x_min, cx), max(x_max, cx)
            y_min, y_max = min(y_min, cy), max(y_max, cy)
        cv.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        mp_drawing.draw_landmarks(frame_bgr, hand, mp_hands.HAND_CONNECTIONS)
    return crop_hand_frame(frame_bgr, (x_min, y_min, x_max, y_max))


def crop_hand_frame(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]):
    """Mask to the hand bounding box, threshold, crop, resize and pad (as app2_video.py)."""
    h, w = frame_bgr.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    mask = np.zeros((h, w), np.uint8)
    cv.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    masked = cv.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, THRESH_VALUE, 255, cv.THRESH_BINARY)

    cropped = thresh[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None

    ch, cw = cropped.shape
    if ch < HAND_SIZE_MAX or cw < HAND_SIZE_MAX:
        if ch < HAND_SIZE_MAX and cw < HAND_SIZE_MAX:
            pass
        elif ch < HAND_SIZE_MAX:
            cropped = cv.resize(cropped, (HAND_SIZE_MAX, ch), interpolation=cv.INTER_LINEAR)
        else:  # cw < HAND_SIZE_MAX: the author's code passes (height, 370) here
            cropped = cv.resize(cropped, (ch, HAND_SIZE_MAX), interpolation=cv.INTER_LINEAR)
    else:
        cropped = cv.resize(cropped, (HAND_SIZE_MAX, HAND_SIZE_MAX), interpolation=cv.INTER_LINEAR)

    cy, cx = cropped.shape
    if cy > CAPTURE_SIZE or cx > CAPTURE_SIZE:  # author's code would fail here; shrink to fit
        s = CAPTURE_SIZE / max(cy, cx)
        cropped = cv.resize(cropped, (max(1, int(cx * s)), max(1, int(cy * s))), interpolation=cv.INTER_LINEAR)
        cy, cx = cropped.shape
    py, px = (CAPTURE_SIZE - cy) // 2, (CAPTURE_SIZE - cx) // 2
    return cv.copyMakeBorder(cropped, py, CAPTURE_SIZE - cy - py, px, CAPTURE_SIZE - cx - px,
                             cv.BORDER_CONSTANT, None, value=PAD_VALUE)
