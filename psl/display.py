"""Draw Urdu text on OpenCV frames.

OpenCV's putText cannot draw Urdu (no Arabic shaping, wrong direction). This uses
Pillow with arabic-reshaper (joins the letters) and python-bidi (right-to-left
order). It needs an Urdu-capable font, e.g. Noto Nastaliq Urdu. Set PSL_URDU_FONT
to its .ttf path. Without a font or the libraries, the text is printed in the
terminal only.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import URDU_FONT_ENV


@lru_cache(maxsize=None)
def _font(size: int):
    path = os.environ.get(URDU_FONT_ENV)
    if path and os.path.exists(path):
        return ImageFont.truetype(path, size)
    return None


def shape_urdu(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return text
    return get_display(arabic_reshaper.reshape(text))


def urdu_available() -> bool:
    try:
        import arabic_reshaper  # noqa: F401
        import bidi  # noqa: F401
    except ImportError:
        return False
    return _font(32) is not None


def draw_urdu(frame_bgr: np.ndarray, text: str, lines: list[str] | None = None, size: int = 34) -> np.ndarray:
    """Return a copy of the frame with the Urdu sentence (and optional lines) drawn at the bottom."""
    font = _font(size)
    if font is None:
        return frame_bgr
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    w, h = img.size
    rows = [shape_urdu(text)] + [l for l in (lines or [])]
    y = h - 14 - size * len(rows) - 10
    draw.rectangle([0, y - 8, w, h], fill=(0, 0, 0))
    for i, row in enumerate(rows):
        f = font if i == 0 else _font(max(16, size - 12))
        bbox = draw.textbbox((0, 0), row, font=f)
        draw.text((w - (bbox[2] - bbox[0]) - 12, y + i * size), row, font=f, fill=(255, 255, 255))
    return np.array(img)[:, :, ::-1].copy()
