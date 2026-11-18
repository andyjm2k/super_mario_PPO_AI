"""Turn MAME screen dumps into the small RGB frames the policy uses."""

from __future__ import annotations

from typing import Optional

import numpy as np


def decode_rgb_hex(picture: Optional[str], width: int, height: int) -> np.ndarray:
    """Turn a BGR hex string into an RGB uint8 image."""
    # No image on this sample.
    if not picture:
        # Black frame of the reported size.
        return np.zeros((max(int(height), 1), max(int(width), 1), 3), dtype=np.uint8)
    # Hex pairs back to bytes.
    raw = bytes.fromhex(picture)
    # B, G, R for every pixel.
    expected = int(width) * int(height) * 3
    # A short or long buffer is not a frame.
    if len(raw) != expected or width <= 0 or height <= 0:
        # Keep the caller from crashing on a torn line.
        return np.zeros((max(int(height), 1), max(int(width), 1), 3), dtype=np.uint8)
    # Packed BGR.
    bgr = np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 3))
    # MAME's bitmap is B, G, R. The booth and the policy expect RGB.
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _bgra_rgb(raw: bytes, width: int, height: int):
    """Decode a BGRA dump to RGB, or None when the buffer is the wrong size."""
    # Four bytes per pixel.
    expected = int(width) * int(height) * 4
    # A short read is not a frame.
    if len(raw) != expected or width <= 0 or height <= 0:
        # Caller keeps the previous picture.
        return None
    # Height, width, BGRA.
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 4))
    # Drop alpha and swap to RGB.
    return np.ascontiguousarray(bgra[:, :, [2, 1, 0]])


def letterbox(rgb, width: int, height: int, target_w: int = 240, target_h: int = 160):
    """Fit an RGB frame into a box and pad the rest black. No stretch."""
    # Local import so callers that never draw a frame skip OpenCV startup.
    import cv2
    # Uniform scale. A 384x224 arcade picture stays 12:7 inside the box.
    scale = min(float(target_w) / float(width), float(target_h) / float(height))
    # Fitted width.
    fitted_w = max(1, int(round(width * scale)))
    # Fitted height from the same scale.
    fitted_h = max(1, int(round(height * scale)))
    # Clamp so rounding cannot exceed the box.
    fitted_w = min(fitted_w, int(target_w))
    # Clamp height too.
    fitted_h = min(fitted_h, int(target_h))
    # Resize the picture only.
    fitted = cv2.resize(rgb, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
    # Black cabinet around the picture.
    canvas = np.zeros((int(target_h), int(target_w), 3), dtype=np.uint8)
    # Vertical offset centers a wide arcade frame.
    top = (int(target_h) - fitted_h) // 2
    # Horizontal offset centers a tall frame.
    left = (int(target_w) - fitted_w) // 2
    # Paste without stretching.
    canvas[top : top + fitted_h, left : left + fitted_w] = fitted
    # Letterboxed RGB.
    return canvas


def bgra_to_rgb(raw: bytes, width: int, height: int):
    """Shrink a BGRA screen dump to the 128x96 image the policy uses."""
    # Full-size RGB, or nothing.
    rgb = _bgra_rgb(raw, width, height)
    # A short read is not a frame.
    if rgb is None:
        # Caller keeps the previous picture.
        return None
    # Local import so unit tests that never load a frame skip OpenCV startup.
    import cv2
    # The CNN still sees the same 128x96 stretch it was trained on.
    return cv2.resize(rgb, (128, 96), interpolation=cv2.INTER_AREA)


def display_bgra(raw: bytes, width: int, height: int):
    """Booth copy of a BGRA dump, fitted without stretching."""
    # Full-size RGB, or nothing.
    rgb = _bgra_rgb(raw, width, height)
    # A short read is not a frame.
    if rgb is None:
        # No booth picture this tick.
        return None
    # 240x160 glass, arcade aspect preserved.
    return letterbox(rgb, int(width), int(height), 240, 160)


def read_exact(stream, count: int) -> bytes:
    """Read exactly ``count`` bytes from a binary pipe."""
    # Accumulated chunks.
    buf = bytearray()
    # Stop early if the pipe closes.
    while len(buf) < count:
        # One slice.
        chunk = stream.read(count - len(buf))
        # EOF.
        if not chunk:
            # Whatever arrived.
            break
        # Append.
        buf.extend(chunk)
    # Frozen bytes.
    return bytes(buf)
