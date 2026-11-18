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


def bgra_to_rgb(raw: bytes, width: int, height: int):
    """Shrink a BGRA screen dump to the 128x96 image the policy uses."""
    # Four bytes per pixel.
    expected = int(width) * int(height) * 4
    # A short read is not a frame.
    if len(raw) != expected or width <= 0 or height <= 0:
        # Caller keeps the previous picture.
        return None
    # Height, width, BGRA.
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 4))
    # Drop alpha and swap to RGB.
    rgb = np.ascontiguousarray(bgra[:, :, [2, 1, 0]])
    # Local import so unit tests that never load a frame skip OpenCV startup.
    import cv2
    # The booth and the CNN both use this size.
    return cv2.resize(rgb, (128, 96), interpolation=cv2.INTER_AREA)


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
