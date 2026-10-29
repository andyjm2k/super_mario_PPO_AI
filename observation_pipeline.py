"""Image observation helpers for Mario Deluxe CNN training."""

import cv2
import numpy as np


# Native Game Boy Color screen width in pixels.
NATIVE_WIDTH = 160
# Native Game Boy Color screen height in pixels.
NATIVE_HEIGHT = 144
# Default pixels cropped from the top to remove the HUD / status bar.
DEFAULT_CROP_TOP = 16
# Default CNN width preserving half of the native aspect ratio.
DEFAULT_OBS_WIDTH = 80
# Default CNN height: half of (144 - 16) playfield rows.
DEFAULT_OBS_HEIGHT = 64


def preprocess_frame(
    rgb_frame,
    crop_top=DEFAULT_CROP_TOP,
    obs_width=DEFAULT_OBS_WIDTH,
    obs_height=DEFAULT_OBS_HEIGHT,
):
    """Convert an RGB GBC frame into a single-channel CNN observation.

    Pipeline: crop HUD → grayscale → INTER_AREA resize (aspect preserved).
    Returns an array with shape (obs_height, obs_width, 1) and dtype uint8.
    """
    # Ensure we are working with a contiguous RGB uint8 array.
    frame = np.asarray(rgb_frame, dtype=np.uint8)
    # Guard against unexpected channel layouts from the emulator.
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape {frame.shape}")
    # Keep only the first three color channels when an alpha channel is present.
    rgb = frame[:, :, :3]
    # Clamp the crop so a misconfigured value cannot empty the playfield.
    safe_crop = max(0, min(int(crop_top), rgb.shape[0] - 1))
    # Drop the top HUD rows so the CNN focuses on gameplay pixels.
    playfield = rgb[safe_crop:, :, :]
    # Convert the cropped playfield to a single grayscale channel.
    gray = cv2.cvtColor(playfield, cv2.COLOR_RGB2GRAY)
    # Downscale with area interpolation (best for shrinking game frames).
    resized = cv2.resize(
        gray,
        (int(obs_width), int(obs_height)),
        interpolation=cv2.INTER_AREA,
    )
    # Add a trailing channel axis so VecFrameStack can stack along channels.
    return resized[:, :, np.newaxis]


def resolve_frame_stack(algorithm, configured_stack=None):
    """Pick a frame-stack depth appropriate for the learning algorithm.

    RecurrentPPO already models temporal context with an LSTM, so a shallow
    stack (2) is enough for velocity cues. Feed-forward PPO / QRDQN need a
    deeper stack (4) in the classic Atari style.
    """
    # Honor an explicit override from the training config / UI.
    if configured_stack is not None:
        # Convert to int so UI string values still work.
        return int(configured_stack)
    # Recurrent policies only need a light stack for instantaneous motion.
    if algorithm == "RecurrentPPO":
        # Two frames give velocity without duplicating the LSTM memory.
        return 2
    # Feed-forward CNN policies rely on the stack for temporal context.
    return 4
