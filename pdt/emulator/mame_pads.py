"""MAME ioport names for the arcade packs.

Each entry is (port tag, field name) for one logical button. Street Fighter
III uses the CPS3 names. Street Fighter Alpha 3 uses the CPS2 6-button map
from MAME 0.289 (``cps2_2p6b``): punches and kicks are Button 1..6.
"""

from __future__ import annotations

from typing import Dict, Tuple

# One held control: MAME port tag, then the field name on that port.
Field = Tuple[str, str]
# Player index -> button name -> field.
Pad = Dict[int, Dict[str, Field]]

# CPS3 3rd Strike. Kicks other than P2 roundhouse live on the EXTRA port.
_SFIII: Pad = {
    # Player 1 joystick and punches.
    1: {
        # Insert a credit.
        "coin": (":INPUTS", "Coin 1"),
        # Start a 1P game.
        "start": (":INPUTS", "P1 Start"),
        # Walk / jump directions.
        "up": (":INPUTS", "P1 Up"),
        "down": (":INPUTS", "P1 Down"),
        "left": (":INPUTS", "P1 Left"),
        "right": (":INPUTS", "P1 Right"),
        # Jab, strong, fierce.
        "lp": (":INPUTS", "P1 Jab Punch"),
        "mp": (":INPUTS", "P1 Strong Punch"),
        "hp": (":INPUTS", "P1 Fierce Punch"),
        # Short, forward, roundhouse.
        "lk": (":EXTRA", "P1 Short Kick"),
        "mk": (":EXTRA", "P1 Forward Kick"),
        "hk": (":EXTRA", "P1 Roundhouse Kick"),
    },
    # Player 2, same layout.
    2: {
        # Second coin slot.
        "coin": (":INPUTS", "Coin 2"),
        # Join as player 2.
        "start": (":INPUTS", "P2 Start"),
        # Directions.
        "up": (":INPUTS", "P2 Up"),
        "down": (":INPUTS", "P2 Down"),
        "left": (":INPUTS", "P2 Left"),
        "right": (":INPUTS", "P2 Right"),
        # Punches.
        "lp": (":INPUTS", "P2 Jab Punch"),
        "mp": (":INPUTS", "P2 Strong Punch"),
        "hp": (":INPUTS", "P2 Fierce Punch"),
        # Kicks. Roundhouse is on INPUTS in this driver.
        "lk": (":EXTRA", "P2 Short Kick"),
        "mk": (":EXTRA", "P2 Forward Kick"),
        "hk": (":INPUTS", "P2 Roundhouse Kick"),
    },
}

# CPS2 Alpha 3 / Zero 3. Directions and buttons 1-3 are on IN0.
# Buttons 4 and 5 are on IN1. P2 button 6 is on IN2 with the coins.
_SFA3: Pad = {
    # Player 1.
    1: {
        # Coin and start share IN2 with service.
        "coin": (":IN2", "Coin 1"),
        "start": (":IN2", "P1 Start"),
        # 8-way stick on IN0.
        "up": (":IN0", "P1 Up"),
        "down": (":IN0", "P1 Down"),
        "left": (":IN0", "P1 Left"),
        "right": (":IN0", "P1 Right"),
        # Light, medium, heavy punch.
        "lp": (":IN0", "P1 Button 1"),
        "mp": (":IN0", "P1 Button 2"),
        "hp": (":IN0", "P1 Button 3"),
        # Light, medium, heavy kick.
        "lk": (":IN1", "P1 Button 4"),
        "mk": (":IN1", "P1 Button 5"),
        "hk": (":IN1", "P1 Button 6"),
    },
    # Player 2.
    2: {
        # Second player coin and start.
        "coin": (":IN2", "Coin 2"),
        "start": (":IN2", "P2 Start"),
        # Stick on the high byte of IN0.
        "up": (":IN0", "P2 Up"),
        "down": (":IN0", "P2 Down"),
        "left": (":IN0", "P2 Left"),
        "right": (":IN0", "P2 Right"),
        # Punches.
        "lp": (":IN0", "P2 Button 1"),
        "mp": (":IN0", "P2 Button 2"),
        "hp": (":IN0", "P2 Button 3"),
        # Kicks. Heavy kick is the leftover bit on IN2.
        "lk": (":IN1", "P2 Button 4"),
        "mk": (":IN1", "P2 Button 5"),
        "hk": (":IN2", "P2 Button 6"),
    },
}

# Drivers that share the CPS2 6-button map above.
_SFA3_IDS = {
    "sfa3",
    "sfa3u",
    "sfa3ur1",
    "sfa3us",
    "sfa3h",
    "sfa3hr1",
    "sfa3b",
    "sfa3ud",
    "sfz3j",
    "sfz3jr1",
    "sfz3jr2",
    "sfz3a",
    "sfz3ar1",
}


def pad_for(game_id: str) -> Pad:
    """Return the ioport table for a MAME driver id."""
    # Normalise so "SFA3" and "sfa3" share a table.
    key = str(game_id or "").strip().lower()
    # Alpha 3 and its clones.
    if key in _SFA3_IDS:
        # CPS2 ports.
        return _SFA3
    # 3rd Strike and any unnamed test session.
    return _SFIII
