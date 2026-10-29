"""Compatibility shim — Mario memory helpers now live in game packs.

Prefer ``pdt.memory_bank.MemoryBank`` + ``games/<id>/memory.json`` for new code.
"""

from pdt.memory_bank import MemoryBank, digits_to_int
from pdt.memory_bank import tile_to_digit as _tile_to_digit


# Default Mario pack memory bank (lazy-loaded on first attribute use).
_BANK = None


def _bank() -> MemoryBank:
    """Load the Mario Deluxe memory.json once."""
    # Global cache so tests can patch addresses cheaply.
    global _BANK
    # Load on first use.
    if _BANK is None:
        # Import here to avoid circular imports at module load.
        from pdt.game_pack import load_game_pack

        # Load the migrated Mario pack memory map.
        _BANK = load_game_pack("mario_deluxe").load_memory()
    # Return the cached bank.
    return _BANK


class MarioMemoryMap:
    """Address constants mirrored from games/mario_deluxe/memory.json."""

    # Score digit tiles.
    SCORE_DIGIT_0 = 0xC103
    # Second score digit.
    SCORE_DIGIT_1 = 0xC104
    # Third score digit.
    SCORE_DIGIT_2 = 0xC105
    # Fourth score digit.
    SCORE_DIGIT_3 = 0xC106
    # Fifth score digit.
    SCORE_DIGIT_4 = 0xC107
    # Level progress tile.
    LEVEL_PROGRESS = 0xC24B
    # Motion marker.
    MARIO_MOTION = 0xC175
    # Remaining lives counter.
    LIVES_REMAINING = 0xC17F
    # Alias used by older code.
    LIFE_STATE = LIVES_REMAINING
    # Flag marker.
    FLAG_STATE = 0xC02E
    # Timer digits.
    TIMER_DIGIT_0 = 0x9811
    # Timer tens.
    TIMER_DIGIT_1 = 0x9812
    # Timer ones.
    TIMER_DIGIT_2 = 0x9813
    # Digit-0 tile id.
    TILE_DIGIT_BASE = 208
    # Max progress segment.
    MAX_PROGRESS_TILE = 12
    # Flag reached sentinel.
    FLAG_REACHED_VALUE = 80


def tile_to_digit(tile_value, tile_base=None, max_digit=None):
    """Map a HUD tile id onto a digit (defaults match Mario Deluxe)."""
    # Default base matches the Mario pack HUD font.
    base = MarioMemoryMap.TILE_DIGIT_BASE if tile_base is None else tile_base
    # Default max digit / progress segment.
    maximum = MarioMemoryMap.MAX_PROGRESS_TILE if max_digit is None else max_digit
    # Delegate to the shared decoder.
    return _tile_to_digit(tile_value, base, maximum)


def read_score_digits(memory):
    """Read five score digit tiles via the declarative SCORE addresses."""
    # Pull each digit using the legacy constants.
    digits = [
        tile_to_digit(
            memory[MarioMemoryMap.SCORE_DIGIT_0],
            MarioMemoryMap.TILE_DIGIT_BASE,
            MarioMemoryMap.MAX_PROGRESS_TILE,
        ),
        tile_to_digit(
            memory[MarioMemoryMap.SCORE_DIGIT_1],
            MarioMemoryMap.TILE_DIGIT_BASE,
            MarioMemoryMap.MAX_PROGRESS_TILE,
        ),
        tile_to_digit(
            memory[MarioMemoryMap.SCORE_DIGIT_2],
            MarioMemoryMap.TILE_DIGIT_BASE,
            MarioMemoryMap.MAX_PROGRESS_TILE,
        ),
        tile_to_digit(
            memory[MarioMemoryMap.SCORE_DIGIT_3],
            MarioMemoryMap.TILE_DIGIT_BASE,
            MarioMemoryMap.MAX_PROGRESS_TILE,
        ),
        tile_to_digit(
            memory[MarioMemoryMap.SCORE_DIGIT_4],
            MarioMemoryMap.TILE_DIGIT_BASE,
            MarioMemoryMap.MAX_PROGRESS_TILE,
        ),
    ]
    # Return digit list.
    return digits


def progress_percent_from_tile(tile_value):
    """Convert a progress tile into a 0-100 percentage."""
    # Decode segment index.
    segment = tile_to_digit(
        tile_value,
        MarioMemoryMap.TILE_DIGIT_BASE,
        MarioMemoryMap.MAX_PROGRESS_TILE,
    )
    # Normalize.
    return (segment / float(MarioMemoryMap.MAX_PROGRESS_TILE)) * 100.0


def read_lives(memory):
    """Return remaining lives from WRAM."""
    # Raw lives counter.
    return int(memory[MarioMemoryMap.LIVES_REMAINING])


def is_life_lost(memory, previous_lives=None):
    """Return True when lives dropped since the previous sample."""
    # Need a baseline.
    if previous_lives is None:
        # No history yet.
        return False
    # Compare counters.
    return read_lives(memory) < int(previous_lives)


def is_flag_reached(memory):
    """Return True when the flag sentinel is set."""
    # Compare against known completion value.
    return memory[MarioMemoryMap.FLAG_STATE] == MarioMemoryMap.FLAG_REACHED_VALUE


def is_game_timer_critical(memory):
    """Return True when the on-screen timer shows 001."""
    # Hundreds digit tile.
    hundreds = memory[MarioMemoryMap.TIMER_DIGIT_0]
    # Tens digit tile.
    tens = memory[MarioMemoryMap.TIMER_DIGIT_1]
    # Ones digit tile.
    ones = memory[MarioMemoryMap.TIMER_DIGIT_2]
    # Match historical critical pattern.
    return (
        hundreds == MarioMemoryMap.TILE_DIGIT_BASE
        and tens == MarioMemoryMap.TILE_DIGIT_BASE
        and ones == MarioMemoryMap.TILE_DIGIT_BASE + 1
    )
