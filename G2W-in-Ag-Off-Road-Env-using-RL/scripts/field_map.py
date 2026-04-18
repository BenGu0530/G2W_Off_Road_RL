"""
Agricultural Field Map Definition
===================================
Pure data — no Isaac Lab dependency. Defines zone layout, safe corridors,
and connectivity for path planning.

Field layout (Y axis, world coords centered at 0):
  offroad -18 | CROP -16 | offroad -14 | pavement -12 | offroad -10 |
  CROP -8 | offroad -6 | pavement -4 | offroad -2 |
  CROP 0 | offroad 2 | pavement 4 | offroad 6 |
  CROP 8 | offroad 10 | pavement 12 | offroad 14 |
  CROP 16 | offroad 18
"""

# Dimensions
BORDER_X = 2.0
INNER_X = 20.0
TOTAL_X = INNER_X + 2 * BORDER_X  # 24m
BORDER_Y = 2.0
CROP_Y = 2.0
OFFROAD_Y = 2.0
PAVEMENT_Y = 2.0

# Compute total Y
_y = BORDER_Y
for _i in range(5):
    _y += OFFROAD_Y + CROP_Y + OFFROAD_Y
    if _i < 4:
        _y += PAVEMENT_Y
_y += BORDER_Y
TOTAL_Y = _y  # 42.0m

# Zone ranges in map coords (0 to TOTAL_Y)
CROP_RANGES_MAP = []
PAVEMENT_RANGES_MAP = []
OFFROAD_RANGES_MAP = []

_y = BORDER_Y
OFFROAD_RANGES_MAP.append((0, BORDER_Y))  # bottom border
for _i in range(5):
    OFFROAD_RANGES_MAP.append((_y, _y + OFFROAD_Y))
    _y += OFFROAD_Y
    CROP_RANGES_MAP.append((_y, _y + CROP_Y))
    _y += CROP_Y
    OFFROAD_RANGES_MAP.append((_y, _y + OFFROAD_Y))
    _y += OFFROAD_Y
    if _i < 4:
        PAVEMENT_RANGES_MAP.append((_y, _y + PAVEMENT_Y))
        _y += PAVEMENT_Y
OFFROAD_RANGES_MAP.append((_y, _y + BORDER_Y))  # top border

# Convert to world coords (centered at 0)
CROP_RANGES = [(s - TOTAL_Y/2, e - TOTAL_Y/2) for s, e in CROP_RANGES_MAP]
PAVEMENT_RANGES = [(s - TOTAL_Y/2, e - TOTAL_Y/2) for s, e in PAVEMENT_RANGES_MAP]
OFFROAD_RANGES = [(s - TOTAL_Y/2, e - TOTAL_Y/2) for s, e in OFFROAD_RANGES_MAP]

# Zone centers in world coords
CROP_CENTERS = [(s + e) / 2.0 for s, e in CROP_RANGES]
PAVEMENT_CENTERS = [(s + e) / 2.0 for s, e in PAVEMENT_RANGES]
OFFROAD_CENTERS = [(s + e) / 2.0 for s, e in OFFROAD_RANGES]

# Safe zones: offroad + pavement (robot can walk here)
SAFE_CENTERS = sorted(OFFROAD_CENTERS + PAVEMENT_CENTERS)

# Connected groups: safe zones separated by crop rows
# Within a group, robot can walk freely. Between groups, must go around crop.
GROUPS = [
    [-18.0],                    # border bottom
    [-14.0, -12.0, -10.0],     # between crop 1 and crop 2
    [-6.0, -4.0, -2.0],        # between crop 2 and crop 3
    [2.0, 4.0, 6.0],           # between crop 3 and crop 4
    [10.0, 12.0, 14.0],        # between crop 4 and crop 5
    [18.0],                     # border top
]

# Field X boundaries
X_MIN = -TOTAL_X / 2.0   # -12.0
X_MAX = TOTAL_X / 2.0    # +12.0
INNER_X_MIN = -INNER_X / 2.0  # -10.0 (crop zone starts here)
INNER_X_MAX = INNER_X / 2.0   # +10.0 (crop zone ends here)

# Spawn location: first pavement corridor
SPAWN_Y = PAVEMENT_CENTERS[0] if PAVEMENT_CENTERS else 0.0


def get_zone_type(y_world: float) -> str:
    """Return zone type at given world Y position."""
    for s, e in CROP_RANGES:
        if s <= y_world < e:
            return "crop"
    for s, e in PAVEMENT_RANGES:
        if s <= y_world < e:
            return "pavement"
    return "offroad"


def find_group(y_world: float) -> int:
    """Find which connectivity group a Y position belongs to."""
    for i, g in enumerate(GROUPS):
        if min(g) - 1.5 <= y_world <= max(g) + 1.5:
            return i
    # Fallback: nearest group
    return min(range(len(GROUPS)),
               key=lambda i: min(abs(y_world - v) for v in GROUPS[i]))


def is_in_crop(y_world: float) -> bool:
    """Check if Y position is inside a crop zone."""
    return get_zone_type(y_world) == "crop"


def nearest_safe_y(y_world: float) -> float:
    """Find nearest safe (non-crop) Y center."""
    return min(SAFE_CENTERS, key=lambda c: abs(c - y_world))