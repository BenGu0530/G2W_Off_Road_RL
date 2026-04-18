# Go2-W Stage A config — minimal override of base
# Now includes farm terrain and safe spawn for demo compatibility

from __future__ import annotations
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)

# ─── Map constants ────────────────────────────────────────────────────────────
BORDER_X   = 2.0
INNER_X    = 20.0
TOTAL_X    = INNER_X + 2 * BORDER_X
BORDER_Y   = 2.0
CROP_Y     = 2.0
OFFROAD_Y  = 2.0
PAVEMENT_Y = 2.0

CROP_Y_CENTERS = []
PAVEMENT_RANGES = []
CROP_RANGES = []

_y = BORDER_Y
for _i in range(5):
    _y += OFFROAD_Y
    CROP_RANGES.append((_y, _y + CROP_Y))
    CROP_Y_CENTERS.append(_y + CROP_Y / 2.0)
    _y += CROP_Y + OFFROAD_Y
    if _i < 4:
        PAVEMENT_RANGES.append((_y, _y + PAVEMENT_Y))
        _y += PAVEMENT_Y
_y += BORDER_Y
TOTAL_Y = _y

# Safe spawn: first pavement corridor
_ps, _pe = PAVEMENT_RANGES[0]
SAFE_SPAWN_Y_CENTER = (_ps + _pe) / 2.0 - TOTAL_Y / 2.0
SAFE_SPAWN_Y_HALF = (_pe - _ps) / 2.0 - 0.3


# ─── Env Config ───────────────────────────────────────────────────────────────
@configclass
class Go2WAgEnvCfg(UnitreeGo2WRoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # Fix broken zero-weight terms
        self.rewards.feet_air_time_variance = None
        self.rewards.feet_distance_y_exp = None

        self.scene.num_envs = 4096

        # Use custom farm terrain
        from .ag_terrain import AG_TERRAIN_CFG
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = AG_TERRAIN_CFG
        self.curriculum.terrain_levels = None

        # Spawn in first pavement corridor (safe zone, not in crops)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (SAFE_SPAWN_Y_CENTER - SAFE_SPAWN_Y_HALF,
                      SAFE_SPAWN_Y_CENTER + SAFE_SPAWN_Y_HALF),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }