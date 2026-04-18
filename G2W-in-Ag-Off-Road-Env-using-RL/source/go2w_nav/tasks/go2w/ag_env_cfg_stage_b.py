# Go2-W Stage B: Mode Switching on Agricultural Field
#
# CONSERVATIVE TUNING for overnight run / demo:
#   - mode_switch weight 0.3 (gentle nudge, won't destroy walking)
#   - crop_penalty weight 0.5 (moderate — physical barriers do most work)
#   - boundary_penalty weight 0.1 (light touch)
#   - Everything else inherited from base

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)

from .ag_terrain import AG_TERRAIN_CFG, TOTAL_X, TOTAL_Y
from .ag_env_cfg import (
    CROP_Y_CENTERS, PAVEMENT_RANGES, CROP_RANGES,
    INNER_X, BORDER_X, BORDER_Y, CROP_Y, OFFROAD_Y, PAVEMENT_Y,
)


# =============================================================================
# Safe spawn Y range — first pavement corridor
# =============================================================================
def _compute_safe_y_range():
    if PAVEMENT_RANGES:
        pave_start, pave_end = PAVEMENT_RANGES[0]
        y_center = (pave_start + pave_end) / 2.0 - TOTAL_Y / 2.0
        y_half = (pave_end - pave_start) / 2.0 - 0.2
        return (y_center - y_half, y_center + y_half)
    return (-1.0, 1.0)

SAFE_Y_MIN, SAFE_Y_MAX = _compute_safe_y_range()


# =============================================================================
# Reward functions — CONSERVATIVE versions
# =============================================================================
def get_zone(y_pos: torch.Tensor, device: str):
    y = y_pos + TOTAL_Y / 2.0
    is_crop = torch.zeros_like(y, dtype=torch.bool)
    for y_start, y_end in CROP_RANGES:
        is_crop |= (y >= y_start) & (y < y_end)
    is_pave = torch.zeros_like(y, dtype=torch.bool)
    for y_start, y_end in PAVEMENT_RANGES:
        is_pave |= (y >= y_start) & (y < y_end)
    return is_crop, is_pave, ~is_crop & ~is_pave


def mode_switch_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CONSERVATIVE mode switch — reward-only, no penalty for walking on pavement.

    On off-road: small bonus for using legs
    On pavement: small bonus for using wheels (but NO penalty for walking)
    This way the robot never gets punished for doing what Stage A taught it.
    """
    asset = env.scene[asset_cfg.name]
    y_pos = asset.data.root_pos_w[:, 1]
    is_crop, is_pave, is_offroad = get_zone(y_pos, env.device)

    leg_vel = asset.data.joint_vel[:, :12].abs().mean(dim=1)
    wheel_vel = asset.data.joint_vel[:, -4:].abs().mean(dim=1)

    gravity_z = asset.data.projected_gravity_b[:, 2]
    is_upright = (gravity_z < -0.7).float()

    # Off-road: bonus for legs (no wheel penalty)
    offroad_bonus = is_offroad.float() * is_upright * leg_vel * 0.3

    # Pavement: bonus for wheels (NO leg penalty — this is the key fix)
    pave_bonus = is_pave.float() * is_upright * wheel_vel * 0.3

    return offroad_bonus + pave_bonus


def crop_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    y_pos = asset.data.root_pos_w[:, 1]
    is_crop, _, _ = get_zone(y_pos, env.device)
    return -30.0 * is_crop.float()


def boundary_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    x_pos = asset.data.root_pos_w[:, 0]
    y_pos = asset.data.root_pos_w[:, 1]
    y_map = y_pos + TOTAL_Y / 2.0
    out_x = (x_pos.abs() > TOTAL_X / 2.0).float()
    out_y = ((y_map < 0) | (y_map > TOTAL_Y)).float()
    return -10.0 * (out_x + out_y)


# =============================================================================
# Stage B Env Config
# =============================================================================
@configclass
class Go2WAgEnvCfg(UnitreeGo2WRoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # Fix broken zero-weight terms
        self.rewards.feet_air_time_variance = None
        self.rewards.feet_distance_y_exp = None

        self.scene.num_envs = 4096

        # ── Terrain: custom farm ─────────────────────────────────────────
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = AG_TERRAIN_CFG

        # ── Disable terrain curriculum ───────────────────────────────────
        self.curriculum.terrain_levels = None

        # ── Safe spawn ───────────────────────────────────────────────────
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-3.0, 3.0),
                "y": (SAFE_Y_MIN, SAFE_Y_MAX),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ── Stage B rewards: CONSERVATIVE ────────────────────────────────

        # Mode switch: gentle reward-only, weight 0.3
        self.rewards.mode_switch = RewTerm(
            func=mode_switch_reward,
            weight=0.3,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Crop penalty: moderate (physical barriers do the heavy lifting)
        self.rewards.crop_penalty = RewTerm(
            func=crop_penalty,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Boundary penalty: light
        self.rewards.boundary_penalty = RewTerm(
            func=boundary_penalty,
            weight=0.1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )