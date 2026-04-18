"""
Agricultural Field Terrain for Isaac Lab
=========================================
Generates a 24m × 38m farm terrain as a single sub-terrain.
Zones: off-road (rough), pavement (flat), crop (barrier), border (rough).

Usage:
    from .ag_terrain import AG_TERRAIN_CFG
    self.scene.terrain.terrain_type = "generator"
    self.scene.terrain.terrain_generator = AG_TERRAIN_CFG
"""

from __future__ import annotations
import numpy as np
import trimesh
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

# Map constants
BORDER_X, INNER_X = 2.0, 20.0
TOTAL_X = INNER_X + 2 * BORDER_X
BORDER_Y, CROP_Y, OFFROAD_Y, PAVEMENT_Y = 2.0, 2.0, 2.0, 2.0


def _compute_zones():
    zones, y = [], 0.0
    zones.append(("offroad", y, y + BORDER_Y)); y += BORDER_Y
    for i in range(5):
        zones.append(("offroad", y, y + OFFROAD_Y)); y += OFFROAD_Y
        zones.append(("crop", y, y + CROP_Y)); y += CROP_Y
        zones.append(("offroad", y, y + OFFROAD_Y)); y += OFFROAD_Y
        if i < 4:
            zones.append(("pavement", y, y + PAVEMENT_Y)); y += PAVEMENT_Y
    zones.append(("offroad", y, y + BORDER_Y)); y += BORDER_Y
    return zones, y

ZONES, TOTAL_Y = _compute_zones()


def _make_flat_box(x0, y0, x1, y1, z=0.0, thickness=0.5):
    dx, dy = x1 - x0, y1 - y0
    return trimesh.creation.box(
        extents=[dx, dy, thickness],
        transform=trimesh.transformations.translation_matrix(
            [x0 + dx/2, y0 + dy/2, z - thickness/2]
        ),
    )

def _make_rough_surface(x0, y0, x1, y1, amplitude, resolution, rng):
    nx = max(int((x1-x0)/resolution)+1, 3)
    ny = max(int((y1-y0)/resolution)+1, 3)
    xs, ys = np.linspace(x0,x1,nx), np.linspace(y0,y1,ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = rng.uniform(-amplitude, amplitude, size=(nx,ny))
    zz[0,:]=zz[-1,:]=zz[:,0]=zz[:,-1]=0.0
    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    faces = []
    for i in range(nx-1):
        for j in range(ny-1):
            v00, v10, v01, v11 = i*ny+j, (i+1)*ny+j, i*ny+(j+1), (i+1)*ny+(j+1)
            faces += [[v00,v10,v11],[v00,v11,v01]]
    surface = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    base = _make_flat_box(x0, y0, x1, y1, z=-amplitude, thickness=0.5)
    return [surface, base]

def _make_crop_barrier(x0, y0, x1, y1, height):
    dx, dy = x1-x0, y1-y0
    return trimesh.creation.box(
        extents=[dx, dy, height],
        transform=trimesh.transformations.translation_matrix(
            [x0+dx/2, y0+dy/2, height/2]
        ),
    )


def ag_field_terrain(difficulty, cfg):
    rng = np.random.default_rng(seed=cfg.seed)
    amp = cfg.offroad_noise_min + difficulty*(cfg.offroad_noise_max - cfg.offroad_noise_min)
    meshes = []
    x_off = (cfg.size[0] - TOTAL_X)/2
    y_off = (cfg.size[1] - TOTAL_Y)/2

    for ztype, ys, ye in ZONES:
        x0, x1 = x_off, x_off + TOTAL_X
        y0, y1 = y_off + ys, y_off + ye
        if ztype == "offroad":
            meshes.extend(_make_rough_surface(x0,y0,x1,y1, amp, cfg.noise_resolution, rng))
        elif ztype == "pavement":
            meshes.append(_make_flat_box(x0,y0,x1,y1, z=0.0))
        elif ztype == "crop":
            ix0, ix1 = x_off+BORDER_X, x_off+BORDER_X+INNER_X
            meshes.append(_make_crop_barrier(ix0,y0,ix1,y1, cfg.crop_barrier_height))
            meshes.extend(_make_rough_surface(x0,y0,ix0,y1, amp, cfg.noise_resolution, rng))
            meshes.extend(_make_rough_surface(ix1,y0,x1,y1, amp, cfg.noise_resolution, rng))

    meshes.append(_make_flat_box(x_off, y_off, x_off+TOTAL_X, y_off+TOTAL_Y, z=-0.1, thickness=0.5))
    origin = np.array([cfg.size[0]/2, cfg.size[1]/2, 0.0])
    return meshes, origin


@configclass
class AgFieldTerrainCfg(SubTerrainBaseCfg):
    function = ag_field_terrain
    offroad_noise_min: float = 0.05
    offroad_noise_max: float = 0.10
    noise_resolution: float = 0.3
    crop_barrier_height: float = 0.25
    seed: int | None = 42


AG_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(TOTAL_X + 4.0, TOTAL_Y + 4.0),
    border_width=2.0,
    border_height=-1.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    difficulty_range=(0.5, 0.5),
    color_scheme="none",
    sub_terrains={
        "ag_field": AgFieldTerrainCfg(
            proportion=1.0,
            offroad_noise_min=0.05,
            offroad_noise_max=0.18,
            noise_resolution=0.2,    # ← 加这行
            crop_barrier_height=0.25,
            seed=42,
        ),
    },
)