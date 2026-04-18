"""
Agricultural Field Terrain for Isaac Lab
=========================================

Generates a 24m × 38m farm terrain as a single sub-terrain that plugs into
Isaac Lab's TerrainGeneratorCfg. Each zone has distinct physical properties:

  - Off-road (brown):  Random height noise 2-5cm — forces leg stepping
  - Pavement (grey):   Perfectly flat — wheels work here
  - Crop rows (red):   Tall bumps 20-30cm — physical barriers
  - Border (brown):    Same roughness as off-road

The terrain is generated as trimesh objects, compatible with the standard
Isaac Lab terrain pipeline. No USD hacks needed.

Usage in env cfg:
    from ag_terrain import AG_TERRAIN_CFG
    self.scene.terrain.terrain_type = "generator"
    self.scene.terrain.terrain_generator = AG_TERRAIN_CFG
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

# =============================================================================
# Map constants — must match ag_env_cfg.py
# =============================================================================
BORDER_X   = 2.0
INNER_X    = 20.0
TOTAL_X    = INNER_X + 2 * BORDER_X   # 24 m
BORDER_Y   = 2.0
CROP_Y     = 2.0
OFFROAD_Y  = 2.0
PAVEMENT_Y = 2.0

# Terrain physics parameters
OFFROAD_NOISE_AMP  = 0.03    # 3cm random bumps for off-road
OFFROAD_NOISE_RES  = 0.2     # noise resolution (m per sample)
CROP_BARRIER_H     = 0.25    # 25cm tall crop barriers
MESH_RESOLUTION    = 0.1     # triangulation resolution (m)


# =============================================================================
# Zone layout computation
# =============================================================================
def _compute_zones():
    """Compute the (name, y_start, y_end) for all zones.

    Returns list of (zone_type, y_start, y_end) tuples.
    Zone coordinates are in LOCAL sub-terrain frame: y goes from 0 to TOTAL_Y.
    """
    zones = []
    y = 0.0

    # Bottom border
    zones.append(("offroad", y, y + BORDER_Y))
    y += BORDER_Y

    for i in range(5):
        # Off-road strip above crop
        zones.append(("offroad", y, y + OFFROAD_Y))
        y += OFFROAD_Y

        # Crop row
        zones.append(("crop", y, y + CROP_Y))
        y += CROP_Y

        # Off-road strip below crop
        zones.append(("offroad", y, y + OFFROAD_Y))
        y += OFFROAD_Y

        # Pavement between crop groups (4 corridors)
        if i < 4:
            zones.append(("pavement", y, y + PAVEMENT_Y))
            y += PAVEMENT_Y

    # Top border
    zones.append(("offroad", y, y + BORDER_Y))
    y += BORDER_Y

    total_y = y
    return zones, total_y


ZONES, TOTAL_Y = _compute_zones()


# =============================================================================
# Trimesh generation helpers
# =============================================================================
def _make_flat_box(x0, y0, x1, y1, z=0.0, thickness=0.5):
    """Create a flat box (extruded rectangle) as a trimesh.

    The top surface sits at height z, box extends down by thickness.
    """
    dx = x1 - x0
    dy = y1 - y0
    box = trimesh.creation.box(
        extents=[dx, dy, thickness],
        transform=trimesh.transformations.translation_matrix(
            [x0 + dx / 2.0, y0 + dy / 2.0, z - thickness / 2.0]
        ),
    )
    return box


def _make_rough_surface(x0, y0, x1, y1, amplitude, resolution, rng):
    """Create a rough surface as a heightfield mesh.

    Generates a grid of vertices with random z-noise, then triangulates.
    The mesh sits on top of a solid base box so there are no gaps.
    """
    dx = x1 - x0
    dy = y1 - y0

    # Number of grid points
    nx = max(int(dx / resolution) + 1, 3)
    ny = max(int(dy / resolution) + 1, 3)

    # Create grid
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    # Random height noise
    zz = rng.uniform(-amplitude, amplitude, size=(nx, ny))

    # Zero out the edges so they connect cleanly to neighbors
    zz[0, :] = 0.0
    zz[-1, :] = 0.0
    zz[:, 0] = 0.0
    zz[:, -1] = 0.0

    # Build vertices (nx*ny, 3)
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Build triangles from grid (two triangles per cell)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces)
    surface = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Add a solid base box underneath so physics doesn't fall through
    base = _make_flat_box(x0, y0, x1, y1, z=-amplitude, thickness=0.5)

    return [surface, base]


def _make_crop_barrier(x0, y0, x1, y1, height):
    """Create a crop row as a tall box — physical barrier the robot can't cross."""
    dx = x1 - x0
    dy = y1 - y0
    box = trimesh.creation.box(
        extents=[dx, dy, height],
        transform=trimesh.transformations.translation_matrix(
            [x0 + dx / 2.0, y0 + dy / 2.0, height / 2.0]
        ),
    )
    return box


# =============================================================================
# Main terrain generation function
# =============================================================================
def ag_field_terrain(
    difficulty: float, cfg: "AgFieldTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the agricultural field terrain.

    This function follows the Isaac Lab SubTerrainBaseCfg.function signature:
        (difficulty, cfg) -> (list[trimesh.Trimesh], origin)

    The difficulty parameter controls off-road roughness amplitude:
        difficulty=0 → minimal bumps (easy walking)
        difficulty=1 → maximum bumps (hard walking)

    Args:
        difficulty: Terrain difficulty, 0 to 1.
        cfg: AgFieldTerrainCfg configuration.

    Returns:
        Tuple of (list of trimesh meshes, terrain origin as np.ndarray(3,)).
    """
    rng = np.random.default_rng(seed=cfg.seed)

    # Interpolate roughness amplitude based on difficulty
    amp = cfg.offroad_noise_min + difficulty * (
        cfg.offroad_noise_max - cfg.offroad_noise_min
    )

    meshes: list[trimesh.Trimesh] = []

    # The sub-terrain coordinate system: (0,0) to (size[0], size[1])
    # We center the field at (size[0]/2, size[1]/2) in the sub-terrain frame
    # x_offset and y_offset shift our local field coords into sub-terrain coords
    x_offset = (cfg.size[0] - TOTAL_X) / 2.0  # center horizontally
    y_offset = (cfg.size[1] - TOTAL_Y) / 2.0  # center vertically

    for zone_type, y_start, y_end in ZONES:
        # Field coordinates → sub-terrain coordinates
        # x: field goes from 0 to TOTAL_X
        x0_local = x_offset
        x1_local = x_offset + TOTAL_X
        y0_local = y_offset + y_start
        y1_local = y_offset + y_end

        if zone_type == "offroad":
            # Rough surface with random bumps
            rough_meshes = _make_rough_surface(
                x0_local, y0_local, x1_local, y1_local,
                amplitude=amp,
                resolution=cfg.noise_resolution,
                rng=rng,
            )
            meshes.extend(rough_meshes)

        elif zone_type == "pavement":
            # Flat surface — wheels drive here
            flat = _make_flat_box(x0_local, y0_local, x1_local, y1_local, z=0.0)
            meshes.append(flat)

        elif zone_type == "crop":
            # Crop barriers with physical height
            # Inner crop zone (between borders)
            inner_x0 = x_offset + BORDER_X
            inner_x1 = x_offset + BORDER_X + INNER_X
            barrier = _make_crop_barrier(
                inner_x0, y0_local, inner_x1, y1_local,
                height=cfg.crop_barrier_height,
            )
            meshes.append(barrier)

            # The side borders in crop rows are still off-road
            left_rough = _make_rough_surface(
                x0_local, y0_local, inner_x0, y1_local,
                amplitude=amp, resolution=cfg.noise_resolution, rng=rng,
            )
            right_rough = _make_rough_surface(
                inner_x1, y0_local, x1_local, y1_local,
                amplitude=amp, resolution=cfg.noise_resolution, rng=rng,
            )
            meshes.extend(left_rough)
            meshes.extend(right_rough)

    # Add a base plane underneath everything to catch any gaps
    base_plane = _make_flat_box(
        x_offset, y_offset,
        x_offset + TOTAL_X, y_offset + TOTAL_Y,
        z=-0.1, thickness=0.5,
    )
    meshes.append(base_plane)

    # Color the meshes by zone type for visual debugging
    for mesh in meshes:
        if not hasattr(mesh.visual, "vertex_colors") or mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.full(
                (len(mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8
            )

    # Assign colors based on height heuristic + explicit coloring
    _color_by_zone(meshes, x_offset, y_offset)

    # Origin: where the robot spawns (center of the field)
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])

    return meshes, origin


def _color_by_zone(meshes, x_offset, y_offset):
    """Color mesh vertices based on which zone they belong to.

    Colors:
        Offroad = brown (140, 90, 40)
        Pavement = grey (140, 140, 140)
        Crop = red (200, 50, 50)
    """
    brown = np.array([140, 90, 40, 255], dtype=np.uint8)
    grey  = np.array([140, 140, 140, 255], dtype=np.uint8)
    red   = np.array([200, 50, 50, 255], dtype=np.uint8)

    for mesh in meshes:
        verts = mesh.vertices
        # Determine zone by average y position of mesh
        avg_y = np.mean(verts[:, 1]) - y_offset
        avg_z = np.mean(verts[:, 2])

        # If vertices are mostly above ground → crop barrier
        if avg_z > 0.05:
            color = red
        # If very flat (low z variance) → pavement
        elif np.std(verts[:, 2]) < 0.005 and avg_z > -0.2:
            color = grey
        # Otherwise → off-road or base
        else:
            color = brown

        mesh.visual.vertex_colors = np.tile(color, (len(verts), 1))


# =============================================================================
# Configuration class — plugs into TerrainGeneratorCfg
# =============================================================================
@configclass
class AgFieldTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the agricultural field terrain."""

    function = ag_field_terrain

    # Off-road roughness range (interpolated by difficulty)
    offroad_noise_min: float = 0.01
    """Minimum noise amplitude in meters (easy difficulty). Default 1cm."""

    offroad_noise_max: float = 0.05
    """Maximum noise amplitude in meters (hard difficulty). Default 5cm."""

    noise_resolution: float = 0.2
    """Spatial resolution of roughness noise in meters. Default 0.2m."""

    crop_barrier_height: float = 0.25
    """Height of crop row barriers in meters. Default 25cm."""

    seed: int | None = 42
    """Random seed for reproducible terrain generation."""


# =============================================================================
# Pre-built terrain generator config — drop-in for env cfg
# =============================================================================
AG_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    # Sub-terrain size must be >= (TOTAL_X, TOTAL_Y)
    # We add 2m padding on each side for the TerrainGenerator border
    size=(TOTAL_X + 4.0, TOTAL_Y + 4.0),  # (28m, 42m)
    border_width=2.0,
    border_height=-1.0,
    # Single row/col — one copy of the farm per terrain tile
    num_rows=1,
    num_cols=1,
    # No curriculum for terrain difficulty — fixed at 0.5
    curriculum=False,
    difficulty_range=(0.5, 0.5),
    # Color by height for visual debugging
    color_scheme="none",  # we handle coloring ourselves
    # The farm terrain
    sub_terrains={
        "ag_field": AgFieldTerrainCfg(
            proportion=1.0,
            offroad_noise_min=0.01,
            offroad_noise_max=0.05,
            crop_barrier_height=0.25,
            seed=42,
        ),
    },
)


# =============================================================================
# Convenience: visual-only terrain overlay (your existing approach)
# =============================================================================
def build_ag_terrain_visual():
    """Build visual-only colored cubes for the terrain (USD overlay).

    This adds colored, non-physics cubes to the stage for visual reference.
    The actual physics terrain is handled by AG_TERRAIN_CFG above.

    Call this from ag_env.py after the sim starts if you want the
    color overlay on top of the physics mesh.
    """
    try:
        import omni.usd
        from pxr import UsdGeom, Gf, UsdShade, Sdf
    except ImportError:
        print("[AgTerrain] USD libraries not available, skipping visual overlay")
        return

    # Colors
    RED   = (0.7, 0.1, 0.1)
    BROWN = (0.5, 0.3, 0.1)
    GREY  = (0.5, 0.5, 0.5)

    stage = omni.usd.get_context().get_stage()
    root_path = "/World/AgTerrainVisual"
    if not stage.GetPrimAtPath(root_path):
        stage.DefinePrim(root_path, "Xform")
        stage.DefinePrim(f"{root_path}/Materials", "Scope")

    color_map = {"offroad": BROWN, "pavement": GREY, "crop": RED}
    thickness = 0.005  # Very thin — visual only, sits on top of physics mesh

    for idx, (zone_type, y_start, y_end) in enumerate(ZONES):
        color = color_map.get(zone_type, BROWN)
        x_start_local = 0.0
        x_end_local = TOTAL_X
        size_x = x_end_local - x_start_local
        size_y = y_end - y_start
        center_x = size_x / 2.0 - TOTAL_X / 2.0  # center around origin
        center_y = (y_start + y_end) / 2.0 - TOTAL_Y / 2.0
        center_z = 0.001  # Just above ground

        prim_path = f"{root_path}/{zone_type}_{idx}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, center_z))
        cube.AddScaleOp().Set(Gf.Vec3d(size_x / 2.0, size_y / 2.0, thickness / 2.0))

        # Material
        mat_name = f"{zone_type}_visual"
        mat_path = f"{root_path}/Materials/{mat_name}"
        if not stage.GetPrimAtPath(mat_path):
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path))
        )

    print(f"[AgTerrain] Built visual overlay: {len(ZONES)} zones, "
          f"{TOTAL_X}m x {TOTAL_Y}m")