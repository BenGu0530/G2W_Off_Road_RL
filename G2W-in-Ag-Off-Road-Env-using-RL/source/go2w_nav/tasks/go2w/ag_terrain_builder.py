import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf

# Colors
STRAWBERRY_RED = (0.85, 0.15, 0.18)
GREEN_LEAF     = (0.2, 0.55, 0.15)

# Dimensions
BORDER_X   = 2.0
INNER_X    = 20.0
TOTAL_X    = INNER_X + 2 * BORDER_X

BORDER_Y   = 2.0
CROP_Y     = 2.0
OFFROAD_Y  = 2.0
PAVEMENT_Y = 2.0

# Only crop zones — we skip offroad/pavement so the physics terrain shows through
CROP_ZONES = []

y = 0.0
y += BORDER_Y  # bottom border

for i in range(5):
    y += OFFROAD_Y  # offroad above crop
    # Crop row (center only, between borders)
    CROP_ZONES.append((y, y + CROP_Y))
    y += CROP_Y
    y += OFFROAD_Y  # offroad below crop
    if i < 4:
        y += PAVEMENT_Y  # pavement

y += BORDER_Y  # top border
TOTAL_Y = y


def create_material(stage, mat_path, color):
    if stage.GetPrimAtPath(mat_path):
        return mat_path
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat_path


def build_ag_terrain():
    """Build ONLY crop row visual overlays.

    Off-road and pavement zones are NOT rendered as overlays —
    the physics terrain mesh shows through with its natural bumps.
    Crop rows are rendered as visible red blocks above the ground.
    """
    stage = omni.usd.get_context().get_stage()

    root_path = "/World/AgTerrain"
    if not stage.GetPrimAtPath(root_path):
        stage.DefinePrim(root_path, "Xform")
    stage.DefinePrim(f"{root_path}/Materials", "Scope")

    # Create materials
    crop_mat = create_material(stage, f"{root_path}/Materials/crop_mat", STRAWBERRY_RED)

    crop_height = 0.3  # 30cm tall red blocks for crops

    for idx, (y_start, y_end) in enumerate(CROP_ZONES):
        # Crop block — center portion only (between side borders)
        prim_path = f"{root_path}/crop_{idx}"
        size_x = INNER_X
        size_y = y_end - y_start
        center_x = 0.0  # centered
        center_y = (y_start + y_end) / 2.0 - TOTAL_Y / 2.0
        center_z = crop_height / 2.0  # sits above ground

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, center_z))
        cube.AddScaleOp().Set(Gf.Vec3d(size_x / 2.0, size_y / 2.0, crop_height / 2.0))

        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(crop_mat))
        )

    print(f"[AgTerrain] Built {len(CROP_ZONES)} crop rows (strawberry red), "
          f"map size {TOTAL_X}m x {TOTAL_Y}m")