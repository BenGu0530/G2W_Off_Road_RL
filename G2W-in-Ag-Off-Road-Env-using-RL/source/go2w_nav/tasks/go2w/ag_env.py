import torch
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf
from isaaclab.envs import ManagerBasedRLEnv
from .ag_terrain_builder import build_ag_terrain
from .ag_env_cfg import CROP_Y_CENTERS, TOTAL_Y, INNER_X, PAVEMENT_RANGES


class Go2WAgEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        build_ag_terrain()

        self.goal_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._randomize_goals(torch.arange(self.num_envs, device=self.device))
        self._spawn_goal_sphere()

    def _randomize_goals(self, env_ids: torch.Tensor):
        n = len(env_ids)
        corridor_ids = torch.randint(0, len(PAVEMENT_RANGES), (n,), device=self.device)
        x_goals = (torch.rand(n, device=self.device) - 0.5) * INNER_X
        pave_centers = [(s + e) / 2.0 for s, e in PAVEMENT_RANGES]
        y_goals = torch.tensor(
            [pave_centers[c] for c in corridor_ids.cpu().tolist()],
            device=self.device
        ) - TOTAL_Y / 2.0
        self.goal_pos[env_ids, 0] = x_goals
        self.goal_pos[env_ids, 1] = y_goals

    def _spawn_goal_sphere(self):
        stage = omni.usd.get_context().get_stage()
        sphere_path = "/World/GoalSphere"
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(1.2)
        sphere.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
        mat = UsdShade.Material.Define(stage, "/World/GoalSphereMat")
        shader = UsdShade.Shader.Define(stage, "/World/GoalSphereMat/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(sphere_path)).Bind(mat)

    def _update_goal_sphere(self, visible=True):
        usd_stage = omni.usd.get_context().get_stage()
        prim = usd_stage.GetPrimAtPath("/World/GoalSphere")
        if prim:
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    if visible:
                        op.Set(Gf.Vec3d(
                            float(self.goal_pos[0, 0].item()),
                            float(self.goal_pos[0, 1].item()),
                            0.3
                        ))
                    else:
                        op.Set(Gf.Vec3d(0.0, 0.0, -10.0))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, 'goal_pos'):
            self._randomize_goals(env_ids)