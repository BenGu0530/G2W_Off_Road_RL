"""
Go2-W Agricultural Field Navigation Demo
=========================================
Usage:
    python demo_nav.py --checkpoint <path_to_stage_a.pt>
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2-W Farm Navigation Demo")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
import numpy as np
import gymnasium as gym
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import go2w_nav.tasks  # noqa: F401

from go2w_nav.tasks.go2w.ag_env_cfg_stage_b import Go2WAgEnvCfg
from go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg import Go2WAgPPORunnerCfg
from go2w_nav.tasks.go2w.ag_env_cfg import PAVEMENT_RANGES, TOTAL_Y, INNER_X


# ── Waypoints ─────────────────────────────────────────────────────────────────
def generate_waypoints(n, device):
    wps = []
    pave_centers = [(s + e) / 2.0 - TOTAL_Y / 2.0 for s, e in PAVEMENT_RANGES]
    for i in range(n):
        ci = i % len(pave_centers)
        y = pave_centers[ci]
        x = np.random.uniform(2.0, INNER_X/2 - 1) if i % 2 == 0 else np.random.uniform(-INNER_X/2 + 1, -2.0)
        wps.append([x, y])
    return torch.tensor(wps, dtype=torch.float32, device=device)


# ── Goal Sphere ───────────────────────────────────────────────────────────────
def create_goal_sphere():
    stage = omni.usd.get_context().get_stage()
    path = "/World/GoalSphere"
    if stage.GetPrimAtPath(path):
        return
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(0.3)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, -10))
    mat = UsdShade.Material.Define(stage, "/World/GoalMat")
    sh = UsdShade.Shader.Define(stage, "/World/GoalMat/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.05, 0.05))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(mat)


def move_goal_sphere(x, y):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/GoalSphere")
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(x), float(y), 0.5))


# ── Heading ───────────────────────────────────────────────────────────────────
def quat_to_yaw(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    env_cfg = Go2WAgEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Go2W-Ag-StageB-v0", cfg=env_cfg)

    # Fix gymnasium Box dtype issue before wrapping
    # The action space high/low might be bool instead of float
    if hasattr(env, 'single_action_space'):
        space = env.single_action_space
        if space is not None and hasattr(space, 'high'):
            env.single_action_space = gym.spaces.Box(
                low=np.float32(space.low),
                high=np.float32(space.high),
                shape=space.shape,
                dtype=np.float32,
            )
    if hasattr(env.unwrapped, 'single_action_space'):
        space = env.unwrapped.single_action_space
        if space is not None and hasattr(space, 'high'):
            env.unwrapped.single_action_space = gym.spaces.Box(
                low=np.float32(space.low),
                high=np.float32(space.high),
                shape=space.shape,
                dtype=np.float32,
            )

    # Wrap for RSL-RL
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=True)

    # Load policy via OnPolicyRunner (same as play.py)
    agent_cfg = Go2WAgPPORunnerCfg()
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    print(f"[DEMO] Policy loaded from: {args_cli.checkpoint}")

    # Setup
    create_goal_sphere()
    waypoints = generate_waypoints(5, device)
    wp_idx = 0

    kp_lin, kp_ang = 1.5, 2.0
    max_lin, max_ang = 1.0, 1.0
    arrive_dist = 1.5

    print(f"\n{'='*60}")
    print(f"  Go2-W Farm Navigation Demo")
    print(f"{'='*60}")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i}: ({wp[0]:.1f}, {wp[1]:.1f})")
    print()

    # Reset
    obs, _ = env_wrapped.get_observations()
    step = 0

    while simulation_app.is_running() and step < 10000:
        if wp_idx >= len(waypoints):
            print("\n[DEMO] All waypoints reached! New route...")
            waypoints = generate_waypoints(5, device)
            wp_idx = 0
            for i, wp in enumerate(waypoints):
                print(f"  Waypoint {i}: ({wp[0]:.1f}, {wp[1]:.1f})")

        goal = waypoints[wp_idx]
        move_goal_sphere(goal[0].item(), goal[1].item())

        # Robot state
        robot = env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w
        quat = robot.data.root_quat_w

        # Navigation
        dx = goal[0] - pos[:, 0]
        dy = goal[1] - pos[:, 1]
        dist = torch.sqrt(dx**2 + dy**2)
        heading = quat_to_yaw(quat)
        desired = torch.atan2(dy, dx)
        err = torch.atan2(torch.sin(desired - heading), torch.cos(desired - heading))

        vx = (kp_lin * dist).clamp(-max_lin, max_lin) * torch.cos(err).clamp(0.0, 1.0)
        wz = (kp_ang * err).clamp(-max_ang, max_ang)

        # Override velocity command
        cmd = env.unwrapped.command_manager.get_term("base_velocity")
        cmd.vel_command_b[:, 0] = vx
        cmd.vel_command_b[:, 1] = 0.0
        cmd.vel_command_b[:, 2] = wz

        # Step
        actions = policy(obs)
        obs, _, _, _, _ = env_wrapped.step(actions)

        # Arrival check
        if (dist < arrive_dist).any():
            print(f"  [ARRIVED] Waypoint {wp_idx}: ({goal[0]:.1f}, {goal[1]:.1f})")
            wp_idx += 1

        step += 1
        if step % 300 == 0:
            p = pos[0]
            print(f"  Step {step}: pos=({p[0]:.1f}, {p[1]:.1f}), "
                  f"dist={dist[0]:.1f}m, wp={wp_idx}/{len(waypoints)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()