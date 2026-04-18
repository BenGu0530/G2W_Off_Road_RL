"""
Go2-W Stage B Mode-Switching Demo
===================================
Loads the Stage B environment (with mode_switch reward) and a Stage B
checkpoint. Uses the PAVEMENT-PREFERRING nav planner (from nav_controller_v2.py)
so the robot actively routes through pavement when possible.

Usage:
    cd /home/rml2/IsaacLab/scripts/reinforcement_learning/rsl_rl
    python <path>/demo_stage_b.py --checkpoint <stage_b.pt>

Key differences from demo_compare.py:
1. Loads Go2W-Ag-StageB-v0 (Stage B env, different rewards)
2. Imports Go2WAgEnvCfg from ag_env_cfg_stage_b
3. Uses NavControllerV2 with pavement preference
4. Logs extra fields for analysis: leg_action_mag, wheel_action_mag
"""

import argparse
import sys
import os
import torch
import time
import csv
from isaaclab.app import AppLauncher
import numpy as np

parser = argparse.ArgumentParser(description="Go2-W Stage B Mode-Switching Demo")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", type=str, choices=["walk", "drive"], default="walk",
                    help="'walk' uses full policy, 'drive' scales leg actions by 0.5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import go2w_nav.tasks
# STAGE B: import Stage B config
from go2w_nav.tasks.go2w.ag_env_cfg_stage_b import Go2WAgEnvCfg
from go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg import Go2WAgPPORunnerCfg

# Add scripts dir to path for nav_controller import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
from nav_controller_v2 import NavController   # use pavement-preferring version
from field_map import CROP_RANGES, TOTAL_Y, INNER_X, get_zone_type

ROBOT_MASS = 19.5
GRAVITY = 9.81


def generate_goals(device):
    """Fixed goals at crop centers for reproducible comparison."""
    goals = []
    crop_centers = [(s + e) / 2.0 for s, e in CROP_RANGES]
    torch.manual_seed(42)
    for cy in crop_centers:
        x = (torch.rand(1).item() - 0.5) * INNER_X * 0.8
        goals.append(torch.tensor([[x, cy]], device=device, dtype=torch.float32))
    return goals


def create_goal_sphere():
    import omni.usd
    from pxr import UsdGeom, Gf, UsdShade, Sdf
    stage = omni.usd.get_context().get_stage()
    path = "/World/GoalSphere"
    if stage.GetPrimAtPath(path):
        return
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(1.0)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, -10))
    mat = UsdShade.Material.Define(stage, "/World/GoalMat")
    sh = UsdShade.Shader.Define(stage, "/World/GoalMat/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0.2, 0.2))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.05, 0.05))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(mat)


def move_goal_sphere(x, y):
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/GoalSphere")
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(x), float(y), 0.5))


def main():
    mode = args_cli.mode
    device = "cuda:0"

    print(f"\n{'='*60}")
    print(f"  Go2-W STAGE B — {mode.upper()}")
    print(f"{'='*60}\n")

    # ── Environment (STAGE B) ───────────────────────────────────────
    env_cfg = Go2WAgEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    env = gym.make("Go2W-Ag-StageB-v0", cfg=env_cfg)
    space = env.unwrapped.single_action_space
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=space.low.astype(np.float32),
        high=space.high.astype(np.float32),
        shape=space.shape, dtype=np.float32,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=Go2WAgPPORunnerCfg().clip_actions)

    # ── Command injection patch ─────────────────────────────────────
    env.unwrapped._nav_cmd = torch.zeros(env.unwrapped.num_envs, 3, device=device)

    _cmd_manager = env.unwrapped.command_manager
    _original_compute = _cmd_manager.compute

    def _patched_compute(dt):
        _original_compute(dt)
        cmd = _cmd_manager.get_command("base_velocity")
        cmd[:] = env.unwrapped._nav_cmd

    _cmd_manager.compute = _patched_compute

    cmd_term = _cmd_manager.get_term("base_velocity")
    cmd_term._resample_command = lambda env_ids: None

    # ── Policy ──────────────────────────────────────────────────────
    agent_cfg = Go2WAgPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    # ── Goals & nav ─────────────────────────────────────────────────
    goals = generate_goals(device)
    nav = NavController(device=device)
    create_goal_sphere()

    print("Goals:")
    for i, g in enumerate(goals):
        print(f"  {i+1}: ({g[0,0]:.1f}, {g[0,1]:.1f})")

    # ── Data ────────────────────────────────────────────────────────
    data_log = []
    start_time = time.time()
    obs, _ = env.reset()

    task_idx = 0
    success_count = 0
    step = 0
    goal_step = 0
    max_steps_per_goal = 3000
    total_energy = 0.0
    total_distance = 0.0
    prev_pos = None
    cot = 0.0

    # First goal
    robot = env.unwrapped.scene["robot"]
    start_pos = robot.data.root_pos_w[0, :2]
    nav.set_goal(goals[task_idx], start_pos)
    env.unwrapped.goal_pos[0] = goals[task_idx][0]
    env.unwrapped._update_goal_sphere(visible=True)
    move_goal_sphere(goals[task_idx][0, 0].item(), goals[task_idx][0, 1].item())
    print(f"\n[{mode.upper()}] Goal 1/{len(goals)}")

    # ── Warmup to prevent spawn instability ─────────────────────────
    print(f"[{mode.upper()}] Warmup: 50 steps at min command...")
    for warmup_step in range(50):
        env.unwrapped._nav_cmd[:, 0] = 0.3
        env.unwrapped._nav_cmd[:, 1] = 0.0
        env.unwrapped._nav_cmd[:, 2] = 0.0
        with torch.no_grad():
            action = policy(obs)
        if mode == "drive":
            ramp = warmup_step / 50.0
            leg_scale = 1.0 - 0.5 * ramp
            action[:, :12] *= leg_scale
        obs, _, _, _ = env.step(action)
    print(f"[{mode.upper()}] Warmup done, starting navigation")

    # ── Main loop ───────────────────────────────────────────────────
    while simulation_app.is_running() and task_idx < len(goals) and step < 20000:

        robot = env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w
        quat = robot.data.root_quat_w

        vx, vy, wz, wp_dist = nav.update(pos, quat)

        if nav.is_goal_reached():
            success_count += 1
            print(f"\n[{mode.upper()}] Goal {task_idx+1}/{len(goals)} REACHED! "
                  f"dist={total_distance:.1f}m CoT={cot:.3f}")
            task_idx += 1
            goal_step = 0
            if task_idx >= len(goals):
                break
            nav.set_goal(goals[task_idx], pos[0, :2])
            env.unwrapped.goal_pos[0] = goals[task_idx][0]
            env.unwrapped._update_goal_sphere(visible=True)
            move_goal_sphere(goals[task_idx][0, 0].item(), goals[task_idx][0, 1].item())
            print(f"[{mode.upper()}] Goal {task_idx+1}/{len(goals)}")
            continue

        env.unwrapped._nav_cmd[:, 0] = vx
        env.unwrapped._nav_cmd[:, 1] = vy
        env.unwrapped._nav_cmd[:, 2] = wz

        with torch.no_grad():
            action = policy(obs)
        if mode == "drive":
            action[:, :12] *= 0.5

        obs, _, done, _ = env.step(action)

        # ── Logging with action magnitudes (for Stage B analysis) ───
        p = robot.data.root_pos_w[0, :2].clone()
        speed = torch.norm(robot.data.root_lin_vel_w[0, :2]).item()

        if prev_pos is not None:
            total_distance += torch.norm(p - prev_pos).item()
        prev_pos = p.clone()

        jt = robot.data.applied_torque
        jv = robot.data.joint_vel
        power = (jt * jv).abs().sum().item() if jt is not None else 0.0
        total_energy += power * env.unwrapped.step_dt
        cot = total_energy / (ROBOT_MASS * GRAVITY * max(total_distance, 0.01))

        # NEW: action magnitude per joint group
        leg_action_mag = action[0, :12].abs().mean().item()
        wheel_action_mag = action[0, 12:16].abs().mean().item()

        terrain = get_zone_type(p[1].item())
        gdist = torch.norm(goals[task_idx][0] - p).item()

        data_log.append({
            "step": step, "mode": mode, "goal_idx": task_idx,
            "x": p[0].item(), "y": p[1].item(),
            "speed": speed, "power": power,
            "energy_cumulative": total_energy,
            "distance_cumulative": total_distance,
            "cot_cumulative": cot, "terrain": terrain,
            "goal_dist": gdist, "wp_dist": wp_dist[0].item(),
            "leg_action_mag": leg_action_mag,
            "wheel_action_mag": wheel_action_mag,
        })

        if step % 300 == 0:
            print(f"  step={step} g={task_idx+1}/{len(goals)} "
                  f"spd={speed:.2f} d={total_distance:.1f}m "
                  f"CoT={cot:.3f} {terrain} gd={gdist:.1f}m "
                  f"leg={leg_action_mag:.2f} whl={wheel_action_mag:.2f}")

        if goal_step > max_steps_per_goal:
            print(f"[{mode.upper()}] Timeout goal {task_idx+1}")
            task_idx += 1
            goal_step = 0
            if task_idx < len(goals):
                nav.set_goal(goals[task_idx], pos[0, :2])
                env.unwrapped.goal_pos[0] = goals[task_idx][0]
                env.unwrapped._update_goal_sphere(visible=True)
                move_goal_sphere(goals[task_idx][0, 0].item(), goals[task_idx][0, 1].item())

        if done.any():
            obs, _ = env.reset()

        step += 1
        goal_step += 1

    # ── Save ────────────────────────────────────────────────────────
    csv_path = f"comparison_data_stage_b_{mode}.csv"
    if data_log:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data_log[0].keys())
            w.writeheader()
            w.writerows(data_log)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  RESULTS — STAGE B {mode.upper()}")
    print(f"{'='*60}")
    print(f"  Goals: {success_count}/{len(goals)}")
    print(f"  Distance: {total_distance:.1f}m")
    print(f"  Energy: {total_energy:.1f}J")
    print(f"  CoT: {cot:.4f}")
    print(f"  Avg speed: {total_distance/max(elapsed,0.01):.2f} m/s")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()