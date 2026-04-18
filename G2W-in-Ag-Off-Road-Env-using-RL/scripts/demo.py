"""
Demo script for Go2-W Agricultural Navigation
Uses waypoint-based navigation routing through pavement corridors.
Usage:
    cd /home/rml2/IsaacLab/scripts/reinforcement_learning/rsl_rl
    python /home/rml2/Documents/ben_gu/ben_Ag/G2W-in-Ag-Off-Road-Env-using-RL/scripts/demo.py \
        --checkpoint <path_to_checkpoint>
"""

import argparse
import torch
from isaaclab.app import AppLauncher
import numpy as np

parser = argparse.ArgumentParser(description="Go2-W Agricultural Demo")
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import go2w_nav.tasks  # noqa: F401
from go2w_nav.tasks.go2w.ag_env_cfg import (
    PAVEMENT_RANGES, CROP_RANGES, TOTAL_Y, INNER_X, TOTAL_X, CROP_Y, OFFROAD_Y, PAVEMENT_Y, BORDER_Y
)
from go2w_nav.tasks.go2w.ag_env_cfg import Go2WAgEnvCfg
from go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg import Go2WAgPPORunnerCfg


def update_nav_command(env_unwrapped, target_pos):
    """Body-frame velocity with yaw correction — walk forward AND turn."""
    robot = env_unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[:, :2]

    to_target = target_pos - pos
    dist = torch.norm(to_target, dim=1, keepdim=True).clamp(min=0.01)
    direction = to_target / dist

    # Robot orientation from quaternion
    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    cos_yaw = 1.0 - 2.0 * (y**2 + z**2)
    sin_yaw = 2.0 * (w * z + x * y)

    # Body frame velocity
    vx_body = cos_yaw * direction[:, 0] + sin_yaw * direction[:, 1]
    vy_body = -sin_yaw * direction[:, 0] + cos_yaw * direction[:, 1]

    # Heading error for yaw correction
    robot_yaw = torch.atan2(sin_yaw, cos_yaw)
    target_yaw = torch.atan2(to_target[:, 1], to_target[:, 0])
    heading_err = torch.atan2(torch.sin(target_yaw - robot_yaw), torch.cos(target_yaw - robot_yaw))

    # Yaw rate to turn toward target
    yaw_rate = (1.5 * heading_err).clamp(-1.0, 1.0)

    # Forward speed — always move, but slower when misaligned
    facing = torch.cos(heading_err).clamp(0.3, 1.0)
    speed = 1.0 * facing

    cmd = env_unwrapped.command_manager.get_command("base_velocity")
    cmd[:, 0] = vx_body * speed
    cmd[:, 1] = vy_body * 0.3     # minimal lateral
    cmd[:, 2] = yaw_rate           # turn head toward target

    return dist.squeeze(1)


def plan_waypoints(start_pos, goal_pos, device):
    """Plan path avoiding crop rows. Route through field edges to cross between groups."""
    
    sx, sy = start_pos[0].item(), start_pos[1].item()
    gx, gy = goal_pos[0].item(), goal_pos[1].item()
    
    # Approach point: offroad strip adjacent to goal crop
    approach_below = gy - (CROP_Y / 2.0 + OFFROAD_Y / 2.0)
    approach_above = gy + (CROP_Y / 2.0 + OFFROAD_Y / 2.0)
    if abs(sy - approach_below) < abs(sy - approach_above):
        approach_y = approach_below
    else:
        approach_y = approach_above
    
    # Connected groups (safe zones that can reach each other without crossing crops)
    groups = [
        [-18.0],
        [-14.0, -12.0, -10.0],
        [-6.0, -4.0, -2.0],
        [2.0, 4.0, 6.0],
        [10.0, 12.0, 14.0],
        [18.0],
    ]
    
    # Crop Y centers (obstacles between groups)
    crop_centers = [-16.0, -8.0, 0.0, 8.0, 16.0]
    
    def find_group(y):
        for i, g in enumerate(groups):
            if min(g) - 1.5 <= y <= max(g) + 1.5:
                return i
        return min(range(len(groups)), key=lambda i: min(abs(y - v) for v in groups[i]))
    
    start_group = find_group(sy)
    goal_group = find_group(approach_y)
    
    waypoints = []
    
    # X position at field edge for crossing (border area, outside crop inner zone)
    edge_x = INNER_X / 2.0 + 0.5  # just inside the border offroad strip
    
    if start_group == goal_group:
        # Same group — direct path
        waypoints.append(torch.tensor([[gx, approach_y]], device=device, dtype=torch.float32))
    else:
        # Different groups — need to go to field edge, cross, come back
        
        # Step 1: go to edge X in current corridor
        current_y = groups[start_group][len(groups[start_group])//2]  # middle of current group
        nearest_start = min(groups[start_group], key=lambda v: abs(v - sy))
        waypoints.append(torch.tensor([[edge_x, nearest_start]], device=device, dtype=torch.float32))
        
        # Step 2: walk along the edge, crossing crop rows at the border
        # (crops only exist in inner X, border X is all offroad)
        if goal_group > start_group:
            # Moving up (positive Y)
            for g_idx in range(start_group + 1, goal_group + 1):
                entry_y = groups[g_idx][0]  # bottom of target group
                waypoints.append(torch.tensor([[edge_x, entry_y]], device=device, dtype=torch.float32))
        else:
            # Moving down (negative Y)
            for g_idx in range(start_group - 1, goal_group - 1, -1):
                entry_y = groups[g_idx][-1]  # top of target group
                waypoints.append(torch.tensor([[edge_x, entry_y]], device=device, dtype=torch.float32))
        
        # Step 3: come back from edge to goal X at approach Y
        waypoints.append(torch.tensor([[gx, approach_y]], device=device, dtype=torch.float32))
    
    # Filter out waypoints too close together
    if len(waypoints) > 1:
        filtered = [waypoints[0]]
        for wp in waypoints[1:]:
            if torch.norm(wp - filtered[-1]).item() > 1.5:
                filtered.append(wp)
        if len(filtered) == 0:
            filtered = [waypoints[-1]]
        waypoints = filtered
    
    return waypoints




def generate_goals(device):
    """Generate goals AT crop row centers (inspection targets).
    
    The goal sphere appears at the crop center.
    The robot is considered "arrived" when it reaches the 
    adjacent off-road strip (handled by arrival threshold).
    """
    goals = []
    crop_centers = [(s + e) / 2.0 - TOTAL_Y / 2.0 for s, e in CROP_RANGES]
    
    for i, crop_y in enumerate(crop_centers):
        x = (torch.rand(1).item() - 0.5) * INNER_X * 0.8
        goals.append(torch.tensor([[x, crop_y]], device=device, dtype=torch.float32))
    
    return goals


def main():
    env_cfg = Go2WAgEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    agent_cfg = Go2WAgPPORunnerCfg()

    env = gym.make("Go2W-Ag-StageA-v0", cfg=env_cfg)

    # Fix gymnasium Box dtype bug
    space = env.unwrapped.single_action_space
    env.unwrapped.single_action_space = gym.spaces.Box(
    low=space.low.astype(np.float32),
    high=space.high.astype(np.float32),
    shape=space.shape,
    dtype=np.float32,
)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    goals = generate_goals("cuda:0")
    print(f"\n[Demo] {len(goals)} goals generated:")
    for i, g in enumerate(goals):
        print(f"  Goal {i+1}: x={g[0,0].item():.1f}m, y={g[0,1].item():.1f}m")

    obs, _ = env.reset()
    task_idx = 0
    success_count = 0
    step = 0
    max_steps_per_goal = 5000

    # Plan waypoints for first goal
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
    waypoints = plan_waypoints(start_pos, goals[task_idx][0], "cuda:0")
    wp_idx = 0
    current_target = waypoints[wp_idx]

    # Update goal sphere
    env.unwrapped.goal_pos[0] = goals[task_idx][0]
    env.unwrapped._update_goal_sphere(visible=True)

    print(f"\n[Demo] Navigating to Goal 1/{len(goals)}")
    print(f"  Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"    WP{i+1}: ({wp[0,0].item():.1f}, {wp[0,1].item():.1f})")

    goal_step = 0

    while simulation_app.is_running() and task_idx < len(goals):

        # Update command toward current waypoint
        dist = update_nav_command(env.unwrapped, current_target)

        # Policy inference
        with torch.no_grad():
            action = policy(obs)
        obs, _, done, _ = env.step(action)

        # Check if command was overwritten
        if step % 200 == 0:
            cmd_after = env.unwrapped.command_manager.get_command("base_velocity")
            print(f"  [CMD CHECK] vx={cmd_after[0,0]:.2f} vy={cmd_after[0,1]:.2f} wz={cmd_after[0,2]:.2f}")

        # Debug print
        if step % 200 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
            final_dist = torch.norm(goals[task_idx][0] - pos).item()
            print(f"  step={goal_step} wp={wp_idx+1}/{len(waypoints)} "
                  f"wp_dist={dist.item():.1f}m goal_dist={final_dist:.1f}m")

        # Advance waypoint
        wp_threshold = 2.0 if wp_idx < len(waypoints) - 1 else 3.5
        if dist.item() < wp_threshold:
            if wp_idx < len(waypoints) - 1:
                wp_idx += 1
                current_target = waypoints[wp_idx]
                print(f"  → Waypoint {wp_idx+1}/{len(waypoints)} reached, "
                      f"next: ({current_target[0,0].item():.1f}, "
                      f"{current_target[0,1].item():.1f})")
            else:
                # Reached final goal
                success_count += 1
                print(f"\n[Demo] ✓ Goal {task_idx+1}/{len(goals)} reached!")
                task_idx += 1
                goal_step = 0

                if task_idx < len(goals):
                    # Plan waypoints for next goal
                    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
                    waypoints = plan_waypoints(start_pos, goals[task_idx][0], "cuda:0")
                    wp_idx = 0
                    current_target = waypoints[wp_idx]

                    env.unwrapped.goal_pos[0] = goals[task_idx][0]
                    env.unwrapped._update_goal_sphere(visible=True)

                    print(f"[Demo] Navigating to Goal {task_idx+1}/{len(goals)}")
                    print(f"  Waypoints: {len(waypoints)}")
                    for i, wp in enumerate(waypoints):
                        label = "FINAL" if i == len(waypoints)-1 else f"WP{i+1}"
                        print(f"    {label}: ({wp[0,0].item():.1f}, "
                              f"{wp[0,1].item():.1f})")

        # Timeout for current goal
        if goal_step > max_steps_per_goal:
            print(f"[Demo] Timeout on goal {task_idx+1}, skipping...")
            task_idx += 1
            goal_step = 0
            if task_idx < len(goals):
                start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
                waypoints = plan_waypoints(start_pos, goals[task_idx][0], "cuda:0")
                wp_idx = 0
                current_target = waypoints[wp_idx]
                env.unwrapped.goal_pos[0] = goals[task_idx][0]
                env.unwrapped._update_goal_sphere(visible=True)

        if done.any():
            obs, _ = env.reset()
            print("[Demo] Episode reset")

        step += 1
        goal_step += 1

    if success_count >= len(goals):
        print(f"\n[Demo] ✓ All {len(goals)} goals completed!")
    else:
        print(f"\n[Demo] Completed {success_count}/{len(goals)} goals")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()