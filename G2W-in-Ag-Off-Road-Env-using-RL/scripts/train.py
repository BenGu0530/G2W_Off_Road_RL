# Training script for Go2-W Agricultural Environment
# Usage: python scripts/train.py

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Go2-W in agricultural environment")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after app launch ---
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)

def main():
    env_cfg = UnitreeGo2WRoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"[INFO] Environment created with {args_cli.num_envs} envs")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
