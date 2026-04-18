# Copyright 2024 Ben Gu
import gymnasium as gym
from . import agents

# Stage A config
from .ag_env_cfg import Go2WAgEnvCfg

gym.register(
    id="Go2W-Ag-StageA-v0",
    entry_point="go2w_nav.tasks.go2w.ag_env:Go2WAgEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ag_env_cfg:Go2WAgEnvCfg",
        "rsl_rl_cfg_entry_point": f"go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg:Go2WAgPPORunnerCfg",
    },
)

# Stage B config
from .ag_env_cfg_stage_b import Go2WAgEnvCfg as Go2WAgEnvCfgStageB

gym.register(
    id="Go2W-Ag-StageB-v0",
    entry_point="go2w_nav.tasks.go2w.ag_env:Go2WAgEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ag_env_cfg_stage_b:Go2WAgEnvCfg",
        "rsl_rl_cfg_entry_point": f"go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg:Go2WAgPPORunnerCfg",
    },
)

# Legacy name for demo.py
gym.register(
    id="Go2W-Ag-Rough-v0",
    entry_point="go2w_nav.tasks.go2w.ag_env:Go2WAgEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ag_env_cfg:Go2WAgEnvCfg",
        "rsl_rl_cfg_entry_point": f"go2w_nav.tasks.go2w.agents.rsl_rl_ppo_cfg:Go2WAgPPORunnerCfg",
    },
)
