"""TJ right-hand (TJDH21R) SO(3) cube reorientation.

This mirrors the LEAP SO(3) task: uniform random goal orientations, goal
command, goal-rotation progress/success rewards, safety sensors, and a 30 s
episode.  The only TJ-specific pieces are the base yaw embodiment (21 DoF,
3.1 cm half-size cache), the palm subtree name (``Palm_R``), and the 0.05 m
fall threshold.  No TJ expert demonstrations exist yet, so the reverse
curriculum is disabled; training starts from the grasp cache with uniform
SO(3) goals.
"""

from __future__ import annotations

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.tj_right.env_cfgs import (
    CUBE_HALF_SIZE,
    GRASP_CACHE_FILE,
    tj_right_hand_cube_rotate_env_cfg,
)
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.commands import GOAL
from in_hand_rotation_mjlab.tasks.hand_cube.safety import add_safety_sensors


def tj_right_hand_cube_reorient_env_cfg(play: bool = False):
    cfg = tj_right_hand_cube_rotate_env_cfg(
        play=play,
        cube_half_size=CUBE_HALF_SIZE,
        grasp_cache_file=GRASP_CACHE_FILE,
    )
    add_safety_sensors(cfg, palm_pattern="Palm_R")

    act = cfg.actions["joint_pos"]
    act.delta_min, act.delta_max = -0.2, 0.2
    act.max_command_lead = 0.25

    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.commands = {
        GOAL: hand_cube_mdp.CubeGoalCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            init_max_angle=torch.pi if play else 0.8,
            reverse_curriculum=False,
        )
    }

    robot = SceneEntityCfg("robot", joint_names=(".*",))
    actor = {
        "joint_pos": ObservationTermCfg(
            func=hand_cube_mdp.joint_pos_abs,
            params={"asset_cfg": robot},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cmd": ObservationTermCfg(
            func=hand_cube_mdp.joint_pos_commanded,
            params={"action_name": "joint_pos", "asset_cfg": robot},
        ),
        "cube_pos": ObservationTermCfg(
            func=hand_cube_mdp.cube_pos_rel, noise=Unoise(n_min=-0.002, n_max=0.002)
        ),
        "cube_rot": ObservationTermCfg(
            func=hand_cube_mdp.cube_rot6d, noise=Unoise(n_min=-0.02, n_max=0.02)
        ),
        "goal_rot": ObservationTermCfg(func=hand_cube_mdp.goal_rot6d),
        "rel_rot": ObservationTermCfg(
            func=hand_cube_mdp.goal_rel_rot6d, noise=Unoise(n_min=-0.02, n_max=0.02)
        ),
    }
    critic = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()}
    critic |= {
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": robot}),
        "cube_lin_vel": ObservationTermCfg(func=hand_cube_mdp.cube_lin_vel),
        "cube_ang_vel": ObservationTermCfg(func=hand_cube_mdp.cube_ang_vel),
        "cube_mass": ObservationTermCfg(
            func=hand_cube_mdp.cube_mass, params={"object_name": "cube", "body_name": "cube"}
        ),
        "cube_friction": ObservationTermCfg(
            func=hand_cube_mdp.cube_friction_coeff,
            params={"object_name": "cube", "geom_name": "cube_geom", "axis": 0},
        ),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(actor, enable_corruption=not play, history_length=3, flatten_history_dim=True),
        "critic": ObservationGroupCfg(critic, enable_corruption=False, history_length=1, flatten_history_dim=True),
    }

    cfg.scale_rewards_by_dt = False
    cfg.rewards = {
        "rot_progress": RewardTermCfg(func=hand_cube_mdp.goal_rot_progress, weight=5.0, params={"clip": 0.05}),
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.02),
        "cube_linvel": RewardTermCfg(func=hand_cube_mdp.object_linvel_l1, weight=-0.5, params={"object_name": "cube"}),
        "success": RewardTermCfg(func=hand_cube_mdp.goal_success, weight=5.0),
        "rot_dist": RewardTermCfg(func=hand_cube_mdp.goal_rot_dist, weight=-0.02),
        "cube_pos": RewardTermCfg(
            func=hand_cube_mdp.object_position_l2_from_reset, weight=-20.0,
            params={"object_name": "cube"},
        ),
        "fallen": RewardTermCfg(func=envs_mdp.is_terminated, weight=-30.0),
        "cube_force": RewardTermCfg(
            func=hand_cube_mdp.contact_force_excess_sq, weight=-2e-3,
            params={"sensor_name": "safety_hand_cube", "free": 10.0},
        ),
        "self_force": RewardTermCfg(
            func=hand_cube_mdp.contact_force_excess_sq, weight=-2e-3,
            params={"sensor_name": "safety_hand_self", "free": 2.0},
        ),
        "torque": RewardTermCfg(
            func=hand_cube_mdp.joint_torque_l2, weight=-0.02, params={"asset_cfg": robot}
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.002),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "cube_fell": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("cube")},
        ),
        "cube_too_fast": TerminationTermCfg(
            func=hand_cube_mdp.object_linear_speed_above,
            params={"max_linear_speed": 1.0, "asset_cfg": SceneEntityCfg("cube")},
        ),
        "cube_far": TerminationTermCfg(
            func=hand_cube_mdp.object_pose_rp_position_deviation_from_reset,
            params={"max_position_error": 0.08, "max_tilt_error": 1e9, "asset_cfg": SceneEntityCfg("cube")},
        ),
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }
    cfg.episode_length_s = 1e9 if play else 30.0
    return cfg
