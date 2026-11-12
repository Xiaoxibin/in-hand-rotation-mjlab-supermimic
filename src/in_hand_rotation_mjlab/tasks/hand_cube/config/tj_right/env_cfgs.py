"""TJ Right (TJDH21R, 21 DoF) cube-rotation environment for the SuperMimic-style project.

The hand model and constants live in ``robots/tj_hand``; this config only binds
the generic hand-cube task to that embodiment.  TJ's MJCF already contains its
own position actuators (21 ``<position>`` elements), so the LEAP-specific
gear-servo / actuator-dynamics randomization events are removed and the XML
actuator dynamics are used as-is.
"""

from __future__ import annotations

from in_hand_rotation_mjlab.robots import get_tj_right_hand_cfg
from in_hand_rotation_mjlab.robots.tj_hand.tj_right_constants import TJ_HOME_JOINT_POS
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg

from .tuning import TJ_TUNED_REWARD

CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)
GRASP_CACHE_FILE = "tasks/hand_cube/cache/tj_right_grasp_cache_31_filtered.npz"
CUBE_HALF_SIZE = 0.031

# The TJ MJCF carries its own position-actuator gains, force limits and joint
# damping.  Randomizing them around the hand-written XML values was not
# validated, so the TJ branch explicitly disables the actuator-dynamics DR.
_TJ_REMOVE_DR_EVENTS = (
    "dr_reflected_inertia",
    "dr_effort_limits",
    "dr_robot_pd_gains",
    "dr_action_delay",
    "dr_motor_friction",
    "dr_motor_damping",
)


def tj_right_hand_cube_rotate_env_cfg(
    play: bool = False,
    cube_half_size: float = CUBE_HALF_SIZE,
    grasp_cache_file: str = GRASP_CACHE_FILE,
    cube_face_colors: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_tj_right_hand_cfg,
        grasp_init_joint_pos=TJ_HOME_JOINT_POS,
        cube_spawn_pos=CUBE_SPAWN_POS,
        grasp_cache_file=grasp_cache_file,
        cube_half_size=cube_half_size,
        cube_face_colors=cube_face_colors,
        negate_yaw_rate=False,  # right hand: counter-clockwise is positive
        play=play,
        palm_body_name="Palm_R",
        palm_center_geom_expr="Palm_R_collision",
        fingertip_geom_expr=r"tip_(?:thumb|index|middle|ring|pinky)_collision",
        enable_domain_randomization=True,
    )

    # TJ uses the XML position actuators; the LEAP gear-servo DR events do not
    # apply.  The cube / contact / link-mass randomization above is kept.
    for name in _TJ_REMOVE_DR_EVENTS:
        cfg.events.pop(name, None)

    # The TJ cache was collected at a single 3.1 cm half-size.  Keep the cube
    # size fixed and use the nearest cache pose; no scale jitter.
    grasp_event = cfg.events.get("reset_from_grasp_cache")
    if grasp_event is not None:
        grasp_event.params["scale_list"] = (1.0,)
        grasp_event.params["scale_jitter"] = 0.0
        grasp_event.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # TJ's grasp sits lower than the LEAP mount (cache cube z ~0.14 m), so the
    # LEAP 0.2 m "fallen" threshold would terminate every episode immediately.
    # The TJ branch uses 0.05 m for both the reward and the termination.
    cfg.rewards["object_fallen"].params["minimum_height"] = 0.05
    cfg.terminations["cube_fell"].params["minimum_height"] = 0.05

    # The curriculum metric must use the same yaw sign as the reward.
    cfg.metrics["rotation_progress"].params["negate_yaw_rate"] = False
    return cfg


_ROTATE_KEYS = (
    "clip_min",
    "clip_max",
    "low_rate_boost",
    "negative_yaw_rate_penalty",
    "tumble_penalty_weight",
    "tumble_dead_zone",
    "drift_mode",
    "drift_decay_k_pos",
    "drift_decay_k_tilt",
    "drift_tilt_threshold",
    "drift_gate_positive_only",
)


def tj_right_hand_cube_rotate_env_cfg_tuned(play: bool = False) -> ManagerBasedRlEnvCfg:
    """TJ cube yaw with the state-only part of the TJ reward tuning.

    The TJ branch's ``cumulative_yaw`` and ``thumb_duty_cycle`` terms need
    episode-level history / contact readings and are not included here; they
    are also not yet reproducible by the world-model adapter.
    """
    cfg = tj_right_hand_cube_rotate_env_cfg(play=play)
    rotate = cfg.rewards["rotate_finite_diff"].params
    rotate.update({key: TJ_TUNED_REWARD[key] for key in _ROTATE_KEYS})
    cfg.rewards["torque_penalty"].params.update(
        max_abs_torque=TJ_TUNED_REWARD["max_abs_torque"], normalize=True
    )
    cfg.rewards["work_penalty"].params.update(
        max_abs_power=TJ_TUNED_REWARD["max_abs_power"]
    )
    return cfg
