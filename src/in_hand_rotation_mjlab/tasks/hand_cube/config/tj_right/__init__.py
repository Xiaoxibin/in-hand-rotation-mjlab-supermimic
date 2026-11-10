from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import tj_right_hand_cube_rotate_env_cfg
from .rl_cfg import tj_right_hand_cube_rotate_ppo_cfg

register_mjlab_task(
    task_id="Mjlab-TJ-Right-HandCube-Rotate",
    env_cfg=tj_right_hand_cube_rotate_env_cfg(),
    play_env_cfg=tj_right_hand_cube_rotate_env_cfg(play=True),
    rl_cfg=tj_right_hand_cube_rotate_ppo_cfg(),
)
