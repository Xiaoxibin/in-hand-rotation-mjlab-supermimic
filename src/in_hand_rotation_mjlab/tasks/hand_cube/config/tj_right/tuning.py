"""TJ yaw reward tuning ported from the TJ branch (state-only terms).

These are the knobs that both the simulator reward and the differentiable
world-model adapter can reproduce from the object/hand state.  The TJ branch's
extra ``cumulative_yaw`` and ``thumb_duty_cycle`` terms need episode-level
history and contact readings, so they are not included in this profile yet.
"""

TJ_TUNED_REWARD = dict(
    # rotate_finite_diff
    clip_min=-0.25,
    clip_max=1.0,
    low_rate_boost=2.0,
    negative_yaw_rate_penalty=2.0,
    tumble_penalty_weight=0.5,
    tumble_dead_zone=0.15,
    drift_mode="exp",
    drift_decay_k_pos=50.0,
    drift_decay_k_tilt=8.0,
    drift_tilt_threshold=0.3,
    drift_gate_positive_only=True,
    # torque_penalty / work_penalty
    max_abs_torque=1.0,
    max_abs_power=1.0,
)
