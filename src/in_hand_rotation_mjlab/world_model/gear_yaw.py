"""The palm-up cube-yaw task (``Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear``) as seen from world-model states: the
policy's observation and the env's reward terms, computed from (predicted) states so that imagined rollouts train the
policy on the objective PPO had.

Env definitions reproduced (``tasks/hand_cube``), per second, the reward manager multiplies by dt = 0.05:
  rotate_finite_diff  +1.25 clip(-mean of the last 4 world-yaw differences / dt, +-0.25) x drift factor
                      (0.1 when the cube is > 2 cm or > 0.35 rad roll / pitch from its anchor = the state after the
                      episode's first step, else 1)
  object_linvel       -0.3 |v_world|_1
  pose_diff           -0.1 sum(max(|q - q_reset| - 0.4, 0)^2)       (q_reset: the episode's reset joints)
  torque, work        -0.1 sum tau^2, -0.05 (sum tau qdot)^2, tau from the gear servo law at the end of the step:
                      K_t clamp(G (q + clamp(cmd - q, +-0.15) - q) - Gd qdot, +-I_max), per-env per-joint law
                      parameters ``plant`` = [G, Gd, I_max, K_t] (the driver's 80 Hz goal latch and the gravity
                      compensation are left out)
  object_fallen       -10 [cube centre below z = 0.2]
Actor observation (history 10, per term oldest -> newest): joint_pos - default (+U(+-0.01), lag 0 or 1 step drawn
every step), the commanded target.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_mul, wrap_to_pi

from .model import quat_to_rot6d
from .record import state_slices

H_OBS = 10
DRIFT_POS, DRIFT_TILT, DRIFT_OUT = 0.02, 0.35, 0.1
YAW_CLIP, W_ROT, W_LIN, W_POSE, POSE_TOL, W_FALL, FALL_Z = 0.25, 1.25, -0.3, -0.1, 0.4, -10.0, 0.2
W_TORQUE, W_WORK, LEAD_CLAMP = -0.1, -0.05, 0.15
OBS_NOISE = 0.01


class GearYaw:
  def __init__(
    self,
    nj: int,
    q_default: torch.Tensor,
    base_p: torch.Tensor,
    base_q: torch.Tensor,
    dt: float,
    negate_yaw_rate: bool = True,
    torque_law: str = "gear",
    fall_z: float = FALL_Z,
    clip_min: float = -YAW_CLIP,
    clip_max: float = YAW_CLIP,
    low_rate_boost: float = 1.0,
    negative_yaw_rate_penalty: float = 1.0,
    tumble_penalty_weight: float = 0.0,
    tumble_dead_zone: float = 0.0,
    drift_position_threshold: float = DRIFT_POS,
    drift_tilt_threshold: float = DRIFT_TILT,
    drift_outside_factor: float = DRIFT_OUT,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.1,
    drift_gate_positive_only: bool = False,
    max_abs_torque: float | None = None,
    normalize_torque: bool = False,
    max_abs_power: float | None = None,
  ):
    self.nj = nj
    self.S = state_slices(nj)
    self.q_default = q_default
    self.base_p, self.base_q = base_p, base_q
    self.dt = dt
    self.negate_yaw_rate = bool(negate_yaw_rate)
    self.torque_law = str(torque_law)
    self.fall_z = float(fall_z)
    self.clip_min, self.clip_max = float(clip_min), float(clip_max)
    self.low_rate_boost = float(low_rate_boost)
    self.negative_yaw_rate_penalty = float(negative_yaw_rate_penalty)
    self.tumble_penalty_weight = float(tumble_penalty_weight)
    self.tumble_dead_zone = float(tumble_dead_zone)
    self.drift_position_threshold = float(drift_position_threshold)
    self.drift_tilt_threshold = float(drift_tilt_threshold)
    self.drift_outside_factor = float(drift_outside_factor)
    self.drift_mode = str(drift_mode)
    self.drift_decay_k_pos = float(drift_decay_k_pos)
    self.drift_decay_k_tilt = float(drift_decay_k_tilt)
    self.drift_min_factor = float(drift_min_factor)
    self.drift_gate_positive_only = bool(drift_gate_positive_only)
    self.max_abs_torque = max_abs_torque
    self.normalize_torque = bool(normalize_torque)
    self.max_abs_power = max_abs_power
    self.obs_dim = H_OBS * 2 * nj
    self.info_dim = nj + 3 + 2  # q_reset, anchor position (world), anchor roll / pitch

  # --- geometry ---
  def world_quat(self, s):
    return quat_mul(self.base_q.expand(s.shape[0], 4), s[..., self.S["quat"]])

  def world_pos(self, s):
    return self.base_p + quat_apply(self.base_q.expand(s.shape[0], 4), s[..., self.S["p"]])

  def rpy(self, s):
    return euler_xyz_from_quat(self.world_quat(s))

  # --- observation ---
  def obs_frame(self, q, cmd, noise: bool = True):
    x = q - self.q_default
    if noise:
      x = x + (torch.rand_like(x) * 2 - 1) * OBS_NOISE
    return torch.cat([x, cmd], -1)

  def flatten(self, hist: torch.Tensor) -> torch.Tensor:
    """(B, H_OBS, 2 nj) frames oldest -> newest -> the env's (B, 2 H_OBS nj) per-term layout."""
    B, nj = hist.shape[0], self.nj
    return torch.cat([hist[..., :nj].reshape(B, -1), hist[..., nj:].reshape(B, -1)], -1)

  # --- episode info ---
  def anchor(self, s):
    roll, pitch, _ = self.rpy(s)
    return torch.cat([self.world_pos(s), torch.stack([roll, pitch], -1)], -1)

  # --- reward ---
  def dyaw(self, s, s2):
    return wrap_to_pi(self.rpy(s2)[2] - self.rpy(s)[2])

  def servo_torque(self, s2, cmd, plant):
    S, nj = self.S, self.nj
    q, qd = s2[..., S["q"]], s2[..., S["qd"]]
    if self.torque_law == "position":
      # XML position actuator: tau = clip(kp * (cmd - q) - kd * qd, +-force_limit).
      kp, kd, flim = plant[:, :nj], plant[:, nj:2 * nj], plant[:, 2 * nj:3 * nj]
      tau = kp * (cmd - q) - kd * qd
      return torch.maximum(torch.minimum(tau, flim), -flim)
    G, Gd, Imax, Kt = plant[:, :nj], plant[:, nj:2 * nj], plant[:, 2 * nj:3 * nj], plant[:, 3 * nj:]
    lead = (cmd - q).clamp(-LEAD_CLAMP, LEAD_CLAMP)
    cur = torch.maximum(torch.minimum(G * lead - Gd * qd, Imax), -Imax)
    return Kt * cur

  def torque_work_rate(self, s2, cmd, plant):
    tau = self.servo_torque(s2, cmd, plant)
    if self.max_abs_torque is not None:
      tau = tau.clamp(-self.max_abs_torque, self.max_abs_torque)
    qd = s2[..., self.S["qd"]]
    torque_val = tau.pow(2).sum(-1)
    if self.normalize_torque:
      denom = self.nj * (self.max_abs_torque**2) if self.max_abs_torque else max(self.nj, 1)
      torque_val = torque_val / denom
    power = (tau * qd).sum(-1)
    if self.max_abs_power is not None:
      power = power.clamp(-self.max_abs_power, self.max_abs_power)
    return W_TORQUE * torque_val + W_WORK * power.pow(2)

  def reward_rates(self, s, s2, info, dyaw_prev3, aux_rate):
    """Per-term reward rates (per second) of the step s -> s2; ``dyaw_prev3`` (B, 3) the previous three world-yaw
    differences, ``aux_rate`` (B,) the torque + work penalty rate (learned). Returns (terms dict, new dyaw window)."""
    S, nj = self.S, self.nj
    d = self.dyaw(s, s2)
    win = torch.cat([dyaw_prev3, d[:, None]], -1)
    rate = (-win.mean(-1) if self.negate_yaw_rate else win.mean(-1)) / self.dt

    if self.low_rate_boost != 1.0:
      boosted = (rate * self.low_rate_boost).clamp(0.0, self.clip_max)
      yaw_reward = torch.where(rate >= 0.0, boosted, rate.clamp(self.clip_min, 0.0))
    else:
      yaw_reward = rate.clamp(self.clip_min, self.clip_max)
    if self.negative_yaw_rate_penalty != 1.0:
      yaw_reward = torch.where(
        yaw_reward < 0.0, yaw_reward * self.negative_yaw_rate_penalty, yaw_reward
      )

    if self.tumble_penalty_weight > 0.0:
      w_world = quat_apply(self.world_quat(s2), s2[..., S["w"]])
      z_axis = torch.zeros_like(w_world)
      z_axis[..., 2] = 1.0
      z_axis_w = quat_apply(self.world_quat(s2), z_axis)
      spin = (w_world * z_axis_w).sum(-1, keepdim=True)
      tumble = (w_world - spin * z_axis_w).norm(dim=-1) - self.tumble_dead_zone
      yaw_reward = yaw_reward - self.tumble_penalty_weight * tumble.clamp_min(0.0)

    roll, pitch, _ = self.rpy(s2)
    pos_err = (self.world_pos(s2) - info[:, nj:nj + 3]).norm(dim=-1)
    tilt = torch.stack([wrap_to_pi(roll - info[:, nj + 3]).abs(), wrap_to_pi(pitch - info[:, nj + 4]).abs()], -1).norm(dim=-1)
    if self.drift_mode == "step":
      inside = (pos_err <= self.drift_position_threshold) & (tilt <= self.drift_tilt_threshold)
      drift = torch.where(inside, torch.ones_like(rate), torch.full_like(rate, self.drift_outside_factor))
    elif self.drift_mode == "exp":
      pos_excess = (pos_err - self.drift_position_threshold).clamp_min(0.0)
      tilt_excess = (tilt - self.drift_tilt_threshold).clamp_min(0.0)
      drift = torch.exp(-self.drift_decay_k_pos * pos_excess - self.drift_decay_k_tilt * tilt_excess)
      drift = drift.clamp(min=self.drift_min_factor, max=1.0)
    else:
      raise ValueError(f"Unknown drift_mode '{self.drift_mode}'")
    if self.drift_gate_positive_only:
      drift = torch.where(yaw_reward >= 0.0, drift, torch.ones_like(drift))

    v_w = quat_apply(self.base_q.expand(s2.shape[0], 4), s2[..., S["v"]])
    over = (s2[..., S["q"]] - info[:, :nj]).abs() - POSE_TOL
    terms = dict(
      rotate_finite_diff=W_ROT * yaw_reward * drift,
      object_linvel_penalty=W_LIN * v_w.abs().sum(-1),
      pose_diff_penalty=W_POSE * over.clamp_min(0).pow(2).sum(-1),
      torque_work=aux_rate,
      object_fallen=W_FALL * (self.world_pos(s2)[:, 2] < self.fall_z).float(),
    )
    return terms, win[:, 1:]

  # --- critic input ---
  def critic_features(self, s, info, dyaw_prev3, ctx):
    S, nj = self.S, self.nj
    roll, pitch, _ = self.rpy(s)
    parts = [s[..., S["q"]], s[..., S["qd"]], s[..., S["cmd"]] - s[..., S["q"]], s[..., S["p"]],
             quat_to_rot6d(s[..., S["quat"]]), s[..., S["v"]], s[..., S["w"]],
             s[..., S["q"]] - info[:, :nj], self.world_pos(s) - info[:, nj:nj + 3],
             torch.stack([wrap_to_pi(roll - info[:, nj + 3]), wrap_to_pi(pitch - info[:, nj + 4])], -1), dyaw_prev3]
    if ctx is not None:
      parts.append(ctx)
    return torch.cat(parts, -1)

  def aux_features(self, s2, cmd_next, ctx):
    """Inputs of the learned torque / work penalty head: the servo law's arguments at the end of the step."""
    S = self.S
    parts = [cmd_next - s2[..., S["q"]], s2[..., S["qd"]]]
    if ctx is not None:
      parts.append(ctx)
    return torch.cat(parts, -1)
