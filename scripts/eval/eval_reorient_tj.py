"""Evaluate a hand-agnostic cube SO(3) reorientation checkpoint (TJ or LEAP).

This is the LEAP-specific ``eval_reorient.py`` without the plant/geometry
perturbations: it loads the play env for the given task, runs the deterministic
PPO actor, and reports goals/min, drops and time-to-goal.  Useful for the TJ
task, where the LEAP-only geometry assertions do not apply.

  MUJOCO_GL=egl uv run --frozen python scripts/eval/eval_reorient_tj.py \
      --task Mjlab-TJ-Right-HandCube-Reorient \
      --ckpt logs/rsl_rl/tj_right_hand_cube_reorient/<run>/model_899.pt \
      --num-envs 512 --steps 1200 --out outputs/eval/tj_so3.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.commands import GOAL
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


class TorchActor:
  """rsl-rl actor: normalizer (eps 1e-2) + ELU MLP + deterministic mean clipped to [-1, 1]."""

  def __init__(self, ckpt: str, device: str):
    sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    self.mean = sd["obs_normalizer._mean"].reshape(-1)
    self.std = sd["obs_normalizer._std"].reshape(-1)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    self.layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in idx]

  @torch.no_grad()
  def __call__(self, x):
    x = (x - self.mean) / (self.std + 1e-2)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x.clamp(-1.0, 1.0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-TJ-Right-HandCube-Reorient")
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--warmup", type=int, default=60)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  B, dev = args.num_envs, env.device
  cmd = env.command_manager.get_term(GOAL)
  actor = TorchActor(args.ckpt, dev)

  ev = {"phase": None, "succ": torch.zeros(B, dtype=torch.bool, device=dev),
        "tout": torch.zeros(B, dtype=torch.bool, device=dev)}
  orig_resample, orig_update, orig_compute = cmd._resample, cmd._update_command, cmd.compute

  def _resample(env_ids):
    if ev["phase"] == "update":
      ev["succ"][env_ids] = True
    elif ev["phase"] == "compute":
      ev["tout"][env_ids] = True
    orig_resample(env_ids)

  def _update():
    ev["phase"] = "update"
    orig_update()
    ev["phase"] = "compute"

  def _compute(dt):
    ev["phase"] = "compute"
    orig_compute(dt)
    ev["phase"] = None

  cmd._resample, cmd._update_command, cmd.compute = _resample, _update, _compute
  tm = env.termination_manager
  tnames = list(tm.active_terms)

  obs, _ = env.reset()
  t0 = time.time()
  n_succ = n_tout = 0
  drops = {n: 0 for n in tnames}
  last_change = torch.zeros(B, dtype=torch.long, device=dev)
  ttg = []
  with torch.no_grad():
    for t in range(args.warmup + args.steps):
      ev["succ"].zero_()
      ev["tout"].zero_()
      obs, _, term_b, tout_b, _ = env.step(actor(obs["actor"]))
      if t < args.warmup:
        last_change[:] = t
        continue
      s, to = ev["succ"].clone(), ev["tout"].clone()
      n_succ += int(s.sum())
      n_tout += int(to.sum())
      for n in tnames:
        if n != "time_out":
          drops[n] += int((tm.get_term(n) & term_b).sum())
      ids = s.nonzero().flatten()
      if len(ids):
        ttg.append(((t - last_change[ids]) * env.step_dt).cpu().numpy())
      last_change[s | to | term_b] = t

  minutes = B * args.steps * env.step_dt / 60.0
  ttg = np.concatenate(ttg) if ttg else np.zeros(0)
  n_drop = sum(drops.values())
  res = dict(task=args.task, ckpt=args.ckpt, num_envs=B, steps=args.steps, seed=args.seed,
             env_minutes=minutes, goals=n_succ, goals_per_min=n_succ / minutes,
             goal_timeouts=n_tout, timeouts_per_min=n_tout / minutes,
             drops=n_drop, drops_by_reason=drops, drops_per_env_hour=n_drop / minutes * 60.0,
             drops_per_100_goals=100.0 * n_drop / max(n_succ, 1),
             time_to_goal_median_s=float(np.median(ttg)) if len(ttg) else None,
             time_to_goal_n=int(len(ttg)), wall_s=time.time() - t0)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=1))
  print("[eval_reorient_tj] " + json.dumps(res))


if __name__ == "__main__":
  main()
