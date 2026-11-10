"""Convert an older RSL-RL checkpoint to the current GaussianDistribution key layout.

Older checkpoints store the policy's scalar action std as ``actor_state_dict["std"]``;
the RSL-RL version pinned by this project expects
``actor_state_dict["distribution.std_param"]``.  Only that key is renamed, so the
converted checkpoint can be loaded by ``scripts/play.py`` / ``scripts/eval/eval_yaw.py``.

  uv run --frozen python scripts/convert_legacy_checkpoint.py \
      /path/to/tj_right_model_31497.pt ckpts/tj/ppo_tj_right_yaw.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(state: dict) -> bool:
  actor = state.get("actor_state_dict")
  if not isinstance(actor, dict):
    return False
  if "std" in actor and "distribution.std_param" not in actor:
    actor["distribution.std_param"] = actor.pop("std")
    return True
  return False


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("src", help="legacy checkpoint (.pt)")
  ap.add_argument("dst", help="output checkpoint (.pt)")
  args = ap.parse_args()
  state = torch.load(args.src, map_location="cpu", weights_only=False)
  changed = convert(state)
  out = Path(args.dst)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, out)
  print(("[converted] " if changed else "[no std key; copied] ") + str(out))


if __name__ == "__main__":
  main()
