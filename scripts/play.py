"""Play a PPO checkpoint (or a zero / random agent) in a task's play env, in the native viewer, a viser web viewer, or to
an mp4.

  uv run python scripts/play.py Mjlab-Leap-Left-Custom-HandCube-Rotate-Gear --checkpoint-file ckpts/yaw/ppo_gear_yaw.pt
  MUJOCO_GL=egl uv run python scripts/play.py <task> --checkpoint-file <ckpt> --num-envs 1 --video True --viewer none

World-model actors (``actor_*.pt``) use their own observation adapters: see scripts/render/ and scripts/eval/.
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from in_hand_rotation_mjlab.utils.plant import check_checkpoint_plant
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class CleanNativeViewer(NativeMujocoViewer):
  """Native viewer with MuJoCo's joint/inertia/COM/mocap debug frames disabled."""

  def setup(self) -> None:
    sim = self.env.unwrapped.sim
    # MuJoCo's native viewer draws mocap bodies as large RGB frames.  With one
    # environment the robot base is at the world origin, so make the viewer's
    # copy of the mocap base a fixed body; this hides the frame without moving
    # the hand.  Multi-env viewer playback keeps the mocap frames.
    if self.env.unwrapped.num_envs == 1:
      mjm = sim.mj_model
      for body_id in range(mjm.nbody):
        if mjm.body_mocapid[body_id] >= 0:
          mjm.body_mocapid[body_id] = -1

    super().setup()
    if self.viewer is None:
      return
    flags = self.viewer.opt.flags
    for name, flag in mujoco.mjtVisFlag.__members__.items():
      if int(flag) < len(flags):
        flags[flag] = 1 if name == "mjVIS_TRANSPARENT" else 0


class GoalCounterNativeViewer(CleanNativeViewer):
  """Native viewer with a cumulative "Goals reached" line for SO(3) tasks."""

  def _set_status_overlay(self, viewer) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    try:
      cmd = self.env.unwrapped.command_manager.get_term("goal")
      goals = int(getattr(cmd, "_total_successes", 0))
    except Exception:
      goals = 0
    text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\nGoals"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n"
      f"{goals}"
    )
    viewer.set_texts(
      (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        text_1,
        text_2,
      )
    )


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  """"none": no viewer; with --video True, record the clip and exit (headless machines)."""


class TaskCenteredViserPlayViewer(ViserPlayViewer):
  """Viser viewer that honors the task viewer body as its tracked center."""

  def setup(self) -> None:
    super().setup()
    self._center_on_viewer_body()

  def _center_on_viewer_body(self) -> None:
    viewer_cfg = self.cfg
    scene = getattr(self, "_scene", None)
    server = getattr(self, "_server", None)
    if scene is None or server is None:
      return

    body_id = -1
    if getattr(viewer_cfg, "entity_name", None) and getattr(viewer_cfg, "body_name", None):
      model_body_name = f"{viewer_cfg.entity_name}/{viewer_cfg.body_name}"
      body_id = mujoco.mj_name2id(
        scene.mj_model, mujoco.mjtObj.mjOBJ_BODY, model_body_name
      )
      if body_id < 0:
        body_id = mujoco.mj_name2id(
          scene.mj_model, mujoco.mjtObj.mjOBJ_BODY, viewer_cfg.body_name
        )

    if body_id >= 0:
      scene._tracked_body_id = body_id
      scene.request_update()

    azimuth = np.deg2rad(float(viewer_cfg.azimuth))
    elevation = np.deg2rad(float(viewer_cfg.elevation))
    distance = float(viewer_cfg.distance)
    camera_offset = np.array(
      [
        -np.cos(elevation) * np.cos(azimuth),
        -np.cos(elevation) * np.sin(azimuth),
        # ViewerConfig follows MuJoCo's convention: negative elevation means
        # above the look-at point. Viser uses an explicit Cartesian offset.
        -np.sin(elevation),
      ]
    ) * distance

    def apply_camera(client) -> None:
      client.camera.position = camera_offset
      client.camera.look_at = np.zeros(3)

    for client in server.get_clients().values():
      apply_camera(client)

    @server.on_client_connect
    def _(client) -> None:
      apply_camera(client)


def _prepare_agent_cfg(agent_cfg) -> dict:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      class_name = model_cfg.get("class_name", "MLPModel")
      if class_name != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
      if class_name != "RNNModel":
        model_cfg.pop("rnn_type", None)
        model_cfg.pop("rnn_hidden_dim", None)
        model_cfg.pop("rnn_num_layers", None)
  return cfg_dict


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    del log_root_path
    if cfg.checkpoint_file is None:
      raise ValueError("--checkpoint-file is required for the trained agent (or use --agent zero / random)")
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    # a training run's directory keeps its videos; a loose checkpoint (e.g. ckpts/) writes under outputs/videos/<name>
    log_dir = resume_path.parent if (resume_path.parent / "params").is_dir() else Path("outputs/videos") / resume_path.stem

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    elif cfg.agent == "random":

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
    else:
      raise RuntimeError(f"Unsupported dummy agent type: {cfg.agent}")
  else:
    ck = torch.load(resume_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and ck.get("kind") == "wm_actor":
      if "Reorient" in task_id or "Fingertip" in task_id:
        raise ValueError(
          "play.py can load world-model actors only for cube-yaw tasks; "
          "use eval_ftrot.py / render_ftrot.py for fingertip and eval_reorient.py for SO(3)."
        )
      from in_hand_rotation_mjlab.world_model.policy import Actor

      actor = Actor.from_checkpoint(ck).to(device).eval()
      policy = lambda obs: actor(obs["actor"] if not torch.is_tensor(obs) else obs)  # noqa: E731
    else:
      runner_cls = load_runner_cls(task_id) or OnPolicyRunner
      runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=device)
      # the run recorded its training cfg, so compare against the task's training plant
      check_checkpoint_plant(resume_path, load_env_cfg(task_id), task=task_id)
      runner.load(str(resume_path), map_location=device)
      policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "none":
    obs = env.get_observations()
    with torch.no_grad():
      for _ in range(cfg.video_length + 1):
        obs, _, _, _ = env.step(policy(obs))
    print(f"[INFO]: Ran {cfg.video_length} steps without a viewer" + (f"; video in {log_dir / 'videos' / 'play'}" if cfg.video and log_dir else ""))
  elif resolved_viewer == "native":
    viewer_cls = GoalCounterNativeViewer if "Reorient" in task_id else CleanNativeViewer
    viewer_cls(env, policy).run()
  elif resolved_viewer == "viser":
    TaskCenteredViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = [t for t in list_tasks() if t.startswith(("Mjlab-Leap", "Mjlab-TJ"))]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
