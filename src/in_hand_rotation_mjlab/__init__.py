"""in_hand_rotation_mjlab: LEAP-hand in-hand rotation tasks (yaw, SO(3), palm-down fingertip rotation) on mjlab, trained
with PPO or with a learned world model (SuperMimic-style)."""

from pathlib import Path

PACKAGE_PATH = Path(__file__).parent
"""Path to the installed in_hand_rotation_mjlab package directory (src/in_hand_rotation_mjlab)."""

# Compatibility alias used by the TJ-hand constants ported from the TJ branch.
MYMJLAB_SRC_PATH = PACKAGE_PATH
