"""TJ Hand 1 Right constants and EntityCfg for mjlab.

This module binds the 21-DoF TJDH21R right hand to mjlab.

The TJ MJCF defines all 21 position actuators and keeps the actuator-side
joint limits, control ranges, gains, and force limits in the XML. This module
therefore does not recreate the PD parameters. XmlPositionActuatorCfg only
selects the XML-defined position actuators.

Robot structure:
    Thumb:  CMC1, CMC2, CMC3, MCPT, DIPT
    Index:  MCPI2, MCPI1, PIPI, DIPI
    Middle: MCPM2, MCPM1, PIPM, DIPM
    Ring:   MCPR2, MCPR1, PIPR, DIPR
    Pinky:  MCPS2, MCPS1, PIPS, DIPS

Collision design follows the LEAP/Wuji project pattern while matching the
current TJ MJCF:
    - visual STL geoms are non-colliding;
    - 21 link collision proxies are named *_collision;
    - 5 fingertip spheres are named tip_*_collision;
    - ordinary collision friction is 0.2;
    - fingertip friction is 0.5;
    - fingertip contacts use condim=6, ordinary contacts use condim=3;
    - the palm has no collision geom in this in-hand-rotation model.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import update_assets
from mjlab.actuator import XmlActuatorCfg as XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


# =============================================================================
# MJCF + assets
# =============================================================================

TJ_RIGHT_HAND_XML: Path = MYMJLAB_SRC_PATH / "robots" / "tj_hand" / "xmls" / "right.xml"
assert TJ_RIGHT_HAND_XML.exists(), f"Missing TJ right-hand MJCF: {TJ_RIGHT_HAND_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Embed TJ mesh assets into the MjSpec asset table."""
    assets: dict[str, bytes] = {}
    update_assets(assets, TJ_RIGHT_HAND_XML.parent / "assets", meshdir)
    return assets


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> None:
    """Give deterministic names to any unexpected unnamed geoms."""
    counter: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.name:
            continue
        body_name = geom.parent.name if geom.parent else "world"
        is_visual = geom.contype == 0 and geom.conaffinity == 0
        suffix = "visual" if is_visual else "collision"
        key = f"{body_name}_{suffix}"
        idx = counter.get(key, 0)
        counter[key] = idx + 1
        geom.name = f"{key}_{idx}" if idx else key


def get_spec() -> mujoco.MjSpec:
    """Load the TJ MJCF and attach its mesh assets without rebuilding actuators."""
    spec = mujoco.MjSpec.from_file(str(TJ_RIGHT_HAND_XML))
    _name_unnamed_geoms(spec)
    spec.assets = get_assets(spec.meshdir)
    return spec


# =============================================================================
# Collision configuration
# =============================================================================

TJ_ALL_COLLISION_GEOMS = r".*_collision.*$"
TJ_FINGERTIP_GEOMS = r"tip_(?:thumb|index|middle|ring|pinky)_collision$"
TJ_PALM_GEOMS = r"Palm_R_collision$"

TJ_COLLISION = CollisionCfg(
    geom_names_expr=(TJ_ALL_COLLISION_GEOMS,),
    condim={
        TJ_FINGERTIP_GEOMS: 6,
        TJ_PALM_GEOMS: 3,
        TJ_ALL_COLLISION_GEOMS: 3,
    },
    friction={
        TJ_FINGERTIP_GEOMS: (1.0, 5e-3, 5e-4),
        TJ_PALM_GEOMS: (0.35,),
        TJ_ALL_COLLISION_GEOMS: (0.6,),
    },
    solref={
        TJ_ALL_COLLISION_GEOMS: (0.01, 1.0),
    },
    priority={
        TJ_FINGERTIP_GEOMS: 1,
    },
)


# =============================================================================
# Initial state
# =============================================================================

# Neutral diagnostic home pose. Replace these values later with the validated
# TJ grasp home pose used by the cube grasp-cache collector.
TJ_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.08)
TJ_HOME_ROT: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0)

TJ_JOINT_ORDER: tuple[str, ...] = (
    "CMC1", "CMC2", "CMC3", "MCPT", "DIPT",
    "MCPI2", "MCPI1", "PIPI", "DIPI",
    "MCPM2", "MCPM1", "PIPM", "DIPM",
    "MCPR2", "MCPR1", "PIPR", "DIPR",
    "MCPS2", "MCPS1", "PIPS", "DIPS",
)

TJ_HOME_JOINT_POS_1 = {
    joint: 0.0
    for joint in TJ_JOINT_ORDER
}

TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -1.45,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -0.7,     # 原 -1.12,更大对掌
    "DIPT":  -0.65,     # 原 -0.752,拇指伸直一点抬高指尖

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,     # 原 0.05
    "MCPI1": -1.1,     # 原 -0.8
    "PIPI":  -0.9,     # 原 -1.25
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,     # 原 0.0616
    "MCPM1": -0.80,     # 原 -0.509
    "PIPM":  -1.0,     # 原 -1.34
    "DIPM":  -0.6,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,     # 原 -0.0621
    "MCPR1": -0.81,
    "PIPR":  -0.7,
    "DIPR":  -0.611,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.15,
    "MCPS1": -1.30,  # -1.30
    "PIPS":  -0.82,
    "DIPS":  -0.611,
}


def make_home_state(pos: tuple[float, float, float] = TJ_HOME_POS, rot: tuple[float, float, float, float] = TJ_HOME_ROT) -> EntityCfg.InitialStateCfg:
    """Return a fresh TJ initial-state configuration."""
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(TJ_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


# =============================================================================
# XML actuator configuration
# =============================================================================

# All 21 actuators are already present in right.xml. Keep their XML kp/kv,
# ctrlrange, and forcerange and let mjlab reference those actuators by joint.
TJ_XML_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=(
        r"(?:CMC[123]|MCPT|DIPT|MCPI[12]|PIPI|DIPI|MCPM[12]|PIPM|DIPM|MCPR[12]|PIPR|DIPR|MCPS[12]|PIPS|DIPS)",
    ),
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(TJ_XML_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


# =============================================================================
# Final EntityCfg
# =============================================================================

def get_tj_right_hand_cfg() -> EntityCfg:
    """Return the mjlab EntityCfg for the TJDH21R right hand."""
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(TJ_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# Backward-friendly alias for places that use a shorter robot factory name.
get_tj_hand_cfg = get_tj_right_hand_cfg


# =============================================================================
# Validation helpers
# =============================================================================

def validate_tj_mjcf() -> dict[str, object]:
    """Perform static XML validation independent of the installed mjlab version."""
    import xml.etree.ElementTree as ET

    root = ET.parse(TJ_RIGHT_HAND_XML).getroot()
    assert root.tag == "mujoco", f"Unexpected root tag: {root.tag}"
    assert root.attrib.get("model") == "TJDH21R", f"Unexpected model: {root.attrib.get('model')}"

    joints = [element.attrib["name"] for element in root.findall(".//joint") if "name" in element.attrib]
    actuators = [element.attrib["joint"] for element in root.findall("./actuator/position") if "joint" in element.attrib]
    actuator_names = [element.attrib["name"] for element in root.findall("./actuator/position") if "name" in element.attrib]
    geoms = [element for element in root.findall(".//geom") if "name" in element.attrib]
    collision_geoms = [g for g in geoms if "_collision" in g.attrib["name"]]
    visual_geoms = [g for g in geoms if g.attrib["name"].endswith("_visual")]
    fingertips = [g for g in collision_geoms if g.attrib["name"].startswith("tip_")]

    expected_joints = list(TJ_JOINT_ORDER)
    assert len(joints) == 21, f"Expected 21 joints, found {len(joints)}"
    assert joints == expected_joints, f"Joint order mismatch: {joints}"
    assert len(actuators) == 21, f"Expected 21 position actuators, found {len(actuators)}"
    assert set(actuators) == set(expected_joints), "Actuator joint mapping is not 1-to-1 with the 21 TJ joints"
    assert len(actuator_names) == 21, f"Expected 21 actuator names, found {len(actuator_names)}"
    assert len(collision_geoms) == 22, f"Expected 22 collision geoms, found {len(collision_geoms)}"
    assert len(visual_geoms) == 22, f"Expected 22 visual geoms, found {len(visual_geoms)}"
    assert len(fingertips) == 5, f"Expected 5 fingertip collision geoms, found {len(fingertips)}"
    assert all(g.attrib.get("contype") == "0" and g.attrib.get("conaffinity") == "0" for g in visual_geoms), "A visual geom is collision-enabled"
    assert all(g.attrib.get("class") in {"collision", "tip"} for g in collision_geoms), "Unexpected collision geom class"
    assert all("_collision" in g.attrib["name"] for g in collision_geoms), "Not all collision geoms contain _collision"

    # The palm uses a single mesh collision geom (wuji-style).
    palm_collision = [g for g in collision_geoms if g.attrib["name"].startswith("Palm_R")]
    assert len(palm_collision) == 1, f"Expected 1 palm collision geom, found {len(palm_collision)}"

    return {
        "model": root.attrib["model"],
        "n_joints": len(joints),
        "n_actuators": len(actuators),
        "n_visual_geoms": len(visual_geoms),
        "n_collision_geoms": len(collision_geoms),
        "n_fingertip_geoms": len(fingertips),
        "actuator_mapping_ok": set(actuators) == set(expected_joints),
        "collision_names_ok": all("_collision" in g.attrib["name"] for g in collision_geoms),
        "palm_collision_geoms": len(palm_collision),
    }


def validate_compiled_tj_model(model: mujoco.MjModel) -> dict[str, object]:
    """Validate the compiled MuJoCo model; run this inside the project uv environment."""
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_joint_names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "")

    collision_names: list[str] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) and int(model.geom_conaffinity[geom_id]):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            collision_names.append(name or "")

    assert model.njnt == 21, f"Compiled model has {model.njnt} joints, expected 21"
    assert model.nu == 21, f"Compiled model has {model.nu} actuators, expected 21"
    assert set(actuator_joint_names) == set(TJ_JOINT_ORDER), "Compiled actuator-to-joint mapping is incorrect"
    assert collision_names and all(name.endswith("_collision") for name in collision_names), "Compiled collision geom names are invalid"

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "n_collision_geoms": len(collision_names),
        "joint_order_ok": joint_names == list(TJ_JOINT_ORDER),
        "actuator_mapping_ok": set(actuator_joint_names) == set(TJ_JOINT_ORDER),
        "collision_names_ok": all(name.endswith("_collision") for name in collision_names),
    }


if __name__ == "__main__":
    result = validate_tj_mjcf()
    print("TJ MJCF static validation: PASS")
    for key, value in result.items():
        print(f"  {key}: {value}")