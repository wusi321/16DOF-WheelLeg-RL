"""go2w robot assembly for the ported 16-DOF wheel-leg RL task.

The physics model is the stock Unitree go2w MJCF (``go2w/go2w.xml``, copied
unchanged into this project).  Only the articulation layer lives here:

* which joints are position-controlled (hip / thigh / calf) and which are
  velocity-controlled (wheel),
* PD gains, effort limits and action scaling,
* the nominal stance and the crouched crawl stance.

Compared with the original 16-DOF task, the robot-specific constants were
replaced by values measured from the go2w model, see :mod:`robot.naming`.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from . import naming

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = _PROJECT_ROOT / "go2w" / "go2w.xml"

# ---------------------------------------------------------------------------
# Control gains (ported unchanged from the 16-DOF task; tune for go2w if the
# policy turns out too soft/stiff on hardware).
# ---------------------------------------------------------------------------
STIFFNESS_LEG = 50.0
DAMPING_LEG = 1.5
DAMPING_WHEEL = 1.0

#: Maximum joint speed of the original actuators (kept for reference; the
#: builtin actuators created below do not clamp velocity).
MAX_JOINT_VEL = 13.0


@functools.lru_cache(maxsize=1)
def _mesh_assets() -> dict[str, bytes]:
    """Read every mesh/texture next to the MJCF into memory.

    MuJoCo's file loader cannot open non-ASCII paths on Windows, and this
    project may live under a Chinese directory name.  Loading the XML text and
    the assets ourselves keeps the model independent of the install path.
    """
    assets: dict[str, bytes] = {}
    root = ROBOT_XML.parent
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".xml":
            continue
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        assets[rel] = data  # e.g. "assets/base_0.obj"
        assets[path.name] = data  # e.g. "base_0.obj"
    return assets


def get_spec() -> mujoco.MjSpec:
    """Build the go2w spec, with the MJCF actuators removed.

    The actuators declared in ``go2w.xml`` are plain torque motors that keep the
    ``ctrlrange`` limits from Unitree.  They are dropped here because the task
    drives the robot through the ``BuiltinPosition`` / ``BuiltinVelocity``
    actuator groups configured below (same approach as the original task).

    Two bookkeeping fixes are applied to the spec:

    * the wheel *visual* meshes are moved from geom group 0 to the visual group
      2.  mjlab's generated terrain also lives in group 0 and the height scanner
      only raycasts group 0, so leaving the wheels there would make the critic's
      height scan hit the robot's own wheels.  Physics is unaffected (the
      collision twins in group 3 keep their ``contype``/``conaffinity``);
    * the ``margin=0.001`` that go2w sets on every geom is cleared.  mujoco-warp
      enables MULTICCD (``ccd_iterations=80`` in the task's ``SimulationCfg``)
      and refuses geom pairs with a non-zero margin; the contact margin is not
      part of the ported control/reward design;
    * the MJCF motors are removed (see above).
    """
    xml = ROBOT_XML.read_text(encoding="utf-8")
    spec = mujoco.MjSpec.from_string(xml, assets=dict(_mesh_assets()))
    for geom in spec.geoms:
        geom.margin = 0.0
    for body in spec.bodies:
        if body.name and body.name.endswith("_wheel_link"):
            for geom in body.geoms:
                if geom.group == 0:
                    geom.group = 2
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------
#: Hip abduction and hip pitch share the 23.7 Nm motor of the go2w model.
HIP_THIGH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=naming.HIP_JOINT_PATTERNS + naming.THIGH_JOINT_PATTERNS,
    stiffness=STIFFNESS_LEG,
    damping=DAMPING_LEG,
    effort_limit=naming.EFFORT_HIP,
)

#: The knee uses the stronger 45.43 Nm motor.
CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=naming.CALF_JOINT_PATTERNS,
    stiffness=STIFFNESS_LEG,
    damping=DAMPING_LEG,
    effort_limit=naming.EFFORT_CALF,
)

#: Wheels are velocity controlled (15 Nm limit from the model).
WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
    target_names_expr=naming.WHEEL_JOINT_PATTERNS,
    damping=DAMPING_WHEEL,
    effort_limit=naming.EFFORT_WHEEL,
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(HIP_THIGH_ACTUATOR_CFG, CALF_ACTUATOR_CFG, WHEEL_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)


# ---------------------------------------------------------------------------
# Nominal states
# ---------------------------------------------------------------------------
def _joint_pos(hip: float, thigh: float, calf: float) -> dict[str, float]:
    """Build a joint-position spec from per-joint-type values."""
    return {
        ".*_hip_joint": hip,
        ".*_thigh_joint": thigh,
        ".*_calf_joint": calf,
        ".*_wheel_joint": 0.0,
    }


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, naming.STAND_BASE_HEIGHT),
    joint_pos=_joint_pos(**naming.STAND_JOINT_POS),
    joint_vel={".*": 0.0},
)

CRAWL_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, naming.CRAWL_BASE_HEIGHT),
    joint_pos={
        "(FL|RL)_hip_joint": naming.CRAWL_JOINT_POS["hip"],
        "(FR|RR)_hip_joint": -naming.CRAWL_JOINT_POS["hip"],
        ".*_thigh_joint": naming.CRAWL_JOINT_POS["thigh"],
        ".*_calf_joint": naming.CRAWL_JOINT_POS["calf"],
        ".*_wheel_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

# The go2w MJCF already carries Unitree's collision setup: torso / leg boxes and
# cylinders on group 3, wheel meshes with condim=6, priority=1 and
# friction=(0.8, 0.02, 0.01).  Nothing is overridden here.
#
# The original 16-DOF task instead disabled robot self-collision globally with
# ``CollisionCfg(geom_names_expr=(".*",), contype=0, conaffinity=1, ...)``.
# If self-collisions turn out to slow training down, pass that cfg in
# ``collisions=(...)`` below (its condim/friction entries can be dropped
# because go2w already defines them on the wheel geoms).


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )


def get_robot_crawl_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=CRAWL_INIT_STATE,
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )


# Action scale: leg position target ±0.25 rad (±0.125 on abduction),
# wheel target velocity ±5.0 rad/s.
LEG_POS_SCALE = 0.25
WHEEL_VEL_SCALE = 5.0
