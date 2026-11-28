"""go2w naming, geometry and pose contract for the ported 16-DOF task.

The original 16-DOF wheel-leg task shipped with a bespoke MJCF whose names were
``fl_hip_abduction_joint`` / ``fl_hip_pitch_joint`` / ``fl_knee_joint`` /
``fl_wheel_joint`` and bodies named ``fl_wheel_Link``.  This port drives the
stock Unitree go2w model (``go2w/go2w.xml``) instead, which uses

    joints : FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL_wheel_joint
    bodies : base_link, FL_hip, FL_thigh, FL_calf, FL_wheel_link

Every model-dependent name, pose and geometric constant lives in this module so
that swapping the robot again only touches ``naming.py`` + ``robot_cfg.py``.

The numbers below were measured from ``go2w/go2w.xml`` (see docs/PORTING.md).
"""

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
BASE_BODY = "base_link"

#: Joint order inside the MJCF (= go2w motor/leg order), each leg has 4 joints.
LEG_ORDER = ("FL", "FR", "RL", "RR")

#: Abduction joint (roll about x).
HIP_JOINT_PATTERNS = (".*_hip_joint",)
#: Hip pitch joint (about y).
THIGH_JOINT_PATTERNS = (".*_thigh_joint",)
#: Knee joint (about y).
CALF_JOINT_PATTERNS = (".*_calf_joint",)
#: Wheel joint (about y, continuous).
WHEEL_JOINT_PATTERNS = (".*_wheel_joint",)

LEG_JOINT_PATTERNS = (
    ".*_hip_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
)
SAGITTAL_JOINT_PATTERNS = (".*_thigh_joint", ".*_calf_joint")

#: Bodies that must not touch the terrain (torso / legs).
BODY_COLLISION_PATTERNS = (".*_hip", ".*_thigh", ".*_calf")

#: Wheel bodies in joint order (FL, FR, RL, RR).
WHEEL_BODY_PATTERNS = tuple(f"{leg}_wheel_link" for leg in LEG_ORDER)

#: Diagonal mirrors used by the gait-shaping rewards: FL<->RR, FR<->RL.
MIRROR_SAGITTAL_PAIRS = (
    ("FL_(thigh|calf)_joint", "RR_(thigh|calf)_joint"),
    ("FR_(thigh|calf)_joint", "RL_(thigh|calf)_joint"),
)
#: Same-side front/back abduction mirrors: FL<->RL, FR<->RR.
MIRROR_ABDUCTION_PAIRS = (
    ("FL_hip_joint", "RL_hip_joint"),
    ("FR_hip_joint", "RR_hip_joint"),
)

# ---------------------------------------------------------------------------
# Geometry and nominal poses measured from the model
# ---------------------------------------------------------------------------
#: Wheel radius, measured as the vertical distance from wheel centre to the
#: lowest wheel-mesh vertex in the nominal stance.
WHEEL_RADIUS = 0.086
#: Distance between the left and right wheel contact patches (|y| = 0.1924).
WHEEL_TRACK = 0.385

#: Nominal stance used for the flat / rough tasks (thigh 0.9, knee -1.8).
STAND_JOINT_POS = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
#: Base height for that stance (forward kinematics gives 0.359 m).
STAND_BASE_HEIGHT = 0.36

#: Crouched stance used by the crawl task (fits go2w knee range -2.72..-0.84).
CRAWL_JOINT_POS = {"hip": 0.4, "thigh": 1.65, "calf": -2.55}
CRAWL_BASE_HEIGHT = 0.20
#: Reward target for ``crawl_height_reward``.
CRAWL_TARGET_HEIGHT = 0.22

# ---------------------------------------------------------------------------
# Wheel command signs
# ---------------------------------------------------------------------------
#: Sign that converts a commanded forward speed into a wheel joint velocity.
#: All four go2w wheel joints spin about +y, so ``+v / R`` drives the robot
#: forward for every leg.  For a yaw rate ``wz`` the left wheels (FL, RL, y > 0)
#: must run slower than the right wheels, hence the side factor below.
WHEEL_SIDE_SIGNS = (-1.0, 1.0, -1.0, 1.0)  # FL, FR, RL, RR

# ---------------------------------------------------------------------------
# Actuator limits (from go2w.xml <motor ctrlrange=...>)
# ---------------------------------------------------------------------------
EFFORT_HIP = 23.7
EFFORT_THIGH = 23.7
EFFORT_CALF = 45.43
EFFORT_WHEEL = 15.0
