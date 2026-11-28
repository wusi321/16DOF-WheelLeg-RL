"""Step 0: environment and model sanity check.

Run with the shared venv:
    <venv>\\Scripts\\python.exe tools\\check_env.py

Checks:
  * Python / torch / CUDA versions
  * the new training MJCF loads and compiles
  * nq=23, nu=16, total mass ~25.1 kg
  * prints joint order / qpos addresses for later tools
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "16DOF-URDF" / "mujoco" / "16DOF-URDF-training.xml"

EXPECTED_NQ = 23
EXPECTED_NU = 16
EXPECTED_MASS = 25.1
MASS_TOL = 0.3


def main() -> int:
    print("=" * 72)
    print("Step 0: environment check")
    print("=" * 72)
    print(f"python      : {sys.version.split()[0]}")
    print(f"executable  : {sys.executable}")
    print(f"project root: {PROJECT_ROOT}")
    print(f"model xml   : {XML_PATH}")
    print()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import torch: {exc}")
        return 1

    print(f"torch       : {torch.__version__}")
    print(f"torch cuda  : {torch.version.cuda}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"gpu memory  : {props.total_memory / (1024 ** 3):.1f} GiB")
    print()

    try:
        import mujoco
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import mujoco: {exc}")
        return 1

    print(f"mujoco      : {mujoco.__version__}")

    if not XML_PATH.is_file():
        print(f"[FAIL] model xml not found: {XML_PATH}")
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] load model: {exc}")
        return 1

    mass = float(model.body_mass.sum())
    print()
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  nkey={model.nkey}")
    print(f"timestep={model.opt.timestep:.4f}  mass={mass:.3f} kg")
    print()

    print("joints (id, name, qposadr, range):")
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        adr = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        print(f"  {jid:2d}  {name:18s} qpos[{adr:2d}]  [{lo:+.4f}, {hi:+.4f}]")

    print()
    print("actuators (id, name, trnid):")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"act_{aid}"
        print(f"  {aid:2d}  {name:18s} joint={int(model.actuator_trnid[aid, 0])}")

    ok = True
    if model.nq != EXPECTED_NQ:
        print(f"[FAIL] nq={model.nq}, expected {EXPECTED_NQ}")
        ok = False
    if model.nu != EXPECTED_NU:
        print(f"[FAIL] nu={model.nu}, expected {EXPECTED_NU}")
        ok = False
    if abs(mass - EXPECTED_MASS) > MASS_TOL:
        print(f"[FAIL] mass={mass:.3f}, expected {EXPECTED_MASS} +/- {MASS_TOL}")
        ok = False
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available")
        ok = False

    print()
    if ok:
        print("[PASS] environment and model are ready.")
        return 0
    print("[FAIL] see messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())