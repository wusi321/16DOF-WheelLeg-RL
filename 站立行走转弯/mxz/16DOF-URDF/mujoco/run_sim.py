from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "16DOF-URDF-sim.xml"
LOG_PATH = HERE / "mujoco_sim.log"


def log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def main() -> None:
    log(f"Loading {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    summary = (
        f"nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, "
        f"mass={float(model.body_mass.sum()):.3f} kg, time={data.time:.3f}s"
    )
    if "--check" in sys.argv:
        log("Check OK: " + summary)
        return

    log("Open viewer: " + summary)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.30]
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.sync()

        wall_start = time.perf_counter()
        sim_start = data.time
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = (data.time - sim_start) - (time.perf_counter() - wall_start)
            if delay > 0:
                time.sleep(delay)

    log("Viewer closed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        raise
