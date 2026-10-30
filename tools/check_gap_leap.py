"""回归检查：跃沟奖励必须"真的跳过去"才给，原地小跳不能被刷。

    uv run python tools/check_gap_leap.py

做法：把 `wheel_contact_any` 换成脚本化的接触序列（真实仿真里 sensor 只在
step 后更新，瞬移测不出来），再按脚本给 x 轨迹，验证：

1. 原地起跳、落回原处（前进 0）      -> 0 次
2. 原地小跳、前进 0.10 m（< 0.25）   -> 0 次
3. 起跳后前进 0.35 m 落地（真跃沟）  -> 1 次
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import robot.mdp.gap_course as gc


def run_case(env, x_seq: list[float], contact_seq: list[bool]) -> float:
    """按脚本喂 (x, contact)，返回最后的 leaps。"""
    env.reset()
    env.episode_length_buf[:] = 0
    xs = iter(x_seq)
    cs = iter(contact_seq)

    def fake_contact(_env, sensor_name: str = "feet_ground_contact"):  # noqa: ARG001
        return torch.tensor([next(cs)], dtype=torch.bool, device=_env.device)

    gc.wheel_contact_any = fake_contact
    robot = env.scene["robot"]
    last = 0.0
    for buf, (x, _c) in enumerate(zip(x_seq, contact_seq), start=1):
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, 3] = 1.0
        pose[:, 0] = env.scene.env_origins[:, 0] + x
        pose[:, 1] = env.scene.env_origins[:, 1]
        pose[:, 2] = env.scene.env_origins[:, 2] + 0.36
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        env.episode_length_buf[:] = buf
        last = float(gc.gaps_crossed(env).max())
        xs.__next__()
    return last


def main() -> int:
    cfg = load_env_cfg("Robot-GapCourse-v0")
    cfg.scene.num_envs = 1
    cfg.scene.terrain.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()

    cases = [
        ("原地跳（x 不变）", [1.0] * 5, [True, False, False, True, True], 0.0),
        ("小跳 + 0.10 m", [1.0, 1.0, 1.0, 1.05, 1.10], [True, False, False, False, True], 0.0),
        ("真跃沟 + 0.35 m", [1.0, 1.0, 1.0, 1.2, 1.35], [True, False, False, False, True], 1.0),
    ]
    ok = True
    for name, xs, cs, expect in cases:
        got = run_case(env, xs, cs)
        flag = "OK" if abs(got - expect) < 1e-6 else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"{name:16s} -> leaps={got:.0f} (expect {expect:.0f}) [{flag}]")

    env.close()
    print("OK: 跃沟判定不会被原地小跳刷" if ok else "FAILED: 跃沟判定仍有漏洞")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
