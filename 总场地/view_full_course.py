"""预览完整赛场（14m x 6m，8 个障碍按官方布局拼装）。

直接加载 build_full_course.py 生成的 full_course_2027.xml，不需要重新生成。

用法：
    python view_full_course.py            # 打开交互窗口
    python view_full_course.py --no-window  # 只加载并打印统计，不开窗口
    python view_full_course.py --rebuild    # 先重新生成再预览
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
XML = HERE / "full_course_2027.xml"


def summary(model: mujoco.MjModel) -> None:
    print("=" * 60)
    print("总场地模型已加载")
    print("=" * 60)
    print("  geom   = %d" % model.ngeom)
    print("  body   = %d" % model.nbody)
    print("  场地   = %.1f x %.1f m" % (14.0, 6.0))
    print("  障碍   = 8 个（单木桥/高墙/梅花桩/限高框架/沟壑组合/绕桩/轮胎阵/砖木坑）")

    # 统计各类颜色，粗略确认 8 个障碍都在
    from collections import Counter

    def name(rgba):
        table = {
            (0.13, 0.55, 0.24): "绿色构件",
            (0.85, 0.80, 0.30): "场地黄",
            (0.90, 0.10, 0.10): "红色标记",
            (0.10, 0.10, 0.12): "轮胎黑",
            (0.72, 0.74, 0.78): "钢条",
            (0.62, 0.42, 0.36): "碎砖",
            (0.45, 0.33, 0.20): "松树皮",
            (0.18, 0.18, 0.20): "地面",
        }
        key = tuple(round(float(c), 2) for c in rgba[:3])
        return table.get(key, str(key))

    cnt = Counter(name(model.geom_rgba[g]) for g in range(model.ngeom))
    print()
    print("  颜色分布（前 8 项）：")
    for k, v in cnt.most_common(8):
        print("    %-10s %d" % (k, v))


def main() -> int:
    args = sys.argv[1:]
    do_rebuild = "--rebuild" in args
    no_window = "--no-window" in args

    if do_rebuild or not XML.exists():
        print("重新生成模型 ...")
        import importlib.util

        spec_path = HERE / "build_full_course.py"
        spec = importlib.util.spec_from_file_location("build_full_course", spec_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()
        print()

    if not XML.exists():
        print("[ERROR] 找不到 %s" % XML)
        return 1

    # 注意：MuJoCo 的 from_xml_path 打不开含中文的路径（本目录名是「总场地」），
    # 所以先读成字符串再解析。
    model = mujoco.MjModel.from_xml_string(XML.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    summary(model)

    if no_window:
        print()
        print("（--no-window：不打开窗口）")
        return 0

    print()
    print("打开预览窗口…… （关闭窗口即退出）")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 俯视整个场地
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.cam.distance = 13.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -55.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    return 0


if __name__ == "__main__":
    sys.exit(main())
