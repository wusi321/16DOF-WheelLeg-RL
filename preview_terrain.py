"""场地预览：打开一个 3D 窗口，直接看这个障碍场地长什么样。

不用训练、不用显卡、不用加载机器人——只是把地形的几何体单独生成出来看一眼。

用法（在 16dof-rl 目录下）：

    # 列出所有能看的场地
    python preview_terrain.py

    # 看某一个场地（默认看比赛难度 1.0）
    python preview_terrain.py 梅花桩
    python preview_terrain.py 单木桥
    python preview_terrain.py 高墙 --level 0.5

    # 看某个场地从易到难的四个难度，一排摆开对比
    python preview_terrain.py 轮胎阵 --levels 4

窗口操作（MuJoCo 查看器）：
    左键拖动 = 旋转视角      右键拖动 = 平移
    滚轮 = 拉近拉远          双击 = 聚焦到那个点
    Esc = 关闭

环境变量：PYTHONPATH 要包含项目的 src 和 mjlab/src。
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 基础场景 XML：灯光、默认视角、焦点全部在这里定。
#
# 为什么不用 Python API 建灯：这个 MuJoCo 版本的 MjSpec light 没有可写的
# `mode` 属性（设 mjtLightType 会抛 TypeError），只能用 XML 声明。
# launch() 也不接受 camera 参数，所以视角靠这里的 <global> 和 <statistic> 定。
# ---------------------------------------------------------------------------
_BASE_XML = """<mujoco model="venue_preview">
  <statistic center="{cx:.3f} {cy:.3f} {cz:.3f}" extent="{extent:.2f}"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.15 0.15 0.15"/>
    <global azimuth="132" elevation="-34"/>
  </visual>
  <worldbody>
    <light pos="4 0 9" dir="0 0 -1" directional="true" diffuse="1 1 1"/>
    <light pos="0 0 6" dir="0.4 0.5 -1" directional="true" diffuse="0.45 0.45 0.55"/>
    <body name="terrain"/>
  </worldbody>
</mujoco>"""


def _venue_table():
    """场地登记表：名字 -> (地形类, 构造参数, 中文说明)。

    参数全部来自赛事官方规则《2027 年足式机器人挑战赛 V1.0》
    和官方场地模型 Robocon2027.xml。
    """
    from robot.terrains import (
        BoxSteppingStonesTerrainCfg,
        BoxGapCourseTerrainCfg,
        BoxSlalomTerrainCfg,
        BoxTireArrayTerrainCfg,
        BoxSingleBeamTerrainCfg,
        BoxBrickPitTerrainCfg,
        RCWallTerrainCfg,
        RCLowBarTerrainCfg,
    )

    return {
        "梅花桩": (
            BoxSteppingStonesTerrainCfg,
            dict(size=(8.0, 1.2)),
            "200×200×200 方块交错，排间净距 200mm、同排净距 400mm，两侧 800×800×200 平台",
        ),
        "单木桥": (
            BoxSingleBeamTerrainCfg,
            dict(size=(8.0, 1.6), beam_length=3.0),
            "官方 3000×400×100mm 窄木面，两侧无护栏，要求持续走直线",
        ),
        "高墙": (
            RCWallTerrainCfg,
            dict(size=(8.0, 1.6), wall_centers_x=(2.1, 3.2, 4.3, 5.4, 6.5),
                 wall_height_range=(0.10, 0.30)),
            "官方 高 300mm、宽 800mm、厚 80mm，连续 5 道",
        ),
        "限高框架": (
            RCLowBarTerrainCfg,
            dict(size=(8.0, 1.6), clearance_range=(0.29, 0.36)),
            "官方 内高 290mm、内宽 800mm、内长 800mm 的方框",
        ),
        "轮胎阵": (
            BoxTireArrayTerrainCfg,
            dict(size=(8.0, 1.6), start_length=0.8, num_x=4, num_y=2, pitch=0.56),
            "官方 165/70R13 轮胎 8 个，外径 550mm、内径 330mm，2×4 矩阵",
        ),
        "绕桩": (
            BoxSlalomTerrainCfg,
            dict(size=(6.0, 3.0)),
            "官方 L 形 6 桩（直径 100mm、高 100mm、中心距 900mm）+ 3 个必达区（直径 300mm）",
        ),
        "沟壑组合": (
            BoxGapCourseTerrainCfg,
            dict(size=(5.2, 3.2)),
            "官方 L 形：平台 360mm 高 / 800×800、沟宽 360mm、台阶 320×120、斜坡 20°",
        ),
        "砖木坑": (
            BoxBrickPitTerrainCfg,
            dict(size=(4.0, 3.0)),
            "官方 L 形 2000×2000、臂宽 1000mm、护栏高 150mm，坑内填碎砖/树皮",
        ),
    }


#: 英文别名 -> 中文场地名。
#:
#: 存在的唯一理由是**让 .bat 启动器保持纯 ASCII**：Windows 命令行代码页在
#: 936(GBK) 和 65001(UTF-8) 之间不确定，批处理文件里一旦出现中文，
#: 换个代码页就会把命令字节吞掉（实测 `set PYTHONPATH` 被解析成 `HONPATH`，
#: 于是窗口一闪就退）。用英文别名就能彻底绕开这个坑。
_ALIASES = {
    "stepstone": "梅花桩", "plum": "梅花桩", "meihuazhuang": "梅花桩",
    "beam": "单木桥", "bridge": "单木桥", "danmuqiao": "单木桥",
    "wall": "高墙", "gaoqiang": "高墙",
    "lowbar": "限高框架", "limitframe": "限高框架", "digan": "限高框架",
    "tire": "轮胎阵", "tires": "轮胎阵", "luntaizhen": "轮胎阵",
    "slalom": "绕桩", "raozhuang": "绕桩",
    "gap": "沟壑组合", "gapcombo": "沟壑组合", "gouhe": "沟壑组合",
    "brick": "砖木坑", "brickpit": "砖木坑", "zhuanmu": "砖木坑",
}


def _resolve_venue(name: str) -> str:
    """把英文别名翻译成中文场地名；已经是中文或未知的直接原样返回。"""
    return _ALIASES.get(name.strip().lower(), name)


def _apply_colors(out) -> None:
    """把地形自带的颜色刷到 MuJoCo 几何体上（否则全是默认灰色）。"""
    for tg in out.geometries:
        if tg.geom is not None and tg.color is not None:
            tg.geom.rgba = tg.color


def _new_spec(cx: float, cy: float, cz: float, extent: float) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        _BASE_XML.format(cx=cx, cy=cy, cz=cz, extent=extent)
    )


def build_model(venue: str, difficulty: float):
    """生成只含该地形的 MuJoCo model。"""
    table = _venue_table()
    if venue not in table:
        raise KeyError(f"没有这个场地：{venue}")
    cls, kwargs, _desc = table[venue]

    x0 = float(kwargs.get("size", (8.0, 1.6))[0])
    spec = _new_spec(cx=x0 / 2.0, cy=0.8, cz=0.15, extent=x0 * 0.75)
    out = cls(**kwargs).function(float(difficulty), spec, np.random.default_rng(0))
    _apply_colors(out)
    return spec.compile(), out


def build_comparison(venue: str, levels: int):
    """把 0..1 的若干难度沿 y 方向并排摆开。

    注意：各地形的 function() 内部固定用 ``spec.body("terrain")``，
    所以这里必须只建一个名为 terrain 的 body，把多批几何体都堆进去、
    再按批次做 y 偏移（不能建多个 body，会取不到）。
    """
    table = _venue_table()
    cls, kwargs, _desc = table[venue]
    patch_y = float(kwargs.get("size", (8.0, 1.6))[1])
    pitch_y = patch_y + 1.0

    total_y = levels * pitch_y
    spec = _new_spec(cx=4.0, cy=0.0, cz=0.15, extent=max(8.0, total_y * 0.9))
    spots = []
    # 必须复用同一个 cfg 实例：梅花桩的桩名靠实例内的 _stone_counter 递增，
    # 每次 new 一个实例都会从 0 重新开始，导致 "repeated name 'stepstone_0'"。
    cfg = cls(**kwargs)
    for i in range(levels):
        d = i / max(1, levels - 1)
        y_off = (i - (levels - 1) / 2.0) * pitch_y
        out = cfg.function(d, spec, np.random.default_rng(0))
        _apply_colors(out)
        for tg in out.geometries:
            if tg.geom is not None:
                tg.geom.pos = (
                    float(tg.geom.pos[0]),
                    float(tg.geom.pos[1] + y_off),
                    float(tg.geom.pos[2]),
                )
        spots.append((d, np.asarray(out.origin) + np.array([0.0, y_off, 0.0])))
    return spec.compile(), spots


def main() -> int:
    table = _venue_table()

    ap = argparse.ArgumentParser(
        description="看障碍场地的 3D 样子（不需要训练、不需要显卡）",
    )
    ap.add_argument("venue", nargs="?", help="场地名；留空则列出所有场地")
    ap.add_argument("--level", type=float, default=1.0,
                    help="难度 0~1（0 最简单，1 = 比赛标准），默认 1.0")
    ap.add_argument("--levels", type=int, default=0,
                    help="一次看 N 个难度并排对比（例如 4）")
    ap.add_argument("--no-window", action="store_true",
                    help="只生成模型、不打开窗口（自检用）")
    args = ap.parse_args()

    if args.venue is not None:
        args.venue = _resolve_venue(args.venue)

    if args.venue is None:
        print("能看的场地：\n")
        for name, (_cls, _kw, desc) in table.items():
            print(f"  {name:<6} {desc}")
        print("\n用法：python preview_terrain.py 梅花桩")
        print("      python preview_terrain.py 轮胎阵 --levels 4")
        return 0

    if args.venue not in table:
        print(f"没有这个场地：{args.venue}")
        print("可选：" + "、".join(table))
        return 1

    if args.levels and args.levels > 1:
        model, spots = build_comparison(args.venue, args.levels)
        print(f"并排显示 {args.levels} 个难度：")
        for d, o in spots:
            print(f"   难度 {d:.2f}  起点 x={o[0]:.2f} y={o[1]:.2f}")
    else:
        model, out = build_model(args.venue, args.level)
        print(f"场地：{args.venue}   难度：{args.level:.2f}")
        print(f"几何体 {model.ngeom} 个，起点 {np.round(out.origin, 2)}")

    if args.no_window:
        print("（--no-window：只生成不显示）")
        return 0

    data = mujoco.MjData(model)
    print("\n窗口操作：左键旋转 / 右键平移 / 滚轮缩放 / Esc 关闭")
    try:
        mujoco.viewer.launch(model, data)
    except Exception as exc:  # noqa: BLE001
        print(f"打不开 3D 窗口：{exc}")
        print("如果是无显示器环境，请在有屏幕的机器上运行。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())