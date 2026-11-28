"""把 8 个独立障碍拼成 14m x 6m 的完整赛场，导出一个可直接预览的 MJCF。

设计原则
--------
1. **复用已验证的地形代码**：每个障碍直接调用 ``src/robot/terrains`` 里对应的
   ``SubTerrainCfg.function()``，难度固定 1.0（= 官方表 1 规格）。
   这样"总场地"里的每个障碍跟单独训练时跑的是同一份代码，不会出现两套实现。
2. **每个障碍只放 1 个实例**：训练用的课程地形里有重复实例（高墙 5 面、大 patch），
   但比赛现场每个障碍只有一个，所以这里按最小配置生成。
3. **布局**：位置基准参考 uwvwko-zzz/27rc-dog-race 的 14x6m 排布
   （见 参考-官方赛场/），并按我们自己的障碍实际尺寸重新避让，满足：
     - 硬约束：障碍互不重叠；距场地边界 >= 600mm（规则 3.1）
     - 软目标：障碍之间 >= 800mm（规则 3.1）
4. **地面只铺一块**：每个 terrain 的 geometries[0] 都是自带地面，这里全部剔除，
   最后统一铺 14x6m 的黄色场地地面 + 蓝色边界线 + 中心红色虚线启动区。

运行（在 16dof-rl 目录下）：
    $env:PYTHONPATH = "$root\\src;$root\\mjlab\\src"
    python 总场地/build_full_course.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from robot.terrains import (
    BoxBrickPitTerrainCfg,
    BoxGapCourseTerrainCfg,
    BoxSingleBeamTerrainCfg,
    BoxSlalomTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    BoxTireArrayTerrainCfg,
    RCLowBarTerrainCfg,
    RCWallTerrainCfg,
)

# ---------------------------------------------------------------------------
# 场地与规则约束
# ---------------------------------------------------------------------------
ARENA_HALF_X = 7.0          # 14000mm / 2
ARENA_HALF_Y = 3.0          # 6000mm / 2
BORDER_MARGIN = 0.60        # 障碍距场地边界 >= 600mm（硬约束）
OBSTACLE_GAP = 0.80         # 障碍之间 >= 800mm（软目标）

FLOOR_COLOR = (0.85, 0.80, 0.30, 1.0)      # 场地黄色
BOUNDARY_COLOR = (0.05, 0.15, 0.95, 1.0)   # 边界线蓝色
START_MARK_COLOR = (0.90, 0.10, 0.10, 1.0) # 启动区红色虚线

# ---------------------------------------------------------------------------
# 8 个障碍
#
# target = 希望该障碍【几何包围盒中心】落到的坐标。
# 每个障碍的包围盒尺寸（实测，含必达区/虚线等标记物）：
#   单木桥 3.00x0.32 / 高墙 0.08x0.60 / 梅花桩 3.50x1.10 / 限高框架 0.88x0.88
#   沟壑   2.62x1.82 / 绕桩 3.80x2.87 / 轮胎阵 2.24x1.16 / 砖木坑   2.22x2.02
#
# 排布思路（严格按官方图 1-2 的相对位置，场地可用 x∈[-6.4,6.4], y∈[-2.4,2.4]）：
#   上排  砖木坑(左) → 单木桥(中) → 高墙(中右) → 梅花桩(右)
#   下排  沟壑组合(左，含启动区2) → 轮胎阵(中) → 限高框架(中右) → 绕桩(右)
#   中央  启动区1（场地中心）
#   绕桩包围盒 2.87m 最高，占右侧整列；梅花桩放它正上方。
# ---------------------------------------------------------------------------
PLACEMENTS = [
    dict(
        name="balance_beam",
        label="单木桥",
        cls=BoxSingleBeamTerrainCfg,
        kwargs=dict(size=(3.6, 1.2), beam_length=3.0),
        target=(-1.60, 2.20),
    ),
    dict(
        name="high_wall",
        label="高墙",
        cls=RCWallTerrainCfg,
        # 训练用 5 面墙，比赛现场只有 1 面
        kwargs=dict(size=(2.0, 1.2), wall_centers_x=(1.0,)),
        target=(1.80, 2.00),
    ),
    dict(
        name="stepping_stones",
        label="梅花桩",
        cls=BoxSteppingStonesTerrainCfg,
        kwargs=dict(size=(3.5, 1.2)),
        target=(4.40, 1.84),
    ),
    dict(
        name="brick_bark_pit",
        label="砖木坑",
        cls=BoxBrickPitTerrainCfg,
        kwargs=dict(size=(3.6, 3.6)),
        target=(-5.28, 1.38),
    ),
    dict(
        name="gap_assembly",
        label="沟壑组合",
        cls=BoxGapCourseTerrainCfg,
        kwargs=dict(size=(6.0, 4.0)),
        target=(-5.08, -1.48),
    ),
    dict(
        name="tire_array",
        label="轮胎阵",
        cls=BoxTireArrayTerrainCfg,
        kwargs=dict(size=(3.2, 2.2)),
        target=(-1.30, -1.78),
    ),
    dict(
        name="clearance_frame",
        label="限高框架",
        cls=RCLowBarTerrainCfg,
        kwargs=dict(size=(2.0, 1.2)),
        target=(1.10, -1.87),
    ),
    dict(
        name="slalom",
        label="绕桩",
        cls=BoxSlalomTerrainCfg,
        kwargs=dict(size=(6.0, 3.8)),
        target=(4.48, -0.95),
    ),
]


def _geom_aabb(geom) -> tuple[float, float, float, float]:
    """粗略的 geom 水平包围盒 (xmin, xmax, ymin, ymax)。

    用 pos ± size 近似，忽略旋转。对斜坡这类倾斜件会略微偏大，用于排布自检足够。
    """
    px, py = float(geom.pos[0]), float(geom.pos[1])
    sx, sy = float(geom.size[0]), float(geom.size[1])
    return (px - sx, px + sx, py - sy, py + sy)


def build_full_course(difficulty: float = 1.0):
    """构建完整赛场，返回 (model, xml_string, 每个障碍的包围盒信息)。"""
    spec = mujoco.MjSpec()
    spec.modelname = "robocon_2027_full_course"
    # 离屏渲染帧缓冲（截图/录屏用），默认只有 640x480 会渲不出大图
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    # 名字必须是 "terrain"：各 SubTerrainCfg 内部都用 spec.body("terrain") 取这个 body
    body = spec.worldbody.add_body(name="terrain")

    # ---- 光源：默认光照偏暗，场地黄色会发橄榄色，加一盏主光 + 一盏补光 ----
    key = spec.worldbody.add_light()
    key.pos = (0.0, 0.0, 9.0)
    key.dir = (-0.25, -0.35, -1.0)
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.castshadow = True
    key.ambient = (0.45, 0.45, 0.45)
    key.diffuse = (0.85, 0.85, 0.85)
    key.specular = (0.10, 0.10, 0.10)

    fill = spec.worldbody.add_light()
    fill.pos = (0.0, 0.0, 8.0)
    fill.dir = (0.4, 0.2, -1.0)
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.castshadow = False
    fill.ambient = (0.0, 0.0, 0.0)
    fill.diffuse = (0.30, 0.30, 0.30)
    fill.specular = (0.0, 0.0, 0.0)

    # ---- 统一地面：14m x 6m ----
    floor = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(ARENA_HALF_X, ARENA_HALF_Y, 0.05),
        pos=(0.0, 0.0, -0.05),
    )
    floor.rgba = list(FLOOR_COLOR)

    rng = np.random.default_rng(2027)
    reports: list[dict] = []

    for item in PLACEMENTS:
        cfg = item["cls"](**item["kwargs"])
        out = cfg.function(difficulty, spec, rng)

        # geometries[0] 是该地形自带的地面，删掉，避免 8 块地面叠在一起
        floor_geom = out.geometries[0].geom
        if floor_geom is not None:
            spec.delete(floor_geom)

        obstacles = [tg for tg in out.geometries[1:] if tg.geom is not None]
        for tg in obstacles:
            if tg.color is not None:
                tg.geom.rgba = list(tg.color)

        boxes = [_geom_aabb(tg.geom) for tg in obstacles]
        xmin = min(b[0] for b in boxes)
        xmax = max(b[1] for b in boxes)
        ymin = min(b[2] for b in boxes)
        ymax = max(b[3] for b in boxes)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

        tx, ty = item["target"]
        dx, dy = tx - cx, ty - cy
        for tg in obstacles:
            g = tg.geom
            g.pos = (float(g.pos[0]) + dx, float(g.pos[1]) + dy, float(g.pos[2]))

        reports.append(
            dict(
                name=item["name"],
                label=item["label"],
                n_geom=len(obstacles),
                xmin=xmin + dx,
                xmax=xmax + dx,
                ymin=ymin + dy,
                ymax=ymax + dy,
            )
        )

    # ---- 场地边界线（不参与碰撞）----
    for p, q in [
        ((-ARENA_HALF_X, -ARENA_HALF_Y), (ARENA_HALF_X, -ARENA_HALF_Y)),
        ((ARENA_HALF_X, -ARENA_HALF_Y), (ARENA_HALF_X, ARENA_HALF_Y)),
        ((ARENA_HALF_X, ARENA_HALF_Y), (-ARENA_HALF_X, ARENA_HALF_Y)),
        ((-ARENA_HALF_X, ARENA_HALF_Y), (-ARENA_HALF_X, -ARENA_HALF_Y)),
    ]:
        g = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=(0.008,),
            fromto=(p[0], p[1], 0.004, q[0], q[1], 0.004),
            contype=0,
            conaffinity=0,
        )
        g.rgba = list(BOUNDARY_COLOR)

    # ---- 启动区 1：场地中心红色虚线方格 1000x800（规则 3.1）----
    sx, sy = 1.0, 0.8
    for p, q in [
        ((-sx / 2, -sy / 2), (sx / 2, -sy / 2)),
        ((sx / 2, -sy / 2), (sx / 2, sy / 2)),
        ((sx / 2, sy / 2), (-sx / 2, sy / 2)),
        ((-sx / 2, sy / 2), (-sx / 2, -sy / 2)),
    ]:
        for i in range(10):
            x0 = p[0] + (q[0] - p[0]) * i / 10.0
            y0 = p[1] + (q[1] - p[1]) * i / 10.0
            x1 = p[0] + (q[0] - p[0]) * (i + 0.55) / 10.0
            y1 = p[1] + (q[1] - p[1]) * (i + 0.55) / 10.0
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=(0.006,),
                fromto=(x0, y0, 0.006, x1, y1, 0.006),
                contype=0,
                conaffinity=0,
            )
            g.rgba = list(START_MARK_COLOR)

    xml = spec.to_xml()
    model = spec.compile()
    return model, xml, reports


def check_layout(reports: list[dict]) -> tuple[list[str], list[str], float]:
    """检查布局。返回 (硬性问题, 软性警告, 全局最小间距)。"""
    hard: list[str] = []
    soft: list[str] = []

    for r in reports:
        if r["xmin"] < -ARENA_HALF_X + BORDER_MARGIN - 1e-6 or r["xmax"] > ARENA_HALF_X - BORDER_MARGIN + 1e-6:
            hard.append(
                "%-6s x 距边界不足 %.2fm（实际 x∈[%.2f,%.2f]）"
                % (r["label"], BORDER_MARGIN, r["xmin"], r["xmax"])
            )
        if r["ymin"] < -ARENA_HALF_Y + BORDER_MARGIN - 1e-6 or r["ymax"] > ARENA_HALF_Y - BORDER_MARGIN + 1e-6:
            hard.append(
                "%-6s y 距边界不足 %.2fm（实际 y∈[%.2f,%.2f]）"
                % (r["label"], BORDER_MARGIN, r["ymin"], r["ymax"])
            )

    min_dist = 1e9
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            a, b = reports[i], reports[j]
            gx = max(0.0, max(a["xmin"], b["xmin"]) - min(a["xmax"], b["xmax"]))
            gy = max(0.0, max(a["ymin"], b["ymin"]) - min(a["ymax"], b["ymax"]))
            dist = math.hypot(gx, gy)
            min_dist = min(min_dist, dist)
            if dist <= 0.0:
                hard.append("%-6s 与 %-6s 重叠" % (a["label"], b["label"]))
            elif dist < OBSTACLE_GAP:
                soft.append(
                    "%-6s <-> %-6s 间距 %.2fm（建议 >= %.2fm）"
                    % (a["label"], b["label"], dist, OBSTACLE_GAP)
                )
    return hard, soft, min_dist


def main() -> int:
    model, xml, reports = build_full_course(1.0)

    print("=" * 76)
    print("总场地：14.0m x 6.0m，难度 1.0（= 官方表 1 规格），每个障碍 1 个")
    print("=" * 76)
    print("%-8s %-5s %-24s %-24s" % ("障碍", "geom", "x 范围", "y 范围"))
    print("-" * 76)
    for r in reports:
        print(
            "%-8s %-5d [%6.2f, %6.2f]          [%6.2f, %6.2f]"
            % (r["label"], r["n_geom"], r["xmin"], r["xmax"], r["ymin"], r["ymax"])
        )
    print("-" * 76)
    print("总 geom = %d    总 body = %d" % (model.ngeom, model.nbody))

    hard, soft, min_dist = check_layout(reports)
    print()
    print("硬约束（不重叠 + 距边界 >= %.2fm）：" % BORDER_MARGIN)
    if hard:
        for p in hard:
            print("  [失败] " + p)
    else:
        print("  全部通过")
    print()
    print("间距（建议 >= %.2fm，全局最小 %.2fm）：" % (OBSTACLE_GAP, min_dist))
    if soft:
        for p in soft:
            print("  [偏小] " + p)
    else:
        print("  全部满足")

    out = Path(__file__).resolve().parent / "full_course_2027.xml"
    out.write_text(xml, encoding="utf-8")
    print()
    print("已导出: %s" % out)
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
