"""采样 Robot-StepStone 训练日志的关键曲线（UTF-16 日志 -> 简明窗口趋势）。

用法: uv run python tools/watch_stepstone_train.py [日志路径]
每次运行打印最近 4 个 25 迭代窗口的均值，以及当前迭代号。
"""

import re
import sys
import pathlib

LOG = pathlib.Path(r"D:\Project\wheelleg-rl\train_2048_log.txt")

FIELDS = [
    ("coverage", "Episode_Metrics/stone_coverage"),
    ("goal_term", "Episode_Termination/goal_reached"),
    ("terrain", "Curriculum/terrain_levels/mean"),
    ("cap", "Curriculum/terrain_levels/level_cap"),
    ("reward", "Mean reward"),
    ("ep_len", "Mean episode length"),
    ("base_low", "Episode_Termination/base_too_low"),
    ("timeout_rew", "Episode_Reward/time_out"),
    ("airborne_rew", "Episode_Reward/airborne"),
    ("goal_rew", "Episode_Reward/goal_reached"),
]


def parse():
    txt = LOG.read_bytes().decode("utf-16", errors="ignore")
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    parts = re.split(r"Learning iteration\s+(\d+)/\d+", txt)
    rows = []
    for i in range(1, len(parts) - 1, 2):
        num, body = int(parts[i]), parts[i + 1]
        row = {"iter": num}
        for name, key in FIELDS:
            m = re.search(re.escape(key) + r":\s*(-?[0-9.]+)", body)
            row[name] = float(m.group(1)) if m else None
        rows.append(row)
    return rows


def main():
    global LOG
    if len(sys.argv) > 1:
        LOG = pathlib.Path(sys.argv[1])
    rows = parse()
    if not rows:
        print("no iterations yet")
        return
    print(f"latest_iter={rows[-1]['iter']}  parsed_rows={len(rows)}  "
          f"last_5: " + ", ".join(str(r["iter"]) for r in rows[-5:]))
    fmt = "{:>7} " + " ".join("{:>9}" for _ in FIELDS)
    print(fmt.format("window", *[n for n, _ in FIELDS]))
    for start in range(max(0, len(rows) - 100), len(rows), 25):
        chunk = [r for r in rows[start:start + 25]
                 if all(r[n] is not None for n, _ in FIELDS)]
        if not chunk:
            continue
        avg = {n: sum(r[n] for r in chunk) / len(chunk) for n, _ in FIELDS}
        print(fmt.format(f"{chunk[0]['iter']}-{chunk[-1]['iter']}",
                         *[f"{avg[n]:.3f}" for n, _ in FIELDS]))


if __name__ == "__main__":
    main()
