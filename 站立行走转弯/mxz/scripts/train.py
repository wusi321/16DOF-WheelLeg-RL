"""训练入口：平地直行 + 步态转向。

用法（在工程根目录下）::

    .\\run_train.ps1                              # 默认 1 卡、2048 环境
    .\\run_train.ps1 --env.scene.num-envs 1024    # 显存紧张时减半
    .\\run_train.ps1 --agent.max-iterations 3000

等价的手写命令::

    $env:PYTHONPATH = "$PWD\\src"
    & $Python scripts\\train.py Robot-WalkTurn-v0
"""

import sys
from pathlib import Path

# 把本工程的 src 放到最前面，确保 import robot 时拿到的是本工程的任务定义，
# 而不是 venv 里 editable 安装的 RL-main 的 robot 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 显式导入本工程的 robot 包来注册任务。mjlab 也会通过 venv 里 robot 包的
# entry point（mjlab.tasks）自动导入，但那依赖旧工程的 editable 安装；
# 换一台干净的机器时这个 entry point 不存在，任务 ID 就会找不到。
import robot  # noqa: E402,F401

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()