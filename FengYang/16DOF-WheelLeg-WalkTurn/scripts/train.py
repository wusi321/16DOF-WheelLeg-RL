"""训练入口：平地直行 + 步态转向。

用法（在工程根目录下）::

    .\\run_train.ps1                      # 默认 1 卡、2048 环境
    .\\run_train.ps1 --num-envs 1024      # 显存紧张时减半
    .\\run_train.ps1 --max-iterations 3000

等价的手写命令::

    $env:PYTHONPATH = "$PWD\\src"
    & $Python scripts\\train.py Robot-WalkTurn-v0
"""

import sys
from pathlib import Path

# 把本工程的 src 放到最前面，确保 import robot 时拿到的是本工程的任务定义，
# 而不是 venv 里 editable 安装的 RL-main 的 robot 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()