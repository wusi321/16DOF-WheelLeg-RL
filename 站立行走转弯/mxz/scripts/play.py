"""回放入口：加载训练好的 checkpoint，在浏览器里看机器人跑直行 / 转向。

用法（在工程根目录下）::

    .\\run_play.ps1                                     # 默认用 checkpoints/model_1795.pt
    .\\run_play.ps1 --checkpoint-file checkpoints\\model_1795.pt
    .\\run_play.ps1 --agent zero                        # 不做策略，只看默认站姿
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 显式导入本工程的 robot 包来注册任务（不依赖 venv 里的 mjlab.tasks entry point）。
import robot  # noqa: E402,F401

from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    main()