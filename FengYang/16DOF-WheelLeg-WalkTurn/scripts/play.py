"""回放入口：加载训练好的 checkpoint，在浏览器里看机器人跑直行 / 转向。

用法（在工程根目录下）::

    .\\run_play.ps1                                     # 用最新一次的 checkpoint
    .\\run_play.ps1 --checkpoint-file logs\\...\\model_5000.pt
    .\\run_play.ps1 --agent zero                        # 不做策略，只看默认站姿
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjlab.scripts.play import main

if __name__ == "__main__":
    main()