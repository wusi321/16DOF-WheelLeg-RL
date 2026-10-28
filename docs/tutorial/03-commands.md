# 03 · 常用命令速查

按"日常开发顺序"排列。所有命令都在仓库根目录（`~/16DOF-WheelLeg-RL`）执行。

---

## 0. 同步代码与环境

```bash
git pull                                   # 拉最新（SSH 走 443，见文末）
uv sync                                    # 同步依赖（改了 pyproject 之后必跑）
```

---

## 1. 环境/契约检查（不需要 GPU，秒级）

```bash
uv run python tools/check_go2w_contract.py    # 名称模式 / 关节限位 / 站立高度
uv run python tools/smoke_build_env.py        # 四个任务 build+reset+step + 100 步站立回归
uv run python tools/smoke_train_ppo.py        # 1 次 PPO 迭代，验证训练链路
```

---

## 2. 梅花桩专项（改地形前必做）

```bash
# 按难度画俯视图（不训练、不占 GPU，输出 docs/images/terrain_levels.png）
uv run python tools/plot_stepstone_terrain.py
uv run python tools/plot_stepstone_terrain.py --out /tmp/terrain.png

# 单环境验证“每根桩都要踩到”的判定与奖励（16/42 根、+2、踩满 +20、坑底 -2）
uv run python tools/check_stepstone_coverage.py
```

> 改任何地形参数之后，**先出图确认净空**，再决定要不要开训。

---

## 3. 训练

```bash
uv run train Robot-StepStone-v0 \
  --env.scene.num-envs 20480 \
  --env.sim.mujoco.ccd-iterations 16 \
  --agent.algorithm.num-mini-batches 32 \
  --agent.save-interval 25 \
  --agent.max-iterations 650 \
  --agent.logger wandb --agent.wandb-project 16dof-wheelleg
```

| 参数 | 建议 | 说明 |
| --- | --- | --- |
| `--env.scene.num-envs` | **20480** | 8 m 梅花桩地形下 40960 会 OOM（48 GiB 卡） |
| `--env.sim.mujoco.ccd-iterations` | **16** | 低于 8 时轮-地接触完全消失；80 时 EPA 缓冲 22.8 GB 会 OOM |
| `--env.scene.terrain.terrain-generator.num-rows` | 10（默认） | 想跑 40960 时降到 6（geom 391→255） |
| `--agent.save-interval` | 25 | 默认 100 太稀，取不到 600–700 的最佳点 |
| `--agent.max-iterations` | 650 | 梅花桩 600–700 是姿态最佳点，之后只变晃 |
| `--agent.logger` | wandb | 想本地跑就去掉，用默认 logger |

**微调地形的开关**（不用改代码）：

```bash
--env.scene.terrain.terrain-generator.sub-terrains.box_stepping_stones.pitch-scale 0.85
--env.scene.terrain.terrain-generator.sub-terrains.box_stepping_stones.stagger-scale 0.8
--env.scene.terrain.terrain-generator.sub-terrains.box_stepping_stones.max-step 0.50
```

从 checkpoint 续训：

```bash
uv run train Robot-StepStone-v0 --resume --checkpoint-file logs/rsl_rl/robot_stepstone/<run>/model_400.pt
```

---

## 4. 回放 / 录视频

```bash
# 普通回放（no display 会自动用 viser 网页）
uv run play Robot-Rough-v0
uv run play Robot-StepStone-v0 --checkpoint-file <ckpt> --num-envs 12

# 按难度回放：每个环境钉一个等级（10 个环境 = 0..9 各一个）
uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt>
uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt> --levels 9
uv run python tools/play_stepstone_levels.py --checkpoint-file <ckpt> --levels 0,5,9 --num-envs 9

# 录视频（存到 logs/.../videos/play/）
uv run play Robot-StepStone-v0 --checkpoint-file <ckpt> --video --video-length 300

# 内置 play 的等级采样（默认 0..9 随机；同等级的环境会叠在一个出生点）
STEPSTONE_PLAY_MAX_LEVEL=9 uv run play Robot-StepStone-v0 --checkpoint-file <ckpt> --num-envs 10
```

> `play` 会关掉课程（`play=True`）：地形等级不推进、回合不超时（`1e9`），
> 所以它展示的是"当前策略一直跑"，不是训练时的回合统计。

---

## 5. wandb

```bash
wandb login                                     # 首次
wandb sync wandb/offline-run-*                  # 补传离线 run
wandb sync --sync-all
```

```python
import wandb
api = wandb.Api()
run = api.run("scunf1001-shandonghuayu/16dof-wheelleg/<run_id>")
print(run.summary["Train/mean_reward"])
```

---

## 6. git（本机常见坑）

```bash
git pull
git add -A
git -c user.name="WUSI321" -c user.email="19589917063@163.com" commit -F msg.txt   # 中文/引号多时用 -F
git push origin main
```

- **SSH 卡住 / TLS 报错**时走 443：

```bash
~/.ssh/config
Host github.com
  HostName ssh.github.com
  Port 443
  User git
```

- 大文件（视频）先压到 < 10 MB 再提交；`assets/videos/` 已放行，
  其他位置的 `*.mp4` 仍在 `.gitignore` 里。

---

## 7. 故障速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Warp CUDA error 2: out of memory`（`wp_cuda_graph_launch`） | 并行环境 × geom 数太大 | 降 `num-envs`（40960→20480）或降 `num-rows`（10→6） |
| 机器人穿地、轮子不受力 | `ccd-iterations < 8` | 设为 16 |
| `Failed to allocate ... bytes (epa_face)` | `ccd-iterations` 太大 | 设为 16 |
| `TypeError: can't convert cuda:0 device type tensor to numpy` | CUDA 张量直接转 numpy | 代码里 `.detach().cpu().numpy()` |
| `nefc overflow` | 接触/约束配额不足 | `nconmax=128`、`njmax=256` |
| 训练不涨、`stone_coverage` 卡住 | 大概率是奖励被绕过 | 先 play 看行为，再对照 `02-wandb.md` |
