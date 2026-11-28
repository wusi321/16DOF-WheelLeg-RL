# 16DOF-URDF MuJoCo 模型

## 当前默认模型

`16DOF-URDF-sim.xml`

- 视觉：原始 STL
- 碰撞：原始 STL，未使用 box/cylinder 或凸包替换
- `base_link.STL` 拆成两个 197,564 面原始三角形子网格
- 根 body：`base`
- 根关节：`root_free`
- 16 路执行器，每腿 `hip → thigh → calf → wheel`
- 12 路位置执行器，4 路轮子速度执行器
- 所有执行器都有 `ctrlrange/forcerange`

## 备用优化模型

- `16DOF-URDF-convex.xml`：显式凸包碰撞
- `16DOF-URDF-training.xml`：box/cylinder 图元碰撞

## 执行器顺序

见 `actuator_order.json`

## 启动默认模型

```powershell
.\start_mujoco.cmd
```
