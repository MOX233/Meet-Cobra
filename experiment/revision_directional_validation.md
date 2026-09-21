# 新训练与方向性服务入口：验证记录（2026-09-21）

本记录只证明所列软件路径通过了小规模检查，不代表正式模型已训练完，或全负载结果已得到。
本轮未修改论文正文、回复信或结果图片。

## 已完成

| 检查 | 结果 |
|---|---|
| 修改前 Git 快照 | `c0c7029`；标签 `pre-directional-prediction-rerun-20260921` |
| CPU unittest | 56 项执行通过，含既有 GAP、MTS、O-MAPPO、TBPTT 回归检查 |
| CUDA unittest | 6 项通过，RTX 3090 上验证方向性增益、显式 RB 服务和预测时序 |
| 两阶段训练 | 合成数据各 2 epoch，两阶段中间分别停止并恢复 |
| 恢复正确性 | 恢复训练与不中断训练的最终最佳 checkpoint SHA-256 相同 |
| 已完成训练重复启动 | 两阶段跳过，最佳 checkpoint 未改写 |
| 缓存重复生成 | 完整缓存校验后跳过，文件 SHA-256 不变 |
| 短轨迹系统仿真 | 13 Mbps、seed 1，8 个方案各 8 个服务帧，全部成功 |
| 双 GPU 后台运行 | 先完成一个 case，再用 `cuda:0,cuda:1 --detach` 完成剩余 7 个 |
| case 续跑 | 重复 `run` 后已完成 raw 文件哈希不变 |
| 汇总 | `summary.json`、`curves.csv`、功率专用松弛参考均成功导出 |
| 防误用 | 拒绝 smoke 模型进入正式 grid，拒绝缓存、冻结代码或结果被篡改后的复用 |
| 代码静态检查 | `py_compile` 和 `git diff --check` 通过 |

集成检查目录：`experiment/results/revision_directional_smoke_20260921_v2/`。
`check_summary.json` 为通过标志；原始数据、模型与日志仍保留用于复核，但不作为论文证据。
上一轮初步检查目录 `revision_directional_smoke_20260921` 也保留，不删除历史记录。

最终集成检查的 protocol SHA-256：
`6ccbe0c60e12ccc46b4e5034e3b9c4f295ec25dd2515e65e4e351cf09eb1d8f9`。
该 protocol 保存了 36 个代码文件的哈希。

八个方案的 traffic SHA-256 完全一致：
`dfae7f12b6199f5160c8d811be80a7f83cccb4127ba8d91aefe82b7700ef97e1`。

方向性服务检查明确验证：服务计算不读取代理干扰字典；HO 中断期间不分配 RB，队列仅增加到达量；
RB 不能为负数、非整数或超出 BS 容量。BF/RA 的报告目标帧和当前帧匹配。

## 正式训练数据已准备，但正式训练尚未启动

- 文件：`experiment/results/revision_directional_20260921/training_data.npz`。
- 原始时间范围：200–800 s，6,001 个 frame；646,747 个 next-frame 样本。
- 车辆级划分复核：494 个训练车辆、212 个验证车辆；对应 507/215 条有效连续轨迹。
- 干扰标签全部有限；beam 与 desired-gain 标签生成方式未改。
- 数据 SHA-256：`cd3af55dba0f8aa930459e912e9ed6856ab6cb697a82281a21a1e17e015ce9bf`。
- split SHA-256：`6af64204659e247f692634efc9db5ecb9d9d1beaefa94986a0265acb7770dc7c`。

## 正式结果解释前需注意

1. 新模型性能和所有方案排序必须等待完整训练及 720 个 case 完成；小规模检查不能预测排序。
2. Reactive-OBRA 与 w/o GAP-HO 保留的旧 greedy HO 实现与正文 RSS-first 描述存在差异，
   本轮仅记录事实，未擅自替换算法。运行入口中已明确其真实实现和信息来源。
3. Oracle-CR-LB 是指定 P2 实例的连续松弛参考；不可行时的满功率占位值单独标识。
4. 比值诊断按负载、seed、方案成对保存；不能预先假定干扰代理始终高估真实干扰。
5. 命令详见 [revision_directional_runbook.md](revision_directional_runbook.md)。
