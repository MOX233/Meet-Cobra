# 正式两阶段 checkpoint 的跨帧递推推理计时

日期：2026-09-13。对应 R1C7 与 R2C6。

## 1. 结果

在 AMD EPYC 7742 CPU、PyTorch 2.6.0+cu118、FP32、batch size 1、单线程条件下，
三个预测器按每车每帧顺序运行。16 条真实连续轨迹每轮共 4,722 帧，重复三轮，
共保存 14,166 次 wall-clock 推理时延。

| 统计量 | 三模型合计时延（ms） |
|---|---:|
| 中位数 | 5.328958 |
| 95 分位 | 5.743067 |
| 99 分位 | 10.422766 |
| 均值 | 5.616233 |
| 最大值 | 121.652054 |

论文及回复信采用 **5.33 ms（中位数）、5.74 ms（95 分位）**。
所有测量样本均保留，没有删除较慢样本或按轮择优。少量长尾样本也计入统计，
因此这些分位数不应被解读为最坏情况时延上界或实时截止期保证。

三轮中位数依次为 5.377732、5.289667、5.295938 ms；95 分位依次为
5.868149、5.701247、5.632430 ms。

按已经递推的历史长度分组，1--10、11--100、101--200、201 帧以上的中位数
分别为 5.330218、5.332279、5.328843、5.326472 ms。在本次观察范围内，没有
随历史增长的中位数时延趋势。每次调用均只输入当前一帧，状态大小固定。

## 2. 模型与实现

使用 `experiment/results/stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt/`
下各任务的 `best.pth`，未使用 `last.pth` 或旧版有限窗口 checkpoint。

| 预测器 | 第二阶段所选 epoch | checkpoint SHA-256 |
|---|---:|---|
| Beam | 88 | `3ab0a2defe9f4795eceb2cb3d8de0bb2265a8d074e39c353a1cda6a8748abd4b` |
| Desired-link gain | 59 | `8c8bc19f2bb30214742f44b3fb1626029a549ecbe6d66249098480f11ce627eb` |
| Interfering-link gain | 91 | `620905d03a2d82a52ede2b8e31f383b5fb6b01fb253e5e1b0bfe4c51bf5619df` |

调用既有精度评估脚本 `compare_stateful_prediction.py` 的 `forward_with_state`，
执行原模型的 LSTM、shared layers 和 BS-specific heads，并返回新的 hidden/cell
states。三个模型分别保存其状态，下一帧直接使用；只有新的观察轨迹开始时清零，
**不会在十帧边界重置，也不会重放滑动窗口**。全部调用使用 `eval()` 和
`torch.inference_mode()`。计时前后验证权重、BN buffers 和 checkpoint 文件均未改变。

矩阵运算量通过实际有状态单步调用的解析计数及 CPU profiler 交叉核验，
三模型合计仍为 4,198,416 FLOPs，即表中取两位小数的 4.20 MFLOPs。
参数量和 FP32 参数存储量仍为 2,127,912 和 8.511648 MB。

## 3. 输入与计时边界

- 使用与此前 CPU 审计相同的 `data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl`。
- 以 seed 20260913 从至少 200 帧的连续轨迹中随机选择 16 条，实际长度为
  222--301 帧。只取各帧已经预处理的最新 128 维 FP32 CSI，保持时间顺序。
  vehicle ID 与逐帧时间戳、输入文件哈希均存入 `metadata.json`。
- 每轮先执行 100 次不计时的软件预热，再按随机顺序处理所有选定轨迹。
  每条轨迹的首帧推理也纳入正式统计；轨迹之间不传递状态。
- intra-op 和 inter-op threads 均为 1，CPU affinity 为 0--3，与旧审计一致；
  正式计时启用 oneDNN。CPU 操作同步完成，不需要 GPU 同步。本轮没有测量 GPU。
- 包含一次共用的 NumPy 到 Tensor 转换、三个模型的依次前向、状态更新、
  各 BS 的排序 top-5 波束索引选择、两类增益反归一化为 dB，以及 NumPy 输出。
- 不包含模型及数据加载、CSI 采集和预处理、报文序列化与传输、HO/BF/RA 决策、
  轨迹筛选和结果记录。输入数据加载完毕后才开始计时。

旧的 5.26/5.72 ms 是旧 checkpoint、单帧输入、完整概率输出的另一组基准。
本次实测替换其在论文中的用途；二者的差值不用于单独量化跨帧状态缓存的收益。

## 4. 正确性检查

- 在已选模型和真实轨迹上，对前缀长度 1、10、11、21、100、301 帧分别比较
  递推输出与原生模型一次处理完整前缀的输出，排序 top-5 索引完全相同。
- 两类 gain 输出的最大差异分别为 0.000030518 和 0.000015259 dB，处于 FP32
  数值误差范围；新轨迹的重置输出与全新 pipeline 完全一致。
- 新增 5 项测试验证完整前缀一致性、跨十帧保持状态、单步 FLOPs、缺帧分段和
  计时样本计数。连同既有预测和开销测试，共 17 项通过。

## 5. 复现与文件

```bash
python -m unittest test_stateful_nn_overhead test_stateful_prediction test_nn_overhead -v
python experiment/benchmark_stateful_nn_overhead.py --output experiment/results/nn_overhead_stateful_20260913/cpu_1thread --threads 1 --affinity 0,1,2,3 --trajectories 16 --min-frames 200 --rounds 3 --warmup 100
```

复跑时使用新的 `--output` 路径；脚本拒绝覆盖已有目录。

结果目录为 `experiment/results/nn_overhead_stateful_20260913/cpu_1thread/`：

- `timings.json`：合并、分轮、按历史长度分组及首帧的统计量。
- `raw_latency_ms.npz`：全部时延及对应轮次、轨迹、帧索引。
- `model_inventory.json`：三个选定 checkpoint、哈希、epoch 和参数统计。
- `metadata.json`：输入来源、轨迹身份、软硬件设置、源码哈希及计时边界。
- `correctness.json`：真实轨迹前缀一致性与重置检查。
- `arithmetic.json`：单步矩阵 FLOPs 和逐模型 profiler 交叉核验。

本轮没有重新训练模型或运行系统性能仿真，没有更改原有仿真入口、模型类或旧计时脚本。
已将论文 Section V-B 和回复信 R1C7/R2C6 的时延更新为本次实测值，撤下“新模型尚待计时”
的工作备注。Fig.5--Fig.8 的 HO-aware、有状态模型系统级重跑仍待后续完成。
