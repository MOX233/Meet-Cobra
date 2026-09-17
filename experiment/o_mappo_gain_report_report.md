# O-MAPPO 预测报告去除波束索引消融

## 范围

按用户要求，在上一轮 57 维报告输入版本上，只删去四个微基站各自上报的五个预测波束对索引，得到 `state_variant="gain_report"`、`information_mode="shared_prediction"`。

- 修改前 Git 快照：`280532d`。
- Actor：29 维原有基础状态 + 8 维预测增益 = 37 维。
- Critic：仍采用所有车辆 local state 的均值、最小值、最大值及归一化车辆数，输入从 172 维变为 112 维。训练时也不接收预测波束索引。
- 保留当前实际使用的 TX/RX 波束记录及其 4 维正余弦编码；删除的是预测候选索引，而不是全部与波束有关的状态。
- 8 个增益的顺序、数值、固定归一化 `(gain_dB + 100)/40` 与上一轮完全相同。测试已验证 37 维向量逐项等于原 57 维向量删除对应 20 项后的结果。
- Actor 和 target optimizer 不读取预测波束索引、原始高维导频或真实候选信道；原有目标选择优化器本就不使用预测索引，因此此次不改变其输入和规则。
- 保留 64 单元隐藏层、二元 HO 触发、奖励函数、候选 BS 数、优化目标、波束全搜索和局部跟踪、公共 OTR-RA 和 10 ms HO 中断。
- 不修改论文、回复信、正式图片或原有 checkpoint。

从部署接口需求看，该版本只需要四个微基站的两个 FP32 增益，即 `4×2×32=256 bit/vehicle/frame`。本轮仿真保持既有导频与服务时间计算不变，没有另行引入反馈字节数减少对应的额外吞吐增益。预测缓存仍含上一轮计算好的所有字段，但 gain-only 路径不会读取 beam 字段；现场实现可以不运行该基线不需要的波束预测模型，本轮未重新测量这一推理开销。

## 训练与测试协议

- 共用冻结前端和缓存：`experiment/results/o_mappo_shared_frontend_20260917/{train,test}_prepared.pkl`，增益预测不重新生成。
- 从头训练两个 seeds（20、21），每个 72 episodes。各 episode 的时间片段、负载、随机种子与上一轮报告输入版本相同。
- 训练轨迹：200–700 s，随机 30 s 片段；负载循环 1、7、13、19、27、35 Mbps。
- 验证：700.1–710 s，7、19、35 Mbps，每 12 episodes 评估一次。
- 沿用 `max(validation U[%]) + 0.01×mean(validation power[W])` 选择跨 seed 的单一 checkpoint；不根据测试结果选模型。
- 完整测试：800–830 s，四个代表性负载 1、13、27、35 Mbps，各 3 个配对 seeds；去掉前两帧，10 ms HO interruption。
- 1、13 Mbps 使用 GPU 3，27、35 Mbps 使用 GPU 0；这两块 GPU 与先前 GPU 5 的物理随机数及完整参考实验已核验一致。
- 与上一轮 57 维报告输入版本直接配对，另复用原 O-MAPPO、导频输入版本、MEET-COBRA 的已完成数据。到达哈希、前端和策略 checkpoint 均核验后再汇总。
- 输入减少使 actor 和 critic 的输入层参数量减少，重新训练时随机初始化也不相同。因此本实验考察当前编码和训练配置下的特征消融，不宣称独立控制所有网络容量因素。

## 验证

新增测试核验向量逐项子集关系、无预测索引时的接口和端到端训练/仿真、改变或污染预测索引不影响输入、真实 SINR/原始导频不进入 actor、checkpoint 保存与加载。结合既有回归测试共 38 项通过。

## 结果

两个训练种子均完成 72 episodes，每个约 12 min。seed 20 的最佳验证分数为 16.0141062519，seed 21 为 18.2878441423，均出现在 episode 72。按预先固定准则选择 seed 20、episode 72，checkpoint SHA-256 为 `c6216dbc14a79c78fc6f8f57c035f093dbd005fece33f78c7ad0079b0fa6c754`。完整报告版本所选模型的验证分数为 17.6207669794；这些分数属于帧级近似验证环境，不是最终逐时隙测试的违反概率。

12 组逐时隙测试已全部完成，并与 48 组已有参考实验一起汇总。核验包括相同负载与 seed 下的到达过程哈希、冻结预测前端、训练配置与 72 个训练片段，以及各方法在所有测试负载上使用同一个选定 checkpoint。实现提交为 `074dfc9`，汇总核验脚本提交为 `3086646`。

### 与上一轮完整报告输入的直接比较

下表是三个测试 seeds 的均值。完整报告输入包括预测增益和预测波束索引；仅增益输入删除了后者。这里的 U 与时延分位数均沿用论文中的队列长度时延代理指标，不是逐包测量的时延。

| 每车负载 (Mbps) | 完整报告功率 (W) | 仅增益功率 (W) | 完整报告 U (%) | 仅增益 U (%) | 完整报告 p99 (ms) | 仅增益 p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 14.8340 | 13.4111 | 0.2600 | 0.1978 | 18.01 | 18.02 |
| 13 | 81.1060 | 107.5654 | 3.1064 | 1.9810 | 168.23 | 82.94 |
| 27 | 176.9136 | 183.9479 | 20.9981 | 26.0351 | 6961.52 | 10248.57 |
| 35 | 183.6317 | 185.5749 | 61.2094 | 56.3848 | 15015.49 | 17958.49 |

- **1 Mbps：功率与违反概率同时改善。** 平均功率降低 9.59%，U 降低 0.0622 个百分点，相对降低 23.91%。
- **13 Mbps：改善时延指标，但增加功率。** U 降低 1.1254 个百分点，p99 约减半，平均功率却增加 32.62%。因此不能认为该负载下整体优于完整报告版本。
- **27 Mbps：功率与时延指标均退化。** 功率增加 3.98%，U 增加 5.0370 个百分点，p99 从约 6.96 s 增加到 10.25 s。
- **35 Mbps：违反概率下降，不代表整体时延改善。** U 降低 4.8246 个百分点，但功率增加 1.06%，平均时延代理值从 1393.79 ms 增加到 1644.20 ms，p99 从约 15.02 s 增加到 17.96 s。更少的阈值违反可以与更严重的部分车辆队列积压并存。

完整的均值、样本标准差以及其他参考方法结果见 `results/o_mappo_gain_report_20260917/result_table.md` 和 `comparison_summary.json`。样本标准差仅反映固定选定策略在三个测试 seeds 上的变化，不是训练随机性与场景泛化不确定性的完整估计。

### 策略行为与判断

删除预测波束索引后，四个负载下都出现了较少的 HO 和较高的宏基站关联比例：

| 每车负载 (Mbps) | 完整报告 HO (次/车/s) | 仅增益 HO (次/车/s) | 完整报告宏基站关联 (%) | 仅增益宏基站关联 (%) |
|---|---:|---:|---:|---:|
| 1 | 0.0259 | 0.0124 | 1.29 | 5.96 |
| 13 | 0.0662 | 0.0357 | 9.61 | 19.85 |
| 27 | 0.2490 | 0.2106 | 22.74 | 33.77 |
| 35 | 0.6572 | 0.5254 | 33.18 | 41.20 |

这些统计表明，输入简化后重新训练的策略改变了关联与切换行为，而不只是减少了输入维数。它们有助于理解性能权衡，但本实验不能单独证明某一行为变化就是功率或时延变化的原因，也不能据此认定预测波束索引本身无用。

与 MEET-COBRA 相比，仅增益版本在 1、13、27 Mbps 的平均功率仍分别为 13.41、107.57、183.95 W，而 MEET-COBRA 为 4.93、29.67、125.73 W；在 35 Mbps，两者功率接近，但 U 分别为 56.38% 和 39.07%。因此，本次删除索引没有解决此前共享预测前端 O-MAPPO 与 MEET-COBRA 之间的主要性能差距。

**结论：保留仅增益版本作为消融候选，不据此直接替换完整报告版本。** 其反馈接口更小，低负载部分指标改善，但跨负载没有一致优势。当前结果只覆盖四个代表性负载和固定场景下的三测试 seeds，尚不是全部论文负载的正式重跑。原版 O-MAPPO 的参考结果使用更强的候选信道信息，不应作为本轮信息条件相同的消融对照；本轮主要对照是上一轮完整报告版本。本轮没有修改论文、回复信、正式结果图或既有模型选择。

## 保存的产物

目录：`experiment/results/o_mappo_gain_report_20260917/`。

- `selected_policy.json`：固定的最终 checkpoint 及选择信息。
- `training_seed20/`、`training_seed21/`：训练协议、训练记录、选择结果和模型权重；最终选用 `training_seed20/best_policy.pt`。
- `runs/`、`raw/`：12 组新测试的指标与原始数组。
- `comparison_summary.json`、`comparison_summary.csv`：全部 60 组运行及五个方法的汇总。
- `paired_differences.csv`：与完整报告版本之间的配对差异。
- `result_table.md`：含样本标准差的结果表。
- `comparison_curves.pdf`、`comparison_curves.png`：实验比较图，仅保存在本实验目录，没有替换论文图片。

代码、报告、选择与协议元数据及 JSON 汇总纳入 Git；大体积 checkpoint、原始数组和可再生绘图保留在上述实验目录中。

## 复现

```bash
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_gain_report_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant gain_report --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_gain_report_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant gain_report --training-seed 21 --episodes 72
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_gain_report_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods gain_report --rates 1,13 --seeds 1,2,3 --workers 1 --rician --physics-gpu 3
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_gain_report_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods gain_report --rates 27,35 --seeds 1,2,3 --workers 1 --rician --physics-gpu 0
python experiment/summarize_o_mappo_gain_report.py
python -m unittest test_o_mappo_gain_report test_o_mappo_report_input test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement test_gpu_phy test_compiled_matching -q
```

为并行运行，可将上述测试命令的 seeds 分别设为 1、2、3，在独立进程中启动，避免 fork 后的执行开销。已经完成的结果会自动复用，重新训练应使用新目录。
