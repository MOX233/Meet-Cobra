# O-MAPPO：恢复原版状态组成，以预测增益计算链路与需求特征

## 范围及版本

按用户确认的方案，恢复原版 actor 的31维特征组成、次序和归一化，将依赖真实CSI的SINR、干扰噪声比与BS需求比例改为根据现有预测报告计算。不新增预测候选索引、原始导频或额外派生特征。

- 修改前代码：`106e141`，快照标签 `pre-o-mappo-predicted-state-20260917`。
- 新实现：`3ebbd84`；配置 `state_variant="predicted_adapted"`、`information_mode="shared_prediction"`。
- 不改目标BS优化器、候选集合、奖励、BF、OTR-RA、HO中断或物理层开销。新的需求重估仅用于actor/critic观察，不改写优化器接收的上一帧实际RB反馈。
- 不修改论文、回复信、正式图片，不覆盖旧checkpoint。

## 状态计算

Actor为31→64 ReLU→2，共2,178参数；critic为94→64 ReLU→1，共6,145参数。两者均恢复原版维数与结构；与上一轮37维actor/112维critic不同，因此不宣称本实验严格只改变actor而保持critic输入不变。输出仍为是否触发HO，而非目标BS。

所有车辆共享actor。原有位置、运动、队列、平均到达率、关联、BS车辆数、实际波束编码和已有服务反馈等状态保留。7个依赖CSI的槽位改为：预测服务SINR、预测INR、5个BS的预测需求比例。不给网络额外拼接原始增益。

微BS的SINR由预测最优波束增益、预测干扰链路增益、发射功率和干扰占用比例计算；宏BS仍采用已有位置/路径损耗模型，按本系统模型忽略跨层干扰。增益先转为线性值；INR为干扰功率除以噪声功率，再转dB。

需求重估沿用原代码的平均到达率口径，每车需求为平均到达率除以每RB速率，按当前关联聚合。保持原估计器最多10轮、1 RB绝对容差的迭代规则和初始值。它不是GAP-HO的2轮迭代，也不改变该算法。为保留特征语义，本估计不新增队列、导频或HO修正；这些仍在物理服务、奖励及目标优化器中按原有规则处理。

需求比例截断为[0,1.5]，与原精确仿真一致；用于物理干扰的占用比例另行截断为[0,1]，避免把过载需求解释成超过全部RB的实际干扰占用。这是相对旧估计器的一项明确物理修正，并非完全逐字复用旧数值计算。无过载测试中，新旧需求计算在浮点误差范围内一致。

当前NN输出是下一帧最优波束增益预测，不等于当前正在跟踪的波束增益。这里恢复的是特征组成和计算含义，不是恢复旧版真实信道数值，也不声称预测误差是两版的唯一差异。

## 实验协议

- 两个训练seeds 20、21，各72 episodes，从随机初始化训练，不加载旧actor权重。相同seed下，31维新模型的初始actor和critic权重与原版31维随机模型逐项相同。
- 冻结预测前端及缓存复用 `o_mappo_shared_frontend_20260917`；200–700 s随机30 s片段，循环1/7/13/19/27/35 Mbps，与37维对照逐episode配对。
- 验证700.1–710 s、7/19/35 Mbps，每12 episodes一次。选择准则仍为 `max(validation U[%]) + 0.01 mean(validation P[W])`，跨两个训练seeds选择单一checkpoint。
- 奖励为 `qos_energy020_load1`，PPO超参数不变。保留原有帧级近似训练环境以及其他基础反馈的既有时序，不混入额外调整。
- 唯一测试点：13 Mbps、seed 1、800–830 s、10 ms HO中断、20 ms时延阈值，前两帧不计入统计。使用相同到达、初始队列和FP64 CUDA Rician随机数。
- 在本轮代码/GPU上复现37维gain_report对照，检查历史结果一致性。真实CSI原版和上一轮派生特征结果仅作为历史参照。

## 验证与结果

56项相关测试通过，包括原版网络与初始化一致、报告接口不读取真实CSI/波束索引、覆盖真实SINR等传参、无过载需求数值核验、过载需求与物理干扰区分、checkpoint加载、训练路径和精确仿真路径。相同脚本化HO动作下，37维与31维预测状态版本的优化器和所有物理结果逐项一致。

两个训练seeds均完成72 episodes，耗时分别为660.52 s和643.43 s（约11.0和10.7 min）。两个seeds的最佳验证模型都位于第72轮，分数分别为17.3497531282和18.1031212110。按既定准则选择seed 20，第72轮checkpoint的SHA-256为 `50daa7cf4307a666ca8a0c58d7774e31c5f048eed98c780d8bce34394742cf0c`。未以测试结果选模或额外重训。

37维对照在本轮代码/GPU1上逐项复现历史结果。新旧测试到达哈希相同，训练轨迹片段、负载序列、奖励、超参数和冻结前端也通过配对核验。完整精确测试结果如下，其中U和时延统计沿用论文的队列长度时延代理。

| 指标 | 37维 gain_report | 上一轮62维 gain_derived | 本次31维 predicted_adapted |
|---|---:|---:|---:|
| 平均发射功率 (W) | 108.14093 | 90.72834 | 106.78217 |
| U (%) | 1.97600 | 3.13555 | 1.55226 |
| 平均时延代理 (ms) | 5.32159 | 8.73889 | 4.88402 |
| p99时延代理 (ms) | 82.97474 | 205.08978 | 59.48646 |
| 宏BS关联比例 (%) | 20.00788 | 13.10717 | 19.86509 |
| HO次数 / 车辆 / 秒 | 0.03496 | 0.09429 | 0.03668 |
| 平均BF导频数 / 车辆 / 时隙 | 0.81243 | 0.89080 | 0.81416 |

相对37维对照，功率下降1.26%；U下降0.42374个百分点（相对下降21.44%）；平均时延代理下降8.22%，p99下降28.31%。宏BS关联比例只下降0.14279个百分点，HO频率增加4.93%，BF导频数增加0.21%。因此，这次没有出现上一轮功率下降而时延恶化的权衡，但主要收益在时延，功率改善很有限。

在同一测试点，原版真实CSI输入模型为60.68850 W / 0.96509%，MEET-COBRA为29.66426 W / 0.24215%。本次仍未恢复原版的功率水平；其宏BS关联比例仍接近20%，而原版为5.82%。这与较高功率相一致，但不能据此唯一解释差距，或证明某个特征单独导致了全部改善。原版真实信息模型也不是当前信息边界内的公平对照，列出仅供历史定位。

**结论：保留31维预测状态版本作为有正向初步结果的候选，不自动替换正式baseline。** 在本次13 Mbps、一个测试seed下，它同时改善了37维对照的功率和时延，但尚无多seed、全负载证据。相对上一轮62维版本，它时延更好、功率更高，不能称为全面优于所有报告型方案。恢复特征组成有价值，但尚未解决报告型策略的高功率问题；本轮未继续调整奖励、训练时序或目标优化器。

所有结果保存于 `experiment/results/o_mappo_predicted_state_20260917/`。`architecture_audit.json`记录网络及信息边界，`training_seed*/`保存协议、训练过程和权重，`selected_policy.json`记录最终选模，`runs/`和`raw/`保存精确仿真输出，`summary.json`和`result_table.md`保存核验及对比。

## 复现

```bash
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_predicted_state_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant predicted_adapted --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_predicted_state_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant predicted_adapted --training-seed 21 --episodes 72
python experiment/summarize_o_mappo_predicted_state.py --audit
python -u experiment/summarize_o_mappo_predicted_state.py --reproduce-baseline --gpu 1
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_predicted_state_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods predicted_adapted --rates 13 --seeds 1 --workers 1 --rician --physics-gpu 1
python experiment/summarize_o_mappo_predicted_state.py --gpu 1
python -m unittest test_o_mappo_predicted_state test_o_mappo_actor_derived test_o_mappo_optimizer_information test_o_mappo_gain_report test_o_mappo_report_input test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement test_gpu_phy test_compiled_matching -q
```
