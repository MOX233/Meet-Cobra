# O-MAPPO 边缘侧预测报告输入实验

## 目的与版本

将 O-MAPPO 的 HO actor 部署接口与 MEET-COBRA 的车辆上报接口对齐。CSI 输入只采用各微基站的有序 top-5 波束对索引、desired-link 和 interfering-link 增益预测，不上传原始叠加导频观测或完整波束概率向量。

- 修改前代码快照：`a7cf3b4`；报告输入实现提交：`51547bb`。
- 新配置：`state_variant="report"`、`information_mode="shared_prediction"`；旧 `pilot` 和 `legacy` 路径保留。
- 结果目录：`experiment/results/o_mappo_report_input_20260917/`。
- 共用预测缓存：`experiment/results/o_mappo_shared_frontend_20260917/{train,test}_prepared.pkl`。缓存和冻结的三个 stateful NN checkpoint 不重新生成或选取。
- 本轮不修改论文、回复信和正式图片。

## 输入与算法边界

Actor 的 57 维状态包含原有 29 维非 CSI 状态和 28 维预测报告。非 CSI 状态包括队列、业务率、关联、负载、车辆移动信息、既往 RB 分配和所用波束等，沿用上一轮。对每个微基站依次编码两个预测增益和五个按预测概率排序的波束对索引。增益采用固定仿射缩放 `(gain_dB + 100)/40`，索引除以 255；不重新排序、不输入概率、不用测试集统计量做归一化。

四个微基站共上报 `4 × (5 × 8 + 2 × 32) = 416 bit/vehicle/frame`，与论文的 CSI 预测报告一致。57 维 FP32 张量是服务器端编码后的 actor 输入，不代表上行传输 57 个浮点数。416 bit 仅指该 CSI 预测报告，不包含其他状态、协议封装或控制信令。

增益和波束 NN 仍在车辆侧递推。Actor 和目标选择优化器在边缘侧；目标选择优化器沿用上一轮共享增益输入。Actor 隐藏层仍为 64 单元，动作仍是二元 HO 触发。未改变目标优化器候选数、目标函数、奖励、决策距离门限、OTR-RA、HO 中断设置或波束执行方式。预测波束索引本轮作为 actor 状态使用；没有将 O-MAPPO 的全搜索/局部跟踪替换成 PET-BF。

真实 RT 信道仅作为环境和共同底层物理接口的输入；新 actor 接口读取不到原始 H、导频观测或真实 SINR。公共 OTR-RA 的既有 CSI 和导频计费假设未在本轮重构，不能据此宣称整个系统已经消除了所有理想 CSI 假设。

## 固定协议

- 两个训练 seeds：20、21；各 72 episodes。每个 episode 从 200–700 s 随机抽取 30 s，负载依次循环 1、7、13、19、27、35 Mbps。
- 验证：700.1–710 s，负载 7、19、35 Mbps，每 12 episodes 验证一次。
- 沿用上一轮选模准则：`max(validation U[%]) + 0.01 × mean(validation P[W])`。不使用测试集挑选 checkpoint。
- 奖励、PPO 超参数、帧级近似训练环境与上一轮一致；最终测试采用逐时隙模拟器。
- 测试：800–830 s，负载 1、13、27、35 Mbps，seeds 1、2、3；10 ms HO interruption；相同到达、初始队列、Rician 随机数流。统计去掉前两帧。
- 对照：原有 O-MAPPO、上一轮 pilot-input O-MAPPO、MEET-COBRA。上一轮完整结果保持不变；只有物理后端及配对随机数一致性核验通过才复用其他 GPU 上的结果。
- 与 pilot-input 版本相比，训练数据抽样、seed、验证和奖励相同，但输入维数改变使网络参数量和初始化不同。因此这是部署输入设计的对比，不是固定策略上替换一个数值输入的消融。

## 验证

新旧接口、序列化、报告顺序与缩放、报告 416-bit 大小、缺少导频字段时的训练与逐时隙仿真、真实 SINR 改变不影响 actor 输入等测试均通过。合并既有基线、HO、GAP-HO、GPU 物理层与 matching 回归测试共 34 项通过。

## 进度与结果

两个训练种子均完成 72 episodes，耗时约 14 min/seed。seed 20 的最佳 checkpoint 位于 episode 36，验证分数为 17.6207669794；seed 21 的最佳 checkpoint 位于 episode 60，分数为 17.9310004287。因此按固定准则选择 seed 20、episode 36，SHA-256 为 `cd3c0149d688780c5bf45fd18f1a9cc62df51d2f01ad2ec5992d5c202db244f8`。该分数来自帧级近似验证环境，不是最终逐时隙仿真的违反概率。

GPU 0、3 与 5 的配对信道和 PET 测量结果在三个审计 seeds 下逐项相等；旧 O-MAPPO 在 1 Mbps、seed 1 的完整 30 s 跨 GPU 重跑中，所有性能指标也完全一致。因此新报告输入模型的 1、13、27 Mbps 在 GPU 3 上评估，35 Mbps 在 GPU 0 上评估，对照结果复用上一轮 GPU 5 的完整配对结果。

已另行完成 MEET-COBRA 的 1 Mbps、seed 1 完整 30 s 核验，原 Python matching 与保留相同增广路径顺序和阈值的 Numba 编译版本在所有记录指标上完全一致，因而其余未完成的对照任务使用这一数值加速。最初在 JIT 初始化后 fork 的可选核验未完成，被终止后改为独立单进程核验并通过；驱动已明确禁止这种 fork 用法。此前已完成的结果没有覆盖，物理随机数、算法、模型和配置均未改变。该调整不是改用另一种匹配或优化方法。

四个代表性负载点、三个测试 seeds 已全部完成。新报告输入模型共 12 组完整测试；复用并补齐此前三个对照共 36 组，合计 48 组。汇总脚本核对所有配对到达哈希、模型 checkpoint、物理后端及跨 GPU 完整参考结果后生成统计文件。这不是论文全部负载点的最终曲线实验。

### 三个配对 seeds 的均值

每个单元格依次为系统发射功率 W / 违反概率 %。逐点标准差、p99、HO 和 BF 导频数见结果目录中的 `result_table.md` 和 `comparison_summary.csv`。

| 到达率 Mbps | 原 O-MAPPO | 导频输入 O-MAPPO | 报告输入 O-MAPPO | MEET-COBRA |
|---|---|---|---|---|
| 1 | 5.047 / 0.0328 | 19.748 / 0.5320 | 14.834 / 0.2600 | 4.925 / 0.1940 |
| 13 | 60.658 / 0.9843 | 74.973 / 3.5535 | 81.106 / 3.1064 | 29.665 / 0.2420 |
| 27 | 168.326 / 20.3363 | 175.698 / 19.0127 | 176.914 / 20.9981 | 125.730 / 0.5095 |
| 35 | 183.656 / 60.2419 | 185.456 / 58.2859 | 183.632 / 61.2094 | 185.722 / 39.0737 |

与上一轮导频输入版本相比：

- 1 Mbps：功率降低 24.88%，违反概率相对降低 51.14%（下降 0.2721 个百分点）；三个 seeds 均表现为两项改善。
- 13 Mbps：违反概率相对降低 12.58%，但功率增加 8.18%，属于折中而非支配性提升。
- 27 Mbps：功率增加 0.69%，违反概率增加 1.9854 个百分点；没有改善，p99 也明显增大。
- 35 Mbps：功率降低约 0.98%，但违反概率增加 2.9235 个百分点；不能把较低功率单独解释为更好的能效。此时所有方法均出现较大的队列违反概率。

### 判断与后续方向

保留报告输入设计是合理的，因为边缘侧 actor 直接消费车辆已上报的预测结果，不再要求中心侧获得原始高维导频。信息接口合理不等于性能必然更好；本轮结果只支持低负载改善，并未显示全负载优势。当前 checkpoint 不宜因为接口更合理就直接冻结为最终论文 baseline。原正式 checkpoint 和论文图没有被覆盖。

在 35 Mbps 时，新模型的 HO 频率为 0.6572 次/车辆/秒，导频输入版本为 0.5869 次/车辆/秒。更频繁切换与更高违反概率同时出现，但本轮没有隔离因果关系，不能断言退化完全由 HO 中断造成。两个模型均由相同帧级近似环境训练；验证轨迹、时长及环境与最终逐时隙测试不同，也不能把验证到测试的落差单独归因于输入设计。

若继续优化，应保持同一预测报告接口，优先检查训练与部署的队列/负载状态时序和奖励反馈是否一致，再考虑从同一报告派生候选可行性或波束变化特征、调整训练方式。本轮未据测试结果追加训练或重新选择模型，也没有恢复真实候选信道输入。

已保存：`training_seed*/`、`selected_policy.json`、`runs/`、`raw/`、`gpu_pairing_audit.json`、`gpu_crosscheck/`、`comparison_summary.json/csv`、`paired_differences.csv`、`result_table.md`、`comparison_curves.pdf/png`。实验图只写入结果目录，不影响论文图片。

## 复现

```bash
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant report --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant report --training-seed 21 --episodes 72
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods report --rates 1,13,27 --seeds 1,2,3 --workers 1 --rician --physics-gpu 3
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods report --rates 35 --seeds 1,2,3 --workers 1 --rician --physics-gpu 0
python experiment/audit_report_input_gpus.py
python experiment/summarize_o_mappo_report_input.py
python -m unittest test_o_mappo_report_input test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement test_gpu_phy test_compiled_matching -q
```
