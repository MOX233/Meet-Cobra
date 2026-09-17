# O-MAPPO 共用增益预测前端实验

## 实现范围

本次只修改实验代码并保存结果；没有修改论文、回复信或论文图片。

- 旧实现回退标签：`pre-omappo-shared-frontend-20260917`，对应 `cdd095a58dc99504629522532a0f7cb751a99f81`。
- 初始实现提交：`d6a4cee`；后续测试、线程设置和汇总脚本另行提交。
- 主入口：`experiment/o_mappo_shared_frontend.py`。
- 结果目录：`experiment/results/o_mappo_shared_frontend_20260917/`。

旧接口仍为默认配置 `information_mode="legacy"`。新接口采用 `information_mode="shared_prediction"`、`state_variant="pilot"`。关闭 HO 中断的原实现路径仍可运行。新增测试已核对其数值输出与上述 Git 快照一致，不比较程序计时值。

## 新旧版本的实际区别

| 部分 | 旧 O-MAPPO-adapted | 新版共享前端 O-MAPPO |
|---|---|---|
| Actor 的 CSI 输入 | 根据原始 RT 信道计算的服务链路 SINR 和干扰信息 | 同 MEET-COBRA 的 8 次叠加导频观测，经相同幅度和相位预处理得到 128 维向量 |
| 其他状态 | 队列、业务率、位置、方向、速度、关联、负载、既往分配和 HO 等 | 保留这些可观测量；BS RB 负载改为此前实际分配反馈 |
| Actor 网络 | 31 维输入、64 单元隐藏层、二元 HO 触发动作 | 157 维输入、64 单元隐藏层、相同二元动作，重新训练 |
| 目标基站评估 | 扫描各候选 BS 的完整真实信道以取得最优波束增益 | 只使用共享的期望链路和干扰链路增益预测；无需原始 H 或最优波束标签 |
| 波束选择 | 到目标微基站后全搜索；保持连接时局部追踪 | 保持原规则，不使用 MEET-COBRA 的波束预测器 |
| RB 分配 | OTR-RA | OTR-RA |

新 Actor 直接使用当前观测；本次没有为 Actor 另加循环网络。跨帧记忆来自两个已固定的 stateful LSTM 增益预测器，供目标基站优化器使用。目标优化器仍保留原先最多三个候选 BS 的筛选及容量/负载/能耗目标，没有改用 GAP-HO。

### 信息边界

两种方法共用当前论文选定的两阶段 stateful 增益模型，参数不再训练。对同一车辆和时刻使用完全相同的增益预测值。缓存按时间顺序只输入当前 CSI，并为每辆车分别保留 LSTM 状态；车辆离开或轨迹中断时清除状态。不输入未来信道标签。

O-MAPPO 目标优化器不读取缓存中的波束预测字段。真实 H 仍用于仿真环境生成真实服务速率、物理干扰以及实际执行的波束探测，不作为新版 Actor 或候选基站优化器的可用输入。单元测试在删除 H 及所有真实波束标签后仍能运行目标优化器，并验证修改真实 SINR 和干扰不会改变 Actor 输入。

MEET-COBRA 比较项使用当前选定的三个 stateful checkpoint，不使用 Oracle。共享增益前端不意味着两个算法相同：HO 决策机制、目标选择目标函数、决策时机和 BF 过程仍然不同。

## 训练及评估协议

- 固定增益 checkpoint：`stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt/{desired_gain,interfering_gain}/best.pth`。MEET-COBRA 另用同目录的 `beam/best.pth`。
- GPU 5 用于生成共同的递推预测缓存；小型 MAPPO 网络及仿真环境在 CPU 上运行。
- RL 训练：200–700 s 时间段内随机抽取 30 s 片段；每个训练种子 72 个片段，训练种子为 20、21；负载循环为 1、7、13、19、27、35 Mbit/s。
- 保留旧版本的帧级流体近似训练方式、PPO 超参数和 `qos_energy020_load1` 奖励。训练中增加 HO 的可服务时间比例。该训练近似并不等于最终逐时隙仿真。
- RL 验证：700.1–710 s，负载 7、19、35 Mbit/s；每 12 个训练片段验证一次。验证缓存实际准备到 730 s，但 710 s 之后未用于选 checkpoint。
- 选择准则事先固定为 `max(validation U[%]) + 0.01 × mean(validation power[W])`。选定 seed 20、更新 48 的 checkpoint，验证分数 14.9118976892。此分数属于训练近似环境，不是下面测试曲线的违反概率。
- 选定 checkpoint SHA-256：`57717aa1e24a01c5ec5c5aa7531ff4f08f4499f9afe9b560e9c2fb85738a1226`。
- 测试：800–830 s 原有轨迹，1、13、27、35 Mbit/s，随机种子 1、2、3；不据测试结果重新选模型。每组去掉前两帧，保持既有统计口径。
- 统一 100 ms 帧、1 ms 时隙、20 ms latency threshold、10 ms HO 中断；切换车辆前十个时隙不分配 RB、不进行 BF 探测，其到达流量继续入队。新旧 O-MAPPO 的目标容量估计都考虑可服务时间损失；MEET-COBRA 使用已讨论的 GAP-HO 容量修正。
- Poisson 到达和初始队列逐车配对；Rician 扰动按帧、时隙及排序后的车辆配对。对相同 rate/seed，三种方法保存相同 traffic SHA-256。
- 旧 O-MAPPO 使用 `o_mappo/final_load1/final_policy.pt`，不重训；因此它是“旧策略在统一新环境下”的对照，不是原论文旧结果的原样复刻，也不是严格隔离训练预算的单因素消融。
- 最终评估的目标分配均使用相同设置的 MILP 求解器；MEET-COBRA 使用原 GAP-HO 的两轮实现。为避免服务器每个求解器生成 128 个线程，评估工作进程内显式设置 HiGHS 单线程，不修改求解器目标或可行域。

## 导频开销及解释边界

O-MAPPO 切换到微基站时执行 256 个波束对的搜索，保持连接时采用邻域追踪；实际探测数按原模拟器规则计入可用于数据传输的时隙比例。HO 中断期间暂停相应探测，搜索开销从恢复服务后开始计算。MEET-COBRA 采用 top-5 候选及提前停止规则。

新版候选基站评估不再做旧版不计费的全候选真实信道扫描。这解决的是本次讨论的额外 CSI 信息优势，不意味着完整控制平面开销问题均已解决：本实验沿用现有模拟器的波束搜索开销摊分方式，没有增加逐项信令协议或推理计算能耗模型。共同叠加导频观测作为现有前端输入，没有在 BF pilot 统计之外另加一次独立控制开销。因此表中 pilot 指标是 BF 探测开销，不是全部控制开销。

干扰增益标签继续采用既有训练代码中的最大信道元素幅度定义；没有在本轮擅自改变该定义。该标签与其他建模定义的一致性问题需要在相应审稿意见下单独处理。

## 数值实现核验

新增批量 PET-BF 只合并数值运算，保留候选顺序、提前停止判据、所选波束和收费的导频数。随机输入下，增益误差在测试容差 `1e-10 dB` 内，波束及探测数相同。系统级短片段可能因浮点累计及优化器处理产生微小数值差异；不宣称所有完整仿真输出逐位一致。

本轮测试集合：旧 O-MAPPO、HO 中断、GAP-HO 及新增共享前端测试，共 28 项通过。测试包含旧版本回归、候选优化器信息隔离、Actor 信息隔离、HO 容量因子、到达和中断处理、配对 Rician 扰动以及批量测量核验。

## 测试结果

完整的三种方法、四个负载点、三个种子的评估正在运行。完成后在此补充结果和结论；当前不将短片段调试结果作为性能结论。

## 复现命令

```bash
python -u experiment/o_mappo_shared_frontend.py prepare --gpu 5
python -u experiment/o_mappo_shared_frontend.py train --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --training-seed 21 --episodes 72
python -u experiment/o_mappo_shared_frontend.py evaluate --methods legacy,shared,meet_cobra --rates 1,13,27,35 --seeds 1,2,3 --workers 12 --rician --vectorized-pet
python experiment/summarize_o_mappo_shared_frontend.py
python -m unittest test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement -q
```

`frontend_manifest.json` 保存前端 checkpoint 和输入数据的 SHA-256；`training_seed*/` 保存训练、验证日志和模型；`runs/` 保存逐项统计；`raw/` 保存逐帧和逐车队列。最终 `comparison_summary.json/csv`、`paired_differences.csv`、`result_table.md` 及 `comparison_curves.pdf/png` 由汇总脚本生成。论文图不在该脚本的写入范围内。
