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

## 进度与结果

实现和测试完成，两个训练种子正在运行。选模和完整逐时隙测试完成后补充结果；不把训练或短片段调试指标作为正式性能结论。

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
