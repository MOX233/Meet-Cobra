# O-MAPPO 边缘侧预测报告输入实验

## 目的与版本

将 O-MAPPO 的 HO actor 部署接口与 MEET-COBRA 的车辆上报接口对齐。CSI 输入只采用各微基站的有序 top-5 波束对索引、desired-link 和 interfering-link 增益预测，不上传原始叠加导频观测或完整波束概率向量。

- 修改前代码快照：`a7cf3b4`。
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

代码改造和测试完成，两个训练种子正在运行。正式选模和配对测试结果完成后补充，不以短片段调试结果作为性能结论。

## 复现

```bash
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant report --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant report --training-seed 21 --episodes 72
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_report_input_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods report --rates 1,13,27,35 --seeds 1,2,3 --workers 3 --rician --physics-gpu 3
python -m unittest test_o_mappo_report_input test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement test_gpu_phy test_compiled_matching -q
```
