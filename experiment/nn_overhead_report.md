# R1C7 与 R2C6：NN 推理开销与控制面上报预算

> 2026-09-13 更新：本文保留 2026-09-11 的历史审计及原始测量口径。
> 当前论文已采用两阶段训练模型和跨帧递推，权威计时结果见
> [新的有状态推理报告](nn_overhead_stateful_report.md)，中位数 5.33 ms、95 分位 5.74 ms。
> 下文旧模型的有限窗口和单帧测量不再作为当前模型的正式时延结果。

审计日期：2026-09-11。仅运行独立的模型统计和推理基准，没有修改 NN、HO、BF、RA 实现或 checkpoint，也没有重跑系统性能曲线。经作者明确，实际上报的是排序后的候选波束对索引，而非完整概率分布；现已更正预算和 R2C6，并在正文 Section IV-E 统一说明上报内容，删去 Section IV-A 的重复说明。

## 1. 结论与统计边界

- 论文所用三个独立预测模型合计 **2,127,912 个参数**，FP32 参数占 **8.511648 MB**（1 MB = 10^6 bytes，等于约 8.11734 MiB）。正文和回复信统一报告 8.51 MB 的参数存储量，不将其称为推理总内存占用。
- 按实际代码的 10 帧历史输入，三个模型合计 **5,638,152 MACs / 11,276,304 matrix FLOPs**。这里一个乘累加计两次浮点运算；不将归一化、非线性激活等未统计的操作冒充为已包含。
- AMD EPYC 7742 CPU 单线程、batch size 1、FP32、关闭梯度记录时，三个预测模型生成 top-5 候选索引和两类增益的合计时延中位数 **5.644 ms**，95 分位 **5.910 ms**，99 分位 **6.142 ms**。这些样本已在原始测量中保存，本轮没有重新计时。
- 每个微 BS 上报 5 个 8-bit 波束对索引和两个 FP32 增益，四个微 BS 共需 **416 bit/vehicle/frame = 4.16 kbit/s/vehicle**，其中索引 160 bit、增益 256 bit。这是载荷预算，不是实际空口吞吐量测量。
- 因此可以补充可核查的推理复杂度、参考硬件时延和预测上报预算；不能据此声称已完成车载硬件验证、端到端控制周期测试或包含所有开销的系统评估。

## 2. 可复现文件与命令

脚本：`experiment/benchmark_nn_overhead.py`；检查：`test_nn_overhead.py`。

```bash
python experiment/benchmark_nn_overhead.py --output experiment/results/nn_overhead_20260911/cpu_1thread --threads 1
python experiment/benchmark_nn_overhead.py --output experiment/results/nn_overhead_20260911/cpu_4threads --threads 4 --lengths 10
python -m unittest test_nn_overhead -v
```

脚本不覆盖已有输出；复跑时请指定新的输出目录。两个正式结果目录分别包含：

- `metadata.json`：软硬件、线程数、CPU affinity、输入样本编号、数据和代码 SHA-256。
- `model_inventory.json`：checkpoint 路径和 SHA-256、结构、参数及模型状态字节数。
- `arithmetic.json`：每个执行到的 Linear/LSTM 层的 MAC 数及算子 profiler 交叉检查。
- `timings.json`：各轮和合并后的统计量。
- `raw_latency_ms.npz`：全部逐次时延样本，单位 ms。
- `control_payload.json`：原始上报预算，保留用于追溯；其中采用完整分布的主口径已被下述修正版取代。

更正后的权威预算保存在 `experiment/results/nn_overhead_20260911/control_payload_ranked_indices.json`。原始测量目录及哈希不作追溯性改写；当前脚本已默认使用排序索引上报，未来复跑会输出更正后的预算。

`cpu_1thread_profiler_diagnostic/` 是初次检查算子计数口径时留下的诊断目录，不是正式计时结果。6 项单元检查全部通过；正式运行中各模型、各输入长度的矩阵 FLOPs 均通过独立 profiler 交叉检查。

## 3. 模型和历史序列

直接使用 `utils/NN_utils.py` 中的 `BeamPredictionLSTMModel` 和 `BestGainPredictionLSTMModel`，严格加载 `utils/sim_utils.py:get_default_sim_params()` 指定的三个 checkpoint。每个模型有一个任务内共享的 LSTM 和四个 BS 输出头，不是总共一个共享于三任务的模型，也不是十二个独立模型。

输入每帧为 128 个实数（`2 * M_R * N_P`），LSTM hidden dimension 为 128，batch size 为 1。当前 `predict()` 对完整的历史窗口重新执行 LSTM，不保留跨调用的 hidden/cell state。因此主结果统计 10 帧窗口，不能用一次单帧递推的计算量代替。

| 模型 | 参数数 | 10 帧 MACs | 10 帧 matrix MFLOPs |
|---|---:|---:|---:|
| Beam predictor | 1,060,864 | 2,228,224 | 4.456448 |
| Desired-link gain predictor | 533,524 | 1,704,964 | 3.409928 |
| Interfering-link gain predictor | 533,524 | 1,704,964 | 3.409928 |
| 三模型合计 | 2,127,912 | 5,638,152 | 11.276304 |

FP32 参数共 8,511,648 bytes（8.511648 MB）；包括 BN 等 buffers 的三个 state dictionaries 共 8,615,040 bytes（8.61504 MB）；原始三个 `.pth` 文件共 8,863,966 bytes。三者不可混为运行时峰值内存或完整模型传输空口开销。模型安装及更新的下发频率没有在系统模型中规定，故不将模型大小按每帧反复计费。正文仅报告参数存储量，其他实现细节保留在本报告中。

对于长度为 T 的历史输入，每个 LSTM 的矩阵 MAC 数为 `4*T*H*(D+H)`；各输出头按实际执行的线性层计数，包括改变维度时的残差投影支路。三模型合计：

`matrix FLOPs(T) = 786,432*T + 3,411,984`。

因此 T=1、5、10 分别为 4.198416、7.344144、11.276304 MFLOPs。T=1 仅代表较短输入，不代表已经验证了与当前滑动窗口推理等价的缓存状态优化。

计数覆盖 LSTM 的四门矩阵运算和所有 Linear 层，排除 bias、BN、激活、残差逐元素加法、softmax、top-K 和增益反归一化。计时则包含调用范围内真实执行的这些操作，不能把计时边界与 FLOPs 边界混同。PyTorch 官方说明 `with_flops` 对部分算子提供估算，因此这里另以解析计算校验，而不直接依赖融合 LSTM 的 profiler 汇总。计数检查临时关闭 oneDNN，正式计时保留 oneDNN；额外报告的逐元素 `mul` FLOPs 不纳入矩阵计数。[PyTorch profiler 文档](https://docs.pytorch.org/docs/stable/profiler)，[LSTM 定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html)。

## 4. 实测 inference latency

硬件为 AMD EPYC 7742，PyTorch 2.6.0+cu118，FP32，CPU affinity 0–3，inter-op threads 为 1。分别测试 intra-op threads 为 1 和 4，两组顺序运行，未同时竞争这些 CPU 核。当前 `nvidia-smi` 无法连接驱动，`torch.cuda.is_available()` 为 False，故没有 GPU 测量值。

使用论文现有 `data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl` 的真实预处理 CSI，以固定 seed 选择 32 个有效 10 帧输入，并取后 1、5、10 帧。每种配置运行三轮，每轮预热 100 次后记录 300 次，共 900 个计时样本。排除模型和数据加载、CSI 采集及预处理、控制信令和传输时间。

主口径是三个模型依次调用现有的预测接口，从 NumPy 预处理 CSI 输入到排序后的 top-5 波束对索引及两类增益输出，包含张量转换、top-K 选择和增益反归一化。使用 `eval()` 和 `torch.inference_mode()`，没有改变模型权重及前向计算结构。代码直接对 logits 作 top-K 选择；softmax 不改变排序，故不必为取得候选索引另外计算完整概率分布。下面同时保留完整分布输出的原始计时，供对照。

| 配置 | 中位数 (ms) | 95 分位 (ms) | 99 分位 (ms) |
|---|---:|---:|---:|
| 单线程，1 帧输入，三模型完整输出 | 5.262 | 5.721 | 8.905 |
| 单线程，5 帧输入，三模型完整输出 | 5.465 | 5.794 | 7.857 |
| 单线程，10 帧输入，三模型完整输出 | 5.585 | 5.848 | 5.989 |
| 四线程，10 帧输入，三模型完整输出 | 5.554 | 5.843 | 6.980 |
| 单线程，10 帧输入，现有 top-5 `predict()` 接口（主结果） | 5.644 | 5.910 | 6.142 |
| 单线程，10 帧输入，保留历史接口的梯度记录 | 7.772 | 8.221 | 9.513 |

最后一行是对历史调用方式的诊断，不是训练，也不是主推理口径。正式计时关闭梯度记录；已有网络仿真程序没有因此被修改。完整概率分布与 top-5 接口之间的候选索引及增益结果通过一致性检查。

单模型的 resident-tensor forward 中位数依次为 1.728、1.638、1.627 ms（单线程、T=10）。它们不含 NumPy 转换和输出后处理，不应简单相加后当作完整三模型流程的实测时延。

主结果约为 100 ms PHO 帧长的 5.64%，但只代表这一服务器 CPU 上的推理阶段。计时不含报文序列化与传输；不能据此保证特定 UE 上的时延、完整控制链路能在帧内完成、20 ms 业务约束或能耗收益。

## 5. 控制面预测上报预算和同口径比较

每车向宏 BS 上报各微 BS 的 top-M_P 波束对索引及两类增益，不上传概率值。索引按预测概率降序排列，因此列表本身保留探测顺序，不需要额外的概率值或排序字段。令每个增益占 b_G bit，则：

`B_pred = M * (M_P*ceil(log2(M_T*M_R)) + 2*b_G)` bit/vehicle/frame。

在 `M=4, M_T=32, M_R=8, M_P=5, b_G=32` 下，每个索引需 8 bit，共得到 `4*(5*8+2*32)=416` bit/vehicle/frame。仅波束索引占 `4*5*8=160` bit，其余 256 bit 是增益。该预算假设紧凑索引编码与 FP32 增益，不是 Python int64 索引数组的内存大小；具体控制协议没有实现，故仍是 payload 预算，不是抓包统计。

在同样的 32-bit 实数精度、100 ms 上报周期和接收端采样能力下：

| 信息获取架构 | 上行 payload (bit/vehicle/frame) | 上行速率 (kbit/s/vehicle) | 每帧探测次数 |
|---|---:|---:|---:|
| 车辆推理，上报排序候选索引和两类增益 | 416 | 4.16 | 8 |
| 上报四个完整复数 MIMO CSI 矩阵 | 65,536 | 655.36 | 128（逐 BS 发射波束扫描） |
| 上报新增叠加导频观测，宏 BS 执行同一 NN | 4,096 | 40.96 | 8 |

第三行假设宏 BS 缓存历史输入，只上报当帧新增的 64 个复数观测，不重复发送整段 10 帧历史；推理工作转移到宏 BS，但计算量没有消失。第二行和第三行是信息获取方式的参照，不是已在相同系统实验中验证过的额外控制算法。

预测 payload 分别约为完整 CSI 和新增叠加导频观测的 0.635% 和 10.156%，即在这些编码假设下分别减少约 99.37% 和 89.84%。这仅比较上行预测载荷，不意味着总控制开销或系统能耗同比减少。8 对 128 的探测数是在相同接收采样能力下的帧级比较，不是所有波束搜索、反馈和训练开销的总量比较。

现有 30 s 数据中平均约 136.296 辆车，对应约 0.567 Mbit/s 的聚合逻辑 payload。没有在当前系统吞吐量、功率或 U 中另行扣除这部分控制开销。这里也不含共同的宏链路测量、消息标识、协议头、信道编码、重传、HO 信令及模型下发等项。

## 6. 更正说明

上一版将完整 NN 输出等同于上传内容，得到 33,024 bit/frame，并把索引上报列为可选方案。这不符合作者明确的实际设计。现有 `BeamPredictionLSTMModel.predict(K=5)` 已返回排序后的五个索引，GAP-HO 使用两个增益模型的输出，因此本次只是纠正通信载荷口径与正文表述，没有更改 NN 或控制算法，也不需要重新运行系统实验。完整概率输出的原始测量仍保留作对照，不能再作为主上报预算。

推理时延同步选用原始结果中已经测得的 `T10_three_models_top5_predict`，而非重新运行实验或将完整分布计时冒充索引输出计时。矩阵 FLOPs 与参数量不变。

## 7. 用于 R1C7 与 R2C6 的范围

可直接用于回复信的是三模型参数及矩阵 FLOPs、带硬件和计时边界的参考 inference latency、带编码假设的预测上报 payload，以及以上同精度、同周期的预算比较。保留现有 HO 模型及 GAP-HO 容量修正说明。

正文已在 Section IV-E 统一说明上报内容，并删去 Section IV-A 的重复说明。在 Section V-B 末尾使用 Table III 和两个精简段落报告模型参数量、矩阵 FLOPs、FP32 参数存储、推理时延及同口径上报预算，不另设 subsubsection。正文不再解释 MB 换算，也不展开计时步骤和开销排除项；这些细节保留在回复信及本报告中。R1C7 与 R2C6 共用这些结果，分别回应各自问题。R2C6 的 HO 说明沿用 R1C2 和 R2C2 的术语。R1C2、R2C2 及对应正文修改已转蓝；新开销内容和 R2C6 的算法修改仍保留红色。

本轮未重跑系统实验。正式提交前仍需根据完整 HO-aware 系统实验更新性能结果和 Conclusion。没有完成空口级控制协议开销、全部 baseline 端到端控制流程时延或车载计算平台性能测量。
