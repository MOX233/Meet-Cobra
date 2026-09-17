# O-MAPPO actor 派生状态特征实验

## 本轮任务与边界

在固定当前报告型目标 BS 优化器的前提下，研究 actor 能否通过更直接的资源需求和节能收益特征学到更好的 HO 触发策略。按用户要求，实验前先核对原版与当前 actor 的输入、网络与输出。

- 修改前版本：`d60496e`；新 actor 实现：`d342f37`。
- 新配置：`state_variant="gain_derived"`，`information_mode="shared_prediction"`。
- 当前对照：最近的 `gain_report`（仅增益报告）版本；原版和含索引报告版作为历史参考。
- 不改变优化器的目标函数、候选数、输入来源、容量处理或求解方式；不改变 BF、OTR-RA、导频计费、10 ms HO 中断、奖励、训练轨迹划分或 PPO 超参数。
- 不修改论文、回复信和正式图片，不覆盖原 checkpoint。

## 原版与当前 actor 的实际差异

下表指本仓库已保存的 O-MAPPO-adapted checkpoint，不是对原始文献网络结构的推测。

| 项目 | 原版 adapted | 当前 gain_report | 本次 gain_derived |
|---|---|---|---|
| Actor 输入维数 | 31 | 37 | 62 |
| Actor 网络 | Linear(31,64), ReLU, Linear(64,2) | Linear(37,64), ReLU, Linear(64,2) | Linear(62,64), ReLU, Linear(64,2) |
| Actor 参数数 | 2,178 | 2,562 | 4,162 |
| Critic 网络 | 94→64→1 | 112→64→1 | 112→64→1，不扩展输入 |
| Critic 参数数 | 6,145 | 7,297 | 7,297 |
| Actor 循环结构 | 无 | 无 | 无 |
| Actor 输出 | 两个动作的 logits | 相同 | 相同 |
| 动作语义 | 0：不 HO；1：触发 HO | 相同 | 相同 |
| 选择方式 | 训练按分类分布采样；评估取 argmax | 相同 | 相同 |

原版与当前共有 29 个基础状态槽位：既往 HO 指示、系统吞吐比例、自身 RB 比例（3 维），五个 BS 的车辆数（5 维），当前关联 one-hot（5 维），位置、方向、速度（5 维），队列与到达率（2 维），五个 BS 的 RB 负载（5 维），当前实际 TX/RX 波束的正余弦编码（4 维）。这些槽位的编码/归一化规则保留，但 BS 负载在原版为基于当前信道的需求估计，当前精确仿真中改为上一帧实际占用反馈。

原版另加当前服务链路 SINR 和干扰噪声比（2 维），由仿真真实信道计算。当前版删除这两项，改为四个微基站各两个增益预测（8 维），采用 `(gain_dB+100)/40` 固定缩放。当前版没有原始导频、完整 CSI、预测波束索引或波束概率输入；保留的四维实际波束编码不是预测候选索引。

上一轮含索引报告版为 57→64→2，增加的是 20 个按概率排序的预测候选波束索引，actor 参数数为 3,842；其 critic 为 172→64→1。该版本不作为本次特征实验的训练起点。

所有车辆共享 actor 参数。Critic 输入由各车基础状态的均值、最小值、最大值，以及归一化车辆数构成。Critic 的值估计用于训练，不负责选择目标 BS；精确模拟器沿用既有接口计算其输出，但该值不参与测试时动作的 argmax。

两版 actor 均不是 LSTM。冻结的 stateful LSTM 增益预测器是独立前端，其预测结果供 actor 和目标选择优化器使用。原版与当前 actor 权重分别训练，不能把二者的区别理解为在相同权重下换几个输入数值。

## 本次新增的 25 个 actor 特征

保留当前 37 维状态的数值和顺序，在末尾增加以下五组特征，每组覆盖宏 BS 和四个微 BS，共 25 维：

1. 预计 RB 需求占 BS 容量的比例；
2. 预计帧平均发射功率代价；
3. 预计可用容量与本车需求之间的余量；
4. 相比维持当前关联的预计功率节省；
5. 相比维持当前关联的预计 RB 需求比例减少量。

令帧时长为 T，当前待服务量为 B=q+λT。每 RB 速率由报告增益、已有 BS 实际占用比例、发射功率和带宽计算，扣除相应的 BF 导频开销。维持关联的估计采用原有每时隙 tracking pilot 预算；切换到微 BS 采用既有全搜索摊销开销。名义需求 k=B/(T r)，预计功率代价为 kp；切换候选的容量需求除以 1−τ_HO/T，但功率代价不乘该因子。

容量余量以 `1−load` 为基础，并在当前 BS 加回本车已有 RB 比例。需求比例截断到 [0,2]，功率除以固定的 10 W 后截断到 [0,5]，容量余量和需求差截断到 [−2,2]，功率节省除以 10 W 后截断到 [−5,5]。所有尺度在训练前固定，不利用测试集统计量。正的 savings 表示候选优于维持关联。

这些是 actor 使用的特征估计，不会改写优化器输入或强制 actor 按特征符号行动。当前 BS 的微链路也使用“预测最优波束增益”，并非实际跟踪波束的测量值；本轮没有借由特征构造偷偷补回这一缺失的真实 CSI。

Critic 仍只池化前 37 维，不接收新增 25 维，从而不把 critic 特征扩展混入本轮。新 actor 输入层参数数仍不可避免地增加，因此这是确定性特征表达的实验，而不是严格保持参数总量的比较。

## 配对初始化与训练协议

- 从随机初始化开始，不加载任何已训练 actor 权重。
- 对同一 seed，新 actor 的共有权重与 37 维模型的随机初始化逐项相同；新增 25 列输入权重初始化为零。Critic 的初始化也逐项相同。由此两者从相同初始决策函数出发，新列通过梯度学习，单元测试已验证梯度非零。
- 沿用训练 seeds 20、21，各 72 episodes；200–700 s 随机抽取 30 s 片段，负载循环 1、7、13、19、27、35 Mbps。
- 共用 `o_mappo_shared_frontend_20260917` 冻结预测缓存；NN 不重新训练或推理。
- 验证仍为 700.1–710 s、7/19/35 Mbps，每 12 episodes 一次。
- 选模准则保持 `max(validation U[%]) + 0.01×mean(validation P[W])`，跨两个训练 seeds 选定单一 checkpoint，不用测试结果选模。
- 奖励仍为 `qos_energy020_load1`；actor/critic 学习率均为 5e−4，PPO clip 0.2，discount 0.9，GAE 0.5，PPO 更新 4 轮，batch 256，entropy coefficient 0.01。
- 保留现有帧级近似训练环境。其状态时序与精确逐时隙测试的既有差异没有在本轮一并修改，以免混入另一个实验因素。
- 测试为 13 Mbps、seed 1、800–830 s，10 ms HO 中断，统计去掉前两帧；到达、初始队列、GPU Rician 随机数与当前报告模型配对。先在当前代码/GPU 上重跑 gain_report 对照，核验历史结果一致性。

## 验证与结果

接口边界、基础状态逐项不变、critic 输入不变、配对初始化、优化器候选逐项不变、checkpoint 保存加载以及两类仿真路径均已测试。训练和测试结果将在完成后更新于此。

保存目录：`experiment/results/o_mappo_actor_derived_20260917/`。`architecture_audit.json` 保存已加载 checkpoint 的完整网络、特征与参数配置；`training_seed*/` 保存训练协议、过程、选择结果及权重；`runs/`、`raw/` 保存测试指标和原始数组；`summary.json`、`result_table.md` 保存配对汇总。

## 复现

```bash
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_actor_derived_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant gain_derived --training-seed 20 --episodes 72
python -u experiment/o_mappo_shared_frontend.py train --output experiment/results/o_mappo_actor_derived_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --state-variant gain_derived --training-seed 21 --episodes 72
python experiment/summarize_o_mappo_actor_derived.py --audit
python -u experiment/summarize_o_mappo_actor_derived.py --reproduce-baseline --gpu 4
python -u experiment/o_mappo_shared_frontend.py evaluate --output experiment/results/o_mappo_actor_derived_20260917 --cache-root experiment/results/o_mappo_shared_frontend_20260917 --methods gain_derived --rates 13 --seeds 1 --workers 1 --rician --physics-gpu 4
python experiment/summarize_o_mappo_actor_derived.py --gpu 4
python -m unittest test_o_mappo_actor_derived test_o_mappo_optimizer_information test_o_mappo_gain_report test_o_mappo_report_input test_o_mappo_shared_frontend test_o_mappo test_ho_interruption test_gap_refinement test_gpu_phy test_compiled_matching -q
```
