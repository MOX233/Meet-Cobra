# E_all 配套训练：与原版等预算续训的配对比较

日期：2026-09-24。状态：正式配对续训已启动，前 10 轮及第一次续训验证完成；最终结果尚待训练及精确评估完成。

## 研究问题与控制变量

第一阶段固定权重时，E_all 在三个验证负载下显著降低功率，但并非所有时延指标和 seed 都改善。本阶段检查配套训练是否能进一步改善 E_all，以及改进是否超过原版等预算续训。

- 两组为 legacy 和 E_all，各训练 seeds 11、22、33，均从当前正式两层模型的同一 actor **及 critic** 权重出发。原 checkpoint 不覆盖。
- 架构、31 维特征及归一化、二元 HO 触发动作、奖励、候选 BS 数量、目标优化函数不变。Actor 为 `31→64→64→2`。
- 保留 minibatch size 256、每轮 PPO epochs 4。两组均重新初始化 Adam；学习率从约 1e-4 余弦下降至第 120 轮的 1e-5，再保持至第 160 轮；熵系数 0.002。
- 六组均训练 160 轮，每轮平衡采样 1、3、…、35 Mbps 各 10 s。相同 seed 和轮次下，两组使用相同轨迹窗口和采样随机种子。
- E_all 同时向 actor 的 SINR、干扰及 BS 占用特征、target BS optimizer、RA 近似模块提供有界占用与波束平均干扰增益。保留当前帧真实 CSI、分层 32 次搜索及局部追踪；本阶段不是预测输入实验。

## 训练环境与精确验证的区别

训练仍使用既有帧级近似环境。E_all 的 RA 近似在共同估计的干扰下执行既有贪心资源分配，不再运行另一套私有负载迭代。分配后的实际 RB 占用继续用于功率、拥塞奖励及历史反馈，不用需求估计替代实际占用。训练环境采用平均到达量，不能把其 reward 当作精确时隙仿真的系统性能。

精确验证和测试均使用上一阶段已验证的 E_all 包装：显式 RB、方向相关干扰、Poisson 到达、10 ms HO 中断、原 BF 开销及时隙级 OTR-RA。所有真实服务量都进入队列更新。代码不修改生产默认路径。

## 数据与模型选择

- 训练：200–650 s 内随机 10 s 片段；每轮每个负载均采样。
- 帧级验证：700–710 s，全部 18 个负载，每 10 轮一次；按原有均值与最差负载合成分数，为每个 seed 选一个正训练轮次的最佳 checkpoint。
- 精确验证：710–720 s，全部 18 个负载，traffic seed 101；六个训练候选及两组未训练控制均评估。每组选择一个全负载统一 checkpoint，不能逐负载选择模型。
- 两阶段选择分数均为负载代价 `U(%) + 0.02 P(W)` 相对于原版未续训模型的比值，取其均值与最大值的等权平均。保留原有评价偏好，不依测试集调权重。
- 精确测试：800–830 s，5、13、25 Mbps × traffic seeds 1、2、3，比较 legacy_zero、E_all_zero、legacy_trained、E_all_trained，共 36 个 case。每组使用精确验证确定的统一模型，不再根据测试结果挑 checkpoint。
- 同一时段的 seeds 共享车辆轨迹和部署，只改变到达、初始队列及配对衰落；不代表独立轨迹。
- 未训练控制独立保留，即使所有续训模型较差，也不覆盖原结果。最终分别报告“E_all 本身的作用”与“配套训练的增量”。

## 收敛判断

不以训练满 160 轮宣称收敛。沿用最近五个验证 checkpoint 的逐负载代价波动、综合分数波动、固定状态动作翻转率及概率漂移检查。若未满足，明确报告稳定性限制。原版训练与 E_all 同预算，不对某一组选择性延长训练。

## 版本与运行

训练前标签：`pre-omappo-eall-training-20260924`。

新增独立入口 `experiment/o_mappo_eall_training.py` 和测试 `test_o_mappo_eall_training.py`，代码提交 `3c8ca14`。44 项相关测试通过。真实数据预检查中，原版 13 Mbps、seed101、10 s 及 E_all 13 Mbps、seed101、30 s 的全部原始数组分别与已有结果完全一致；六组均通过真实数据单轮采样及 PPO 更新。预检查结果保存在 `experiment/results/o_mappo_eall_training_smoke_20260924/`。

正式协议记录代码、模型、验证/测试缓存与训练数据 hash。GPU 用于精确物理仿真，轻量网络及 MILP 采样在 36 个 CPU worker 上训练。

```bash
PYTHONHASHSEED=0 python -u experiment/o_mappo_eall_training.py run
# 完成后重新核验原始数据并生成训练诊断图
python experiment/audit_o_mappo_eall_training.py
```

相同命令支持恢复；训练从最近一个完整的 10 轮验证边界恢复模型、Adam 和 shuffle RNG，未完成的后续轮次重做。精确仿真经协议和原始文件 hash 检查后跳过已完成 case。更改源代码或输入须使用新目录。

结果目录：`experiment/results/o_mappo_eall_training_20260924/`。

- `protocol.json`：冻结实验配置。
- `training.log`、`training_progress.json`、`training/*/seed*/updates/`：训练进度、逐负载 reward 和 PPO 更新信息。
- `training/*/seed*/validation_history.json`：完整验证历史。
- `zero_validation/`、`trained_validation/`：精确模型选择结果。
- `selection.json`：全负载统一模型选择记录。
- `paired_test/`：36 个精确测试 case 的指标、原始数据和逐帧诊断。
- `results_summary.json`：最终配对结果及稳定性检查。
- `audit.json`、`training_diagnostics.pdf`：独立原始数据核验及训练/验证曲线。核验工具也支持 `--wait` 等待主流程完成后自动执行。

本轮不修改论文、回复信、正式图片或正式 baseline 的默认 checkpoint。
