# 最新预测输入与方向性干扰：训练和重跑入口

## 版本与适用范围

修改前快照：`c0c7029`，标签 `pre-directional-prediction-rerun-20260921`。
本入口与旧实验分开保存，不改写论文、图片、旧 checkpoints 或旧结果。
旧接口的默认行为保留；方向性服务计算由新入口显式启用。

本轮只做数据准备和小规模软件验证，没有启动正式 100+100 epoch 训练或全负载任务。
验证结果见 [revision_directional_validation.md](revision_directional_validation.md)。

### 本次固定设置

- 仅重新训练 interfering-gain predictor，标签为 `||H||_F^2/(M_R M_T)` 的 dB 值。
  Beam 和 desired-gain predictor 沿用 `stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt` 的最佳模型。
- 沿用既有车辆级 train/validation split。干扰模型第一阶段为 100 epoch 有限窗口训练，
  第二阶段为 100 epoch stateful TBPTT，截断长度 10；两阶段均按最低验证 MAE 选择 checkpoint。
  第二阶段从第一阶段最佳 checkpoint 初始化，冻结 BN running statistics，初始学习率 `1e-4`，使用 plateau 调度。
- 所有新预测缓存使用跨帧递推。记录在帧 x 中的报告预测 x+1；当前 BF/RA 使用上一帧报告。
  新进入的车辆先接入宏站，不用真实微站 CSI 填补缺失报告。
- HO interruption 为 10 ms；GAP-HO 两轮迭代，启用 HO 容量修正及 `K_m` 占用上限。
- **八种可执行方案都由实际收发波束和显式 RB 索引计算逐时隙服务量，并以该服务量更新队列。**
  在线控制中的预测增益、期望 RB 重叠与模拟器中的真实服务计算分离；保留各方案的导频开销。
- O-MAPPO 按之前确认的方案，保留原始 privileged-information actor、优化器和已训练策略，不重新训练 RL。
  MTS 使用共享预测报告，RA 的干扰输入也不再读取真实微站信道。
- Reactive-OBRA 明确使用当前真实增益测量，不使用 NN。其与 w/o GAP-HO 的 HO 保留现有
  `HO_EE_Greedy_offload`，即按能耗代价排序的既有实现，**不是正文所写的 RSS-first 规则**。
  本轮没有擅自重写该基线。正式解释这两条曲线前须统一代码与正文的这处差异。
- Oracle-CR-LB 不另跑队列轨迹。单独导出 Oracle-MC 轨迹上最终 P2 实例的连续松弛功率参考，
  保存不可行标志；不可行时的满功率替代值不是下界，也不应当据此声称全动态系统的全局最优性。

## 1. 环境和路径

以下命令在服务器普通 shell 中运行；GPU 编号以当时 `nvidia-smi` 为准。
实际验证环境为 `/home/ubuntu/anaconda3/envs/sionna/bin/python`。

```bash
cd /home/ubuntu/niulab/Meet_Cobra
conda activate sionna
export REV_RUN=experiment/results/revision_directional_20260921
nvidia-smi
```

本目录的 `training_data.npz` 已生成，646,747 个样本，标签摘要保存在 `training_data.json`。
**在当前工作目录无需再运行下面的数据生成命令**；另建实验目录时才运行：

```bash
python -u experiment/revision_training.py data --output "$REV_RUN/training_data.npz"
```

生成器拒绝覆盖已有数据。新数据及正式模型文件不纳入 Git，但以 SHA-256 记录来源。

## 2. 训练：100+100 epoch

```bash
python -u experiment/revision_training.py train \
  --data "$REV_RUN/training_data.npz" \
  --output "$REV_RUN/training" \
  --device cuda:0 --stage1-epochs 100 --stage2-epochs 100 \
  --batch-size 128 --seed 20 --resume
```

`--resume` 可从第一次启动就携带。后台训练可以使用：

```bash
nohup python -u experiment/revision_training.py train \
  --data "$REV_RUN/training_data.npz" --output "$REV_RUN/training" \
  --device cuda:0 --stage1-epochs 100 --stage2-epochs 100 \
  --batch-size 128 --seed 20 --resume \
  > "$REV_RUN/training.log" 2>&1 &
```

同一训练目录有互斥锁，勿同时启动前台和后台两份。每个 epoch 保存模型、optimizer、scheduler、
随机数状态和当前最优权重。中途终止后重新执行相同训练命令即可；未保存完的 epoch 从头重做。
已完成阶段校验 checkpoint 后跳过。改变 epochs、batch size、split 或冻结代码时必须换输出目录。

查看训练进度：

```bash
tail -f "$REV_RUN/training.log"
python -m json.tool "$REV_RUN/training/training_status.json"
```

## 3. 汇集模型并生成跨帧预测缓存

须先完成两阶段正式训练，不允许拿 smoke 模型代替。

```bash
python experiment/revision_training.py assemble \
  --training "$REV_RUN/training" --output "$REV_RUN/models"

python -u experiment/revision_training.py cache \
  --models "$REV_RUN/models" --output "$REV_RUN/test_predictions.pkl" \
  --device cuda:0 --start 800 --end 830
```

Beam 和 desired-gain checkpoint 原样复制；正式模型 bundle 检查三个模型的 split 哈希一致。
缓存保存 source/target frame、checkpoint 哈希和输入来源。完整缓存重复运行会校验后跳过，
不会重新抽取预测。缓存生成若中断，需要重新生成尚未完成的缓存，并非逐帧恢复。

## 4. 冻结全负载、多 seed 设置

```bash
python experiment/revision_pipeline.py prepare \
  --root "$REV_RUN/grid" --cache "$REV_RUN/test_predictions.pkl" \
  --rates 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35 \
  --seeds 1,2,3,4,5 --backend cuda --reactive-input current
```

默认八种方案：`meet_cobra,oracle_mc,reactive_obra,wo_gap_ho,wo_pet_bf,wo_otr_ra,o_mappo,mts_report`。
合计 **8×18×5=720 个 case**。每个 case 为 30 s、300 个服务帧；统计剔除最初两帧。
同一 seed/load 使用相同轨迹、到达量及配对衰落设置。五个 seed 是固定场景内的重复，
不是五个独立基站部署或交通场景。

`prepare` 冻结代码、输入缓存、旧 RL 策略、方法和参数。若后续改变这些内容，请新建 grid 根目录，
不要编辑已有 `protocol.json` 绕过检查。不能把历史结果复制到新 `runs` 目录充数。

## 5. 启动、查看、续跑

正式全量启动前，可用同一个正式缓存先跑 13 Mbps、seed 1；结果合格后直接纳入后续全量任务：

```bash
python -u experiment/revision_pipeline.py run --root "$REV_RUN/grid" \
  --rates 13 --seeds 1 --devices cuda:0,cuda:1 --detach
```

全量启动（只填写空闲且可用的 GPU；每张卡一个独立 case）：

```bash
python -u experiment/revision_pipeline.py run --root "$REV_RUN/grid" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6 --detach
```

须等待前一个同目录 launcher 退出，再启动下一条。同一目录不能同时运行两个 launcher。
`--detach` 创建独立会话，退出当前 SSH 不会结束任务。

```bash
python experiment/revision_pipeline.py status --root "$REV_RUN/grid"
tail -f "$REV_RUN/grid/queue.log"
tail -f "$REV_RUN/grid/logs/meet_cobra_rate13_seed1.log"
python -m json.tool "$REV_RUN/grid/progress/meet_cobra_rate13_seed1.json"
```

断点续跑：**原样重复 `run` 命令即可**。已经完成且哈希验证通过的 case 会跳过，未完成 case 从头运行，
不是从中间时隙恢复。失败时停止追加任务并记录失败 case；先查看对应日志，不要批量清理结果目录。
单个 case 默认有 3600 s 超时保护。正式运行可能受同时使用 GPU/CPU 的其他作业影响。

## 6. 汇总与原始数据

```bash
python experiment/revision_pipeline.py summarize --root "$REV_RUN/grid"
```

未完成时仅供诊断的汇总：

```bash
python experiment/revision_pipeline.py summarize --root "$REV_RUN/grid" --allow-partial
```

- `raw/*.npz`：逐帧能耗、RB 占用、逐车辆逐时隙队列、关联、HO 和导频记录。
- `runs/*.json`：每个 case 的指标、完成状态及原始数据哈希。
- `diagnostics/*.json`：干扰与服务容量的固定决策比较、HO/GAP 诊断。
- `aggregate/summary.json`：逐负载、逐方案的均值、seed 间标准差及 t-based 95% 区间。
- `aggregate/curves.csv`：功率、违反概率、90/99 分位时延代理量、宏站关联比例的曲线数据。
- `aggregate/oracle_cr_lb_power_only.json`：带可行比例的连续松弛功率参考，无时延指标。

固定决策诊断把同一时隙、关联、波束和 RB 数量下的真实干扰与“码本均匀平均增益 + 期望 RB 重叠”比较。
两个比值分别按对应 RB/服务量加权；统计限于有 RB 分配的微站链路。它量化两种近似的组合影响，
不是把 NN 预测误差也混入其中。闭环功率/队列结果来自真实服务更新，是另一组独立含义的指标。

## 7. 小规模复核和安全回看旧版

```bash
python -m unittest test_revision_pipeline_guards test_revision_directional test_interference_validation test_gpu_phy \
  test_mts_report test_mts_report_bounded test_o_mappo test_gap_refinement \
  test_gap_rb_usage test_stateful_tbptt -v

TEST_PHY_DEVICE=cuda:0 python -m unittest test_revision_directional test_gpu_phy -v

python -u experiment/check_revision_pipeline.py \
  --output experiment/results/revision_directional_smoke_manual --device cuda:0 --second-device cuda:1
```

最后一个命令要求全新目录；使用合成训练数据以及测试轨迹的短片段，只验证软件链路。
正式 `prepare` 拒绝 smoke 模型；不要在正式运行中使用 `--allow-smoke`。

查看旧代码不会影响当前工作：

```bash
git show pre-directional-prediction-rerun-20260921:utils/mts_report_sim.py
git diff pre-directional-prediction-rerun-20260921 -- utils experiment
```

如需实际运行旧版，建议建立独立 worktree，再显式连接所需数据；不要 `git reset --hard` 覆盖当前论文修改。
