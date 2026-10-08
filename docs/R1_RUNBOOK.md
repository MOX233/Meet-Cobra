# R1 复现操作手册

本手册区分“复用已提交模型和结果”与“生成一次新的训练/实验”。所有命令从仓库根目录执行。
**不要覆盖已提交的 `latexCodes/revision1/`，不要在旧实验目录内修改 protocol 或替换模型。**
入口清单见 `configs/r1_workflow.json`；正式资产路径以 `configs/paper_r1_assets.json` 为准。
科学代码仍在原处，新增 `scripts/` 入口只负责隔离输出、调用已有实现和校验结果。

## 0. 环境与路径

```bash
cd /home/ubuntu/niulab/Meet_Cobra
conda activate sionna
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
export MPLCONFIGDIR="$(mktemp -d /tmp/meet-cobra-mpl-XXXXXX)"
nvidia-smi
python -B scripts/check_paper_assets.py
```

下面的 `cuda:0` 只是示例，启动前应选择空闲 GPU。不要使用原脚本中默认的全部 GPU 列表。
2026-10-08 检查环境：Python 3.10.16、PyTorch 2.6.0+cu118、NumPy 1.26.4、
SciPy 1.15.1、Numba 0.61.2、Matplotlib 3.10.0。原始数据生成另需 SUMO 1.24.0、
Sionna/Sionna RT 1.0.1、Mitsuba 3.6.2、Dr.Jit 1.0.3；当前 `SUMO_HOME=/usr/share/sumo`。
详细版本见 `environment-observed-20261008.txt`。本轮没有安装或升级依赖。
这是一份已观察环境记录，并非保证跨平台逐位一致的锁文件。

常用输入：

```bash
R1_RAW='sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl'
R1_PREPARED='data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl'
R1_FORMAL='experiment/results/revision_directional_20260922'
R1_SPLIT='experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz'
R1_UNIFIED='experiment/results/stateful_tbptt_unified_split_20260913'
```

一次新工作的目录名自行设定，例如 `r1_new_20261009`；不要反复用同一目录启动不同配置。
命令尽量单行，避免反斜杠后空格导致 shell 把参数当命令执行。

## 1. 执行顺序与三条常用途径

```text
SUMO 轨迹 → RT 信道 ┬→ 紧凑训练数据 → 固定车辆划分 → 三个 NN 的两阶段训练
                  └→ 系统输入（含导频 CSI）                 ↓
                                             模型组合 → 跨帧预测缓存
                                                               ↓
                          六种主方案 + 两种最终基线 → 补充分析 → 图表

O-MAPPO 上游预训练 → E_all 初始化权重 → 预测输入微调 → 验证选择 → 固定策略评估
```

- **只重画已提交结果**：直接到第 6 节，无需训练、缓存或仿真。
- **复用正式模型重跑系统结果**：使用现有缓存，直接到第 4 节。
- **重新训练 NN**：第 2 节的数据若已保留就不必重建；第 3 节之后生成新缓存，再到第 4 节。

主网格中历史 `o_mappo`、`mts_report` 的实现不是最终版本。新实验只在主脚本运行六种
主方案，两条最终基线通过 `scripts/r1_baselines.py` 运行。原始已提交曲线则仍从三个
历史结果目录合并，不修改这些目录。

## 2. 数据：保留数据优先，重建写入新目录

`Lambda=1.00` 是 SUMO 车辆流量参数，不是系统实验中 1–35 Mbps 的每车业务到达率。
当前配置为 28 GHz、32 个发射天线、8 个接收天线、4 个微基站。

### 可选：从 SUMO/RT 重建

```bash
python -B scripts/r1_data.py sumo --output experiment/results/data_rebuild_mobility_NEW
python -B scripts/r1_data.py raytrace --source sumo_data/trajectory_Lbd1.00.csv --start 200 --end 800 --gpu 0 --workers 1 --output experiment/results/data_rebuild_rt_train_NEW
python -B scripts/r1_data.py raytrace --source sumo_data/trajectory_Lbd1.00.csv --start 800 --end 830 --gpu 0 --workers 1 --output experiment/results/data_rebuild_rt_test_NEW
python -B scripts/r1_data.py system-input --source 'sionna_result/trajectoryInfo_lbd1.00_800_830_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl' --seed 1 --output experiment/results/data_rebuild_system_NEW
```

上例 RT 命令复用正式轨迹；若使用新 SUMO 轨迹，将 `--source` 指向新输出中的 CSV。
同理，`system-input` 的 `--source` 可以换为新 RT 文件。各阶段的实际输出路径、哈希和
完成标记写入 `generation.json`。只有 `complete: true` 的产物可以继续使用。
新封装拒绝已有输出目录，RT 子任务缺帧会报错，不把不完整信道当作成功数据。
SUMO/完整 RT 在本轮**没有重跑**；实际检查覆盖预处理，并验证与旧预处理逐元素相同。
随机交通、RT 实现和导频噪声可能改变生成文件，重建不等于恢复已提交的随机实现。

### 紧凑训练数据与车辆划分

```bash
R1_NEW='experiment/results/r1_new_20261009'
mkdir -p "$R1_NEW"
python -B experiment/prepare_stateful_trajectories.py --source "$R1_RAW" --output "$R1_NEW/training_legacy_labels.npz" --interference-label legacy-max
python -B experiment/revision_training.py data --source "$R1_RAW" --output "$R1_NEW/training_beam_average.npz"
python -B experiment/vehicle_split.py --data "$R1_NEW/training_legacy_labels.npz" --output "$R1_NEW/vehicle_split_seed20.npz" --seed 20 --train-fraction 0.7
```

也可直接使用保留的 `stateful_tbptt_20260912/training_trajectories.npz`、
`revision_directional_20260922/training_data.npz` 和 `$R1_SPLIT`。
两个紧凑训练文件的干扰标签不同，不能按大小相同认定为重复文件。
波束、期望增益取前者，修订后的干扰增益取后者；三者使用同一车辆划分。
零信道仍按原幅度 epsilon `1e-9` 处理，对应 −180 dB；不得改标签后冒用旧 metadata。
已删除的巨型滑动窗口数据集不需要恢复，窗口由 trainer 在线构造。

## 3. NN 训练、模型选择、缓存

### 波束和期望增益

```bash
for R1_TASK in beam desired_gain; do
  python -B -u experiment/train_finite_window_vehicle_split.py --data "$R1_NEW/training_legacy_labels.npz" --split-file "$R1_NEW/vehicle_split_seed20.npz" --output "$R1_NEW/stage1/$R1_TASK" --task "$R1_TASK" --device cuda:0 --epochs 100 --batch-size 512 --history-length 10 --augmentation-ratio 2 --learning-rate 0.001 --weight-decay 0.0001 --seed 20
  python -B -u experiment/train_stateful_tbptt.py --data "$R1_NEW/training_legacy_labels.npz" --split-file "$R1_NEW/vehicle_split_seed20.npz" --output "$R1_NEW/stage2/$R1_TASK" --task "$R1_TASK" --device cuda:0 --epochs 100 --patience 101 --batch-size 128 --chunk-length 10 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20 --initialization checkpoint --initial-checkpoint "$R1_NEW/stage1/$R1_TASK/best.pth" --input-normalization paper --normalization batchnorm --batch-norm-mode frozen --scheduler plateau
done
```

第一阶段 batch **512**，第二阶段 **128**，不是两个阶段都用 128。
这两个 trainer **不支持通用断点续训**，输出已有时会拒绝覆盖；失败后保留该目录，另建新
目录重新训练该阶段。不要将 `last.pth` 人工冒充为包含完整优化器状态的 resume checkpoint。

### 干扰增益：独立的可续训两阶段入口

```bash
nohup python -B -u experiment/revision_training.py train --data "$R1_NEW/training_beam_average.npz" --split-file "$R1_NEW/vehicle_split_seed20.npz" --output "$R1_NEW/interfering_training" --device cuda:0 --stage1-epochs 100 --stage2-epochs 100 --batch-size 128 --seed 20 --resume >> "$R1_NEW/interfering_training.log" 2>&1 &
tail -n 20 "$R1_NEW/interfering_training.log"
```

两个阶段 batch 都为 **128**，不同于上面两个任务。相同命令重新启动即可从已保存 epoch
继续；配置不一致会拒绝。进度见 `interfering_training/training_status.json`。
第一次创建亦可带 `--resume`；不要同时启动两个同目录训练进程。

### 组合与跨帧缓存

```bash
python -B experiment/revision_training.py assemble --base-models "$R1_NEW/stage2" --training "$R1_NEW/interfering_training" --output "$R1_NEW/models"
python -B -u experiment/revision_training.py cache --models "$R1_NEW/models" --source "$R1_PREPARED" --output "$R1_NEW/test_predictions.pkl" --device cuda:0 --start 800 --end 830
```

波束选最高验证 Top-1，两个增益模型分别选最低验证 MAE；不是分别挑 Top-1、Top-5 的
不同权重来拼成一个部署模型。缓存按车辆持续保存递归状态，记录 x 帧预测 x+1 帧的
source/target 对齐、模型与输入哈希。再次执行相同命令会核验并跳过已完成缓存。
缓存阶段中途失败不会从半个 pickle 续算，而会保留未提交缓存并重新计算。

## 4. 全负载、多 seed 系统实验：固定正式模型的标准入口

正式配置：18 档 `1,3,...,35` Mbps、seeds `1,2,3`、30 s、去掉前 2 帧、HO 中断 10 ms、
GAP-HO 固定两轮且 RB 占用封顶、`M_P=5`。服务量由实际方向波束和显式 RB 干扰计算。
新目录示例：

```bash
R1_GRID='experiment/results/r1_grid_NEW'
R1_BASELINES='experiment/results/r1_baselines_NEW'
R1_CACHE="$R1_FORMAL/test_predictions.pkl"
python -B experiment/revision_pipeline.py prepare --root "$R1_GRID" --cache "$R1_CACHE" --methods meet_cobra,oracle_mc,reactive_obra,wo_gap_ho,wo_pet_bf,wo_otr_ra --rates 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35 --seeds 1,2,3 --reactive-input current --backend cuda
python -B experiment/revision_pipeline.py run --root "$R1_GRID" --devices cuda:0 --detach
python -B experiment/revision_pipeline.py status --root "$R1_GRID"
tail -n 20 "$R1_GRID/queue.log"
```

主网格共 **324** 个 case。需要新 NN 时，将 `R1_CACHE` 改为第 3 节的新缓存，另建新网格。
待主网格结束后：

```bash
python -B experiment/revision_pipeline.py summarize --root "$R1_GRID"
python -B scripts/r1_baselines.py prepare --root "$R1_BASELINES" --base-grid "$R1_GRID"
nohup python -B -u scripts/r1_baselines.py run --root "$R1_BASELINES" --devices cuda:0 >> "$R1_BASELINES/launcher.log" 2>&1 &
python -B scripts/r1_baselines.py status --root "$R1_BASELINES"
tail -n 20 "$R1_BASELINES/launcher.log"
```

两条基线共 **108** 个 case，使用相同缓存、负载和 seeds。O-MAPPO 默认固定已验证选定的
`o_mappo_predicted_cross5_20260925/training/seed11/best_positive.pt`；可用 `prepare --policy`
指定另一个**已选定、配置兼容**的预测输入策略。MTS 直接调用已确定的 H32-cross5 实现。
没有重新训练或自动选优，不复用旧 pilot case，不给旧结果换新 protocol。
若换用重新训练的 NN，原 RL 策略仍是旧报告分布上训练的模型；评估可以运行，但不能称为
“已针对新预测器重新训练”。重新微调须明确其输入模型与初始化路径，参见第 5 节。

断点续跑：重新执行对应 `run`，完整 case 核验哈希后跳过，未完成 case 从头计算。
不要在已有队列仍运行时重复启动。多卡时显式使用 `--devices cuda:0,cuda:1`。
原主脚本即使仅跑六种方案，也会核验保留的 legacy policy 文件，这是入口元数据依赖，
不表示将旧 RL 基线放入这六种方案。

完成后合并绘图：

```bash
python -B scripts/r1_baselines.py plot --root "$R1_BASELINES"
```

输出位于 `$R1_BASELINES/plots/`，包含五张 PDF/PNG、逐指标 CSV、manifest。
复用正式图形样式；合并前检查全 case、输入/源代码哈希和配对流量。
**不会自动更新论文图片或已提交材料。** 小样本输入会在 manifest 中保留 smoke 标记和帧数，
不能因输出了同名图片而当作正式结果。

## 5. O-MAPPO 训练路径（与固定策略评估分开）

最终模型不是从随机权重直接运行最后一个脚本得到的：

| 顺序 | 入口 | 必须保留的输入 / 输出 |
|---|---|---|
| H32 预训练 | `experiment/train_o_mappo_hierarchical32.py train` | 原始训练信道；v4 的 channels/index 数组；v5 预训练 |
| 两层 actor | `experiment/o_mappo_actor_depth_experiment.py` | v5、v4 数组与 exact reference；生成 actor2 selected checkpoint |
| E_all 配套训练 | `experiment/o_mappo_eall_training.py` | actor2 已选权重；生成 E_all seed11 best_positive |
| 预测输入微调 | `experiment/train_o_mappo_predicted_cross5.py` | E_all 权重、正式三个 NN、v4 数组、正式测试缓存 |

历史前三阶段的默认路径嵌入原入口，保留在原地并由资产清单保护。完全从零重建时，必须
在独立工作副本按这些阶段恢复依赖；`--root` **不是**统一的上游输入重定向开关。
不要仅执行最后一个脚本就声称从零训练，更不要把旧目录删除后以为 Git 已存储权重。

在当前保留资产上，重新运行正式“预测输入微调”流程的命令为：

```bash
R1_RL='experiment/results/o_mappo_predicted_retrain_NEW'
python -B -u experiment/train_o_mappo_predicted_cross5.py prepare --root "$R1_RL" --device cuda:0
python -B experiment/train_o_mappo_predicted_cross5.py launch --root "$R1_RL" --devices cuda:0 --rounds 160 --rollout-seconds 5
tail -n 20 "$R1_RL/pipeline.log"
```

流程包含三个训练 seeds `11,22,33`、各 160 PPO updates、每轮全 18 负载、独立验证选择和
54 个测试 case。训练/验证/选择/测试时间段分别为 200–650、700–710、710–720、800–830 s。
进度为 `training_progress.json`，完成标记 `complete.json`，选择记录 `selection.json`。
重新执行相同 launch 会按保存的训练轮次和完整 case 继续，先确认旧进程已退出。
actor/critic 为 CPU 训练，GPU 用于物理仿真和 NN 报告生成，不能把 GPU 使用率低直接解读为未运行。

该历史入口固定 `DATA`、`MODELS`、`TEST`、`INITIAL` 为正式路径，因此上述命令复现的是
**当前正式前端下的微调**，不会自动使用第 3 节的另一套新 NN。更换这些输入应作为新的
实验配置在独立工作副本实施，并记录哈希，不在清理任务中改动科学源代码。
`analyze_o_mappo_predicted_cross5.py` 还要求历史零训练/真 CSI 对照，不是任意新 run 的
通用汇总器；仅评估选定策略时使用第 4 节入口，不需要重跑这些历史对照。

## 6. 只重画正式结果：所有输出隔离

```bash
R1_PLOTS='experiment/results/paper_plot_check_NEW'
mkdir -p "$R1_PLOTS"
python -B experiment/plot_revision_system_results.py --figures "$R1_PLOTS/figures" --report "$R1_PLOTS/report"
python -B experiment/plot_stateful_training_curves.py --stage1-results "$R1_UNIFIED/stage1_finite_window" --stage2-results "$R1_UNIFIED/stage2_stateful_tbptt" --interfering-stage1-results "$R1_FORMAL/training/stage1" --interfering-stage2-results "$R1_FORMAL/training/stage2" --summary "$R1_PLOTS/fig4_summary.json" --figures "$R1_PLOTS/figures"
python -B experiment/plot_gain_error_paper.py --figures "$R1_PLOTS/figures"
```

第一条默认读取主网格以及最终 O-MAPPO/MTS 独立结果，**不要**加 `--legacy-o-mappo` 或
`--legacy-mts`。第三条读取完整已核验的增益误差实验，无需再跑仿真。
速度图和回复信干扰图的旧脚本还会写固定位置的 manifest/图表，即使部分输出可重定向。
归档阶段不要直接无参数运行；其入口与输入见 `PAPER_FILES.md`，确需重画时使用隔离副本。

**旧主网格的 source hash 问题**：旧 protocol 对应 `0e9474f`，其中三个源文件在最终基线
开发中后来扩展。因此对旧网格运行 `revision_pipeline validate` 或绘图 `--audit` 会因
活跃代码不同失败；这是已有版本差异，不是整理损坏。不要改 protocol 或取消校验。
`check_paper_assets.py` 检查旧精确源文件仍存在 Git，另检查已提交源与当前模型；本轮
绘图读取器核验结果及替换曲线哈希。精确旧版重跑要在独立工作副本恢复该提交并恢复其
资产路径。新网格使用新 protocol，后续源文件变化也会被拒绝。

## 7. 补充实验入口

| 项目 | 入口及执行顺序 | 输入与输出注意事项 |
|---|---|---|
| GAP 两轮迭代选择 | `gap_refinement_directional.py prepare → run → summarize` | `--base-grid` 指向匹配当前源代码的网格；21/27/35 Mbps、3 seeds、3 配置 |
| 增益误差闭环实验 | `gain_error_sensitivity.py prepare → run → summarize` | 9/19/29/35 Mbps、3 seeds、期望/干扰分别扰动；sigma 0–10，零噪声复用；252 cases |
| 速度分布 | `audit_mobility_data.py --output NEW` | 读取正式轨迹/划分/测试数据；输出 CSV、JSON |
| 分速度组预测及系统分析 | `mobility_conditioned_evaluation.py prepare → validation → test → system → finalize` | 固定模型验证推理 + 现有缓存/系统结果后处理，不训练 |

所有上述入口在 `experiment/`。例如，对第 4 节**已完成的新网格**：

```bash
python -B experiment/gap_refinement_directional.py prepare --root experiment/results/gap_check_NEW --base-grid "$R1_GRID"
python -B experiment/gap_refinement_directional.py case --root experiment/results/gap_check_NEW --rate 27 --seed 1 --mode fixed2 --smoke --device cuda:0
python -B experiment/gap_refinement_directional.py run --root experiment/results/gap_check_NEW --devices cuda:0 --workers-per-device 1
python -B experiment/gap_refinement_directional.py summarize --root experiment/results/gap_check_NEW
python -B experiment/gain_error_sensitivity.py prepare --root experiment/results/gain_check_NEW --base-grid "$R1_GRID"
python -B experiment/gain_error_sensitivity.py run --root experiment/results/gain_check_NEW --devices cuda:0 --workers-per-device 1 --detach
python -B experiment/gain_error_sensitivity.py status --root experiment/results/gain_check_NEW
```

补充入口的完整配置与保护规则以 `--help` 和保留的 protocol 为准；本轮不重跑这些网格。
旧 GAP 实验依赖旧网格的严格 source hash，不能直接在当前代码上原目录续跑。

## 8. 快速复现检查

```bash
python -B -u scripts/smoke_r1.py --output experiment/results/reproduction_smoke_NEW --device cuda:0 --paper-plots
python -B -m unittest discover -s tests -p 'test_r1_reproduction_tools.py' -v
python -B scripts/check_paper_assets.py
```

实际完成 21 个时间点、32 辆连续车辆、2 s 的检查：紧凑预处理、车辆划分、三个 NN 各两阶段
1 epoch、干扰模型 epoch 边界续跑、跨帧缓存及重复生成跳过、6 方案仿真与续跑、当前两个
基线、一次 PPO 更新和绘图。还使用正式 NN 在同一小轨迹重新推理，确保实际覆盖 MTS 的
分层搜索和 cross-5 分支，避免“全关联宏站”造成空检查。
每次检查必须新目录；`report.json` 记录命令、耗时、返回码和受保护文件是否改变。
小模型用了测试输入的子集，**仅为软件测试**，不能用于论文的训练/测试性能结论。

## Backup and recovery

Git 存代码、配置、测试和小文档；已有提交历史不重写，今后不批量添加二进制成果。
精确恢复已提交结果，需要在 Git 之外备份：

1. `latexCodes/revision1/` 完整提交包，以及手工绘图源、导师批注、投稿截图。
2. `configs/paper_r1_assets.json` 中的全部 retained_datasets：原轨迹、信道、prepared inputs、
   两份紧凑训练集、固定 split、正式预测缓存。
3. 三个正式 NN、计时所用同权副本、最终 RL 权重及其上游初始化依赖。
4. 正式结果与上游结果目录中的 protocol、case JSON、原始数组、诊断、训练历史、
   checkpoint 选择记录及哈希；详见 `docs/EXPERIMENTS.md`。

从代码可生成的缓存/模型/结果并不意味着可无损删去：重训练可能因随机数、库或硬件变化
产生不同权重。目录搬迁也可能使 metadata 内绝对路径失效。迁移服务器时保留原路径映射
或单独制定路径迁移方案，不直接改冻结 protocol。Git + 资产清单不是大型文件的备份。
