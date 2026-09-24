# 正式 O-MAPPO-adapted 两层版本：全负载补充与 Fig.5–8 更新

日期：2026-09-24。状态：54 个 case 全部完成并通过原始数据审计，Fig.5–8 已更新。

## 固定配置

- 用户已确认采用两层 actor 版本，不再按测试结果选模型。
- Actor：31→64→64→2，两个隐藏层使用 ReLU；critic：94→64→1，不变。
- 统一模型：`experiment/results/o_mappo_actor_depth_20260924/checkpoints/actor2_seed33/selected.pt`，训练 seed 33、第 140 轮；SHA-256 为 `9b4d87a4f1505a46cc79675bacd1105ab86c8c7c2e43a35016426bf4604756d9`。
- 该模型在独立验证中选出，在全部负载和 traffic seeds 下使用同一权重。本次不新增训练；不能把此前完成 160 轮写成已经满足收敛判据。
- HO 到微 BS 后，先探测 8 个 TX 宽波束和 2 个 RX 宽波束的 16 个组合，再探测选中扇区内 4×4 个 DFT 细波束对，总计 32 次。保持服务 BS 的决策采用 9 个邻域波束对追踪。搜索遵循决策事件，不改为每时隙邻域搜索。
- 优化器采用当前帧相同分层搜索得到的候选增益，而非全码本最大增益；下一帧执行时再进行 BF。它仍利用真实 CSI，候选 BS 额外 CSI 获取成本未计入，不能声称与 MEET-COBRA 完全相同的信息开销。
- 服务链路 BF 探测开销、10 ms HO 中断、逐时隙 OTR-RA、实际方向性逐 RB 服务评估均保留。

## 补充范围与数据复用

18 个负载 `{1,3,...,35}` Mbps × traffic seeds `{1,2,3}`，每例 30 s（800–830 s），排除前两帧。

- 复用此前两层实验的 18 个 case：负载 1、5、15、21、25、29 Mbps，各三个 seeds。
- 补跑其余 36 个 case：3、7、9、11、13、17、19、23、27、31、33、35 Mbps，各三个 seeds。
- 其余七个方法复用 `experiment/results/revision_directional_20260922/grid` 的 378 个 case。仅替换图中的 O-MAPPO 曲线，不修改历史仿真文件。
- 新旧实验使用相同测试缓存及 traffic hash。新结果保存于 `experiment/results/o_mappo_actor2_full_grid_20260924`，包含冻结协议、复用来源、逐例 JSON、队列等 NPZ 原始数据及汇总。
- 对每个新版本 case 从原始队列重算 U、L90、L99，从能量重算功率，从关联计数重算宏 BS 占比，再校验与报告指标相同。
- 全部 54 个 case 的原始数据审计已通过。绘图 CSV 的 720 行指标中，仅 O-MAPPO 对应的 90 行改变；其他七个方案的 630 行均与更新前完全一致。
- 图中曲线是三 seed 均值，色带是最小值到最大值；不做数值平滑，也不是置信区间。时延指标仍是队列代理量。

## 运行、恢复与绘图

```bash
python -u experiment/o_mappo_actor2_full_grid.py --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6
python experiment/o_mappo_actor2_full_grid.py --audit-only
python experiment/plot_revision_system_results.py
```

第一条命令支持逐 case 恢复，不会重训或覆盖不相容的输入协议。绘图默认采用已批准的新基线，输出为 `latexCodes/figures/*_revision1.pdf` 和 PNG，曲线数据与清单位于新实验目录的 `paper_figures/`。历史原图和 `_WBL` 文件不删除；旧基线数据可通过绘图的 `--legacy-o-mappo` 选项复现（建议另指定输出目录）。

## 版本管理

- 修改前快照：Git `05df16e`，包含论文、回复信、修订策略、当时的 Fig.5–8 和绘图代码。原版 256 次搜索的图可由此恢复。
- 补跑、审计和绘图代码：Git `e7f6afc`。27 项相关测试通过；其中两个旧测试补齐了现有入口要求的模型参数模拟对象，未更改实验算法。
- 两层模型的训练与独立选模记录见 `experiment/o_mappo_actor_depth_report.md`，本轮仅扩展其测试负载。

## 结果与论文更新

以下是新版 O-MAPPO-adapted 的三 seed 均值。U 和宏 BS 占比的单位均为百分比。

| 到达率 Mbps | 功率 W | U % | L90 ms | L99 ms | 宏 BS 占比 % |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.10 | 0.0296 | 14.91 | 18.14 | 2.68 |
| 3 | 15.62 | 0.0162 | 5.03 | 6.98 | 4.24 |
| 5 | 24.56 | 0.0380 | 3.04 | 4.09 | 5.09 |
| 7 | 34.03 | 0.0371 | 2.48 | 3.19 | 6.08 |
| 9 | 47.29 | 0.0265 | 2.03 | 2.55 | 8.89 |
| 11 | 60.54 | 0.0683 | 2.02 | 2.39 | 10.96 |
| 13 | 86.30 | 0.0577 | 1.98 | 2.17 | 17.13 |
| 15 | 109.99 | 0.1227 | 1.40 | 2.03 | 21.14 |
| 17 | 137.04 | 0.5131 | 1.32 | 2.40 | 25.53 |
| 19 | 149.89 | 1.3928 | 1.32 | 85.33 | 25.97 |
| 21 | 160.32 | 3.1044 | 1.39 | 467.53 | 26.49 |
| 23 | 169.49 | 5.3452 | 1.46 | 1148.45 | 26.71 |
| 25 | 177.30 | 7.7719 | 2.33 | 2380.50 | 26.91 |
| 27 | 183.09 | 10.6664 | 30.30 | 3772.33 | 26.53 |
| 29 | 184.72 | 13.7354 | 92.25 | 3661.16 | 25.59 |
| 31 | 185.03 | 17.0269 | 200.81 | 4742.14 | 25.25 |
| 33 | 185.09 | 21.6302 | 453.90 | 6994.18 | 25.31 |
| 35 | 185.31 | 26.9221 | 865.58 | 10432.21 | 26.66 |

### 与 MEET-COBRA 的比较

- 27 Mbps：新基线为 183.0851 W、U=10.6664%；MEET-COBRA 为 95.0290 W、U≈0.16%，功率降低 48.1%。
- 33 Mbps：新基线为 185.0882 W、U=21.6302%、L99=6994.18 ms；MEET-COBRA 为 172.81 W、U=0.524%、L99=5.00 ms。
- 平均 L99 首次超过 20 ms 的负载点是 19 Mbps；MEET-COBRA 仍是 35 Mbps。
- 新版本在中高负载仍存在明显的功率和尾部队列性能差距。这些是完整适配方案的比较，不单独证明某一子模块导致差距。

### 与原图中 256 次搜索版本的区别

本次同时替换了搜索机制、候选增益口径和经重新训练的 actor，不能把差异仅归因于两层网络。减少探测开销没有带来所有负载、所有指标的改善。例如 29 Mbps 的功率由 156.85 W 增至 184.72 W，U 从 11.8167% 增至 13.7354%；35 Mbps 的 U 则从 31.454% 降至 26.922%。受控的单层与两层比较仍以此前两个版本都采用分层搜索及修正优化器的实验为准。

### 更新文件

- `latexCodes/figures/power_comparison_curves_revision1.pdf`：Fig.5。
- `latexCodes/figures/violation_prob_comparison_curves_revision1.pdf`：Fig.6。
- `latexCodes/figures/latency_90th_comparison_curves_revision1.pdf`、`latency_99th_comparison_curves_revision1.pdf`：Fig.7(a)/(b)。
- `latexCodes/figures/BS0_assoc_ratio_comparison_curves_revision1.pdf`：Fig.8。
- 各 PDF 均有同名 PNG；配色、尺度和子图布局保持不变，其他七个方法的数据不变。
- `latexCodes/main_revision1.tex`：更新 Section V-C 的 O-MAPPO 结构、分层 BF、优化器信息口径；更新 V-D 的 27 Mbps 比较和 L99 门限交叉点。
- `response_letter/response_letter.tex`：同步 R2C5 与 R3 Major C3 的方法说明、表中数值及尾部时延比较；当前修改保留红色。
- `response_letter/paper_revision_policy.md`：记录正式模型、补跑来源和新绘图入口。
- 已逐一检查五张图片的显示，并将回复信表格与汇总均值逐项核对。论文与回复信 PDF 均重新编译成功。
