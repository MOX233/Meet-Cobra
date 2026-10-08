# 最终方向性干扰系统实验：结果核验及论文图表更新

日期：2026-09-22。本轮仅核验、后处理和修改图文，没有重新训练或运行系统仿真。

## 1. 唯一数据来源及完整性

- 实验根目录：`experiment/results/revision_directional_20260922/grid`。
- 最终协议：`grid/protocol.json`，SHA-256 为 `6c851155e24935ecaab253b22633c7fb6922ba053bdcd442022eec9ec0068362`。
- 8 个可执行方案 × 18 个负载 `{1,3,...,35}` Mbps × 3 个 seeds `{1,2,3}` = **432 次完整运行**。实际不是早期计划的 5 seeds。
- 每次运行 300 帧、30 s，统计排除最初两帧，计入 29.8 s；HO interruption 为 10 ms，候选数为 5，GAP-HO 固定两轮并使用容量修正和有上限的 RB 占用反馈。
- 同一负载和 seed 的所有方案共用到达流量（54 组 traffic hashes）；所有方案均以实际收发波束、显式 RB 索引和逐 RB 服务量更新队列，计入各自 BF 开销及 HO 中断。
- 已核验冻结代码、预测缓存、O-MAPPO policy、432 个原始结果及诊断文件的哈希；独立从全部原始队列重算功率、违反概率、L90、L99 和宏层关联比例，并重新汇总干扰诊断。全部与已保存结果一致。
- 预测缓存中的三个 checkpoint 与已批准的 `selected_models/bundle.json` 一致：beam 第二阶段 epoch 88，desired gain epoch 59，interfering gain epoch 100。没有改用中间模型。
- 已独立复核 54 次 Oracle-MC 运行导出的 P2 松弛参考及可行比例。

## 2. 统计口径及绘图

- 曲线为逐 seed 指标的算术平均；阴影为 seed 最小值到最大值，**不是置信区间**。未做移动平均或拟合平滑。
- L90、L99 在每个 seed 内直接对车辆时隙的 `q_v/lambda_v` 求分位数，再跨 seeds 平均。不附加一个时隙，不将三个 seed 的样本混合后重新求分位数。
- U 按原定义对各帧车辆时隙求超过 20 ms 阈值的比例，再对帧及 seeds 平均。
- 宏层关联率为统计窗口内宏层车辆帧计数除以全部车辆帧计数，不是逐帧比例的不加权平均。
- 三个 seeds 共享部署和车辆轨迹；图中阴影不表示不同地图、部署或速度分布之间的不确定性。
- 各图统一方法颜色、点形和线型；MEET-COBRA 加粗；图例置于绘图区外。
- U 使用 0.01% 以下线性、以上对数的对称对数坐标，真实零值仍为零，不人为抬到正数。
- 原 Fig.7 拆成两个 PDF，在论文中并排作为 Fig.7(a)、Fig.7(b)，分别显示 L90、L99，维持宏层关联图为 Fig.8。

新增文件（同名 PNG 用于快速预览）：

| 论文图号 | `latexCodes/figures/` 下的新文件 |
|---|---|
| Fig.5 | `power_comparison_curves_revision1.pdf` |
| Fig.6 | `violation_prob_comparison_curves_revision1.pdf` |
| Fig.7(a) | `latency_90th_comparison_curves_revision1.pdf` |
| Fig.7(b) | `latency_99th_comparison_curves_revision1.pdf` |
| Fig.8 | `BS0_assoc_ratio_comparison_curves_revision1.pdf` |

原始四张无后缀 PDF、旧 `_WBL` 图及原始投稿 `main.tex` 均保留。

## 3. 关键结果及解释

27 Mbps 的三 seed 均值：

| 方案 | 功率 W | U % | L99 ms |
|---|---:|---:|---:|
| MEET-COBRA | 95.03 | 0.161 | 1.59 |
| Oracle-MC | 66.00 | 0.091 | 1.60 |
| Reactive-OBRA | 185.39 | 9.841 | 5753.21 |
| w/o GAP-HO | 179.27 | 5.672 | 8332.89 |
| w/o PET-BF | 98.80 | 0.192 | 1.56 |
| w/o OTR-RA | 105.59 | 0.213 | 1.51 |
| O-MAPPO-adapted | 139.46 | 7.955 | 839.75 |
| MTS-GS-HBF-adapted | 89.70 | 0.477 | 1.62 |

- MEET-COBRA 相对 Reactive-OBRA 降低发射功率 **48.74%**，相对 O-MAPPO-adapted 降低 **31.86%**。正文分别用 48.7%、31.9%；摘要和结论用约 49%。
- MTS 不是在所有指标上都差于 MEET。29 Mbps 时 MTS 为 109.01 W、U=0.604%，MEET 为 127.75 W、U=0.194%。正文和两条基线回复均明确这种功率与时延取舍。
- 33 Mbps 时 MEET、MTS 的 L90 分别为 1.24、1.27 ms，但 L99 分别为 5.00、33.54 ms。单看 L90 会掩盖极端尾部差异。
- 平均 L99 首次超过 20 ms 的负载：Reactive 19、O-MAPPO 21、w/o GAP-HO 25、MTS/w/o PET-BF/w/o OTR-RA 33、MEET 35 Mbps；Oracle 在全部负载上均未超过。
- 35 Mbps 时 MEET 为 175.80 W、U=1.150%；MTS 为 175.26 W、U=1.917%；O-MAPPO 为 183.08 W、U=31.454%；Reactive 的 U=34.463%。
- 低负载下 Reactive 和 O-MAPPO 的 U 可能低于 MEET，不能声称 MEET 在全部负载拥有最低违反概率。
- Oracle 的信息优势仍有可见收益，不能将当前预测误差影响笼统写为“可忽略”。

## 4. 干扰验证与参考量解释

R2C1 固定决策比较沿最终 MEET-COBRA 的决策轨迹进行，使用相同关联、波束、RB 数量和矩阵信道。比较中的均匀波束平均值由真实矩阵得到，故衡量两项平均近似的组合误差，不包含 NN 标签预测误差。

- 三 seed 平均干扰比值范围 **0.6987–2.6627**。
- 聚合微基站服务容量低估范围 **6.4906%–19.5025%**，正文写 **6.5%–19.5%**。
- R2C1 回复列出全部 18 个负载的两项成对统计，不再使用先前零散负载的验证表。
- 干扰比值在部分负载小于 1，并不与服务容量低估矛盾；速率依赖非线性 SINR 和空间、时隙分布。不能把聚合服务量的经验保守性写成每个链路的干扰上界。
- 另用全方案闭环性能说明上述估计引导决策后的实际服务与队列结果，不把固定决策重放当作闭环验证。

Oracle-CR-LB 是每帧最后一个固定 P2 实例的连续松弛参考，不是实际方向性服务下闭环系统功率的全局下界，因此它可能高于 Oracle-MC 的实际发射功率。31、33、35 Mbps 下的松弛问题可行比例分别约为 99.33%、86.24%、34.23%，不可行实例采用原协议的功率上限替代；图中叉号标识这些负载。已删除旧稿由该曲线推断系统全局近优性的表述。

Reactive-OBRA 的实际实现按估计功率代价顺序分配可用 BS，并非 RSS-first 或按最近距离卸载。正文已据代码更正描述；本轮没有改变 Reactive 的算法、输入或实验结果。

## 5. 图文联动及未覆盖任务

- 正文：Section V-A 的干扰数据，V-C 的参考量与 Reactive 描述，V-D 的图表、统计方式和结果解读，摘要及结论中的数值。
- 回复：R2C1 的完整误差与闭环证据；R2C2 的分位数定义和 Fig.7；R2C6 的 HO 全方案评估完成状态；补写 R2C5 和 R3 Major C3 的独立方法与结果回复。审稿意见原文均未改动。
- 后续已按用户要求另行完成 R1C6 的 27 次迭代敏感性重跑，使用同一最终方向性服务评估器；新表已替换早期数据。见 `experiment/gap_refinement_directional_report.md`。固定两轮的 9 次原始结果精确复现本报告中的 Oracle-MC，因此 Fig.5–8 无须重新生成。
- 未将本轮数据冒充预测误差注入或速度控制实验；R1C4、R2C3、R3 Major C4 仍待后续工作。
- 本轮新增或更新的文字标红；相对原投稿删去的正文保留 `\currentdel` 痕迹，旧图不删除。
- 最终编译：论文 15 页、回复信 26 页；无未定义引用或引文。26 段编辑/审稿意见逐字保留，原图和 `main.tex` 无改动。已检查论文第 13–14 页的实际排版，图例、曲线及 20 ms 标注互不遮挡。

## 6. 复现

在仓库根目录运行，只重算审计与图表，不调用 GPU 仿真：

```bash
python -u experiment/plot_revision_system_results.py --audit
```

只重新绘图（沿用已有审计清单）：

```bash
python experiment/plot_revision_system_results.py
```

生成的 `paper_figures/figure_data.csv` 保存各指标的均值及 seed 范围；`figure_manifest.json` 保存统计定义、干扰表及 L99 门限交点；`audit.json` 保存完整性核查记录。原始 `grid/aggregate` 未被覆盖。

编译命令：

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error latexCodes/main_revision1.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error response_letter/response_letter.tex
```
