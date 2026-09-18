# GAP-HO RB 占用上限修正（2026-09-18）

按用户要求，Algorithm 1 第 11 步及对应正文改为

\[
\hat k_m^{[x+1]}=\min\left\{K_m,
\sum_{v\in\mathcal V^{[x]}}\widetilde k_{m,v}^{[x+1]}\hat u_{m,v}^{[x+1]}\right\}.
\]

`main_revision1.tex` 的本轮修改标红；以提交稿 `main.tex` 为基准保留删除标记。仅补充一句说明该上限用于干扰计算中的物理 RB 占用，不改 P2 的目标、逐链路需求、HO 容量修正和可行性修复。

## 代码范围

- `utils/gap_refinement.py`：每轮分配之后、下一轮干扰计算之前，对总需求按物理容量截断；修复后返回的占用也有界。
- `utils/alg_utils.py`：当前 MEET 使用的 `HO_EE_GAP_APX_SINR_conservative_adaptive` 显式两轮路径同步修改。
- `utils/sim_utils.py`：转发 `gap_cap_rb_usage`，默认启用新行为；显式设为 `False` 可复现修正前更新。
- 截断上限是物理容量 `K_m`，不是预留资源后的规划容量，也不是乘以 zeta 的放松容量。
- 求解成本、容量约束和修复仍接收未截断需求；HO 因子仍只作用于容量系数，不加入帧平均占用更新。
- 诊断同时保存未截断的 `implied_demand`、有界的 `implied_load`，以及修复后原始容量负载，避免掩盖不可行性。

本轮**没有修改**公共 `estimate_num_RB_allocated_perBS`、OTR-RA、BF 预测报告时序、MTS/O-MAPPO 代码、回复信或既有实验结果。这里只落实 GAP-HO 第 11 步的修正，不表示上一轮发现的所有公共负载估计问题均已解决。

## 验证及回溯

- 38 项测试通过，包括新增的强制过载、物理容量与规划容量区分、需求不截断、不可行性不掩盖、两种 GAP 路径一致、仿真器开关转发，以及既有 HO/MTS 回归测试。
- 从 Git 回溯标签提取修改前的实际函数，在三档到达率、启用或关闭 HO 容量修正、两种实现路径下进行 12 组比较。设 `gap_cap_rb_usage=False` 后，关联结果和返回负载与修改前逐项完全一致。
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main_revision1.tex` 编译通过，15 页；已检查第 8 页 Algorithm 1 和相邻正文，无新增公式越界。
- 修改前代码标签：`pre-gap-occupancy-cap-20260918`（`089aa06`）。该标签保存代码状态；论文原有未提交编辑未被回滚或覆盖。
- 旧实验协议的代码哈希因本轮修改而不再匹配当前实现，这是预期保护行为。不要更新旧协议哈希或把旧结果改称新算法结果；今后的重跑须另建协议和输出目录。

测试命令：

```bash
python -m unittest test_gap_rb_usage test_gap_refinement test_ho_interruption test_mts_report.MTSReportTests test_mts_report_bounded.BoundedLoadTests test_revision_grid test_mts_gs_hbf -v
```

本轮未重跑 Fig.5–Fig.8，也未就截断后的性能或两轮迭代的效果作新结论。
