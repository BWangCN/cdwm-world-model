# Colleague update — two drop-WM extensions (2026-07-26)

Both pushed to `github.com/BWangCN/cdwm-world-model` (branch `restructure-common-grasp-drop`, commit `64b311f`; two new sections in `drop/docs/00_overview.md`). Both significant, robustness-tested, Codex-reviewed.

## 1. CoACD end-to-end geometry — the hidden-CoM mechanism survives faithful geometry
Closes the known single-hull scope limitation. We regenerated Gate B on the real mesh's **CoACD convex decomposition** (faithful multi-hull collision) AND fed the world model a cloud **re-sampled from the same real-mesh surface** (sim + WM geometry-consistent). Retrained object-disjoint on 29 mesh objects; object-bootstrap over the 7 held-out test objects.

| arm | NLL | top-1 | ECE |
|---|---|---|---|
| point | 1.447 | 0.56 | 0.166 |
| no_latent (distribution) | 1.057 | 0.63 | 0.042 |
| abstract-CoM | 1.028 | 0.63 | 0.048 |
| shuffled-CoM (control) | 1.054 | 0.63 | 0.055 |
| **grounded (hidden CoM)** | **0.821** | **0.71** | 0.050 |

- distribution > point **+0.355** [+0.195,+0.537] SIG (7/7 objects)
- grounded > no_latent (hidden CoM causal) **+0.245** [+0.179,+0.296] SIG (7/7)
- grounded > abstract **+0.212**, grounded > shuffled **+0.236**, both SIG 7/7 — abstract and wrong-CoM control BOTH collapse to no_latent, so the effect needs the *correct geometry-grounded CoM*, not extra channels.
- The CoM-causal gain is **larger** than on the single hull (+0.245 vs +0.184): faithful geometry has more resting modes, so the CoM decides the basin more often → the mechanism is not a hull artifact.

## 2. Inferring the CoM from observations — test-time instance adaptation (drop the oracle)
The transfer arms are handed the true CoM. Here the model **infers** the hidden CoM from a few observed drops of the same object instance. Using the frozen grounded WM as a likelihood `P(basin | CoM, release)`, we do analytic Bayesian updating over a CoM grid; predict the next drop by marginalizing. No new training/data (843 fixed-CoM instances grouped from existing episodes). 19 held-out objects.

| observed drops k | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| predictive NLL | 0.678 | 0.655 | 0.631 | 0.597 | **0.566** |

- paired per-object adaptation gain k=0→k=8 **+0.111** [+0.045,+0.212] SIG (17/19 objects); no-latent = 0.744.
- posterior concentrates with evidence (entropy 3.39→2.82; mass on true CoM 0.03→0.05).
- marginalizing beats a single MAP CoM (0.566 vs 0.600) and stays better calibrated (ECE 0.043 vs 0.065); no support/query leakage (0/600).

## 3. What we tried that did NOT work (informative null)
A **learned amortized encoder** `q(CoM | observed drops, geometry)` — a single-forward-pass alternative to the analytic Bayesian inference. It learns a **useful geometry-conditioned CoM prior** (NLL ~0.67, better than no_latent 0.744, ≈ the analytic 1-shot), but it is **FLAT across k** (0.672→0.675, gain −0.003 ns) — it does not extract the *adaptation* signal from observations that the analytic method captures (0.678→0.566). Takeaway: the geometry→CoM-prior is amortizable, but the observation→CoM-*update* is not (with this approach). This actually sharpens ②: the contribution is specifically the **adaptation**, delivered by the analytic WM-as-likelihood Bayesian method. (Systems extension, not required for the claim; code stays in the working repo.)

---

## 中文摘要（发同事）

两个 drop 世界模型扩展，均已推送 GitHub（commit `64b311f`），均显著、通过稳健性检验、Codex 审核。

**① CoACD 端到端几何 —— 隐藏质心机制在更真实几何下依然成立。** 用真实网格的 CoACD 凸分解重建 Gate B 物理，并让世界模型输入点云也从同一真实网格表面采样（仿真与模型几何一致）。29 个网格物体，物体不相交划分，7 个留出测试物体上 bootstrap：
- 分布 > 单点 **+0.355** 显著（7/7）；grounded（隐藏质心）> no_latent **+0.245** 显著（7/7）。
- grounded > abstract **+0.212**、grounded > shuffled（错误质心对照）**+0.236**，均显著 —— abstract 与错误质心都退化到 no_latent，说明要的是**正确的几何锚定质心**，不是多加通道。
- 质心因果增益比单凸包更大（+0.245 vs +0.184）：更真实几何有更多稳定姿态，质心更常决定落定 basin —— 机制不是凸包假象，反而更强。

**② 从观测推断质心 —— 测试时实例自适应（去掉 oracle）。** 用冻结的 grounded 世界模型当似然 `P(basin|质心,释放)`，对同一物体实例的 k 次观测做贝叶斯更新推断隐藏质心，边缘化预测下一次。无需新训练/数据（843 个固定质心实例）。19 个留出物体：
- 预测 NLL 随 k 单调下降 0.678→0.566；配对增益 k=0→k=8 **+0.111** 显著（17/19 物体）。
- 后验随证据收敛（熵 3.39→2.82）；边缘化优于单点 MAP 且更好校准；无泄漏。

**③ 没走通的（有信息量的负结果）：** 学习式摊销编码器（一次前向预测质心）能学到**有用的、几何条件化的质心先验**（NLL≈0.67，优于 no_latent 0.744，≈解析法单次观测），但**随 k 不变**（0.672→0.675，增益 −0.003 不显著）—— 它没能从观测中提取解析法所捕捉的**自适应**信号（0.678→0.566）。结论：几何→质心先验可摊销，但观测→质心更新（自适应）不可摊销。这反而强化了 ②：真正的贡献是**自适应**，由解析式贝叶斯（世界模型当似然）方法实现。
