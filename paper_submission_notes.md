# 投稿定位、Related Work 与 Baseline 清单

## 一、这套方法适合对比的方法有哪些？

如果要投稿，对比方法和投稿 venue 都应该围绕这套方法的定位来选：

> **无预训练、normal-only、reconstruction-based、关注 anomaly map 质量与可解释性。**

---

## 二、应该对比哪些方法

建议分成 4 组 baseline，而不是只和 PatchCore 对比。

### A. 经典重构类
这些是最应该正面对比的同类方法。

#### 1. Autoencoder / CAE
- 最基础 baseline。
- 直接用重构误差做 anomaly map。
- 作用：说明普通重构方法的上限，以及你的方法相对最朴素重构框架的改进。

#### 2. VAE
- 经典概率重构方法。
- 作用：说明为什么使用 VQ-VAE 比连续潜变量建模更适合当前任务。

#### 3. VQ-VAE / VQ-based anomaly detection
- 如果能找到工业缺陷检测中直接使用 VQ-VAE 的工作，必须对比。
- 这是和你最接近的一类方法。

#### 4. GAN-based reconstruction
可选代表：
- AnoGAN
- GANomaly
- Skip-GANomaly

作用：
- 审稿人会自然联想到这一类工作。
- 即使不是最强，也值得作为历史代表方法列入 related work 或 baseline。

---

### B. 更强的重构 / 预测 / 像素级定位方法
这组用于说明你不是只和很老的方法比较。

#### 5. DRAEM
- 工业异常定位中非常常见。
- 虽然使用合成异常，不是严格的 pure normal-only。
- 但 pixel-level localization 很强，通常值得对比。

#### 6. STFPM
- Student-Teacher Feature Pyramid Matching。
- 不是传统重构，但属于 anomaly map 表达能力很强的方法。

#### 7. RD4AD
- reconstruction + distillation 风格的代表方法。
- 与你的“重构 + 结构异常图”有一定相关性。

#### 8. Reverse Distillation
- 与 RD4AD 同类。
- 常作为工业 AD 里中强 baseline。

---

### C. 非重构但工业 AD 的强 baseline
这组用于说明你方法在整个工业异常检测领域中的位置。

#### 9. PatchCore
- 几乎必须对比。
- 即使你的方法不依赖 memory bank，也需要给出对照。

#### 10. PaDiM
- 分布建模类经典方法。
- 很适合做 feature distribution modeling 的代表。

#### 11. SPADE
- 较早期但经典。
- 适合作为 related work 中的代表工作。

#### 12. FastFlow / CFlow-AD
- flow-based 方法代表。
- 用于说明密度建模路线与 reconstruction 路线的差异。

---

### D. 如果主打“无预训练”，建议单独做一个公平对比表
审稿人很容易问：

> 你不如 PatchCore，是不是只是因为你没有使用 pretrained backbone？

所以建议专门做两张表：

#### 表 1：不依赖 ImageNet 预训练的公平对比
只放：
- AE
- VAE
- VQ-VAE baseline
- 你的方法
- 你的方法 + adaptive normalization

目的：
- 证明你在“无预训练、normal-only reconstruction”这条赛道里是有价值的。

#### 表 2：与强工业 AD 方法的总体对比
放：
- PatchCore
- PaDiM
- DRAEM
- RD4AD
- STFPM
- 你的方法

目的：
- 说明你在整个领域里的相对位置，而不是回避强方法。

---

## 三、投稿时必须做的 ablation

### 必做 ablation

#### 1. 只用 MAD
- 只使用 latent quantization error。
- 用于证明 latent-space 信号本身的作用。

#### 2. 只用 1-SSIM
- 只使用图像空间结构重建误差。
- 用于证明结构误差的独立贡献。

#### 3. MAD + (1-SSIM)
- 证明双空间融合比单一信号更有效。

#### 4. MAD + (1-SSIM) + adaptive normalization
- 这是核心创新点。
- 用于证明 anomaly map 校准的贡献。

---

### 强烈建议补充的 ablation

#### 5. 全局归一化 vs 逐像素归一化
建议对比：
- global mean/std normalization
- per-image normalization
- per-pixel normalization（你的方法）

目的：
- 直接证明逐像素统计的设计合理性。

#### 6. texture 类 vs object 类分组分析
建议把 MVTec 分成两组：
- 纹理类：carpet, grid, leather, tile, wood
- 对象类：bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper

目的：
- 强调你的方法主要解决的是纹理底噪问题。

#### 7. 训练轮数影响
建议比较：
- 50 epoch baseline
- 100 epoch baseline
- 50 epoch + normalization
- 100 epoch + normalization

目的：
- 回答“提升到底来自训练更久，还是来自 adaptive anomaly map”。

---

## 四、适合投稿的期刊

如果当前创新点主要包括：
- VQ-VAE + MAD / SSIM 融合
- adaptive anomaly map normalization
- normal-only, no pretrained

那么更适合应用型或工业视觉检测方向期刊，而不是一上来就冲最顶尖模式识别期刊。

### 1. IEEE Transactions on Instrumentation and Measurement (TIM)
推荐程度：高

理由：
- 接受工业视觉检测类工作。
- 接受方法改进 + 完整实验验证。
- 不一定要求绝对 SOTA，但需要清楚的方法创新点。

适合你的切入方式：
- industrial anomaly detection
- dual-space anomaly scoring
- adaptive anomaly map normalization

---

### 2. Engineering Applications of Artificial Intelligence (EAAI)
推荐程度：高

理由：
- 工程背景明确的方法很适合。
- 允许应用导向，但仍要求一定方法创新。
- 对工业异常检测类论文较友好。

---

### 3. Expert Systems with Applications (ESWA)
推荐程度：中高

理由：
- 对应用型 AI 工作友好。
- 接收工业视觉检测类论文较多。
- 需要实验较完整。

---

### 4. Computers in Industry
推荐程度：中高

理由：
- 很适合工业场景导向的问题。
- 如果强调可部署性、只依赖正常样本、无需预训练，会比较匹配。

---

### 5. Journal of Manufacturing Systems
推荐程度：中

理由：
- 如果文章更强调制造场景与工业意义，可以考虑。
- 适合讲工业检测流程与落地价值。

---

### 6. IEEE Access
推荐程度：保底 / 快速发表

理由：
- 对工程型完整工作更友好。
- 包容性较强。
- 但整体认可度通常低于 TIM / EAAI / ESWA。

---

## 五、如果想冲更强一些的期刊

### Pattern Recognition (PR)
当前状态：可以作为冲刺目标，但要求会明显更高。

通常需要：
- 方法创新点更凝练；
- 实验更充分；
- baseline 更强；
- ablation 更完整；
- 最好不止一个数据集。

如果后续能把“adaptive anomaly map normalization”讲成更一般性的 reconstruction AD 后处理框架，并补足更多实验，PR 才更有把握。

---

## 六、目前不太建议直接投的顶刊
- IEEE TPAMI
- IJCV
- TNNLS
- TIP

原因：
- 当前方法的创新性和实验广度大概率还不够支撑。
- 除非后续补上更多数据集、更强理论、更广泛泛化实验。

---

## 七、如果先投会议，可以考虑哪些
比较合适：
- WACV
- BMVC
- ACCV
- ICPR
- ICIP

理由：
- 对工业 AD、重构类方法、应用型视觉任务更友好。

更高要求但可尝试：
- CVPR Workshop
- ECCV Workshop

主会 CVPR / ICCV / ECCV 目前竞争会非常激烈，需要更强的方法新意与更大规模实验支撑。

---

## 八、投稿版 Related Work 清单

下面是更适合写论文时展开的 related work 结构。

### 1. Reconstruction-based anomaly detection
建议覆盖：
- Autoencoder-based AD
- Variational autoencoder-based AD
- GAN-based reconstruction AD
- VQ / discrete latent reconstruction AD

应该强调的主线：
- 这类方法只用正常样本训练；
- 通过重构误差检测异常；
- 优点是直观、可解释；
- 缺点是正常高频纹理也会引入较强重构噪声，导致 anomaly map 不干净。

你自己的切入点可以写成：
> 现有 reconstruction-based 方法大多关注提升重建器本身，而较少显式建模 anomaly map 中由正常纹理引入的空间底噪。

---

### 2. Feature-discrepancy / distillation-based methods
建议覆盖：
- STFPM
- Reverse Distillation
- RD4AD

应该强调：
- 这类方法往往通过 teacher-student 或 feature reconstruction 的差异构造 anomaly map；
- 相比纯像素重构，它们更容易得到干净的定位结果；
- 但通常依赖预训练表征，或在语义特征空间中工作。

你可以借此突出自己的不同：
> 你是 normal-only、从头训练、不依赖外部预训练语义。

---

### 3. Distribution modeling / memory-based methods
建议覆盖：
- PatchCore
- PaDiM
- SPADE
- FastFlow / CFlow-AD

应该强调：
- 这类方法通常在特征空间中建立正常分布或记忆库；
- 在 benchmark 上通常更强；
- 但往往依赖预训练 backbone 或复杂的检索 / 密度建模过程。

你自己的定位：
> 不是追求最强 benchmark，而是构建更纯粹、更可解释的 normal-only reconstruction 框架。

---

### 4. Anomaly map calibration / score normalization
这是你最应该重点补文献的一节。

建议重点搜索和整理的关键词：
- anomaly score normalization
- heatmap calibration for anomaly detection
- statistical normalization of anomaly maps
- per-pixel normalization anomaly localization
- normality modeling in anomaly maps
- residual map calibration

这节应该强调：
- 很多工作关注 anomaly score 的阈值选择；
- 但较少工作直接对最终像素级 anomaly map 做训练后逐像素统计校准；
- 尤其在纹理类类别中，正常样本会产生结构化底噪，这一问题值得单独研究。

你的方法在这节的表述可以是：
> a simple post-hoc pixel-wise anomaly map normalization using the mean and standard deviation estimated from normal training samples.

---

## 九、投稿版 Baseline 清单

### A. 必须复现 / 重点比较
这些建议自己跑或尽量统一实验设置：

1. **AE / CAE**
2. **VAE**
3. **VQ-VAE baseline**（只用 reconstruction 或只用单一 anomaly score）
4. **PatchCore**
5. **PaDiM**
6. **DRAEM**
7. **STFPM**
8. **RD4AD 或 Reverse Distillation**

---

### B. 如果实验资源有限，可作为补充引用的 baseline
如果没有足够算力完全复现，可在 related work 或结果表中引用公开结果，但要注明来源：

- SPADE
- FastFlow
- CFlow-AD
- GANomaly / Skip-GANomaly

注意：
- 如果直接引用他人结果，必须说明实验设置可能不同；
- 核心结论不要只依赖引用结果，最好至少对关键 baseline 自己复现。

---

## 十、你自己的方法应该拆成哪些版本
投稿时建议至少展示以下版本：

1. **Reconstruction-only**
   - 只用重构误差（如 MSE 或 1-SSIM）

2. **MAD-only**
   - 只用 latent quantization error

3. **MAD + (1-SSIM)**
   - 双空间融合版本

4. **MAD + (1-SSIM) + adaptive normalization**
   - 最终完整版本

可选补充：
5. **global normalization**
6. **per-image normalization**
7. **per-pixel normalization**

---

## 十一、实验 section 的推荐组织方式

### 1. Main comparison
- 与 PatchCore、PaDiM、DRAEM、STFPM、RD4AD 等方法比较。

### 2. Fair comparison under no-pretraining setting
- 与 AE、VAE、VQ-VAE baseline 等无预训练方法比较。

### 3. Ablation study
- MAD only
- SSIM only
- MAD + SSIM
- + adaptive normalization

### 4. Category-wise analysis
- texture vs object
- 说明 adaptive normalization 主要改善纹理类 anomaly map。

### 5. Visualization
建议展示：
- input image
- ground truth
- MAD map
- 1-SSIM map
- fused map
- normalized fused map

这样最能体现你的核心贡献。

---

## 十二、写作时最值得强调的卖点
如果投稿，最应该强调的不是“整体分数 SOTA”，而是：

1. **A fully normal-only anomaly detection framework without pretrained semantics or memory retrieval.**
2. **A dual-space anomaly signal combining latent quantization deviation and image-structure reconstruction error.**
3. **A post-hoc adaptive anomaly-map normalization that suppresses normal texture noise and improves localization on texture-dominated categories.**
4. **Improved interpretability of anomaly formation compared with single residual maps.**

---

## 十三、当前最推荐的投稿策略
优先顺序建议：

1. **IEEE TIM**
2. **EAAI**
3. **ESWA**
4. **Computers in Industry**
5. **IEEE Access**

如果后续能补更多数据集、做更完整的归一化分析和更强 baseline，再考虑 **Pattern Recognition**。
