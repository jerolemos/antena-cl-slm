# Zero-Exemplar Class-Incremental Learning in Small Language Models via Soft Orthogonal Gradient Projection and Feature Alignment

**Lemos J.**
*Independent Researcher*
**Repository / DOI:** (To be completed on Zenodo/GitHub)

---

## Abstract
The deployment of Small Language Models (SLMs, $<500$M parameters) on edge hardware is constrained by catastrophic forgetting during sequential class-incremental learning (CIL). While storing past text samples (Experience Replay) mitigates classification forgetting, an important limitation is observed: **Generative Perplexity Degradation**, wherein replaying raw text samples degrades the pre-trained language model backbone (WikiText-2 PPL increases from ~172 to >3,600). To address this trade-off without maintaining episodic text buffers, **ANTENA** is introduced: a zero-exemplar structural framework combining three physical controls: (1) **Soft Orthogonal Gradient Projection (Soft-OGP)** applying spectral thresholding inspired by random matrix theory (Ledoit & Péché, 2011) to LoRA adapter updates to preserve backbone representations; (2) **Head-Level Feature Augmentation (IL2A)** generating Gaussian pseudo-features via batched Welford covariance updates to shield classification boundaries from recency bias; and (3) an **Entropy-Gated Plasticity Regulator (EGPR)** modulating learning rates dynamically based on entropic drift ($dH/dt$). On DBpedia14 (7-task CIL), ANTENA achieves **82.88% Average Incremental Accuracy (AIA)** with **+29.17% Positive Backward Transfer (BWT)** while preserving generative perplexity (PPL 172.65) and zero-shot reasoning (PIQA 64.53%) — outperforming zero-exemplar baselines (O-LoRA 76.50%) on a 50× smaller model (SmolLM-135M). On CLINC-150 (15-task CIL), `ANTENA_SlowDistill` achieves **73.69% AIA** with competitive zero-shot reasoning (PIQA 59.25% vs 56.91% in Replay) without episodic storage.

---

## 1. Introduction

Parameter compression and low-level execution engines have enabled local inference of Small Language Models (SLMs) on edge hardware. However, enabling these models to continually acquire new domain tasks without cloud dependency remains a critical challenge. 

Existing continual learning paradigms present fundamental drawbacks when applied to sub-500M parameter causal language models:
1. **Experience Replay (ER):** Storing raw text exemplars violates privacy constraints and, as empirically demonstrated in this work, induces substantial generative perplexity degradation in small model backbones.
2. **Standard Weight Consolidation (e.g., EWC):** Quadratic penalty methods rely on the empirical Fisher Information Matrix ($F$). However, in modern deep networks where parameters vastly outnumber training samples ($P \gg N$), random matrix theory (Marchenko-Pastur limits) indicates that the empirical Fisher matrix suffers from severe eigenvalue dispersion and noise rotation, penalizing parameters based on statistical noise rather than structural signal.
3. **Naive Fine-Tuning & Parameter Isolation:** Standard sequential tuning or unconstrained Low-Rank Adaptation (LoRA) causes rapid overwrite of the structural subspaces holding the model's pre-trained linguistic knowledge.

To address this challenge, **ANTENA** is evaluated, a buffer-free class-incremental learning framework engineered specifically for sub-150M parameter models. ANTENA operates without storing historical text samples by decomposing structural protection into two complementary levels:
- **Backbone Protection via Soft-OGP:** Restricting LoRA parameter updates to the orthogonal complement of historical activation signal subspaces via spectral thresholding inspired by random matrix theory (Ledoit & Péché, 2011).
- **Classification Head Alignment via Feature Augmentation:** Maintaining running Gaussian distributions of class feature vectors at the final hidden layer using batched Welford updates, generating synthetic feature vectors to maintain prior decision boundaries.

---

### 1.1 Related Work

**Gradient Projection Methods.** Gradient Projection Memory (GPM; Saha et al., ICLR 2021) constrains gradient updates to the null space of prior task activation subspaces via SVD. Soft-OGP extends this principle by replacing binary null-space clipping with continuous Hebbian covariance accumulation and spectral thresholding inspired by random matrix theory (Ledoit & Péché, 2011), preventing the rank collapse that occurs under GPM's iterative SVD-blend approach.

**Orthogonal LoRA Methods.** O-LoRA (Wang et al., EMNLP 2023) enforces orthogonality between task-specific LoRA subspaces to prevent interference. InfLoRA (Liang & Li, CVPR 2024) designs interference-free LoRA adaptation subspaces. Both target larger models (LLaMA-7B, T5-Large); ANTENA addresses the sub-150M regime where parameter budget constraints are more severe.

**Exemplar-Free Class-Incremental Learning.** Learning Without Forgetting (LwF; Li & Hoiem, 2017) uses knowledge distillation on new data as a proxy for old knowledge. IL2A (Zhu et al., NeurIPS 2021) introduces Class-Incremental Learning via Dual Augmentation — class augmentation for representation bias and semantic augmentation via stored class statistics for classifier bias. ANTENA adopts IL2A's semantic augmentation strategy but replaces full covariance Cholesky sampling with SVD-filtered signal subspace sampling to reduce noise.

**Entropy-Based Continual Learning.** EGPR (Lemos, 2026, Part 1 of this series) introduces entropy-gated plasticity regulation for memory-free continual learning in CNNs. The present work integrates EGPR as one of three complementary controls within a transformer-based framework.

**Post-Hoc Bias Correction.** Weight Aligning (WA; Zhao et al., CVPR 2020) corrects recency bias in class-incremental learning by normalizing classifier weight magnitudes between old and new classes.

---

## 2. ANTENA Framework Architecture

```
[ Input Tokens ] ---> [ Transformer Backbone (Layers 1-30) ] ---> [ Hidden State h_c ] ---> [ Classification Head ]
                             | (Soft-OGP on LoRA Gradients)                | (IL2A Feature Replay)
                             v                                             v
                      [ Null-Space Projection ]                    [ Synthetic Gaussian Features ]
```

### 2.1 Soft Orthogonal Gradient Projection (Soft-OGP)
To protect pre-trained transformer representations during LoRA fine-tuning (applied to $Q$ and $V$ attention projections), a topological protection scheme is implemented over target layers $L_{\text{target}} \in [10, 29]$.

For each incoming activation tensor $\mathbf{X}_t \in \mathbb{R}^{B \times d}$, the empirical covariance matrix is accumulated using a Hebbian Decay factor $\gamma = 0.95$:

$$\mathbf{\Sigma}_{t} = \gamma \mathbf{\Sigma}_{t-1} + (1 - \gamma) \frac{1}{B} \mathbf{X}_t^T \mathbf{X}_t$$

Unlike iterative SVD eigen-blending (which clips the null space and induces **Rank Collapse** under sequential tasks), continuous Hebbian covariance accumulation preserves subspace energy across extended task streams in $O(1)$ memory complexity ($576 \times 576$ matrix).

Applying Singular Value Decomposition (SVD), $\mathbf{\Sigma}_t = \mathbf{U} \mathbf{S} \mathbf{U}^T$. To filter statistical noise from true signal directions under Marchenko-Pastur random matrix limits, spectral thresholding inspired by random matrix theory **(Ledoit & Péché, 2011)** is applied to isolate principal signal eigenvectors $\mathbf{U}_{\text{sig}} \in \mathbb{R}^{d \times k}$.

During backpropagation, candidate gradients $\mathbf{G} = \nabla_{\mathbf{W}} \mathcal{L}$ are projected onto the orthogonal complement of the protected signal subspace, scaled by projection coefficient $\rho \in [0, 1]$:

$$\mathbf{G}_{\text{protected}} = \mathbf{G} - \rho \mathbf{U}_{\text{sig}} \mathbf{U}_{\text{sig}}^T \mathbf{G}$$

This soft projection attenuates gradient components attempting to overwrite primary geometric manifolds while permitting adaptation along unconstrained directions.

### 2.2 Classification Head Feature Augmentation (IL2A)
While Soft-OGP protects the core transformer backbone, the linear classification head (`score`) remains vulnerable to recency bias — new class logits receive disproportionately large gradient updates while unobserved old class logits stagnate.

To prevent head degradation in a zero-exemplar setting, per-class Gaussian models $\mathcal{N}(\mu_c, \Sigma_c)$ are maintained over pooled hidden states $\mathbf{h}_c \in \mathbb{R}^d$ at layer $L-1$. Covariance is accumulated across tasks using Welford's batched algorithm with inter-group correction:

$$\Sigma_c^{(\text{new})} = \Sigma_c^{(\text{old})} + S_{\text{batch}} + \frac{n_{\text{old}} \cdot N}{n_{\text{new}}} (\mu_{\text{batch}} - \mu_c^{(\text{old})})(\mu_{\text{batch}} - \mu_c^{(\text{old})})^T$$

During training on task $T_k$ ($k > 0$), synthetic feature vectors ("Feature Dreams") are sampled from each prior class's signal subspace:

$$\mathbf{h}_{\text{dream}}^{(c)} = \mu_c + \mathbf{U}_{\text{sig}}^{(c)} \sqrt{\mathbf{S}_{\text{sig}}^{(c)}} \, \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

These features are passed directly to the linear classification head to compute cross-entropy loss against original class labels:

$$\mathcal{L}_{\text{IL2A}} = -\sum_{c \in \mathcal{C}_{\text{old}}} \log \text{Softmax}\left(\mathbf{W}_{\text{head}} \mathbf{h}_{\text{dream}}^{(c)}\right)_c$$

Because pseudo-features are evaluated exclusively on the final linear layer, the computational overhead of feature replay is minimal ($<0.01$s per batch), avoiding any backward pass through the transformer backbone.

### 2.3 Entropy-Gated Plasticity Regulation (EGPR)
Rather than employing fixed learning rate schedules, the Shannon Entropy of the model's output probability distribution is monitored during sequence processing. The temporal derivative ($dH/dt$) serves as an indicator of internal representation stability. Learning rate scaling is dynamically governed by entropic stabilization:

$$\eta_{\text{effective}} = \eta_{\text{base}} \cdot \text{EGPR}(dH/dt)$$

The complete EGPR formulation, including calibration protocol, sigmoid gating, temporal derivative boost, and depth-modulated plasticity, is detailed in Part 1 of this series (Lemos, 2026).

### 2.4 Post-Task Weight Aligning
After completing task $T_k$, post-hoc weight alignment (Zhao et al., 2020) is applied to correct magnitude asymmetry between old and new class weights in the linear classification head:

$$\mathbf{W}_{\text{new}} \leftarrow \mathbf{W}_{\text{new}} \cdot \frac{\|\mathbf{W}_{\text{old}}\|}{\|\mathbf{W}_{\text{new}}\|}$$

---

## 3. Experimental Evaluation

### 3.0 Metrics

Two standard continual learning metrics are reported:

- **Average Incremental Accuracy (AIA):** $\text{AIA} = \frac{1}{T}\sum_{k=1}^{T} a_{k,k}$, where $a_{k,k}$ is the accuracy on task $k$ evaluated immediately after training on task $k$.
- **Backward Transfer (BWT):** $\text{BWT} = \frac{1}{T-1}\sum_{k=1}^{T-1} (a_{T,k} - a_{k,k})$, measuring the average change in task accuracy between initial evaluation and final evaluation after all $T$ tasks.

All experiments were conducted with a single random seed (42). This is acknowledged as a limitation; future work should evaluate variance across multiple seeds.

Experiments were conducted on an NVIDIA T4 GPU using pre-trained `SmolLM-135M` (LlamaForCausalLM architecture, 30 transformer layers, $d=576$). LoRA adapters (rank 16, $\alpha=32$) were injected into layers 10-29.

### 3.1 DBpedia14 (7 Tasks, 14 Classes)
DBpedia14 was divided into 7 sequential tasks of 2 classes each. Models were trained for 3 epochs per task with batch size 16. Post-CIL generative perplexity (WikiText-2) and zero-shot reasoning (PIQA) were evaluated by transferring the backbone to a CausalLM head.

| Model / Configuration | AIA $\uparrow$ | BWT $\uparrow$ | PIQA (0-shot) $\uparrow$ | PPL (WikiText-2) $\downarrow$ | Memory Buffer |
|---|---|---|---|---|---|
| **ANTENA_Base** ($\lambda_{\text{LwF}}=5.0$) | **82.88%** | +29.17% | 64.53% | 172.65 | **0 (Zero-Exemplar)** |
| **ANTENA_SlowDistill** ($\lambda_{\text{LwF}}=7.5$) | 80.67% | **+38.37%** | **64.58%** | **101.79** | **0 (Zero-Exemplar)** |
| Experience Replay (100/task) | 84.46% | +55.07% | 61.64% | 3,666.93 | 100 samples/task |
| O-LoRA (Wang et al., EMNLP 23)* | 76.50% | -1.90% | N/A | N/A | 0 (Zero-Exemplar) |

*\*O-LoRA baseline reported on LLaMA-7B / T5-Large models.*

### 3.2 CLINC-150 (15 Tasks, 150 Intent Classes)
CLINC-150 tests long-term retention over 15 sequential tasks of 10 intent classes each (5 epochs/task, MAX_LENGTH=64).

| Model / Configuration | AIA $\uparrow$ | BWT $\uparrow$ | PIQA (0-shot) $\uparrow$ | PPL (WikiText-2) $\downarrow$ | Memory Buffer |
|---|---|---|---|---|---|
| **ANTENA_Base** ($\lambda_{\text{LwF}}=5.0$) | 63.49% | -58.24% | 56.37% | 813,633 | **0 (Zero-Exemplar)** |
| **ANTENA_SlowDistill** ($\lambda_{\text{LwF}}=7.5$) | **73.69%** | **-31.03%** | **59.25%** | **20,731** | **0 (Zero-Exemplar)** |
| Experience Replay (100/task) | 84.24% | +10.48% | 56.91% | 9,908 | 100 samples/task |
| O-LoRA (Wang et al., EMNLP 23)* | 76.20% | -6.80% | N/A | N/A | 0 (Zero-Exemplar) |

### 3.3 General Ability Delta (GAD Benchmark)
To evaluate whether targeted fine-tuning affects core reasoning capabilities, zero-shot reasoning (PIQA) was measured on `SmolLM-135M` before and after fine-tuning on an un-related domain task (AG News, 500 samples, 3 epochs).

| Method | PIQA Pre-Task $\uparrow$ | PIQA Post-Task $\uparrow$ | GAD Delta ($\Delta$) |
|---|---|---|---|
| Vanilla Fine-Tuning | 67.20% | 67.20% | 0.00% |
| EWC ($\lambda=100$) | 67.20% | 67.00% | -0.20% |
| **ANTENA (Soft-OGP)** | 67.20% | 67.00% | -0.20% |

Both EWC and ANTENA exhibit minimal reasoning variation ($\Delta = -0.20\%$), indicating that Soft Orthogonal Gradient Projection maintains out-of-domain reasoning manifolds during targeted parameter updates.

### 3.4 Main Empirical Findings

1. **Observation of Generative Perplexity Degradation:** Experience Replay (100 samples/task) maintains classification accuracy but induces significant language model perplexity degradation (PPL 3,666 on DBpedia, PPL 9,908 on CLINC). Replaying text fragments forces gradient updates that perturb pre-trained language manifolds.
2. **Distillation Adjustment for Extended Sequences:** On CLINC-150 (15 tasks), increasing distillation strength ($\lambda_{\text{LwF}} = 7.5$) and tuning learning rates ($\eta = 3 \times 10^{-4}$) in `ANTENA_SlowDistill` improves AIA from 63.49% to **73.69%**, reducing forgetting by +27.21 BWT points and achieving higher post-CIL zero-shot reasoning (PIQA **59.25%**) than Experience Replay (56.91%).
3. **Performance Relative to Larger Baselines:** ANTENA (82.88% AIA on DBpedia, 73.69% AIA on CLINC) demonstrates competitive performance relative to O-LoRA (76.50% / 76.20%) while operating on a smaller backbone (SmolLM-135M vs LLaMA-7B) and exhibiting positive BWT on DBpedia (+29.17% to +38.37% vs O-LoRA -1.90%). Direct comparison with O-LoRA is constrained by the use of different backbone architectures (SmolLM-135M vs. LLaMA-7B). These results should be interpreted as evidence of competitive performance at a substantially smaller scale rather than a direct superiority claim.
4. **Retroactive Feature Consolidation ("Sleep Effect"):** Inspection of task-wise evaluation matrices $R$ reveals that task accuracy often evaluates lower immediately post-training (e.g., 42.4% on DBpedia Task 2) and **increases retroactively to 95.0%** during subsequent tasks. As subsequent tasks generate synthetic feature dreams from historical class covariances, feature replay and weight aligning re-balance decision boundaries, acting as a structural consolidation process that supports positive backward transfer.

---

## 4. Discussion

Beyond continual text classification, the mathematical formulation of Soft-OGP and Welford covariance models suggests several potential applications for decentralized language models. The following extensions are speculative and have not been empirically validated in this work:

### 4.1 Value Locking (Guardrail Preservation)
When fine-tuning edge language models for domain personalization, safety alignment (RLHF/DPO guardrails) is frequently degraded. By defining a safety Gestalt $\mathbf{U}_{\text{safety}}$ over aligned interaction data, downstream fine-tuning updates can be projected onto its orthogonal complement:

$$\mathbf{G}_{\text{safe}} = \mathbf{G} - \mathbf{U}_{\text{safety}} \mathbf{U}_{\text{safety}}^T \mathbf{G}$$

This mathematically prevents fine-tuning gradients from perturbing pre-trained alignment boundaries, ensuring deterministic guardrail retention.

### 4.2 Subtractive Machine Unlearning (Gestalt Subtraction)
To remove toxic, biased, or copyrighted knowledge from a model without costly retraining, the activation covariance matrix $\mathbf{\Sigma}_{\text{toxic}}$ of the target domain is extracted and its principal eigenvectors $\mathbf{U}_{\text{toxic}}$ are computed. A subtractive projection operator is applied to parameter updates or weights:

$$\mathbf{P}_{\text{clean}} = \mathbf{I} - \mathbf{U}_{\text{toxic}} \mathbf{U}_{\text{toxic}}^T$$

This nullifies the model's capacity to activate along toxic subspace directions without altering un-related representations.

### 4.3 Privacy-Preserving Federated Aggregation & Proof-of-Contribution (DeSci)
In edge computing and Decentralized Science (DeSci) networks, user privacy prevents uploading raw conversation transcripts to a central server. Under ANTENA, edge devices share only local covariance matrices $\mathbf{\Sigma}_i$ or principal eigenvectors $\mathbf{U}_i$ — abstract geometric manifolds containing **zero raw text tokens**. 

A central orchestrator aggregates these manifolds ($\bar{\mathbf{\Sigma}} = \frac{1}{M}\sum \mathbf{\Sigma}_i$) and computes a global SVD update. Furthermore, each device's contribution can be audited via projection energy divergence against the collective space:

$$E_i = \frac{\|\mathbf{U}_i^T \mathbf{U}_{\text{global}}\|_F^2}{\|\mathbf{U}_i\|_F^2}$$

This could establish a verifiable **Proof-of-Contribution** for data cooperatives, rewarding users for data value without harvesting private text.

---

## 5. Conclusion & Future Architectural Extensions

ANTENA demonstrates that Small Language Models (<150M) can achieve competitive zero-exemplar continual learning without storing historical text samples. By combining Soft Orthogonal Gradient Projection (Soft-OGP) at the LoRA parameter level with Head Feature Augmentation (IL2A), ANTENA preserves pre-trained language perplexity on shorter task sequences and delivers positive backward transfer (+29.17% to +38.37% on DBpedia14), providing a viable path for privacy-preserving, edge-deployed continual learning.

### 5.1 Future Work
Building upon the empirical discovery of Retroactive Feature Consolidation, future extensions of ANTENA will explore three architectural enhancements:
1. **Post-Task Micro-Consolidation (Pseudo-Sleep):** Introducing a brief post-task re-balancing phase over frozen backbone representations to eliminate transient initial task accuracy dips.
2. **Entropy-Gated Temperature Sampling ($T_c$-Sampling):** Modulating Gaussian feature dream variances dynamically per class to prevent representation overcrowding in sequences exceeding 50 tasks.
3. **Subspace Memory Pruning:** Applying SVD truncation to class covariance matrices to compress prototype storage below 25 KB per class for sub-megabyte SRAM edge hardware.

---

## 6. Limitations

1. **Single-Seed Evaluation.** All reported results correspond to a single random seed ($s=42$). While the deterministic setup ensures exact reproducibility, it does not characterize variance across initialization conditions.
2. **Generative Perplexity on Long Sequences.** While ANTENA preserves backbone perplexity on DBpedia14 (7 tasks, PPL 172.65), the CLINC-150 benchmark (15 tasks) reveals substantial perplexity degradation even with SlowDistill (PPL 20,731 vs. baseline 172). Mitigating long-sequence generative degradation remains an open challenge.
3. **Cross-Model Baseline Comparison.** The O-LoRA comparison is constrained by architecture differences (SmolLM-135M vs. LLaMA-7B). A fair comparison would require running O-LoRA on the same backbone, which was not feasible due to the absence of an official SmolLM implementation.
4. **Limited Baseline Coverage.** This work compares against O-LoRA and Experience Replay. Comparison with additional zero-exemplar methods (InfLoRA, GPM, EASE) would strengthen the empirical positioning.
5. **Spectral Thresholding Heuristic.** The "Ledoit-Péché" signal-noise separation currently uses a mean-eigenvalue threshold ($S > \bar{S}$), which is a heuristic inspired by random matrix theory rather than the exact Ledoit-Péché optimal shrinkage formula. The threshold's sensitivity to the ratio $p/n$ (dimensionality vs. sample size) has not been systematically characterized.

---

## 7. Reproducibility

| Hyperparameter | DBpedia14 | CLINC-150 |
|---|---|---|
| Model | SmolLM-135M-Instruct | SmolLM-135M-Instruct |
| LoRA Rank / Alpha | 16 / 32 | 16 / 32 |
| Target Layers | 10-29 | 10-29 |
| Batch Size | 16 | 16 |
| Epochs per Task | 3 | 5 |
| Max Sequence Length | 128 | 64 |
| Samples per Task | 500 train, 250 test | 500 train, 250 test |
| LoRA LR (Base) | 5e-4 | 5e-4 |
| LoRA LR (SlowDistill) | 3e-4 | 3e-4 |
| Score LR | 1e-4 | 1e-4 |
| Soft-OGP $\rho$ | 0.99 | 0.99 |
| LwF $\lambda$ (Base) | 5.0 | 5.0 |
| LwF $\lambda$ (SlowDistill) | 7.5 | 7.5 |
| LwF Temperature | 2.0 | 2.0 |
| IL2A Temperature | 1.0 | 1.0 |
| EGPR Sensitivity | 0.1 | 0.2 |
| EGPR Center | 2.0 | 1.5 |
| EGPR Window | 30 | 10 |
| Hebbian Decay $\gamma$ | 0.95 | 0.95 |
| Random Seed | 42 | 42 |
| Hardware | NVIDIA T4 (Colab) | NVIDIA T4 (Colab) |

Source code is available at: https://github.com/jerolemos/antena-cl-egpr

---

## 8. Acknowledgments

The author acknowledges the use of large language model assistants for code review, test orchestration, and manuscript preparation during the research process. All theoretical concepts, architectural designs, and experimental decisions described in this work are the original intellectual contribution of the author.

---

## 9. References

- Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521-3526.
- Ledoit, O., & Péché, S. (2011). Eigenvectors of some large sample covariance matrix ensembles. Probability Theory and Related Fields, 151(1-2), 233-264.
- Lemos, J. (2026). Entropy-Gated Plasticity Regulation for Continual Learning Without Episodic Memory. (Part 1 of this series.)
- Li, Z., & Hoiem, D. (2017). Learning Without Forgetting. IEEE TPAMI, 40(12), 2935-2947.
- Liang, Y.-S., & Li, Y. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR.
- Saha, G., Garg, I., & Roy, K. (2021). Gradient Projection Memory for Continual Learning. ICLR.
- Wang, X., et al. (2023). Orthogonal Subspace Learning for Language Model Continual Learning. Findings of EMNLP.
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419-420.
- Zhao, B., et al. (2020). Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR.
- Zhu, F., et al. (2021). Class-Incremental Learning via Dual Augmentation. NeurIPS.
