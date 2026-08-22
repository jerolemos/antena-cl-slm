# ANTENA: Zero-Exemplar Class-Incremental Learning in Small Language Models

Official open-source implementation and benchmark suite for **ANTENA** (*Soft Orthogonal Gradient Projection and Feature Alignment for Small Language Models*).

* **Part 2 of the Memory-Free Continual Learning Series:**
  * **[Part 1: EGPR Foundation](https://github.com/jerolemos/antena-cl-egpr)** (Vision benchmarks: Split-MNIST, Permuted-MNIST, Rotated-MNIST, Split-FashionMNIST, Split-CIFAR10, Split-CIFAR100)
  * **[Part 2: SLM](https://github.com/jerolemos/antena-cl-slm)** (Small Language Models & Soft-OGP)
  * **[Part 3: 1-Bit](https://github.com/jerolemos/antena-cl-1bit)** (1-Bit Discrete Subspace Routing)

---

## 📌 Overview

Small Language Models (SLMs, $<500$M parameters) deployed on edge devices face severe catastrophic forgetting during sequential Class-Incremental Learning (CIL). While traditional Experience Replay (ER) mitigates classification forgetting by storing past text samples, a novel failure mode is uncovered: **Generative Perplexity Collapse**, where replaying text exemplars fundamentally degrades the pre-trained language backbone (WikiText-2 PPL explodes from ~172 to >3,600).

**ANTENA** resolves this trade-off in a **zero-exemplar setting** (storing 0 text samples) by decomposing protection into two physical controls:
1. **Backbone Protection (Soft-OGP):** Projects LoRA adapter gradients onto the orthogonal complement of activation signal subspaces, isolated via non-linear **Ledoit-Péché (2011)** random matrix shrinkage.
2. **Classification Head Alignment (IL2A):** Maintains running Gaussian distributions over final hidden states using batched Welford updates, generating synthetic feature vectors to shield decision boundaries from recency bias without backbone forward/backward passes.
3. **Entropy-Gated Regulation (EGPR):** Modulates learning rates dynamically based on entropic stabilization ($dH/dt$).

```
[ Input Tokens ] ---> [ Transformer Backbone (Layers 1-30) ] ---> [ Hidden State h_c ] ---> [ Classification Head ]
                             | (Soft-OGP on LoRA Gradients)                | (IL2A Feature Replay)
                             v                                             v
                      [ Null-Space Projection ]                    [ Synthetic Gaussian Features ]
```

---

## 🏆 Key Benchmark Results

All evaluations were conducted on pre-trained `SmolLM-135M` on an NVIDIA T4 GPU.

### 1. DBpedia14 (7 Tasks, 14 Classes CIL)

| Model / Configuration | AIA $\uparrow$ | BWT $\uparrow$ | PIQA (0-shot) $\uparrow$ | PPL (WikiText-2) $\downarrow$ | Buffer |
|---|---|---|---|---|---|
| **ANTENA_Base** ($\lambda_{\text{LwF}}=5.0$) | **82.88%** | +29.17% | 64.53% | 172.65 | **0 (Zero-Exemplar)** |
| **ANTENA_SlowDistill** ($\lambda_{\text{LwF}}=7.5$) | 80.67% | **+38.37%** | **64.58%** | **101.79** | **0 (Zero-Exemplar)** |
| Experience Replay (100/task) | 84.46% | +55.07% | 61.64% | 3,666.93 | 100 samples/task |
| O-LoRA (Wang et al., EMNLP 23)* | 76.50% | -1.90% | N/A | N/A | 0 (Zero-Exemplar) |

*\*O-LoRA baseline reported on LLaMA-7B / T5-Large models (50× larger).*

### 2. CLINC-150 (15 Tasks, 150 Classes CIL)

| Model / Configuration | AIA $\uparrow$ | BWT $\uparrow$ | PIQA (0-shot) $\uparrow$ | PPL (WikiText-2) $\downarrow$ | Buffer |
|---|---|---|---|---|---|
| **ANTENA_Base** ($\lambda_{\text{LwF}}=5.0$) | 63.49% | -58.24% | 56.37% | 813,633 | **0 (Zero-Exemplar)** |
| **ANTENA_SlowDistill** ($\lambda_{\text{LwF}}=7.5$) | **73.69%** | **-31.03%** | **59.25%** | **20,731** | **0 (Zero-Exemplar)** |
| Experience Replay (100/task) | 84.24% | +10.48% | 56.91% | 9,908 | 100 samples/task |
| O-LoRA (Wang et al., EMNLP 23)* | 76.20% | -6.80% | N/A | N/A | 0 (Zero-Exemplar) |

---

## 🚀 Quick Start & Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Reproducing Benchmarks

#### Option A: Direct Python Scripts
```bash
# Benchmark 1: DBpedia14 (7 Tasks)
python SCRIPTS/benchmark_dbpedia14.py

# Benchmark 2: CLINC-150 (15 Tasks)
python SCRIPTS/benchmark_clinc150.py

# Benchmark 3: General Ability Delta (PIQA Zero-Shot Reasoning)
python SCRIPTS/benchmark_gad.py
```

#### Option B: Google Colab Notebooks
Pre-configured Jupyter notebooks with automatic Drive checkpointing are available in `NOTEBOOKS/`:
- `NOTEBOOKS/01_DBpedia14_Benchmark.ipynb`
- `NOTEBOOKS/02_CLINC150_Benchmark.ipynb`
- `NOTEBOOKS/03_GAD_Benchmark.ipynb`

---

## 📂 Repository Structure

```
.
├── CORE/
│   ├── antena_gestalt_core.py       # EGPR entropy thermostat & Hebbian covariance
│   ├── antena_paso2_lora_ogp.py     # LoRA injection & Soft-OGP Ledoit-Péché projection
│   └── antena_latent_il2a.py     # Feature covariance & Welford IL2A augmentation
├── SCRIPTS/
│   ├── benchmark_dbpedia14.py       # Standalone DBpedia14 CIL runner
│   ├── benchmark_clinc150.py        # Standalone CLINC-150 CIL runner
│   └── benchmark_gad.py            # Standalone GAD PIQA runner
├── NOTEBOOKS/
│   ├── 01_DBpedia14_Benchmark.ipynb # Colab notebook for DBpedia14
│   ├── 02_CLINC150_Benchmark.ipynb  # Colab notebook for CLINC-150
│   └── 03_GAD_Benchmark.ipynb       # Colab notebook for GAD
├── PAPER/
│   └── SLM_PAPER.md             # Full scientific manuscript
├── RESULTS_ARCHIVE/                 # Final empirical logs and R evaluation matrices
├── LICENSE                          # Dual license notice (AGPLv3 / Commercial)
├── requirements.txt                 # Dependencies
└── README.md
```

---

## ⚖️ License & Dual-Licensing

This repository is dual-licensed:

1. **Open-Source & Academic License:** Licensed under the **GNU Affero General Public License v3.0 (GNU AGPLv3)**. Free for research, academic use, and open-source projects. Any derivative network service or software must make its source code publicly available under AGPLv3.
2. **Commercial & Enterprise License:** For commercial deployment, proprietary hardware integration, closed-source enterprise software, or silicon IP integration, a commercial license is required.

For commercial licensing inquiries, please contact:
- **Author:** Lemos J. (jerolemos@proton.me)
- **Repository:** https://github.com/jerolemos/antena-cl-slm
