# Thermodynamic Veto and Latent Representation: Mitigating Catastrophic Forgetting in Edge Language Models

**Author:** Lemos J.

**Part of a series on memory-free continual learning:**
* [Part 1: EGPR Foundation](../01_EGPR_FOUNDATION)
* [Part 2: SLM & Soft-OGP](../antena-cl-slm)
* [Part 3: 1-Bit QUBO Routing](../03_1BIT_QUBO)


## Abstract
The primary limitation for the deployment of language models on peripheral hardware (Edge AI) is associated with the degradation of long-term memory during continual learning. Conventional architectures, oriented toward extended cache modules, increase the computational load, whereas standard fine-tuning often results in catastrophic forgetting of the base latent representation. This document evaluates ANTENA, a structural regulation scheme designed to operate on Small Language Models (SLMs) in resource-constrained environments. The model dispenses with secondary attention networks through the application of two control mechanisms: a thermodynamic regulator based on entropy gradients ($dH/dt$) to condition empirical plasticity, and an orthogonal gradient projection method (Soft-OGP) grounded in Ledoit-Péché shrinkage to isolate orthogonal subspaces. Preliminary evaluations indicate stable retention margins compared to baseline LoRA fine-tuning, which exhibits asymptotic catastrophic forgetting during early phases.

## Files Included
*   `SLM_PAPER.md`: The preprint manuscript detailing the application of Thermodynamic Veto and Soft-OGP in Small Language Models, alongside its implications for AI Safety ("Value Locking").
*   `src/`: Directory containing the core Python implementation (`antena_gestalt_core.py` and `antena_gestalt_optimization.py`).
*   `LICENSE`: Dual-license file (AGPLv3 / Commercial).

## License & Usage
© 2026 Jero Lemos. All rights reserved.

This repository is dual-licensed:
1. **Open Source:** Available under the GNU Affero General Public License v3.0 (AGPLv3) strictly for open-source and non-commercial use.
2. **Commercial:** A proprietary commercial license is available for enterprise integration without source-code disclosure requirements. Contact Jero Lemos (jerolemos@proton.me) for details.

## Citation
If you use this work in your research, please cite it as:
```bibtex
@misc{lemos2026slm,
  title={Thermodynamic Veto and Latent Representation: Mitigating Catastrophic Forgetting in Edge Language Models},
  author={Lemos, J.},
  year={2026},
  howpublished={Preprint}
}
```
