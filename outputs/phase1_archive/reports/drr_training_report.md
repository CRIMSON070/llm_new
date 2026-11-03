# Phase 1: Domain-Specialized Document Re-Ranker (DRR) Training Report

## 1. Paper Specifications and Configuration

| Hyperparameter / Setting | Paper Specification | Implementation | Rationale |
|---|---|---|---|
| Base Checkpoint | `BAAI/bge-reranker-v2-gemma` | `BAAI/bge-reranker-v2-gemma` | Paper Section 3.3, 4.2.2 |
| LoRA Rank ($r$) | 32 | 32 | Paper Appendix B.3 |
| LoRA Alpha ($\alpha$) | 64 | 64 | Paper Appendix B.3 |
| Learning Rate | 1e-4 | 1e-4 | Paper Appendix B.3 |
| Per-Device Batch Size | 1 | 1 | Paper Appendix B.3 |
| Gradient Accumulation | 2 | 2 | Paper Appendix B.3 |
| Epochs | 10 | 10 | Paper Appendix B.3 |
| Quantization | Full Precision (Multi-GPU) | 4-bit NF4 QLoRA | Hardware Adaptation (8 GB VRAM budget) |
| Random Seed | Unspecified | 42 | Implementation Choice for Determinism |
| DPO Beta ($\beta$) | Unspecified | 0.1 | Standard DPO default |
| Max Sequence Length | Unspecified | 512 | FlagEmbedding Cross-Encoder default |
| Negatives / Query Step | Unspecified | 5 | Mined from 15 hard negatives per sample |
| Optimizer | Unspecified | AdamW (weight decay 0.01) | Standard LLM fine-tuning default |
| LR Scheduler | Unspecified | Cosine (5% warmup) | Standard LLM fine-tuning default |

## 2. Hardware and Environment

- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8.55 GB total)
- **Peak VRAM**: 3.64 GB (Headroom: 4.91 GB)
- **Total Training Time**: 90.7 minutes
- **PyTorch**: 2.14.1+cu130
- **Transformers**: 5.19.0
- **PEFT**: 0.21.2

## 3. Epoch-by-Epoch Convergence

| Epoch | Train Loss | Train Pref Acc | Val Loss | Val Pref Acc | Peak VRAM | Epoch Time |
|---|---|---|---|---|---|---|
| 1 | 0.5291 | 80.0% | 0.2675 | 90.6% | 3.64 GB | 552.1s |
| 2 | 0.2660 | 92.2% | 0.2165 | 92.9% | 3.64 GB | 526.2s |
| 3 | 0.0613 | 98.6% | 0.1065 | 95.3% | 3.64 GB | 526.2s |
| 4 | 0.0103 | 99.8% | 0.0997 | 95.3% | 3.64 GB | 528.6s |
| 5 | 0.0023 | 100.0% | 0.0930 | 95.3% | 3.64 GB | 526.8s |
| 6 | 0.0011 | 100.0% | 0.0943 | 95.3% | 3.64 GB | 538.3s |
| 7 | 0.0008 | 100.0% | 0.0946 | 95.3% | 3.64 GB | 539.6s |
| 8 | 0.0006 | 100.0% | 0.0934 | 95.3% | 3.64 GB | 563.4s |
| 9 | 0.0006 | 100.0% | 0.0948 | 95.3% | 3.64 GB | 567.6s |
| 10 | 0.0006 | 100.0% | 0.0954 | 95.3% | 3.64 GB | 568.0s |

**Best Checkpoint**: Val Loss = 0.093 saved to `outputs/phase1_drr_adapter`.
