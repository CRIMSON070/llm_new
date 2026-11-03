# Stage 1 faithful-methodology reproduction with local model adaptations

**Generated:** 2026-11-03 05:55:42 UTC

**Dataset:** 150 questions from `data/financebench/questions.jsonl` (Published paper reports 149 questions; exact excluded question unknown).

**Integrity Safeguards:** Zero heuristic boosts applied (+0.0). Fixed identical candidate pools across Base vs DPO reranker.

## Three-Way Benchmark Comparison (All 150 Questions)

| Evaluation Stage | Metric | Initial Multi-Path RRF | Arm 1: Base Gemma Reranker | Arm 2: DPO Adapted Gemma | RRF -> DPO Delta | Base -> DPO Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Retrieval** | Candidate Pool Recall@40 | 75.33% | 75.33% | 75.33% | 0.00% (Identical Pool) | 0.00% (Identical Pool) |
| **Raw Retrieval** | MRR | 0.3419 | 0.3503 | 0.5379 | +0.1960 | +0.1876 |
| **Raw Retrieval** | nDCG@10 | 0.3800 | 0.4005 | 0.5767 | +0.1967 | +0.1762 |
| **Raw Retrieval** | Recall@1 | 24.67% | 24.0% | 43.33% | +18.66% | +19.33% |
| **Raw Retrieval** | Recall@3 | 36.0% | 38.67% | 62.0% | +26.00% | +23.33% |
| **Raw Retrieval** | Recall@5 | 40.0% | 45.33% | 65.33% | +25.33% | +20.00% |
| **Raw Retrieval** | Recall@10 | 54.67% | 60.0% | 70.67% | +16.00% | +10.67% |
| **Bundled Coverage** | Evidence Coverage Rate | 37.33% | 40.0% | 63.33% | +26.00% | +23.33% |
| **Bundled Coverage** | Avg Chunks per Query | 9.72 | 9.78 | 9.73 | +0.01 | -0.05 |

## DPO Leakage Stratification (Initial RRF vs Base Gemma vs DPO Gemma)

| Sub-Split | Questions | RRF MRR | Base MRR | DPO MRR | RRF R@5 | Base R@5 | DPO R@5 | RRF Cov | Base Cov | DPO Cov |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Strictly Held-Out (Zero Exposure) | 8 | 0.2972 | 0.3115 | 0.3551 | 25.0% | 50.0% | 37.5% | 25.0% | 50.0% | 37.5% |
| DPO Validation Split | 17 | 0.3425 | 0.3902 | 0.5168 | 41.18% | 41.18% | 58.82% | 35.29% | 41.18% | 64.71% |
| DPO Training Split | 125 | 0.3447 | 0.3474 | 0.5524 | 40.8% | 45.6% | 68.0% | 38.4% | 39.2% | 64.8% |

## Methodological Interpretation of DPO Splits & Generalization

- **125 Questions (DPO Training Split):** These questions were mined as positive/negative triplets during the training of the domain reranker adapter (`outputs/phase1_drr_adapter/`). Performance on this subset reflects in-distribution contrastive fitting and memorization capacity rather than generalization.
- **17 Questions (DPO Validation Split):** Used for checkpoint selection and hyperparameter validation during DPO training.
- **8 Questions (Strictly Held-Out):** Questions `financebench_id_03029`, `financebench_id_04672`, `financebench_id_00746`, `financebench_id_00521`, `financebench_id_00601`, `financebench_id_00603`, `financebench_id_00605`, and `financebench_id_00606` had zero exposure during DPO training or validation. They represent the only strictly unseen subset for assessing DPO generalization in this evaluation set.
- **Critical Methodological Note:** The 142 training/validation questions must not be treated as an independent DPO generalization evaluation. Furthermore, results on the eight held-out questions must be interpreted cautiously due to the small sample size ($N=8$) and high statistical variance. They should not be interpreted as definitive proof of broad out-of-distribution generalization.
