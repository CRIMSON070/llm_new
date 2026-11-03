# Stage 1 — RRF Candidate Pool Cutoff Ablation — Diagnostic Report

**Generated:** 2026-11-03 10:01:41 UTC
**Ablation Dataset:** Exactly the 37 questions that failed Candidate Pool Recall@40 in the Stage 1 reproduction benchmark.
**Hardware & Environment:** Device = `cuda`, Peak VRAM = `2257.62 MB`, Total Execution Time = `508.87 s`.

> [!IMPORTANT]
> **Scope Constraint:** This ablation is strictly restricted to the 37 Stage 1 candidate-pool failure questions. Metrics reported herein reflect recovery on the failure subset and **must not be claimed as overall benchmark performance**.

## 1. Executive Summary & Core Findings

1. **Candidate Pool Gold Recovery:**
   - **Baseline (K_pool=40):** Pool Recall = **0.0%** (0/37). Recovered **0/12** (0.0%) of the previously displaced gold chunks.
   - **Test A (K_pool=60):** Pool Recall = **18.92%** (7/37). Recovered **7/12** (58.33%) of the previously displaced gold chunks.
   - **Test B (K_pool=80):** Pool Recall = **21.62%** (8/37). Recovered **8/12** (66.67%) of the previously displaced gold chunks.
   - **Test C (K_pool=120):** Pool Recall = **32.43%** (12/37). Recovered **12/12** (100.0%) of the previously displaced gold chunks.
2. **Cross-Encoder Reranker Surfacing Ability:**
   - Expanding $K_{pool}$ successfully brought gold chunks into the candidate pool. However, the cross-encoder's ability to promote them into top positions (Top 1–5) depends on whether the passage text provides dense contrastive signals against distractor candidates.
3. **Irrelevant Candidate Volume & Noise Trade-off:**
   - Increasing $K_{pool}$ from 40 to 120 introduced an average of **75.1 additional irrelevant candidates per query** (+2,778 total distractor passages across the 37 questions).

## 2. Four-Arm Comparative Scorecard

| Experimental Arm | Pool Size ($K$) | Pool Recall (37 Qs) | Displaced Recovered (12 Qs) | Irrelevant Cands / Q | Reranked MRR | Reranked R@1 | Reranked R@3 | Reranked R@5 | Reranked R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (K_pool=40)** | 40 | **0.0%** (0/37) | **0/12** (0.0%) | 40.0 | **0.0000** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Test A (K_pool=60)** | 60 | **18.92%** (7/37) | **7/12** (58.33%) | 59.76 | **0.0691** | 2.7% | 10.81% | 10.81% | 18.92% |
| **Test B (K_pool=80)** | 80 | **21.62%** (8/37) | **8/12** (66.67%) | 79.68 | **0.0678** | 2.7% | 10.81% | 10.81% | 16.22% |
| **Test C (K_pool=120)** | 120 | **32.43%** (12/37) | **12/12** (100.0%) | 114.59 | **0.0850** | 2.7% | 10.81% | 16.22% | 21.62% |

## 3. Trajectory of the 12 Displaced Questions Across Ablation Arms

| Question ID | Target Doc | Pages | Initial RRF Rank | K=40 Rerank | K=60 Rerank | K=80 Rerank | K=120 Rerank | Status & Surfacing Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `financebench_id_10420` | `AES_2022_10K` | 130;132 | 77 | Out of Pool | Out of Pool | Rank 12 | Rank 15 | Recovered in pool, reranked at 15 |
| `financebench_id_01079` | `AMCOR_2023_10K` | 64 | 96 | Out of Pool | Out of Pool | Out of Pool | Rank 2 | Promoted to Top 5 by DRR |
| `financebench_id_01928` | `AMCOR_2023Q4_EARNINGS` | 12 | 51 | Out of Pool | Rank 6 | Rank 6 | Rank 6 | Promoted to Top 10 by DRR |
| `financebench_id_00563` | `AMD_2022_10K` | 48 | 41 | Out of Pool | Rank 10 | Rank 11 | Rank 16 | Recovered in pool, reranked at 16 |
| `financebench_id_10130` | `CORNING_2020_10K` | 70;72 | 54 | Out of Pool | Rank 3 | Rank 3 | Rank 3 | Promoted to Top 5 by DRR |
| `financebench_id_05915` | `CVSHEALTH_2018_10K` | 302;304 | 44 | Out of Pool | Rank 2 | Rank 3 | Rank 4 | Promoted to Top 5 by DRR |
| `financebench_id_00956` | `JOHNSON_JOHNSON_2022_10K` | 28 | 56 | Out of Pool | Rank 1 | Rank 1 | Rank 5 | Promoted to Top 5 by DRR |
| `financebench_id_01484` | `JOHNSON_JOHNSON_2022Q4_EARNINGS` | 2 | 86 | Out of Pool | Out of Pool | Out of Pool | Rank 19 | Recovered in pool, reranked at 19 |
| `financebench_id_00299` | `JPMORGAN_2021Q1_10Q` | 19 | 100 | Out of Pool | Out of Pool | Out of Pool | Rank 1 | Promoted to Top 5 by DRR |
| `financebench_id_00206` | `JPMORGAN_2022_10K` | 3 | 113 | Out of Pool | Out of Pool | Out of Pool | Rank 88 | Recovered in pool, reranked at 88 |
| `financebench_id_01328` | `PEPSICO_2022_10K` | 78 | 56 | Out of Pool | Rank 3 | Rank 3 | Rank 3 | Promoted to Top 5 by DRR |
| `financebench_id_00605` | `ULTABEAUTY_2023Q4_EARNINGS` | 3 | 55 | Out of Pool | Rank 8 | Rank 6 | Rank 6 | Promoted to Top 10 by DRR |

## 4. Resource Utilization & Operational Overhead

- **Initial Memory (Model Loaded):** 2100.85 MB
- **Peak GPU Memory Allocated:** 2257.62 MB
- **Total Candidate Passages Scored:** 4258 pairs
- **Total Reranking Inference Runtime:** 472.77 seconds
- **Full Pipeline Runtime:** 508.87 seconds

## 5. Architectural Recommendations

1. **Pool Expansion Trade-off:** Expanding $K_{pool}$ from 40 to 80 or 120 successfully recovers genuine gold evidence that was displaced by multi-path consensus noise. However, simply widening the pool also scales the cross-encoder inference workload proportionally and exposes the reranker to more distractor candidates.
2. **Next Step:** Do not promote $K_{pool}=120$ unconditionally to the full benchmark without considering whether upstream retrieval improvements (e.g., stripping evaluation prompt boilerplate or primary statement catalog injection) can prevent displacement at lower $K$.
