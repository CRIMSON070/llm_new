# FinSage Traceability & Empirical Verification Matrix

**Course Project:** FinSage Reproduction & Enhancement  
**Group Number:** Group [NUMBER]  
**Document:** Full Audit Trail Mapping Reported Figures to Workspace Files  

This document certifies that every single numerical figure, hyperparameter, and empirical claim presented in the Project Report (`Group_NUMBER_Project_Report.pdf`) and Presentation (`Group_NUMBER_Presentation.pptx`) is traceable to verifiable physical files in this repository.

---

## 1. Master Traceability Table

| Claim / Metric Dimension | Exact Numerical Value | Physical File Path in Repository | Notes & Verification Scope |
| :--- | :--- | :--- | :--- |
| **Stage 1 Faithful DRR MRR** | `0.5379` | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 7, N=150 questions, zero boosts |
| **Stage 1 Faithful DRR Recall@1** | `43.33%` (65/150) | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 9 |
| **Stage 1 Faithful DRR Recall@5** | `65.33%` (98/150) | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 11 |
| **Stage 1 Faithful DRR Recall@10** | `70.67%` (106/150) | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 12 |
| **Stage 1 Base Gemma MRR** | `0.3503` | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 23, Base cross-encoder baseline |
| **Stage 1 Candidate Pool Recall** | `75.33%` (113/150) | [`outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/scorecard_stage1_full.json) | Line 5, K_pool=40 RRF fusion |
| **DPO Leakage: Held-Out MRR** | `0.3551` (Recall@5: 37.5%) | [`outputs/reproduction_stage1_faithful/stage1_reproduction_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/stage1_reproduction_report.md) | Line 27, N=8 strictly held-out questions |
| **DPO Leakage: Validation MRR** | `0.5168` (Recall@5: 58.82%)| [`outputs/reproduction_stage1_faithful/stage1_reproduction_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/stage1_reproduction_report.md) | Line 28, N=17 validation questions |
| **DPO Leakage: Training MRR** | `0.5524` (Recall@5: 68.00%)| [`outputs/reproduction_stage1_faithful/stage1_reproduction_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/reproduction_stage1_faithful/stage1_reproduction_report.md) | Line 29, N=125 in-distribution training questions |
| **Stage 2 Macro MRR ($N=150$)** | `0.2926` | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 5, Full benchmark macro ranking |
| **Stage 2 Macro Recall@1** | `18.00%` (27/150) | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 6 |
| **Stage 2 Macro Recall@5** | `45.33%` (68/150) | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 8 |
| **Stage 2 Macro Recall@10** | `52.00%` (78/150) | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 9 |
| **Stage 2 Candidate Pool Recall** | `86.00%` (129/150) | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 10, Lifted via table header propagation |
| **Table Header Propagation** | `2,902` continuation chunks| [`scripts/enrich_phase2_chunks.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/enrich_phase2_chunks.py) | Verified chunk count in bm25_index_phase2 |
| **Stage 2 Batch Evaluation Time** | `43.24 s` (0.29s/query) | [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) | Line 12, Fast batch scaling across 150 Qs |
| **Diagnostic Semantic Accuracy** | `83.33%` (5/6) | [`outputs/phase2/evaluation/phase2_diagnostic_results.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_diagnostic_results.json) | Line 4, Evaluated with Python math engine |
| **Q1 3M CapEx Rank Jump** | Rank #21 $\rightarrow$ Rank #1 | [`outputs/phase2/evaluation/phase2_diagnostic_results.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_diagnostic_results.json) | Line 13, Native #1 placement |
| **Phase 3.2 Controlled FB Recall@10** | `39.33%` $\rightarrow$ `47.33%` (+8.0 pp) | [`outputs/experiments/phase3_2_controlled_benchmark/financebench_comparison.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/phase3_2_controlled_benchmark/financebench_comparison.json) | Line 15, K=40 pre-reranker |
| **Phase 3.2 Controlled FB Pool Recall** | `60.00%` $\rightarrow$ `66.00%` (+6.0 pp) | [`outputs/experiments/phase3_2_controlled_benchmark/financebench_comparison.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/phase3_2_controlled_benchmark/financebench_comparison.json) | Line 14, -15% retrieval misses |
| **Phase 4 Coreference FB Recall@5** | `53.33%` $\rightarrow$ `59.33%` (+6.0 pp) | [`outputs/phase4_coreference_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase4_coreference_benchmark_report.md) | Line 47, +9 net questions recovered |
| **Phase 4 Stage 3 Coreference R@10**| `16.00%` $\rightarrow$ `20.00%` (+4.0 pp) | [`outputs/phase4_coreference_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase4_coreference_benchmark_report.md) | Line 28, Held-out brand recovery |
| **Mounjaro Target Query Recovery** | NO_HIT (Rank 999) $\rightarrow$ Rank 6 | [`outputs/phase4_coreference_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase4_coreference_benchmark_report.md) | Line 30, Query `stage3_heldout_0072` |
| **Stage 3 Generalization Pool Recall**| `30.00%` (30/100) | [`outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md) | Line 35, 40 unseen Form 10-K filings |
| **Stage 3 Generalization MRR** | `0.1050` | [`outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md) | Line 40, p=1.000 vs frozen Exp F |
| **Failure Subset Recovery (K=120)** | `12 / 12` displaced chunks (100%) | [`outputs/FINAL_PROJECT_REPORT.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/FINAL_PROJECT_REPORT.md#L149-L169) | Line 157, 37 Stage 1 failure subset |

---
*Generated for Group [NUMBER]. Certified 100% empirical traceability.*
