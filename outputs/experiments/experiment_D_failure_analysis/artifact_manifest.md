# FinSage Experiment D — Artifact Manifest

**Inspection Directory:** `outputs/experiments/experiment_D_failure_analysis/`  
**Governing Standard:** Read-Only Forensic Analysis & Evidence Traceability  
**Workspace Root:** `C:\Users\user\Downloads\finsage-main`  
**Date:** November 02, 2026  

---

## 1. Overview of Inspected Artifacts

This manifest documents every file, dataset, configuration, model checkpoint, and result artifact inspected during the **Experiment D Retrieval Failure Analysis**. No existing files were modified, overwritten, or deleted.

---

## 2. Manifest of Inspected Artifacts

| Artifact Category | File Path | Format / Size | Role & Evidence Provided |
|---|---|---|---|
| **Experiment C Retrieval Checkpoint** | [`outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl) | JSONL (151 lines, 67.8 KB) | Primary ground truth for Experiment C retrieval performance. Contains per-question `financebench_id`, `doc_name`, `target_pages`, `best_rank`, `best_rank_in_pool`, `candidate_pool_hit`, reciprocal rank `rr`, and `top3_retrieved` chunks. Used to independently calculate MRR (0.3513), R@1 (25.33%), R@3 (42.00%), R@5 (48.00%), R@10 (57.33%), pool recall (68.67%), and identify all 64 failed questions. |
| **Experiment C Scorecard** | [`outputs/experiments/experiment_C_company_intent/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/evaluation/phase2_full_scorecard.json) | JSON (324 lines, 13.2 KB) | Official aggregated scorecard for Experiment C, verifying macro metrics ($N=150$) and diagnostic suite results (Q1–Q6). |
| **Experiment C Official Report** | [`outputs/experiments/experiment_C_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_report.md) | Markdown (248 lines, 19.0 KB) | Comprehensive report on Experiment C execution, hypothesis verification, and comparison against Experiment AB and Baseline. |
| **Experiment C Source Code (Backup)** | [`outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py) | Python (316 lines, 13.6 KB) | Exact reranker code executed in Experiment C, defining the 32-company regex dictionary, `detect_query_company()`, `detect_query_statement_intent()`, statement boost (+4.5), negative penalty (-3.0), company boost (+1.5), period boost (+1.5), table boost (+0.5), and 512-character input truncation. |
| **Experiment C Benchmark Runner (Backup)** | [`outputs/experiments/experiment_C_company_intent/source_backup/run_phase2_benchmark.py`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/source_backup/run_phase2_benchmark.py) | Python (598 lines, 28.7 KB) | Exact benchmark pipeline runner used in Experiment C, defining $K_{\text{pool}}=50$, BM25 $k=50$, Dense FAISS $k=50$, RRF $k=60$, target page 1-indexing (`e["evidence_page_num"] + 1`), and retrieval scoring loop. |
| **FinanceBench Benchmark Questions** | [`data/financebench/questions.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/questions.jsonl) | JSONL (151 lines, 962.8 KB) | Ground truth benchmark data containing 150 questions, target SEC filings (`doc_name`), gold evidence text snippets, 0-indexed evidence page numbers, justification, and question types. |
| **Enriched Chunk Index** | [`data/financebench/processed/phase2_enriched_chunks.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/processed/phase2_enriched_chunks.jsonl) | JSONL (54,564 lines, ~120 MB) | Full corpus of PyMuPDF-extracted and enriched chunks. Inspected to verify chunk availability, text integrity, table flags, and token counts for all 74 unique target pages across the 64 failures. |
| **Original SEC PDF Filings** | `data/financebench/pdfs/*.pdf` | 84 PDF documents (~1.2 GB) | Original source 10-K, 10-Q, and 8-K filings. PyMuPDF was used to verify physical file existence and page bounds for all 39 distinct PDFs associated with the 64 failed questions. |
| **Experiment AB Retrieval Checkpoint** | [`outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl) | JSONL (151 lines, 67.8 KB) | Prior best baseline checkpoint ($K_{\text{pool}}=50$, routing fix, 6 hardcoded companies) used for longitudinal regression and movement tracking. |
| **Experiment AB Diagnostic Report** | [`outputs/experiments/failure_analysis_AB/failure_analysis_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/failure_analysis_AB/failure_analysis_report.md) | Markdown (172 lines, 14.9 KB) | Prior failure diagnosis on Experiment AB's 66 failures, establishing baseline methodology and highlighting company contamination. |
| **Phase 1 DRR Adapter Weights** | [`outputs/phase1_drr_adapter/adapter_model.safetensors`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_model.safetensors) | Safetensors (29.5 MB) | Trained LoRA adapter weights for `BAAI/bge-reranker-v2-gemma` using DPO loss. Inspected to assess reranker comparison feasibility. |
| **Phase 1 DRR Training Metadata** | [`outputs/phase1_drr_adapter/drr_training_metadata.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json) | JSON (51 lines, 1.5 KB) | Training specifications, hyper-parameters (rank 32, alpha 64, bfloat16, 4-bit NF4 QLoRA), peak VRAM (3.64 GB), and validation preference accuracy (95.29%). |
| **Phase 1 DRR Evaluation Script** | [`scripts/evaluate_phase1_drr.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py) | Python (452 lines, 22.4 KB) | Reranker evaluation harness used in Phase 1 to compare base BGE, base Gemma, and DRR-adapted Gemma on held-out diagnostic test cases. |
| **Ablation Candidate Data (Diagnostic Suite)** | [`outputs/phase9c/retrieval_ablation_data.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase9c/retrieval_ablation_data.json) | JSON (~85 KB) | Saved candidate pools for the 6 held-out diagnostic queries (Q1–Q6). |

---

## 3. Telemetry Availability and Limitations

In strict compliance with empirical auditing principles, the availability of intermediate pipeline traces was checked:

| Pipeline Stage / Telemetry | Persisted Status | Availability Details |
|---|:---:|---|
| **Per-Path Retrieval Hit Ranks (BM25, Dense FAISS, Metadata)** | ❌ **Not Persisted** | Individual path hit ranks for Path 1 (BM25), Path 2 (Dense FAISS), and Path 3 (Metadata) were not logged into `phase2_retrieval_checkpoint.jsonl` during the batch run. Only the fused candidate pool hit status (`candidate_pool_hit` and `best_rank_in_pool`) was saved. |
| **Complete 50-Candidate Pool Chunk IDs (Full Benchmark N=150)** | ❌ **Not Persisted** | Only the Top-3 reranked chunk IDs (`top3_retrieved`) were written to the checkpoint per question. The remaining 47 candidate chunk IDs were discarded after reranking. |
| **Reranker Raw & Boost Scores for Full Benchmark** | ❌ **Not Persisted** | Reranker raw cross-encoder scores, statement boosts, company boosts, period boosts, and penalties were printed during interactive execution but not written to the JSONL checkpoint file. |
| **Diagnostic Suite (Q1–Q6) Detailed Traces** | ✅ **Persisted** | Complete candidates, bundled contexts, generated answers, and operation-operand breakdowns are available in `phase2_full_scorecard.json` and `retrieval_ablation_data.json`. |

> [!IMPORTANT]
> Because per-path retrieval ranks were not saved in the batch checkpoint, our failure classifications attribute candidate pool misses to **Candidate-Generation Failure** broadly (combining sparse, dense, and reciprocal rank fusion), rather than claiming an unsupported attribution to BM25 or Dense FAISS in isolation.
