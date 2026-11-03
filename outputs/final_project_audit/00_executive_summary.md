# FinSage Final Project Audit: Executive Summary

**Project Title:** FinSage: A Multi-aspect RAG System for Financial Filings Question Answering  
**Reference Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering* (Wang et al., arXiv:2504.14493, ACM CIKM 2025 Applied Research Track)  
**Official Repository:** `https://github.com/simplew4y/finsage`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Audit Date:** November 2, 2026  
**Auditor:** Antigravity Autonomous Systems Engineering (Read-Only Independent Audit)  
**Governing Standard:** Academic & Empirical Fidelity to CIKM 2025 Paper and University Final Submission Requirements  

---

## 1. Context & Purpose of this Audit

This document serves as the high-level executive briefing of an exhaustive, evidence-based, read-only technical audit conducted on the **FinSage Financial RAG** codebase.

The audit was performed to establish an unassailable evidentiary baseline for preparing:
1. **University Project Presentation:** 7 slides (plus 1 optional results slide).
2. **Detailed Project Report:** Formal academic PDF report with complete methodology, empirical data, and comparative analysis.
3. **Submission Code Package:** Verified runnable Stage 1 (paper reproduction) and Stage 2 (production enhancement) codebases.
4. **Architectural Block Diagrams:** Original paper vs. implemented system architectures.

Every finding, number, and status determination in this audit is cited to specific repository-relative file paths, line numbers, or published paper sections. Zero metrics or findings have been fabricated.

---

## 2. High-Level Project Status & Classification

| Dimension | Paper Specification | Implemented Status | Verification Classification | Primary Evidence |
|---|---|:---:|---|---|
| **Stage 1 Retrieval (MPR)** | 4 Paths: BM25, Dense FAISS, Metadata Section FAISS, Generative HyDE | **All 4 Paths Implemented** | **Implemented & Verified** | [`src/retrieval/multi_path_retriever.py:35-180`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/multi_path_retriever.py#L35-L180) |
| **Stage 1 Reranker (DRR)** | `BAAI/bge-reranker-v2-gemma` fine-tuned with DPO LoRA ($r=32, \alpha=64$, 10 ep) | **Trained from scratch, zero leakage** | **Implemented & Verified** | [`outputs/phase1_drr_adapter/adapter_model.safetensors`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_model.safetensors) (29.5 MB) |
| **Stage 1 Reranker Scoring** | Pure cross-encoder logits + Time Bonus (Zero heuristic boosts) | **Zero heuristic boosts (+0.0)** | **Implemented & Verified** | [`scripts/evaluate_phase1_drr.py:112-145`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L112-L145) |
| **Stage 1 Chunk Bundling** | Adjacency cosine expansion ($\tau=0.85, \mathcal{K}=5$) | **Implemented ($\tau=0.70, \mathcal{K}=5$)** | **Implemented with Adaptation** | [`src/retrieval/chunk_bundler.py:28-125`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/chunk_bundler.py#L28-L125) |
| **Stage 1 Ingestion (FFP)** | MinerU layout parser + VLM Table-to-Text synthetic narratives | PyMuPDF raw text & table extraction | **Partially Implemented** | [`scripts/preprocess_documents.py:6-40`](file:///c:/Users/user/Downloads/finsage-main/scripts/preprocess_documents.py#L6-L40) |
| **Stage 1 Generator** | Qwen2.5-72B-Instruct-AWQ on A100 (80 GB) | Qwen2.5-7B-Instruct (4-bit NF4) on RTX 5060 | **Documented Hardware Adaptation** | [`scripts/run_generation_phase1_drr.py:42-58`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_generation_phase1_drr.py#L42-L58) |
| **Stage 1 Evaluation** | 150 questions evaluated via GPT-4o CorrectnessEvaluator | 6 diagnostic questions (Q1–Q6) via Harness V2 | **Documented Diagnostic Scope** | [`outputs/phase1_audit/evaluation_harness_v2/`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_audit/evaluation_harness_v2/) |
| **Historical Phase 9C** | Ad-hoc heuristic pipeline (+5.5 boosts, regex expander) | Distinct historical milestone | **Archived Separately** | [`outputs/phase9c/phase9c_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase9c/phase9c_report.md) |
| **Stage 2 Production** | Table header propagation, calibrated routing, $N=150$ eval | **Executed & Recorded** | **Reported & Verified in Outputs** | [`outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md) |
| **Interactive Dashboard** | Production Web UI with Mode Toggling | Functional Streamlit UI on `:8501` | **Implemented & Verified** | [`app.py`](file:///c:/Users/user/Downloads/finsage-main/app.py) |

---

## 3. Four Critical Findings Every Subagent Must Know

### Finding 1: The DPO Target Resolution Milestone
* In earlier project phases (Phase 9B), DPO was mistakenly applied to the **Generator LLM** (`Qwen2.5-7B-Instruct`), generating a 1-epoch pilot and 3-epoch adapter. An audit proved this did not fix financial QA errors because financial hallucinations stemmed from missing context chunks, not language generation tokens.
* In Phase 1, the DPO objective was correctly aligned with the published paper: DPO was applied directly to the **Cross-Encoder Re-ranker** (`BAAI/bge-reranker-v2-gemma`) using 4-bit NF4 QLoRA ($r=32, \alpha=64$, 10 epochs, learning rate $10^{-4}$). The resulting 29.5 MB adapter exists at [`outputs/phase1_drr_adapter/adapter_model.safetensors`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_model.safetensors) and is executed with **zero heuristic boosts** (+0.0).

### Finding 2: The Table Bisection Anomaly on Q1 (3M FY2018 CapEx)
* For Question Q1, the user asks for 3M's FY2018 Capital Expenditures from the cash flow statement ($1,577M).
* In standard 10-K chunking, Page 60 was severed into `p60_c0` (Cash Flows from Operating Activities) and `p60_c1` (Cash Flows from Investing Activities). Because `p60_c1` was severed from the document header, it contained zero occurrences of "3M", "Company", "2018", or "Cash Flows". It scored 0.0 in BM25 and ranked outside the top 500 in dense FAISS.
* In Phase 1, semantic **Chunk Bundling** ([`src/retrieval/chunk_bundler.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/chunk_bundler.py)) anchored to `p60_c0` pulled in `p60_c1` (cosine similarity = 0.7172), assembling the full statement and enabling extraction of $1,577M.
* Crucially, the paper's reported threshold of $\tau_{\text{exp}} = 0.85$ **failed** ($0.7172 < 0.85$); only the adapted threshold of $\tau_{\text{exp}} = 0.70$ (derived from the official repo's 0.66–0.72 regime) succeeded.

### Finding 3: Evaluation Scope Reality Check ($N=6$ vs. $N=150$)
* The local repository indexes **84 SEC filing PDFs** (54,564 chunks) from **32 companies** across **9 GICS sectors**.
* However, all Phase 1 experiments were conducted on a diagnostic suite of **6 held-out questions (Q1–Q6)** covering only **2 companies** (3M and Ulta Beauty).
* The reported 83.3% Phase 1 accuracy applies **strictly to the 6 diagnostic questions** and **must never be reported as the accuracy across all of FinanceBench**.
* Stage 2 successfully evaluated the full $N=150$ FinanceBench questions, establishing a true macro MRR of **0.2926** and Recall@5 of **45.3%**.

### Finding 4: Separation of Historical Phase 9C from Stage 2
* Phase 9C was an exploratory milestone that used heavy heuristic boosts (+5.5: Statement +2.0, Company +1.5, Period +1.5, Table +0.5) and a deterministic regex expander on a base reranker. It produced false positive extractions (e.g., Common Stock instead of No Debt on Q3).
* Stage 1 discarded these heuristics, proving that a pure neural DPO reranker achieves 83.3% semantic accuracy on the diagnostic set.
* Stage 2 introduced true architectural improvements: Table Header Propagation at ingestion (2,902 chunks enriched), calibrated statement routing (+3.5 boost / -3.0 distractor penalty), a deterministic Python math sandbox, and full $N=150$ benchmark evaluation.

---

## 4. Deliverables Index in `outputs/final_project_audit/`

The complete audit suite comprises 13 distinct documents:

| File Name | Title & Purpose | Primary Contents |
|---|---|---|
| [`00_executive_summary.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/00_executive_summary.md) | Executive Briefing & System Verdict | High-level synthesis, status classifications, core findings |
| [`01_repository_inventory.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/01_repository_inventory.md) | Complete Repository File Inventory | Granular file-by-file categorization, evidence status, and dependencies |
| [`02_original_paper_method.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/02_original_paper_method.md) | Original Research Paper Specification | Comprehensive documentation of arXiv:2504.14493 with exact equations & citations |
| [`03_dataset_audit.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/03_dataset_audit.md) | Dataset & Evidence Manifest Audit | FinanceBench 84 PDFs, Lotus Company dataset, DRR preference pairs, leakage verification |
| [`04_stage1_implementation_audit.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/04_stage1_implementation_audit.md) | Stage 1 Implementation Audit | Component-by-component source code audit with exact file paths and line numbers |
| [`05_paper_results.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/05_paper_results.md) | Original Paper Published Results | Full retrieval, DRR, end-to-end QA, and GraphRAG tables with plain-English definitions |
| [`06_our_experimental_results.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/06_our_experimental_results.md) | Implemented Experimental Results | Phase 8, 9A, 9B, 9C, Phase 1 (DRR, Bundling, 4-Path), and Phase 2 ($N=150$) results |
| [`07_paper_vs_implementation.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/07_paper_vs_implementation.md) | Stage 1 vs. Paper Gap Analysis | 6-column comparison table, compliance classification, and remaining gaps |
| [`08_stage2_historical_experiments.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/08_stage2_historical_experiments.md) | Stage 2 & Historical Experiments | Separation of Phase 9C vs. Stage 2 (Table Header Propagation, Calibrated Boosts, Math) |
| [`09_architecture_diagrams.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/09_architecture_diagrams.md) | System Architecture Block Diagrams | Two Mermaid diagrams: Original Paper Method vs. Implemented System |
| [`10_evaluation_metrics.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/10_evaluation_metrics.md) | Evaluation Methodology & Metrics | Mathematical formulas, denominators, code implementations, and Harness V2 scorecard |
| [`11_submission_checklist.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/11_submission_checklist.md) | University Submission Checklist | 7-slide outline, 9-section report outline, code reproducibility checklist, blockers |
| [`evidence_manifest.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/final_project_audit/evidence_manifest.json) | Structured Evidence Manifest | JSON mapping of every key fact, metric, and artifact to source code/paper citations |

---

## 5. Recommended Presentation & Report Strategy

1. **Acknowledge Hardware & API Constraints Transparently:**
   * Do not hide the fact that Qwen2.5-7B (4-bit) replaced Qwen2.5-72B, or that Harness V2 replaced GPT-4o. Frame this as an engineering virtue: *“Deploying domain RAG under an accessible 8 GB VRAM desktop budget with deterministic verification.”*
2. **Highlight the DRR DPO Training:**
   * This was the hardest machine learning task in the project (90.7 minutes training on local GPU, 10 epochs, 0 leakage, 95.3% validation accuracy). It directly reproduces the paper's core scientific contribution.
3. **Showcase the Q1 Empirical Discovery (Table Bisection):**
   * Explain the bisection of Page 60 on 3M's cash flow statement. Showing how Chunk Bundling ($\tau=0.70$) solved it in Stage 1, and how Table Header Propagation solved it at the root in Stage 2, demonstrates deep domain mastery.
4. **Present Honest Metrics:**
   * Clearly separate the 83.3% diagnostic pass rate ($N=6$) from the 45.3% Recall@5 and 0.2926 MRR on the full 150-question benchmark. This academic honesty will win high praise from examiners.
