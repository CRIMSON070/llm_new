# FinSage Stage 3: Independent Validation Dataset Progress Report

**Target Milestone:** Stage 3 — Build an Independent Validation Dataset for FinSage  
**Report Location:** `outputs/experiments/experiment_stage3_heldout_validation/stage3_progress_report.md`  
**Governing Objective:** Establish a defensible independent validation benchmark to test whether Experiment H2.1 (Correctness Fixes Only) generalizes to unseen companies, SEC filings, and financial questions.  
**Execution Date:** November 2, 2026  
**Current Status:** **EVALUATION COMPLETED — FROZEN BENCHMARK EXECUTED & AUDITED**

---

## 1. Executive Summary

Following explicit user authorization, the frozen, head-to-head comparative evaluation was executed on the **100-question Stage 3 independent held-out benchmark**. The evaluation compared the reference baseline (**Experiment F: Fiscal-Period-Aware Candidate Generation**) against the candidate system (**Experiment H2.1: Statement-Aware Candidate Selection with Strict Correctness Fixes**).

All objectives across **Phase A (Audit and Plan)**, **Phase B (Dataset Construction)**, **Phase C (Isolated Indexing Preparation)**, **Phase D (SEC Filing Ingestion, Gold-Evidence Verification, and Isolated Indexing)**, and **Phase E (Frozen Comparative Benchmark Execution)** are 100% completed:

1. **Preflight Checks:** Verified 100 unique question IDs, 100 verified evidence pages (1-indexed physical PDF pages), verified Alphabet corrected figures ($43,229M Google Cloud revenue and $72,764M Free Cash Flow), zero pending annotations, and verified all 7 protected SHA-256 baseline hashes.
2. **Corpus & Index Isolation:** Executed strictly against `data/heldout_stage3/` (29,335 chunks, 40 Form 10-K filings, 20 unseen companies) with zero interaction with the 54,564 development chunks.
3. **Comparative Evaluation Results:**
   - Both Experiment F and Experiment H2.1 achieved **30.00% Candidate Pool Recall**, **20.00% Recall@10**, **16.00% Recall@5**, **12.00% Recall@3**, **7.00% Recall@1**, and **0.1050 MRR**.
   - Zero net recoveries ($b = 0$), zero regressions ($c = 0$), and 100 ties across all 100 queries ($p = 1.000$ on McNemar's test and Wilcoxon signed-rank test).
   - Average query latency: 180.0 ms (Exp F) vs. 174.0 ms (H2.1).
   - All 7 protected SHA-256 hashes re-verified post-execution with 100% identity.
4. **Primary Evaluation Report:** The complete audit is documented in [`stage3_heldout_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md).

---

## 2. Phase D Ingestion & Validation Results

### 2.1 SEC Filing Ingestion Summary
* **Total Filings Downloaded:** 40 / 40 (100.0%)
* **Total Filings Validated:** 40 / 40 (100.0%)
* **Storage Location:** `data/heldout_stage3/raw_filings/`
* **Formats Preserved:** Both raw SEC EDGAR HTML (`.htm`) and compiled pagination-faithful PDFs (`.pdf`).
* **Total Pages Compiled:** 17,751 physical PDF pages across the 40 filings.
* **Manifest Artifact:** [`downloaded_filings_manifest.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/downloaded_filings_manifest.csv).

| Metric | Target | Realized | Compliance |
| :--- | :---: | :---: | :---: |
| Unseen Companies | 15–20 | **20** | 100% |
| SEC Filings (10-K) | 30–40 | **40** | 100% |
| Fiscal Years Covered | FY23 & FY24 | **FY23 & FY24** | 100% |
| Overlap with Dev Set | 0.0% | **0.00%** | 100% (Zero Overlap) |
| SHA-256 Tracking | 100% | **100% (40/40)** | Complete Fingerprint |

---

## 3. Gold-Evidence & Question Verification Audit

Every question in `heldout_questions.jsonl` was systematically reconciled against the compiled 10-K documents:

* **Total Questions Evaluated:** 100 / 100
* **Questions with Verified Physical PDF Evidence Pages:** **100 / 100 (100.0%)**
* **Questions Corrected During Audit:** **2**
  1. `stage3_heldout_0006` (Alphabet FY2024 Google Cloud Revenue): Corrected draft annualized estimate ($43,025M) to exact 10-K reported segment revenue on physical page 279: **$43,229 million**.
  2. `stage3_heldout_0008` (Alphabet FY2024 Free Cash Flow): Reconciled Operating Cash Flow ($125,299M) and CapEx ($52,535M) from physical page 158 to yield exact reported Free Cash Flow of **$72,764 million**.
* **Questions Excluded:** 0 (all 100 queries were confirmed fully answerable from the primary filings).
* **Audit Trail Log:** Complete question-by-question audit log is archived in [`question_verification_audit.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/question_verification_audit.csv).

---

## 4. Corpus Processing and Isolated Indexing Integrity

### 4.1 Chunking & Table Enrichment Integrity
* **Output Path:** `data/heldout_stage3/processed/heldout_enriched_chunks.jsonl`
* **File Size:** 23,471,849 bytes (23.47 MB)
* **Total Chunks Extracted:** 29,335 chunks
* **Unique Chunk IDs:** 29,335 (0 duplicates detected)
* **Table Continuations Enriched:** 504 continuation chunks enriched with statement titles, units, and comparative fiscal years.
* **Extraction Processing Time:** 9.70 seconds.

### 4.2 Isolated Index Compilation Integrity
* **Sparse BM25 Index:**
  - Directory: `data/heldout_stage3/indexes/bm25/`
  - Total Documents: 29,335
  - Indexing Time: 1.93 seconds
  - Status: Built and verified.
* **Dense BGE-M3 FAISS Index:**
  - Directory: `data/heldout_stage3/indexes/faiss_bge_m3/`
  - Embedding Model: `BAAI/bge-m3` (1024-dimensional embeddings)
  - Hardware Acceleration: NVIDIA GeForce RTX 5060 Laptop GPU (100% CUDA utilization)
  - Total Vectors Encoded: 29,335
  - Encoding & Construction Time: 618.06 seconds (~10.3 minutes)
  - Files Generated: `index.faiss` (120 MB) and `index.pkl`
* **Retrieval Sanity Test:**
  - Query: `"Apple total net sales 2024"`
  - Hit 1: `AAPL_2024_10K` physical page 68 (Consolidated Statements of Operations)
  - Hit 2: `AAPL_2023_10K` physical page 66 (Consolidated Statements of Operations)
  - Hit 3: `AAPL_2024_10K` physical page 90 (Segment Breakdown)
  - Verified: Correct top-ranked retrieval from held-out index.

---

## 5. Protected Baseline Integrity and Hash Verification

All protected baseline source files and prior experiment artifacts were hashed with SHA-256 before and after Phase D execution:

| Protected File Path | Baseline SHA-256 Hash | Post-Phase D SHA-256 Hash | Integrity Status |
| :--- | :---: | :---: | :---: |
| `src/retrieval/phase2_reranker.py` | `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9` | `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9` | **Untouched** |
| `scripts/run_phase2_benchmark.py` | `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1` | `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1` | **Untouched** |
| `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py` | `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED` | `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED` | **Untouched** |
| `outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl` | `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C` | `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C` | **Untouched** |
| `outputs/experiments/experiment_H2_statement_aware/scorecard.json` | `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2` | `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2` | **Untouched** |
| `outputs/experiments/experiment_H2_statement_aware/phase2_retrieval_checkpoint.jsonl` | `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F` | `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F` | **Untouched** |
| `outputs/experiments/experiment_H2_1_correctness_fixes/scorecard.json` | `ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC` | `ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC` | **Untouched** |

---

## 6. Deliverables Inventory

All requested Stage 3 artifacts are generated and archived in `outputs/experiments/experiment_stage3_heldout_validation/`:

1. [`stage3_dataset_construction_plan.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_dataset_construction_plan.md) — Master construction plan.
2. [`candidate_companies_and_filings.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/candidate_companies_and_filings.csv) — Catalog of 20 companies and 40 verified filings.
3. [`downloaded_filings_manifest.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/downloaded_filings_manifest.csv) — Full download manifest with SHA-256 hashes and page counts.
4. [`dataset_provenance_and_overlap_audit.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/dataset_provenance_and_overlap_audit.md) — Overlap audit proving 0.00% overlap.
5. [`heldout_questions.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl) — 100 questions with verified physical PDF page numbers and excerpts.
6. [`question_verification_audit.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/question_verification_audit.csv) — Question-by-question ground truth verification log.
7. [`annotation_guidelines.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/annotation_guidelines.md) — Annotation standards and physical page conventions.
8. [`frozen_evaluation_protocol.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/frozen_evaluation_protocol.md) — Pre-registered comparative evaluation protocol.
9. [`execution_manifest.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/execution_manifest.json) — System metadata and integrity tracking.
10. [`stage3_progress_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_progress_report.md) — This progress report.
11. [`scripts/run_heldout_stage3_benchmark.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_heldout_stage3_benchmark.py) — Reproducible isolated runner script.

12. [`stage3_heldout_benchmark_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_heldout_benchmark_report.md) — Complete comparative evaluation report.
13. [`heldout_metrics.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/heldout_metrics.json) — Master structured metrics.
14. [`heldout_question_level_comparison.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/heldout_question_level_comparison.csv) — Question-level comparative scorecard.

---

## 7. Execution Verdict & Audit Closure

* **Benchmark Status:** **EXECUTION COMPLETED & VERIFIED.**
  - All preflight checks passed.
  - Frozen Experiment F and H2.1 evaluated across all 100 questions.
  - Zero regressions, zero recoveries, 100 ties ($p = 1.000$).
  - Post-execution protected SHA-256 baseline hashes 100% verified.
* **Remaining Blockers:** **NONE.** All Stage 3 objectives are fully completed and archived.
