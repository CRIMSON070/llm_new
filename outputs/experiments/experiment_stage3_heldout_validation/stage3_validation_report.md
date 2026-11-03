# FinSage Stage 3: Independent Held-Out Validation Report

**Experiment Title:** Stage 3 — Independent Held-Out Validation  
**Target Directory:** `outputs/experiments/experiment_stage3_heldout_validation/`  
**Governing Objective:** Evaluate whether Experiment H2.1 (Correctness Fixes Only) generalizes to previously unseen financial questions and SEC filings.  
**Reference Baseline:** Experiment F (`outputs/experiments/experiment_F_period_aware/`)  
**Development Iteration:** Experiment H2.1 (`outputs/experiments/experiment_H2_1_correctness_fixes/`)  
**Date:** November 2026  
**Final Decision:** **VALIDATION BLOCKED — INDEPENDENT DATASET UNAVAILABLE**  

---

## 1. Executive Summary & Audit Verdict

Following the successful completion and verification of **Experiment H2.1 (Correctness Fixes Only)**—which achieved 86.00% Candidate Pool Recall (129/150), 72.00% Recall@10 (108/150), 0.4453 MRR, and zero regressions on the 150-question development benchmark—this Stage 3 investigation was initiated to evaluate empirical generalizability.

The mandate for Stage 3 required executing four systematic phases:
* **Phase A:** Inspect and qualify the proposed held-out dataset against strict independence and zero-data-leakage criteria.
* **Phase B:** Freeze evaluation arms, dataset annotations, and configurations.
* **Phase C:** Execute head-to-head evaluation (Exp F vs H2.1) across all macro metrics.
* **Phase D:** Verify baseline integrity and document limitations.

### Key Finding & Verdict:
Per **Phase A, Rule 6** of the governing experiment instructions:
> *"If no suitable independent dataset exists, STOP and produce a report describing what is missing and propose a defensible held-out dataset construction plan. Do not label reused data as independent."*

An exhaustive audit of all data assets, indexes, and historical experiment artifacts across `c:\Users\user\Downloads\finsage-main` established that:
1. **Zero Unseen SEC Filings on Disk:** The repository contains exactly 84 SEC PDF filings in `data/financebench/pdfs/`, indexed across 54,564 enriched chunks. No additional SEC filings exist in the workspace.
2. **Zero Independent Evaluated Questions on Disk:** The repository contains exactly 150 benchmark questions in `data/financebench/questions.jsonl`. All 150 questions were evaluated, inspected, and utilized during development, failure analysis, and tuning in Experiments A, C, D, E1, F, G, H1, H2, and H2.1.
3. **Pseudo-Held-Out Partitions Disqualified:** The 6-question "diagnostic set" (Q1–Q6) from Phase 1 is a direct subset of the 150 development questions and was repeatedly evaluated during H2/H2.1 debugging. Any k-fold or 80/20 train/test split of the 150 questions is post-hoc contaminated and scientifically invalid.
4. **Final Decision:** **VALIDATION BLOCKED — INDEPENDENT DATASET UNAVAILABLE**. In strict adherence to scientific rigor, metric computation on reused development data was rejected. This report documents the Phase A disqualification evidence and presents a defensible, publication-grade construction plan for a true Stage 3 independent validation dataset.

---

## 2. Phase A — Data Qualification & Overlap Audit

### 2.1 Repository Data Asset Audit

Every data asset in the workspace was inspected to identify potential held-out validation data:

| Data Asset | Path | Format | Size | Total Items | Scope & Provenance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Questions Benchmark** | `data/financebench/questions.jsonl` | JSONL | 962.8 KB | 150 questions | 150 open-source FinanceBench questions spanning 32 companies and 84 SEC filings. |
| **Document Manifest** | `data/financebench/manifest.json` | JSON | 10.5 KB | 84 filings | Metadata mapping for all 84 SEC filings in the repository. |
| **Ingestion Report** | `data/financebench/validation_report.json` | JSON | 312 B | Summary | Confirms exactly 150 questions and 84 PDFs were downloaded at project inception. |
| **PDF Repository** | `data/financebench/pdfs/` | PDF | ~1.4 GB | 84 files | 84 SEC annual (10-K) and quarterly (10-Q) filings. |
| **Enriched Chunk Corpus**| `data/financebench/processed/phase2_enriched_chunks.jsonl` | JSONL | 59.8 MB | 54,564 chunks | Primary chunk corpus enriched with table headers and filing metadata. |
| **Vector Index** | `data/financebench/indexes/faiss_bge_m3/` | FAISS | ~220 MB | 54,564 vectors | BGE-M3 dense embeddings covering the 84 filings. |
| **BM25 Sparse Index** | `data/financebench/bm25_index_phase2/` | MMIndex | ~3.3 MB | 54,564 docs | BM25 index built exclusively on the 84 filings. |
| **DRR Training Data** | `data/financebench/drr/train_drr.jsonl` | JSONL | 2.01 MB | 3,200 pairs | Training pairs derived from the same FinanceBench corpus for Phase 1 DRR adapter. |
| **DRR Validation Data**| `data/financebench/drr/val_drr.jsonl` | JSONL | 272 KB | 400 pairs | Validation pairs derived from the same FinanceBench corpus. |
| **Phase 1 Diagnostic** | `outputs/phase1_archive/.../harness_v2_results.jsonl` | JSONL | 22.6 KB | 6 questions | 6-question diagnostic subset (`03029`, `04672`, `00746`, `00521`, `00499`, `00603`). |

### 2.2 Overlap Matrix with H2.1 Development Benchmark

| Dimension | Total in Workspace | Used in Exp H2.1 Development | Unseen / Held-Out on Disk | Overlap Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Questions** | 150 | 150 | **0** | **100.0%** |
| **SEC Filings** | 84 | 84 | **0** | **100.0%** |
| **Corporate Entities** | 32 | 32 | **0** | **100.0%** |
| **Corpus Chunks** | 54,564 | 54,564 | **0** | **100.0%** |
| **GICS Sectors** | 9 | 9 | **0** | **100.0%** |

### 2.3 Forensic Disqualification of Candidate Pseudo-Held-Out Partitions

The forensic audit evaluated whether any subset of existing data could legitimately serve as a Stage 3 held-out evaluation set:

#### Disqualification 1: The Phase 1 Diagnostic Suite (Q1–Q6, $N=6$)
* **Candidate:** `financebench_id_03029` (3M), `04672` (3M), `00746` (Ulta), `00521` (Ulta), `00499` (3M), `00603` (Ulta).
* **Disqualification Rationale:**
  1. **Subset Contamination:** All 6 questions are elements of `data/financebench/questions.jsonl` and were evaluated in Experiment F, H2, and H2.1.
  2. **Active Debugging Exposure:** Questions Q1 (`03029`), Q2 (`04672`), and Q6 (`00603`) were explicitly audited during the H2 failure analysis and the H2.1 filing resolution redesign.
  3. **Statistical Insignificance:** Evaluating $N=6$ queries across only 2 companies (3M and Ulta Beauty) provides zero statistical power to evaluate generalization across 9 economic sectors.
  4. **Violation of Explicit Constraint:** Violates instruction: *"Do not label reused data as independent."*

#### Disqualification 2: Post-Hoc Split of the 150 Development Questions (e.g. 80/20 Train/Test)
* **Candidate:** Reserving 30 randomly selected questions from the 150 benchmark questions.
* **Disqualification Rationale:**
  1. **Post-Hoc Data Leakage:** All 150 questions were fully inspected during Experiment G (which performed failure traces on all 38 baseline misses) and Experiment H2 (which evaluated statement quota allocations across all 150 questions).
  2. **Hardcoded Corporate Knowledge:** The `COMPANY_MAP` dictionary in `scripts/run_controlled_comparison_h2_1.py` contains exactly the 32 companies present in the 150 benchmark questions. Testing on a subset of these same 32 companies cannot evaluate whether canonical company detection generalizes to unmapped corporate entities.
  3. Slicing an existing evaluation set post-hoc after seeing results and reporting it as "independent validation" is scientifically invalid.

---

## 3. Comparison Arm Freezing (Phase B Verification)

In accordance with Phase B, both comparison arms have been strictly frozen to guarantee zero parameter tuning:

### 3.1 Arm 1: Reference Baseline (Experiment F)
* **Retrieval Depth:** Sparse BM25 ($k=50$), Dense BGE-M3 ($k=50$), Metadata ($k=3$ sections).
* **Fusion:** Reciprocal Rank Fusion (RRF, $k=60$).
* **Candidate Pool Size:** $K_{\text{pool}} = 50$.
* **Temporal Prioritization:** In-period candidate promotion based on extracted fiscal years and quarters.
* **Statement Routing:** **Disabled** (standard RRF fusion).
* **Reranker:** `BAAI/bge-reranker-base` (context = 512, fp16 = True).

### 3.2 Arm 2: Experiment H2.1 (Audited Correctness Fixes)
* **Retrieval Depth:** Identical to Reference ($k=50$ sparse, $k=50$ dense, $k=3$ metadata).
* **Fusion:** Identical RRF ($k=60$), $K_{\text{pool}} = 50$.
* **Statement Routing:** **Enabled**, strictly bounded by quota $Q \le 6$ (preserving $\ge 44$ standard RRF slots).
* **Fix A (Filing Disambiguation):** Strict canonical company prefix matching (`COMPANY_MAP`), strict 10-K (annual) vs 10-Q (interim) form matching, 2-digit FY extraction (`FY22` $\rightarrow$ 2022). Safe fallback to baseline pool if ambiguous or unverified.
* **Fix B (Primary Statement Selection):** Normalized heading and line-item structure qualification. Slices primary consolidated financial statements, rejecting MD&A and preliminary footnote tables. Resolves `financebench_id_10130` regression.
* **Reranker:** `BAAI/bge-reranker-base` with calibrated boosts:
  - Statement boost: $+4.5$
  - Negative disclosure penalty: $-3.0$
  - Company boost: $+1.5$
  - Period boost: $+1.5$
  - Table boost: $+0.5$

Both arms are preserved in [scripts/run_controlled_comparison_h2_1.py](file:///c:/Users/user/Downloads/finsage-main/scripts/run_controlled_comparison_h2_1.py). Zero retrieval logic or boost parameters were modified.

---

## 4. Defensible Stage 3 Held-Out Benchmark Construction Plan

To establish publication-grade generalizability, an independent Stage 3 dataset must be constructed according to the following protocol:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               STAGE 3 INDEPENDENT BENCHMARK CONSTRUCTION BLUEPRINT                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  1. Unseen Document Universe (SEC EDGAR Ingestion):                                    │
│     - 20 New S&P 500 Companies (Zero overlap with the 32 development companies):       │
│       • Tech: Apple (AAPL), Alphabet (GOOGL), NVIDIA (NVDA), Broadcom (AVGO)           │
│       • Financials: Bank of America (BAC), Goldman Sachs (GS), Morgan Stanley (MS)     │
│       • Consumer: Procter & Gamble (PG), Home Depot (HD), Target (TGT), PepsiCo        │
│       • Healthcare: Eli Lilly (LLY), UnitedHealth (UNH), Merck (MRK), AbbVie (ABBV)    │
│       • Industrials & Energy: ExxonMobil (XOM), Chevron (CVX), Caterpillar (CAT)       │
│     - Ingest 40 Annual 10-K Filings for FY2023 and FY2024.                             │
│     - Dedicated PyMuPDF ingestion and table header propagation.                        │
│     - Isolated corpus indexing: `data/heldout_stage3/indexes/faiss_bge_m3`.             │
│                                                                                        │
│  2. Independent Query Generation ($N = 100$ Questions):                                │
│     - Stratified Intent Distribution:                                                  │
│       • 40% Single-Statement Queries (Working Capital, Net PP&E, Operating Margin)     │
│       • 20% Compound Multi-Statement Queries (Cash Conversion Cycle, ROA, DPO)         │
│       • 40% Narrative / Qualitative Queries (Segment growth, M&A agendas, Risks)       │
│     - Ground Truth Annotations: Exact Filing ID, 1-indexed target page, snippet.       │
│                                                                                        │
│  3. Execution & Generalization Test:                                                   │
│     - Run frozen Exp F vs frozen H2.1 on the new index.                                │
│     - Confirm whether Candidate Pool Recall gains (+10.67%) and MRR gains (+0.0955)     │
│       transfer to unmapped company names and novel filing layouts.                     │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Development Benchmark Performance Summary (Exp F vs H2.1)

For reference, the verified performance on the 150-question development benchmark is summarized below:

| Metric | Reference (Exp F) | Audited Exp H2.1 | Net Gain ($\Delta$) | Empirical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Questions Evaluated** | 150 / 150 | 150 / 150 | — | Complete Benchmark |
| **Candidate Pool Recall** | 74.67% (112/150) | **86.00% (129/150)** | **+11.33% (+17)** | Substantial Candidate Elevation |
| **Recall@1** | 23.33% (35/150) | **32.67% (49/150)** | **+9.34% (+14)** | Top-Rank Precision Surge |
| **Recall@3** | 42.67% (64/150) | **51.33% (77/150)** | **+8.66% (+13)** | Context Precision Gain |
| **Recall@5** | 50.67% (76/150) | **60.00% (90/150)** | **+9.33% (+14)** | Context Precision Gain |
| **Recall@10** | 61.33% (92/150) | **72.00% (108/150)** | **+10.67% (+16)** | High-Confidence Answer Pool |
| **MRR** | 0.3493 | **0.4453** | **+0.0960** | Statistically Significant |
| **Regressions vs Ref** | 0 | **0** | 0 | Zero System Regressions |
| **Regressions vs H2** | N/A | **0** | 0 | Single Regression Repaired |
| **Narrative Pool Identity**| 100% | **100% (68/68)** | 0% perturbation | 100% Narrative Invariance |

*Note: Against the internal reference arm executed in H2.1 (which included the 2-digit FY parsing fix), Candidate Pool Recall is +10.67% (113 $\rightarrow$ 129, +16 net recoveries).*

---

## 6. Baseline Integrity & Hash Verification

All protected baseline source files and prior experiment artifacts were verified via SHA-256 hashes:

| Protected File Path | Baseline SHA-256 Hash | Post-Audit SHA-256 Hash | Integrity Status |
| :--- | :---: | :---: | :---: |
| `src/retrieval/phase2_reranker.py` | `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9` | `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9` | **Untouched** |
| `scripts/run_phase2_benchmark.py` | `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1` | `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1` | **Untouched** |
| `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py` | `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED` | `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED` | **Untouched** |
| `outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl` | `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C` | `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C` | **Untouched** |
| `outputs/experiments/experiment_H2_statement_aware/scorecard.json` | `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2` | `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2` | **Untouched** |
| `outputs/experiments/experiment_H2_statement_aware/phase2_retrieval_checkpoint.jsonl` | `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F` | `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F` | **Untouched** |

* Git workspace remains completely clean (`git status --porcelain` is empty).
* Zero production pipeline modules in `src/retrieval/` were modified.
* All Stage 3 artifacts are isolated exclusively in `outputs/experiments/experiment_stage3_heldout_validation/`.

---

## 7. Deliverables & Next Steps

All required deliverables have been generated and archived in `outputs/experiments/experiment_stage3_heldout_validation/`:
1. [`stage3_validation_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/stage3_validation_report.md): Formal Stage 3 audit and plan document.
2. [`dataset_overlap_and_qualification.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/dataset_overlap_and_qualification.md): Exhaustive data qualification and inventory.
3. [`execution_manifest.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/execution_manifest.json): Experiment configuration, hashes, and audit metadata.
4. [`heldout_metrics.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/heldout_metrics.json): Metrics record documenting Phase A halt.
5. [`heldout_question_level_comparison.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/heldout_question_level_comparison.csv): Schema-preserved held-out comparison artifact.
6. [`recovery_regression_audit.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/recovery_regression_audit.csv): Schema-preserved transition audit artifact.

### Actionable Next Steps:
1. **Acquire True Held-Out SEC Filings:** Follow Section 4 of this report to download 40 new 10-K filings from SEC EDGAR for 20 unseen companies.
2. **Synthesize Independent Evaluation Benchmark:** Generate and human-verify 100 new financial questions with page-level gold annotations.
3. **Execute Independent Validation:** Run the frozen Exp F vs H2.1 pipeline on the new benchmark to establish definitive generalization before production release.
