# FinSage Stage 3: Dataset Overlap & Qualification Report

**Document Title:** Stage 3 Held-Out Dataset Overlap and Qualification  
**Location:** `outputs/experiments/experiment_stage3_heldout_validation/dataset_overlap_and_qualification.md`  
**Governing Objective:** Phase A Qualification of Proposed Stage 3 Independent Held-Out Validation Data  
**Audit Scope:** Full repository scan of `c:\Users\user\Downloads\finsage-main`  
**Audit Standard:** Strict Empirical Independence & Zero-Data-Leakage Standard  
**Date:** November 2026  
**Phase A Qualification Verdict:** **DISQUALIFIED — NO SUITABLE INDEPENDENT DATASET AVAILABLE**  

---

## 1. Executive Summary

Phase A of the Stage 3 Independent Held-Out Validation requires qualifying the dataset intended to validate whether Experiment H2.1 (Correctness Fixes Only) generalizes beyond the original 150 FinanceBench benchmark questions.

Per the mandatory qualification protocol:
> *"If no suitable independent dataset exists, STOP and produce a report describing what is missing and propose a defensible held-out dataset construction plan. Do not label reused data as independent."*

An exhaustive forensic scan of all data directories, indexes, corpus files, and historical experiment outputs established that:
1. **Zero Unseen SEC Filings:** The repository contains exactly 84 SEC PDF filings in `data/financebench/pdfs/`, indexed across 54,564 enriched chunks. There are no additional filings present.
2. **Zero Unseen Evaluated Questions:** The repository contains exactly 150 benchmark questions in `data/financebench/questions.jsonl`. All 150 questions were evaluated, inspected, and utilized during development, failure analysis, and tuning in Experiments A–H2.1.
3. **Pseudo-Held-Out Candidate Disqualification:** The 6-question "diagnostic set" (Q1–Q6) used in Phase 1 is a direct subset of the 150 development questions and was repeatedly evaluated during Experiment H2/H2.1.
4. **Data Contamination Risk:** The company normalization table (`COMPANY_MAP`), statement routing heuristics, and temporal patterns implemented in H2.1 were developed with knowledge of the 32 companies and 84 filings in this benchmark.
5. **Conclusion:** No independent, previously unseen dataset exists on disk. Labeling any partition of the existing 150 questions as "independent held-out data" would constitute data leakage. Phase B/C execution is therefore blocked per protocol.

---

## 2. Comprehensive Inventory of Repository Data Assets

Every file containing structured text, questions, annotations, or index embeddings was audited:

| Data Asset Path | Format | Size | Total Items | Content Description & Scope | Development Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `data/financebench/questions.jsonl` | JSONL | 962.8 KB | 150 questions | 150 open-source FinanceBench questions spanning 32 companies and 84 SEC filings. | Development benchmark for Exp C, D, E1, F, G, H1, H2, H2.1. |
| `data/financebench/manifest.json` | JSON | 10.5 KB | 84 filings | Mapping of 84 SEC filings (doc_name, company, filing year, form type, source URL). | Defines active document universe. |
| `data/financebench/validation_report.json` | JSON | 312 B | Summary | Verification report confirming 150 questions downloaded, 84 unique docs, 84 PDFs. | Baseline data verification. |
| `data/financebench/pdfs/` | PDF | ~1.4 GB | 84 files | 84 annual 10-K and quarterly 10-Q SEC PDF filings. | Ingestion source for corpus. |
| `data/financebench/processed/phase2_enriched_chunks.jsonl` | JSONL | 59.8 MB | 54,564 chunks | Enriched chunks with table headers, filing metadata, and statement tags. | Active retrieval corpus for all experiments. |
| `data/financebench/indexes/faiss_bge_m3/` | FAISS | ~220 MB | 54,564 vectors | Dense vector index embedding all 54,564 chunks using BAAI/bge-m3. | Dense retrieval path for all experiments. |
| `data/financebench/bm25_index_phase2/` | MMIndex | ~3.3 MB | 54,564 docs | BM25 sparse index built on phase 2 enriched chunks. | Sparse retrieval path for all experiments. |
| `data/financebench/drr/train_drr.jsonl` | JSONL | 2.01 MB | 3,200 pairs | Training query-chunk pairs derived from FinanceBench training slice. | Phase 1 DRR adapter training. |
| `data/financebench/drr/val_drr.jsonl` | JSONL | 272 KB | 400 pairs | Validation query-chunk pairs derived from FinanceBench training slice. | Phase 1 DRR adapter validation. |
| `outputs/phase1_archive/.../harness_v2_results.jsonl` | JSONL | 22.6 KB | 6 questions | Diagnostic benchmark Q1–Q6 (`03029`, `04672`, `00746`, `00521`, `00499`, `00603`). | Phase 1 diagnostic harness. |

---

## 3. Overlap & Contamination Analysis

### 3.1 Overlap with H2.1 Development Benchmark

| Dimension | Total in Repository | Used in Exp H2.1 Development | Unseen / Held-Out Candidates | Overlap Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Questions** | 150 | 150 | **0** | **100.0%** |
| **SEC Filings** | 84 | 84 | **0** | **100.0%** |
| **Companies** | 32 | 32 | **0** | **100.0%** |
| **Corpus Chunks** | 54,564 | 54,564 | **0** | **100.0%** |
| **Economic Sectors (GICS)** | 9 | 9 | **0** | **100.0%** |

### 3.2 Evaluation of Candidate Pseudo-Held-Out Partitions

Two potential internal data partitions were evaluated and disqualified:

#### Disqualification of Option A: The Phase 1 Diagnostic Suite (Q1–Q6, $N=6$)
* **Composition:** `financebench_id_03029` (3M), `04672` (3M), `00746` (Ulta), `00521` (Ulta), `00499` (3M), `00603` (Ulta).
* **Disqualification Reasons:**
  1. All 6 questions are members of the 150 benchmark questions evaluated in Experiment F, H2, and H2.1.
  2. Questions Q1 (`03029`), Q2 (`04672`), and Q6 (`00603`) were explicitly inspected during the H2 audit and H2.1 filing resolution debugging.
  3. Sample size ($N=6$) across only 2 companies (3M and Ulta Beauty) is statistically meaningless for testing generalized multi-statement routing across 9 sectors.
  4. Violates constraint: *"Do not label reused data as independent."*

#### Disqualification of Option B: K-Fold or 80/20 Holdout Split of the 150 Questions
* **Concept:** Partitioning the 150 questions into 120 development questions and 30 "held-out" validation questions.
* **Disqualification Reasons:**
  1. **Post-Hoc Contamination:** The entirety of the 150 questions was already exposed during Experiment G (which diagnosed all 38 failures across the 150 questions), Experiment H1 (which evaluated all 150 questions), and Experiment H2 (which tuned statement quotas against the 150 questions).
  2. **Rule Leakage:** `COMPANY_MAP` in `scripts/run_controlled_comparison_h2_1.py` contains exactly the 32 companies present in the 150 questions.
  3. Splitting an already-seen benchmark post-hoc does not measure true generalization to unseen filings or novel query formulations.

---

## 4. Defensible Stage 3 Held-Out Benchmark Construction Plan

To conduct a scientifically valid, publication-grade Stage 3 independent validation, a new benchmark must be constructed according to the following specifications:

### 4.1 Target SEC Filing Ingestion (Unseen Universe)
* **Company Universe:** Select **15–20 new Fortune 500 / S&P 500 companies** with zero overlap with the current 32 companies:
  - *Technology:* Apple (AAPL), Alphabet (GOOGL), NVIDIA (NVDA)
  - *Financials:* Bank of America (BAC), Goldman Sachs (GS), Morgan Stanley (MS)
  - *Consumer:* Home Depot (HD), Procter & Gamble (PG), Target (TGT)
  - *Healthcare:* Eli Lilly (LLY), UnitedHealth (UNH), Merck (MRK)
  - *Energy & Industrials:* ExxonMobil (XOM), Chevron (CVX), Caterpillar (CAT)
* **Filing Universe:** Ingest **30–40 new annual 10-K filings** (FY2023 and FY2024) from SEC EDGAR.
* **Corpus Isolation:**
  - Parse PDFs using PyMuPDF and apply Table Header Propagation.
  - Index chunks into a completely separate, isolated vector store (`data/heldout_stage3/indexes/faiss_bge_m3`) and sparse index (`data/heldout_stage3/indexes/bm25_index`).
  - Zero chunk overlap with the development corpus.

### 4.2 Query Synthesis & Gold Annotation Protocol
* **Sample Size:** $N = 100$ financial questions.
* **Intent Stratification:**
  - **40% Single-Statement Queries:** Focused strictly on Balance Sheet (e.g. working capital, inventory, net debt), Income Statement (e.g. operating margin, COGS, tax rate), or Cash Flow (e.g. free cash flow, CapEx).
  - **20% Compound Multi-Statement Queries:** Requiring cross-statement synthesis (e.g. Cash Conversion Cycle, Return on Assets, Days Payable Outstanding).
  - **40% Narrative / Qualitative Queries:** Requiring MD&A, segment disclosure, or risk factor evidence (e.g. business drivers, customer concentration, M&A agendas).
* **Gold Annotations:**
  - Target SEC Document Name (e.g. `AAPL_2023_10K`).
  - Target Form Type (`10-K` vs `10-Q`).
  - Target Page Number (1-indexed physical page in filing).
  - Target Financial Line Item / Text Snippet.
  - Question Category & GICS Sector.

### 4.3 Validation Execution Protocol
* **Zero Modification Policy:** The frozen H2.1 candidate selection engine and Phase 2 Reranker must be run without tuning `COMPANY_MAP`, boost weights, or quotas.
* **Evaluation Metrics:**
  - Candidate Pool Recall (Top 50)
  - Recall@1, Recall@3, Recall@5, Recall@10
  - Mean Reciprocal Rank (MRR)
  - Wrong-Company Injection Rate (Target: 0.0%)
  - Wrong-Filing Injection Rate (Target: $\le 2.0\%$)
  - Narrative Invariance Guarantee (Target: 100.0%)

---

## 5. Phase A Audit Decision

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE A AUDIT DECISION                          │
├────────────────────────────────────────────────────────────────────────┤
│  STATUS: DISQUALIFIED                                                  │
│  VERDICT: VALIDATION BLOCKED — INDEPENDENT DATASET UNAVAILABLE         │
│                                                                        │
│  RATIONALE:                                                            │
│  1. 100% of all repository questions and filings were used in H2.1    │
│     development, tuning, and failure analysis.                         │
│  2. No independent SEC filings or unseen questions exist on disk.     │
│  3. Reusing or partitioning the 150 benchmark questions would violate  │
│     empirical integrity and produce invalid generalization claims.     │
│  4. In accordance with Phase A Rule 6, execution is halted pending     │
│     acquisition or construction of a true held-out benchmark.          │
└────────────────────────────────────────────────────────────────────────┘
```
