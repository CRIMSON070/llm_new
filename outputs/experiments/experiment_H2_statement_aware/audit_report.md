# FinSage Experiment H2: Independent Audit Report

**Audit Target:** `outputs/experiments/experiment_H2_statement_aware/`  
**Evaluation Scope:** 150 FinanceBench Benchmark Questions across 84 SEC Filings  
**Audit Type:** Strict Read-Only Forensic Audit  
**Date:** November 2026  

---

## 1. Executive Summary & Audit Recommendation

This independent audit evaluated the integrity, data consistency, recovery validity, regression causes, and filing safeguards of **Experiment H2: Statement-Aware Candidate Selection**.

### Key Audit Conclusions:
1. **Metric Reconciliation:** All reported metrics (79.33% candidate pool recall, 65.33% Recall@10, 0.3796 MRR, 29.99s runtime) are **100% reconciled and mathematically verified** across `scorecard.json`, `metrics_comparison.json`, `question_level_comparison.csv`, and `report.md`.
2. **Gold Evidence Recovery Audit:** All **8 reported recoveries** are **100% verified**. Every recovered question's candidate pool and reranked top 10 contain the authentic gold page and exact financial evidence line items.
3. **Narrative Invariance Guarantee:** All **68 narrative questions (`intent == NONE`)** experienced **zero pool perturbation**; their candidate pools in H2 were identical to the Experiment F reference baseline.
4. **Vulnerability Identified in Filing Selection:** In 9 out of 67 statement-injected queries, the selected filing was a competing filing of the target company (7 cases) or another company (2 cases) due to ambiguous temporal matching. While none of these caused regressions, this represents an unresolved algorithmic risk.
5. **Mechanistic Cause of Single Regression (`financebench_id_10130`):** Caused by naive first-chunk catalog slicing (`[:3]`) selecting page 44/50 interim tables rather than the primary statement on page 72, while evicting page 72 from pool position #48.

### Audit Verdict: **INVESTIGATE FURTHER & PROMOTE FOR VALIDATION**
Experiment H2 delivers substantial, verified net improvements (+4.66% pool recall, +4.00% Recall@10, +0.0303 MRR). However, before promoting H2 into the active production pipeline, the filing resolution rule and catalog chunk selector should be hardened against competing filings.

---

## 2. Reconciled Benchmark Metrics

Every question record ($N = 150$) was independently re-aggregated from [question_level_comparison.csv](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_H2_statement_aware/question_level_comparison.csv) and reconciled against [scorecard.json](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_H2_statement_aware/scorecard.json) and [metrics_comparison.json](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_H2_statement_aware/metrics_comparison.json):

| Metric | Reference (Exp F) | Experiment H2 | Independent Audit Verification | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Questions Evaluated** | 150 / 150 | 150 / 150 | Exactly 150 rows in CSV, zero null values | Verified |
| **Candidate Pool Recall** | 74.67% (112/150) | **79.33%** (119/150) | $\Delta = +4.66\%$ (+7 net questions) | Verified |
| **Recall@1** | 23.33% (35/150) | **25.33%** (38/150) | $\Delta = +2.00\%$ (+3 net questions) | Verified |
| **Recall@3** | 42.67% (64/150) | **47.33%** (71/150) | $\Delta = +4.66\%$ (+7 net questions) | Verified |
| **Recall@5** | 50.67% (76/150) | **55.33%** (83/150) | $\Delta = +4.66\%$ (+7 net questions) | Verified |
| **Recall@10** | 61.33% (92/150) | **65.33%** (98/150) | $\Delta = +4.00\%$ (+6 net questions) | Verified |
| **Mean Reciprocal Rank (MRR)** | 0.3493 | **0.3796** | $\Delta = +0.0303$ | Verified |
| **Total Benchmark Runtime** | 30.41s | **29.99s** | Overhead = -0.42s (faster due to cached catalog) | Verified |

---

## 3. Forensic Audit of the 8 Recovered Questions

Each of the 8 reported recoveries was audited against the benchmark ground truth in `data/financebench/questions.jsonl` and the corpus text in `data/financebench/processed/phase2_enriched_chunks.jsonl`:

| Question ID | Company | Filing | Detected Intent | Gold Target Page | Matching Chunk ID in H2 Pool | Pool Rank | Final Rank | Evidence Verified in Chunk? |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `financebench_id_00005` | Corning | `CORNING_2022_10K` | `balance_sheet` | 60 | `CORNING_2022_10K_p60_c1` | #49 | **#1** | **YES** (Contains full operating current assets & current liabilities) |
| `financebench_id_04103` | General Mills | `GENERALMILLS_2019_10K` | `balance_sheet+income_statement` | 53, 55 | `GENERALMILLS_2019_10K_p55_c1`, `p53_c1` | #46 | **#1** | **YES** (Contains accounts payable, inventories, net earnings) |
| `financebench_id_04417` | Best Buy | `BESTBUY_2019_10K` | `balance_sheet` | 52 | `BESTBUY_2019_10K_p52_c1` | #45 | **#2** | **YES** (Contains merchandise inventories balance) |
| `financebench_id_05915` | CVS Health | `CVSHEALTH_2018_10K` | `balance_sheet` | 302, 304 | `CVSHEALTH_2018_10K_p304_c1` | #49 | **#2** | **YES** (Contains net property, plant and equipment) |
| `financebench_id_00438` | Adobe | `ADOBE_2022_10K` | `income_statement` | 54 | `ADOBE_2022_10K_p54_c1` | #47 | **#3** | **YES** (Contains operating income and total revenue) |
| `financebench_id_00552` | Microsoft | `MICROSOFT_2023_10K` | `balance_sheet` | 60 | `MICROSOFT_2023_10K_p60_c1` | #46 | **#3** | **YES** (Contains current portion of long-term debt) |
| `financebench_id_00723` | American Express | `AMERICANEXPRESS_2022_10K` | `income_statement` | 96 | `AMERICANEXPRESS_2022_10K_p96_c1` | #48 | **#6** | **YES** (Contains total revenues net of interest expense) |
| `financebench_id_00216` | Verizon | `VERIZON_2022_10K` | `balance_sheet` | 56 | `VERIZON_2022_10K_p56_c1` | #50 | #999 | **YES** (Contains cash, accounts receivable, debt; reranked outside top 10) |

**Audit Finding:** **Zero false recoveries.** All 8 recovered questions contain legitimate gold pages and evidence. In 7 of the 8 cases, the BGE reranker successfully elevated the newly admitted chunk into the Top 6 final positions.

---

## 4. Forensic Investigation of the Single Regression (`financebench_id_10130`)

* **Question:** *"Based on the information provided primarily in the balance sheet and the statement of income, what is FY2020 days payable outstanding (DPO) for Corning?..."*
* **Target Filing:** `CORNING_2020_10K`
* **Gold Target Pages:** Page 70 (Income Statement) and Page 72 (Balance Sheet).
* **Reference Baseline Status (Exp F):**
  - `CORNING_2020_10K_p72_c0` (Page 72 Balance Sheet) entered the candidate pool at **Rank #48**.
  - After reranking, it was ranked **#5** overall ($RR = 0.20$).
* **Experiment H2 Status:**
  - `statement_intents` was detected as `balance_sheet+income_statement`.
  - The statement catalog returned earlier chunks: page 44 (`p44_c0`, `c1`, `c2`) and page 50 (`p50_c0`, `c2`, `c3`) from footnote/interim tables.
  - To inject these 6 chunks, H2 evicted the bottom 6 chunks of the base pool (positions 45–50).
  - Position #48 was `CORNING_2020_10K_p72_c0`, which was evicted from the pool.
* **Root Cause:**
  - **Not a filing error:** The correct filing (`CORNING_2020_10K`) was selected.
  - **Catalog chunk ordering:** The catalog took the first 3 chunks of each statement type in document order (`[:3]`), which matched earlier discussion tables on pages 44 and 50 rather than the primary Consolidated Financial Statements on pages 70 and 72.
  - **Bottom-pool eviction:** Because page 72 entered the base pool at rank 48, evicting 6 bottom slots without checking if the evicted chunk was already a target statement chunk caused the regression.

---

## 5. Audit of Classification & Filing Safeguards

Inspection of [statement_classification_audit.csv](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_H2_statement_aware/statement_classification_audit.csv):

1. **Intent Distribution:**
   - `NONE` (Narrative / Qualitative): **68 questions (45.3%)**
   - Single Statement (`BS`, `IS`, `CF`): **65 questions (43.3%)**
   - Compound Statements (`BS+IS`, `IS+CF`, `BS+CF`): **17 questions (11.3%)**

2. **Narrative Query Protection:**
   - For all 68 questions with `intent == NONE`, the candidate pools between Reference and H2 were **100% bit-for-bit identical** (zero additions, zero displacements, zero rank changes).

3. **Mismatched Filing Injections (Vulnerability Finding):**
   - In **9 out of 67 statement queries**, the injected statement chunks originated from a filing different from the gold document:
     - 7 cases were temporal/period mismatches for the same company (e.g., query asked about 2023, but `AMCOR_2023Q2_10Q` was selected over `AMCOR_2023_10K`; `AMD_2015_10K` was selected over `AMD_2022_10K`).
     - 2 cases were company mismatches (`CORNING_2022_10K` injected into Johnson & Johnson `financebench_id_00669`; `KRAFTHEINZ_2019_10K` injected into Pfizer `financebench_id_00702`).
   - None of these 9 queries caused regressions (the gold page was already absent from the base pool), but injecting chunks from competing filings represents an unresolved precision risk.

---

## 6. Implementation & Preservation Guarantees

* **Baseline Integrity:** All protected files remain identical to the pre-experiment repository state:
  - `src/retrieval/phase2_reranker.py`: `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9`
  - `scripts/run_phase2_benchmark.py`: `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1`
  - `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py`: `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED`
  - `outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl`: `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C`
* **Driver Isolation:** H2 was implemented and run exclusively via [scripts/run_controlled_comparison_h2.py](file:///C:/Users/user/Downloads/finsage-main/scripts/run_controlled_comparison_h2.py). Zero active pipeline modules were modified.

---

## 7. Audit Recommendations & Next Steps

1. **Do Not Promote Immediately to Active Baseline:** While Experiment H2 achieved a benchmark high of 79.33% candidate pool recall and 65.33% Recall@10, the filing selection vulnerability (9 mismatched filings) must be resolved first.
2. **Hardening Path for Production (Experiment H3 / Production Refinement):**
   - **Tighter Filing Resolution:** Enforce company-intent filtering before selecting the target filing to eliminate the 2 cross-company injection bugs.
   - **Form Disambiguation (10-K vs 10-Q):** Prefer annual 10-K filings over interim quarterly 10-Q filings for annual queries.
   - **Primary Statement Prioritization:** In the statement chunk catalog, prioritize chunks containing `"Consolidated Balance Sheets"` and `"Consolidated Statements of Income"` over preliminary footnote tables.
