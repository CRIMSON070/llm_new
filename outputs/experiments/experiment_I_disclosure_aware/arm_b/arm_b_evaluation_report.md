# FinSage Experiment I: Arm I-B Evaluation Report — Disclosure-Aware Candidate Selection

**Target Milestone:** Experiment I — Arm I-B: Disclosure-Aware Candidate Selection ($Q_{\text{disc}} \le 10$, $K_{\text{base}} \ge 40$, $k=50$)  
**Report Artifact:** `outputs/experiments/experiment_I_disclosure_aware/arm_b/arm_b_evaluation_report.md`  
**Author:** Antigravity Research & Engineering Agent  
**Date:** November 2, 2026  
**Status:** **ARM I-B COMPLETED — DECISION GATE REACHED — AWAITING USER REVIEW & AUTHORIZATION**  

---

## 1. Executive Summary

Following user authorization, **Experiment I-B (Disclosure-Aware Candidate Selection)** was implemented and evaluated on the frozen 100-question Stage 3 held-out benchmark and isolated corpus (`data/heldout_stage3/`). 

In strict adherence to the decision gate findings from Arm I-A, **Arm I-B branched directly from the cleaner, un-diluted frozen Experiment F baseline ($k=50$ for BM25 and Dense FAISS)** rather than the diluted depth-100 pool of Arm I-A.

### Headline Findings:
1. **Control & Integrity Verification:**  
   The frozen **Experiment F** baseline was verified 100% reproducible before execution (Pool Recall: 30.00%, Recall@10: 20.00%, MRR: 0.1050). All 7 protected baseline modules and checkpoints were audited before and after execution; **all 7 SHA-256 hashes remained 100% untouched**.
2. **Arm I-B Retrieval Performance vs. Experiment F:**
   - **Candidate Pool Recall ($K=50$):** **30.00%** (+0.00 pp vs. Exp F, exactly 30/100)
   - **Recall@1:** **6.00%** (-1.00 pp vs. Exp F, 6/100 vs. 7/100)
   - **Recall@3:** **11.00%** (-1.00 pp vs. Exp F, 11/100 vs. 12/100)
   - **Recall@5:** **15.00%** (-1.00 pp vs. Exp F, 15/100 vs. 16/100)
   - **Recall@10:** **18.00%** (-2.00 pp vs. Exp F, 18/100 vs. 20/100)
   - **Mean Reciprocal Rank (MRR):** **0.0940** (-0.0110 vs. Exp F, 0.0940 vs. 0.1050)
   - **Mean Query Latency:** **187.0 ms** (+7.0 ms vs. Exp F's 180.0 ms)
3. **Paired Transitions:**
   - **Candidate Pool ($K=50$):** 1 Recovery ($b=1$), 1 Regression ($c=1$), 98 Ties (29 hit-ties, 69 miss-ties). McNemar's exact test $p = 1.000$ (no statistical difference).
   - **Top-10 Retrieval ($K=10$):** 0 Recoveries ($b=0$), 2 Regressions ($c=2$), 98 Ties (18 hit-ties, 80 miss-ties). McNemar's exact test $p = 0.500$, Wilcoxon signed-rank $W=0.0, z=-1.3416, p = 0.1797$ (not statistically significant).
4. **Candidate Pool Stability:**  
   Enforcing the guaranteed baseline floor ($\ge 40$ slots) and quota cap ($\le 10$ novel slots) maintained an average **Jaccard similarity of 0.9731** against the Experiment F baseline, with an average of only **0.72 novel chunks injected** per query.
5. **Core Tradeoff Identified:**  
   While disclosure routing successfully recovered an unseen Alphabet corporate income statement chunk (`stage3_heldout_0007`), the displacement of lower-ranked baseline candidates at ranks 45–50 caused a severe regression on Merck financing disclosures (`stage3_heldout_0081`), and re-ranking competition caused a top-10 displacement on Meta (`stage3_heldout_0029`).

---

## 2. Experimental Architecture & Preflight Validation

### 2.1 Controlled Arm Configuration

| Parameter | Reference Baseline (Exp F) | Arm I-B (Disclosure-Aware) | Status |
| :--- | :---: | :---: | :---: |
| **Sparse BM25 Depth** | $k = 50$ | $k = 50$ | Frozen / Identical |
| **Dense FAISS Depth** | $k = 50$ | $k = 50$ | Frozen / Identical |
| **RRF Fusion Formula** | $k_{\text{rrf}} = 60$ | $k_{\text{rrf}} = 60$ | Frozen / Identical |
| **Temporal Prioritization** | In-period chunks prioritized | In-period chunks prioritized | Frozen / Identical |
| **Disclosure Candidate Selection** | DISABLED | **ENABLED (Unsupervised Catalog Lookup)** | **Variable Under Test** |
| **Max Novel Injection Quota ($Q_{\text{novel}}$)** | 0 | **$\le 10$ unique chunks** | Enforced |
| **Reserved Baseline Floor ($K_{\text{base}}$)** | 50 | **$\ge 40$ unique chunks** | Enforced |
| **Final Candidate Pool Size ($K_{\text{pool}}$)** | Exactly 50 unique chunks | Exactly 50 unique chunks | Invariant |
| **Phase 2 Reranker** | `bge-reranker-base` (fp16) | `bge-reranker-base` (fp16) | Frozen / Identical |
| **Reranker Boost Weights** | Stmt +4.5, Disc -3.0, Co +1.5, Per +1.5, Tab +0.5 | Stmt +4.5, Disc -3.0, Co +1.5, Per +1.5, Tab +0.5 | Frozen / Identical |
| **Corpus & Indexes** | `data/heldout_stage3/` (29,335 chunks) | `data/heldout_stage3/` (29,335 chunks) | Isolated / Frozen |

### 2.2 Preflight Code & Leakage Audit
Prior to benchmark execution, `scripts/validate_arm_b_preflight.py` performed four automated checks:
1. **Leakage Audit:** Confirmed that `sanitized_queries` passes strictly `{question_id, question, company, fiscal_year}` to the retrieval pipeline. Zero access to `evidence_page_num`, `evidence_sources`, `relevant_passage`, `gold_answer`, `target_statement_or_section`, or `target_concept` during candidate selection.
2. **Pool Assembly Invariant Testing:** Validated `assemble_disclosure_aware_pool` on synthetic edge cases (standard injection, zero injection fallback, and quota capping). Confirmed that the baseline floor ($\ge 40$ slots) is strictly respected and the final pool is always exactly 50 unique chunks.
3. **Retrieval Depth Alignment:** Verified that BM25 and Dense FAISS depths are configured to $k=50$.
4. **Baseline Hashes:** Pre-run SHA-256 hashes of all 7 protected modules matched 100%.

---

## 3. Comparative Benchmark Results

### 3.1 Headline Retrieval Metrics

| Metric | Reference Exp F Baseline | Arm I-B (Disclosure-Aware) | Net Delta | Statistical Significance Test |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall ($K=50$)** | **30.00%** (30/100) | **30.00%** (30/100) | **+0.00 pp** | McNemar exact test $p = 1.000$ (Identical) |
| **Recall@1** | **7.00%** (7/100) | **6.00%** (6/100) | **-1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@3** | **12.00%** (12/100) | **11.00%** (11/100) | **-1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@5** | **16.00%** (16/100) | **15.00%** (15/100) | **-1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@10** | **20.00%** (20/100) | **18.00%** (18/100) | **-2.00 pp** | McNemar exact test $p = 0.500$ (Not Significant) |
| **Mean Reciprocal Rank (MRR)** | **0.1050** | **0.0940** | **-0.0110** | Wilcoxon $W=0.0, z=-1.3416, p=0.1797$ |
| **Average Query Latency** | **180.0 ms** | **187.0 ms** | **+7.0 ms** | Negligible overhead (+3.9%) |
| **Total Benchmark Runtime** | **18.0 s** | **18.7 s** | **+0.7 s** | Extremely fast execution |

```
                       RECALL COMPARISON: EXP F vs. ARM I-B (N=100)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Candidate Pool Recall (K=50)                                                           │
  │   Exp F Baseline (k=50)   ██████████████████████████████ 30.00%                        │
  │   Arm I-B (Disc-Aware)    ██████████████████████████████ 30.00% (+0.00 pp)             │
  │                                                                                        │
  │ Recall@10                                                                              │
  │   Exp F Baseline (k=50)   ████████████████████ 20.00%                                  │
  │   Arm I-B (Disc-Aware)    ██████████████████ 18.00% (-2.00 pp)                         │
  │                                                                                        │
  │ Mean Reciprocal Rank (MRR)                                                             │
  │   Exp F Baseline (k=50)   ███████████ 0.1050                                           │
  │   Arm I-B (Disc-Aware)    ██████████ 0.0940 (-0.0110)                                  │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Paired Question-Level Transitions

| Transition Type | Candidate Pool ($K=50$) | Top-10 Retrieval ($K=10$) | Exact Affected Question IDs |
| :--- | :---: | :---: | :--- |
| **Recoveries ($b$)** | **1** | **0** | Pool: `stage3_heldout_0007` (Alphabet Inc.) |
| **Regressions ($c$)** | **1** | **2** | Pool: `stage3_heldout_0081` (Merck & Co.). Top-10: `stage3_heldout_0029`, `0081`. |
| **Ties: Hit $\rightarrow$ Hit ($a$)** | **29** | **18** | 29 pool hits and 18 top-10 hits retained unchanged. |
| **Ties: Miss $\rightarrow$ Miss ($d$)** | **69** | **80** | 69 pool misses and 80 top-10 misses remained unrecovered. |
| **Total Questions** | **100** | **100** | **Net change: +0 pool hits, -2 top-10 hits.** |

---

## 4. Deep-Dive: The Questions Recovered and Lost

An audit of [`disclosure_routing_audit.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_I_disclosure_aware/arm_b/disclosure_routing_audit.csv) and [`arm_b_vs_exp_f_question_comparison.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_I_disclosure_aware/arm_b/arm_b_vs_exp_f_question_comparison.csv) illuminates the exact mechanisms governing the recoveries and regressions:

### 4.1 The Single Recovery: `stage3_heldout_0007` (Alphabet Inc. 2023)
- **Question:** Alphabet Inc. Consolidated Statements of Income line items for fiscal year 2023.
- **Experiment F Baseline:** Missed candidate pool (`best_rank_in_pool = 999`). BM25 and Dense FAISS failed to place the Consolidated Statements of Income chunk (`GOOGL_2023_10K_p143_c0`) in the top 50 due to dense dispersion across hundreds of Alphabet filing pages.
- **Arm I-B Action:**
  - Intent detector identified `income_statement` intent from query phrasing.
  - Target filing was strictly resolved to `GOOGL_2023_10K`.
  - Unsupervised disclosure catalog provided the primary Consolidated Statement of Income chunk (`GOOGL_2023_10K_p143_c0`), which was injected into the pool.
- **Outcome:** **Candidate Pool Recovery ($0 \rightarrow 1$)**. (Gold chunk entered pool at rank 50; missed top 10).

### 4.2 The Severe Regression: `stage3_heldout_0081` (Merck & Co., Inc. 2024)
- **Question:** Merck's credit facility borrowing capacity and commercial paper liquidity in 2024.
- **Experiment F Baseline:** The gold evidence chunk (`MRK_2024_10K_p11_c0`) placed at rank 48 in the RRF fused pool. In Phase 2 cross-encoder reranking, its exact semantic match scored highly and ranked **#1 in top-10 retrieval (MRR = 1.000)**.
- **Arm I-B Action:**
  - Query phrasing "commercial paper" triggered `segment_disclosures` intent.
  - The catalog offered 4 novel segment reporting chunks from `MRK_2024_10K`.
  - Candidate pool assembly injected the 4 novel segment chunks, displacing the bottom 4 candidates from the baseline pool (ranks 47–50).
  - The gold chunk at rank 48 was **evicted from the pool**.
- **Outcome:** **Double Regression ($1 \rightarrow 0$ pool, $1 \rightarrow 0$ top-10)**. Rank 1 hit dropped to a complete miss, incurring an **MRR loss of -1.000**.

### 4.3 The Re-Ranking Regression: `stage3_heldout_0029` (Meta Platforms, Inc. 2024)
- **Question:** Meta Platforms capital expenditures and purchases of property and equipment in 2024.
- **Experiment F Baseline:** Gold evidence chunk (`META_2024_10K_p88_c0`) placed in the pool and was ranked at **rank 10** by the Phase 2 cross-encoder (MRR = 0.1000).
- **Arm I-B Action:**
  - Query phrasing triggered `cash_flow_statement` intent.
  - Catalog injected primary cash flow statement chunk `META_2024_10K_p213_c0` into the pool.
  - During Phase 2 reranking, the cross-encoder scored the primary statement chunk higher than the MD&A narrative table chunk, pushing the gold chunk from rank 10 to **rank 11**.
- **Outcome:** **Top-10 Regression ($1 \rightarrow 0$)**. Gold chunk remained in the pool, but was displaced past the Top-10 boundary by statement boost competition (**MRR loss of -0.1000**).

---

## 5. Results Stratified by Evidence Category

| Evidence Category | Total Questions | Exp F Pool Recall | Arm I-B Pool Recall | Pool Delta | Exp F Top-10 | Arm I-B Top-10 | Top-10 Delta | Exp F MRR | Arm I-B MRR | MRR Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Financial Statements** | 28 | 35.71% | 35.71% | +0.00 pp | 21.43% | 21.43% | +0.00 pp | 0.1658 | 0.1658 | +0.0000 |
| **Other / Qualitative & Risks** | 16 | 18.75% | **25.00%** | **+6.25 pp** | 6.25% | 6.25% | +0.00 pp | 0.0625 | 0.0625 | +0.0000 |
| **Segment Disclosures** | 12 | 16.67% | 16.67% | +0.00 pp | 16.67% | 16.67% | +0.00 pp | 0.0444 | 0.0444 | +0.0000 |
| **MD&A Tables and Narrative** | 32 | **25.00%** | 21.88% | **-3.12 pp** | **12.50%** | 6.25% | **-6.25 pp** | **0.0535** | 0.0191 | **-0.0344** |
| **Footnotes** | 11 | 63.64% | 63.64% | +0.00 pp | 63.64% | 63.64% | +0.00 pp | 0.2371 | 0.2371 | +0.0000 |
| **Debt, Lease, Inventory Detail** | 1 | 0.00% | 0.00% | +0.00 pp | 0.00% | 0.00% | +0.00 pp | 0.0000 | 0.0000 | +0.0000 |
| **Total Benchmark** | **100** | **30.00%** | **30.00%** | **+0.00 pp** | **20.00%** | **18.00%** | **-2.00 pp** | **0.1050** | **0.0940** | **-0.0110** |

### Critical Insight: The MD&A Vulnerability
As predicted by the experimental hypothesis, **disclosure routing helped one category while hurting another**:
- **Gain:** "Other or Uncertain" questions gained +6.25 pp in pool recall due to targeted filing resolution.
- **Loss:** **MD&A disclosures (32% of benchmark) suffered a severe decline (-3.12 pp pool, -6.25 pp top-10, -0.0344 MRR).** When queries ask about complex operational metrics that cross boundaries (e.g. CapEx, margins, financing), heuristic routing injects primary statements or segment disclosures that displace the subtle MD&A table chunks.

---

## 6. Synthesis of Experiment I: Arm I-A vs. Arm I-B vs. Baseline

We now have the complete, head-to-head empirical results for both arms of Experiment I evaluated on the identical 100 held-out questions:

| Experimental Arm | Pool Recall ($K=50$) | Recall@10 | MRR | Pool Recoveries / Regressions | Top-10 Recoveries / Regressions | Avg Latency | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference Baseline (Frozen Exp F)** | **30.00%** | **20.00%** | **0.1050** | — | — | **180.0 ms** | Reference Control |
| **Arm I-A: Depth Ablation ($k=100$)** | **31.00%** (+1.0 pp) | **20.00%** (+0.0 pp) | **0.0966** (-0.008) | 3 / 2 (net +1) | 1 / 1 (net 0) | **188.6 ms** | $p = 1.000$ (Not Sig) |
| **Arm I-B: Disclosure Routing ($k=50$)**| **30.00%** (+0.0 pp) | **18.00%** (-2.0 pp) | **0.0940** (-0.011) | 1 / 1 (net 0) | 0 / 2 (net -2) | **187.0 ms** | $p = 0.500$ (Not Sig) |

```
                       SUMMARY COMPARISON ACROSS EXPERIMENT I (N=100)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Candidate Pool Recall (K=50)                                                           │
  │   Experiment F Reference Baseline    ██████████████████████████████ 30.00%             │
  │   Arm I-A (Depth k=100)              ███████████████████████████████ 31.00% (+1.00 pp) │
  │   Arm I-B (Disclosure-Aware k=50)    ██████████████████████████████ 30.00% (+0.00 pp) │
  │                                                                                        │
  │ Recall@10                                                                              │
  │   Experiment F Reference Baseline    ████████████████████ 20.00%                       │
  │   Arm I-A (Depth k=100)              ████████████████████ 20.00% (+0.00 pp)            │
  │   Arm I-B (Disclosure-Aware k=50)    ██████████████████ 18.00% (-2.00 pp)              │
  │                                                                                        │
  │ Mean Reciprocal Rank (MRR)                                                             │
  │   Experiment F Reference Baseline    ███████████ 0.1050                                │
  │   Arm I-A (Depth k=100)              ██████████ 0.0966 (-0.0083)                       │
  │   Arm I-B (Disclosure-Aware k=50)    ██████████ 0.0940 (-0.0110)                       │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Conclusions & Governance Recommendation

### 7.1 Empirical Verdict
1. **Arm I-A (Depth Expansion):** Increasing BM25 and Dense FAISS depth from 50 to 100 did not solve the candidate pool bottleneck. 75% of depth-truncated chunks failed to survive RRF fusion due to the single-path penalty. The small +1.00 pp pool gain was offset by MRR dilution and is statistically indistinguishable from chance ($p = 1.000$).
2. **Arm I-B (Disclosure Routing):** Introducing unsupervised disclosure routing with a guaranteed floor ($\ge 40$ slots) prevented catastrophic pool turnover (97.3% Jaccard retained), but resulted in **net-neutral candidate pool recall (30.00%)** and a **net decline in Recall@10 (18.00% vs. 20.00%) and MRR (0.0940 vs. 0.1050)** due to displacing valid MD&A baseline evidence.
3. **Core Architectural Insight:** Heuristic keyword/regex routing at query time cannot reliably distinguish between a question requiring primary financial statements, MD&A segment margin tables, or debt footnotes. When heuristic routing injects statement chunks, it displaces borderline MD&A chunks that the downstream cross-encoder would have ranked in the Top 10.

### 7.2 Definitive Recommendation
> **Do NOT promote Arm I-A or Arm I-B to production.**  
> **Retain Frozen Experiment F as the verified reference baseline.**
>
> Neither depth expansion nor heuristic disclosure candidate injection produced a statistically defensible improvement over the frozen Experiment F baseline on the 100-question held-out benchmark.
>
> **Future Architectural Direction:**  
> The remaining 70% recall gap on 440-page Form 10-K filings cannot be bridged by heuristic candidate injection or deeper RRF fusion. A meaningful breakthrough will require:
> 1. **Trained Dense Section Encoders:** Joint document-section hierarchical embeddings that represent filing structure natively during vector indexing.
> 2. **Learned Query-to-Filing Routing:** A lightweight supervised classifier trained on financial query taxonomy rather than keyword regexes.

---

## 8. Status & Next Steps

- **Experiment Status:** Both Arm I-A and Arm I-B are fully completed, audited, and documented.
- **Repository Integrity:** All 7 protected baseline modules and checkpoints remain 100% verified and untouched. Production code under `src/retrieval/` has **zero edits**.
- **Awaiting User Action:** Awaiting your review of this final Arm I-B evaluation report.
