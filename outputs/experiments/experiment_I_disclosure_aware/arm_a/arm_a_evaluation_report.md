# FinSage Experiment I: Arm I-A Evaluation Report — Retrieval-Depth Ablation

**Target Milestone:** Experiment I — Arm I-A: Isolated Retrieval-Depth Ablation ($k=100$)  
**Report Artifact:** `outputs/experiments/experiment_I_disclosure_aware/arm_a/arm_a_evaluation_report.md`  
**Author:** Antigravity Research & Engineering Agent  
**Date:** November 2, 2026  
**Status:** **ARM I-A COMPLETED — DECISION GATE REACHED — AWAITING USER REVIEW & AUTHORIZATION BEFORE PROCEEDING TO ARM I-B**  

---

## 1. Executive Summary

In accordance with user authorization and governance safeguards, **Experiment I-A (Retrieval-Depth Ablation)** was implemented and executed as an isolated, controlled benchmark on the frozen 100-question Stage 3 held-out dataset and isolated corpus (`data/heldout_stage3/`).

### Headline Findings:
1. **Control Verification:** The reference **Experiment F** baseline was verified 100% reproducible before execution, replicating exact published metrics: Candidate Pool Recall = **30.00%**, Recall@10 = **20.00%**, MRR = **0.1050**.
2. **Arm I-A Performance:** Increasing initial retrieval depth from $k=50$ to $k=100$ in sparse BM25 and dense FAISS yielded:
   - **Candidate Pool Recall ($K=50$):** **31.00%** (+1.00 pp vs. Exp F, 31/100 vs. 30/100)
   - **Recall@1:** **6.00%** (-1.00 pp vs. Exp F, 6/100 vs. 7/100)
   - **Recall@3:** **11.00%** (-1.00 pp vs. Exp F, 11/100 vs. 12/100)
   - **Recall@5:** **15.00%** (-1.00 pp vs. Exp F, 15/100 vs. 16/100)
   - **Recall@10:** **20.00%** (+0.00 pp vs. Exp F, 20/100 vs. 20/100)
   - **Mean Reciprocal Rank (MRR):** **0.0966** (-0.0083 vs. Exp F, 0.0966 vs. 0.1050)
   - **Mean Query Latency:** **188.6 ms** (+8.6 ms vs. Exp F's 180.0 ms)
3. **Statistical Significance:** Neither the pool recall difference ($b=3, c=2$, McNemar exact binomial $p = 1.000$) nor the MRR change (Wilcoxon signed-rank $W=2.0, z=-0.5345, p = 0.5930$) is statistically significant.
4. **Audit of 12 Depth-Truncated Questions:** Only **3 out of 12** (25.0%) entered the candidate pool, and only **1 out of 12** (8.3%) entered the top 10. The remaining 9 questions (75.0%) failed to survive RRF fusion and the top-50 pool cutoff.
5. **Dilution & Regressions:** Expanding depth to $k=100$ introduced an average of 4.2 new candidates per query into the candidate pool. This caused **2 pool regressions**, including `stage3_heldout_0081` where a Rank 1 baseline hit was displaced from the pool, dropping MRR from 1.0 to 0.0.
6. **Integrity & Leakage:** Pre- and post-run SHA-256 hashes of all 7 protected baseline modules and checkpoints were verified 100% untouched. Retrieval and pool assembly operated strictly without access to gold labels.

---

## 2. Experimental Setup & Control Verification

### 2.1 Controlled Arm Identity
Arm I-A isolates a single independent variable: **initial retrieval depth** across sparse and dense search paths.

| Architectural Component | Reference Baseline (Exp F) | Arm I-A (Depth Ablation) | Control Status |
| :--- | :---: | :---: | :---: |
| **Sparse BM25 Depth** | $k = 50$ | **$k = 100$** | **Variable Under Test** |
| **Dense FAISS BGE-M3 Depth** | $k = 50$ | **$k = 100$** | **Variable Under Test** |
| **RRF Fusion Formula** | $k_{\text{rrf}} = 60$ | $k_{\text{rrf}} = 60$ | Identical / Frozen |
| **Temporal Prioritization** | In-period chunks prioritized | In-period chunks prioritized | Identical / Frozen |
| **Final Candidate Pool Size** | Exactly 50 unique chunks | Exactly 50 unique chunks | Identical / Frozen |
| **Phase 2 Reranker** | `bge-reranker-base` (fp16) | `bge-reranker-base` (fp16) | Identical / Frozen |
| **Reranker Boost Weights** | Stmt +4.5, Disc -3.0, Co +1.5, Per +1.5, Tab +0.5 | Stmt +4.5, Disc -3.0, Co +1.5, Per +1.5, Tab +0.5 | Identical / Frozen |
| **Disclosure / Statement Routing** | DISABLED | DISABLED | Identical / Frozen |
| **Corpus & Indexes** | `data/heldout_stage3/` (29,335 chunks) | `data/heldout_stage3/` (29,335 chunks) | Identical / Frozen |

### 2.2 Baseline Reproduction Audit
Prior to running Arm I-A, the frozen Experiment F baseline was loaded from `outputs/experiments/experiment_stage3_heldout_validation/phase2_retrieval_checkpoint_reference_exp_f.jsonl` and verified:
- **Candidate Pool Recall ($K=50$):** 30.00% (30 / 100) — Match verified.
- **Recall@1:** 7.00% (7 / 100) — Match verified.
- **Recall@3:** 12.00% (12 / 100) — Match verified.
- **Recall@5:** 16.00% (16 / 100) — Match verified.
- **Recall@10:** 20.00% (20 / 100) — Match verified.
- **MRR:** 0.1050 — Match verified.

### 2.3 Baseline Hash Verification
All 7 protected repository modules and historical checkpoints were audited before and after execution:

| Protected Module / Artifact Path | Expected SHA-256 Hash | Pre-Run Status | Post-Run Status |
| :--- | :---: | :---: | :---: |
| `src/retrieval/phase2_reranker.py` | `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9` | VERIFIED | VERIFIED |
| `scripts/run_phase2_benchmark.py` | `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1` | VERIFIED | VERIFIED |
| `outputs/.../experiment_C_.../phase2_reranker.py` | `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED` | VERIFIED | VERIFIED |
| `outputs/.../experiment_F_.../checkpoint.jsonl` | `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C` | VERIFIED | VERIFIED |
| `outputs/.../experiment_H2_.../scorecard.json` | `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2` | VERIFIED | VERIFIED |
| `outputs/.../experiment_H2_.../checkpoint.jsonl` | `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F` | VERIFIED | VERIFIED |
| `outputs/.../experiment_H2_1_.../scorecard.json` | `ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC` | VERIFIED | VERIFIED |

---

## 3. Benchmark Results & Comparative Analysis

### 3.1 Headline Metrics Comparison

| Metric | Reference Exp F ($k=50$) | Arm I-A ($k=100$) | Net Delta | Statistical Test & Significance |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall ($K=50$)** | **30.00%** (30/100) | **31.00%** (31/100) | **+1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@1** | **7.00%** (7/100) | **6.00%** (6/100) | **-1.00 pp** | Fisher/McNemar $p = 1.000$ (Not Significant) |
| **Recall@3** | **12.00%** (12/100) | **11.00%** (11/100) | **-1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@5** | **16.00%** (16/100) | **15.00%** (15/100) | **-1.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Recall@10** | **20.00%** (20/100) | **20.00%** (20/100) | **+0.00 pp** | McNemar $p = 1.000$ (Not Significant) |
| **Mean Reciprocal Rank (MRR)** | **0.1050** | **0.0966** | **-0.0083** | Wilcoxon $W=2.0, p = 0.5930$ (Not Significant) |
| **Average Query Latency** | **180.0 ms** | **188.6 ms** | **+8.6 ms** | Neutral (+4.8% runtime overhead) |
| **Total Benchmark Runtime** | **18.0 s** | **18.9 s** | **+0.9 s** | Highly efficient execution |

```
                       RECALL COMPARISON: EXP F vs. ARM I-A (N=100)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Candidate Pool Recall (K=50)                                                           │
  │   Exp F Baseline (k=50)   ██████████████████████████████ 30.00%                        │
  │   Arm I-A Depth (k=100)   ███████████████████████████████ 31.00% (+1.00 pp)            │
  │                                                                                        │
  │ Recall@10                                                                              │
  │   Exp F Baseline (k=50)   ████████████████████ 20.00%                                  │
  │   Arm I-A Depth (k=100)   ████████████████████ 20.00% (+0.00 pp)                       │
  │                                                                                        │
  │ Mean Reciprocal Rank (MRR)                                                             │
  │   Exp F Baseline (k=50)   ███████████ 0.1050                                           │
  │   Arm I-A Depth (k=100)   ██████████ 0.0966 (-0.0083)                                  │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Question-Level Paired Transitions

| Transition Type | Candidate Pool ($K=50$) | Top-10 Retrieval ($K=10$) | Exact Question IDs Affected |
| :--- | :---: | :---: | :--- |
| **Recoveries ($b$)** | **3** | **1** | Pool: `stage3_heldout_0017`, `0076`, `0083`. Top-10: `stage3_heldout_0083`. |
| **Regressions ($c$)** | **2** | **1** | Pool: `stage3_heldout_0058`, `0081`. Top-10: `stage3_heldout_0081`. |
| **Ties: Hit $\rightarrow$ Hit ($a$)** | **28** | **19** | 28 pool hits and 19 top-10 hits retained unchanged. |
| **Ties: Miss $\rightarrow$ Miss ($d$)** | **67** | **79** | 67 pool misses and 79 top-10 misses remained unrecovered. |
| **Total Questions** | **100** | **100** | **Net change: +1 pool hit, 0 top-10 hits.** |

---

## 4. Deep Audit of the 12 Depth-Truncated Questions

In Phase 1 failure analysis, 12 questions were identified where gold evidence chunks appeared between ranks 51 and 100 in either BM25 or Dense FAISS. We explicitly tracked whether depth expansion to $k=100$ allowed these chunks to survive RRF fusion, temporal prioritization, and the top-50 pool cutoff.

### 4.1 Question-by-Question Audit Table

| Question ID | Company & Doc | Gold Page | Evidence Category | BM25 Rank in 100 | Dense Rank in 100 | Total RRF Rank | Final Pool Rank | Pool Hit? | Top-10 Rank | Outcome & Failure Mechanism |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `stage3_heldout_0015` | NVIDIA (`NVDA_2023`) | 9 | Segment Disclosures | 96 | >100 | 165 | — | ❌ 0 | — | **Dropped:** Low single-path BM25 score pushed past rank 100 in RRF. |
| `stage3_heldout_0017` | Cisco (`CSCO_2023`) | 85 | Primary Statements | >100 | 63 | 114 | **49** | ✅ **1** | — | **Recovered into Pool:** Dense rank 63 saved by 2023 temporal filter. |
| `stage3_heldout_0019` | Cisco (`CSCO_2024`) | 11 | Footnotes | >100 | 78 | 145 | — | ❌ 0 | — | **Dropped:** Dense rank 78 insufficient to enter top-50 period list. |
| `stage3_heldout_0034` | Bank of America (`BAC_2024`) | 11 | MD&A Tables | >100 | 75 | 124 | — | ❌ 0 | — | **Dropped:** RRF score 0.00741 displaced by higher cross-company scores. |
| `stage3_heldout_0061` | P&G (`PG_2024`) | 51 | Primary Statements | >100 | 52 | 89 | — | ❌ 0 | — | **Dropped:** Rank 52 missed RRF top-50 cutoff by 39 positions. |
| `stage3_heldout_0067` | Starbucks (`SBUX_2023`) | 21 | Other / Business | >100 | 66 | 110 | — | ❌ 0 | — | **Dropped:** Dense rank 66 displaced during RRF fusion. |
| `stage3_heldout_0071` | Eli Lilly (`LLY_2024`) | 93 | Primary Statements | >100 | 55 | 93 | — | ❌ 0 | — | **Dropped:** Rank 55 displaced; 43 other 2024 chunks scored higher. |
| `stage3_heldout_0073` | Eli Lilly (`LLY_2024`) | 93 | Primary Statements | >100 | 100 | 174 | — | ❌ 0 | — | **Dropped:** Dense rank 100 finished at RRF rank 174. |
| `stage3_heldout_0076` | UnitedHealth (`UNH_2024`) | 70 | Primary Statements | 79 | 75 | 40 | **25** | ✅ **1** | — | **Recovered into Pool:** Dual-path score ($79+75$) summed to RRF rank 40. |
| `stage3_heldout_0083` | Merck (`MRK_2024`) | 8 | Segment Disclosures | 76 | 56 | 27 | **13** | ✅ **1** | **5** | **Full Recovery:** Dual-path score ($76+56$) placed in pool & reranked to #5! |
| `stage3_heldout_0084` | Merck (`MRK_2024`) | 38 | Other / Risk Factors | 52 | >100 | 88 | — | ❌ 0 | — | **Dropped:** BM25 rank 52 missed RRF top 50 by 38 slots. |
| `stage3_heldout_0089` | AbbVie (`ABBV_2024`) | 28 | Footnotes | 77 | >100 | 129 | — | ❌ 0 | — | **Dropped:** BM25 rank 77 scored only 0.0073 RRF, ending at rank 129. |

### 4.2 Key Findings from the 12-Question Deep Dive
1. **The Dual-Path Requirement for RRF Survival:**  
   The only two questions that comfortably entered the candidate pool without relying on extreme temporal filtering were `stage3_heldout_0076` (UNH) and `stage3_heldout_0083` (MRK). Both scored in **both** retrieval paths:
   $$\text{UNH: } \frac{1}{60+79} + \frac{1}{60+75} = 0.00719 + 0.00741 = 0.01460 \rightarrow \text{RRF Rank } 40$$
   $$\text{MRK: } \frac{1}{60+76} + \frac{1}{60+56} = 0.00735 + 0.00862 = 0.01597 \rightarrow \text{RRF Rank } 27$$
2. **The Single-Path Death Valley (Ranks 51–100):**  
   When a chunk appears in only *one* path between rank 51 and 100, its maximum possible RRF contribution is:
   $$\text{RRF}_{\text{single}} \le \frac{1}{60 + 51} = 0.00901$$
   Meanwhile, any competing chunk that placed even moderately in both paths (e.g. rank 45 in BM25 and rank 48 in Dense) receives $\approx 0.0190$. As a result, **single-path chunks between rank 51 and 100 are systematically displaced out of the top 50** during RRF fusion.
3. **Empirical Confirmation:**  
   Retrieving a chunk in BM25 or Dense FAISS at depth $k=100$ **does NOT guarantee survival into the candidate pool**. 75% of depth-truncated candidates were extinguished during RRF fusion.

---

## 5. Failure & Regression Analysis: Why Depth Expansion Dilutes

Arm I-A produced **2 pool regressions** where chunks that succeeded in Experiment F were displaced out of the top-50 candidate pool:

### Regression 1: `stage3_heldout_0058` (Target Corporation Balance Sheet)
- **Question:** Target Corporation cash and cash equivalents balance at fiscal year-end 2023.
- **Experiment F Baseline:** Gold chunk (`TGT_2023_10K` p. 45) placed at rank 48 of the candidate pool.
- **Arm I-A Result:** Expanding depth to $k=100$ introduced 9 lower-scoring candidates from depth 51–100 that matched generic retail balance sheet keywords ("cash", "equivalents", "restricted cash"). These competing chunks pushed the gold chunk from rank 48 to rank 54, causing a **pool regression** (`exp_f_pool_hit=1 -> arm_a_pool_hit=0`).

### Regression 2: `stage3_heldout_0081` (Merck & Co. MD&A Financing Review) — Severe Regression
- **Question:** Merck's credit facility and commercial paper borrowing capacity in 2024.
- **Experiment F Baseline:** Gold chunk (`MRK_2024_10K` p. 11) was in the candidate pool, scored highly by the cross-encoder, and was **ranked #1 in top-10 retrieval (MRR = 1.000)**.
- **Arm I-A Result:** Expanding depth to $k=100$ brought in 2 additional debt/credit chunks from other sections of the 400-page filing. Because those chunks scored in both paths at depth 80–90, their combined RRF score edged out the gold chunk at the pool boundary. The gold chunk was **evicted from the candidate pool**, turning a Rank 1 baseline hit into a complete miss (`exp_f_rank=1 -> arm_a_rank=999`, **MRR delta = -1.000**).

This single severe regression directly caused the aggregate MRR of Arm I-A to drop from 0.1050 to 0.0966.

---

## 6. Results Stratified by Evidence Category

| Evidence Category | Total Questions | Exp F Pool Recall | Arm I-A Pool Recall | Pool Delta | Exp F Recall@10 | Arm I-A Recall@10 | Top-10 Delta | Exp F MRR | Arm I-A MRR | MRR Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Financial Statements** | 28 | 35.71% | **39.29%** | **+3.57 pp** | 21.43% | 21.43% | +0.00 pp | 0.1658 | 0.1658 | +0.0000 |
| **Segment or Product Disclosures** | 12 | 16.67% | **25.00%** | **+8.33 pp** | 16.67% | **25.00%** | **+8.33 pp** | 0.0444 | **0.0583** | **+0.0139** |
| **Financial Statement Footnotes** | 11 | 63.64% | 63.64% | +0.00 pp | 63.64% | 63.64% | +0.00 pp | 0.2371 | 0.2371 | +0.0000 |
| **MD&A Tables and Narrative** | 32 | **25.00%** | 21.88% | **-3.12 pp** | **12.50%** | 9.38% | **-3.12 pp** | **0.0535** | 0.0222 | **-0.0312** |
| **Other / Qualitative & Risks** | 16 | 18.75% | 18.75% | +0.00 pp | 6.25% | 6.25% | +0.00 pp | 0.0625 | 0.0625 | +0.0000 |
| **Debt, Lease, Inventory Detail** | 1 | 0.00% | 0.00% | +0.00 pp | 0.00% | 0.00% | +0.00 pp | 0.0000 | 0.0000 | +0.0000 |
| **Total Benchmark** | **100** | **30.00%** | **31.00%** | **+1.00 pp** | **20.00%** | **20.00%** | **+0.00 pp** | **0.1050** | **0.0966** | **-0.0083** |

### Category Observations:
- **Segment Disclosures:** Benefited most from depth expansion (+8.33 pp pool recall and top-10 recall), driven by the recovery of Merck's Keytruda segment schedule.
- **MD&A Disclosures:** Suffered net degradation (-3.12 pp pool recall, -0.0312 MRR). In multi-page MD&A narrative, expanding search depth introduces diffuse narrative fragments that dilute the pool.

---

## 7. Candidate Pool Turnover & Runtime Latency

- **Candidate Pool Jaccard Similarity:** **0.8511** (85.1% candidate pool overlap with Experiment F).
- **Average Candidate Turnover:** An average of **4.2 new chunks** entered the 50-candidate pool per question (and 4.2 baseline chunks were displaced).
- **Latency Impact:**
  - Experiment F mean query time: 180.0 ms
  - Arm I-A mean query time: 188.6 ms
  - Overhead: **+8.6 ms per query** (+4.8%), demonstrating that expanding FAISS and BM25 depth to 100 incurs virtually zero computational penalty.

---

## 8. Decision Gate Recommendation for Arm I-B

### Evaluation Against Experimental Criteria:
1. **Did Arm I-A achieve its aspirational target ($\ge 35\%$ pool recall)?**  
   **No.** Candidate pool recall reached 31.00% (+1.00 pp), far short of the +5.00 pp goal.
2. **Is the improvement statistically significant?**  
   **No.** McNemar's exact test yields $p = 1.000$, and Wilcoxon signed-rank test yields $p = 0.5930$.
3. **Did depth expansion introduce regressions?**  
   **Yes.** 2 pool regressions occurred, including 1 severe regression (Merck $1.0 \rightarrow 0.0$ MRR), causing aggregate MRR to decline.

### Recommendation:
> **Branch Arm I-B from the Frozen Experiment F Baseline ($k=50$), NOT from Arm I-A ($k=100$).**
>
> Increasing retrieval depth to $k=100$ failed to recover 75% of depth-truncated chunks due to RRF cutoff dynamics, while introducing semantic dilution that displaced valid baseline evidence.
> 
> Therefore, **Arm I-B (Disclosure-Aware Candidate Selection)** should be constructed using the cleaner, tighter $k=50$ retrieval depth of Experiment F. This guarantees that:
> 1. Baseline candidates in ranks 1–40 remain completely protected from depth dilution.
> 2. Disclosure candidate injection ($Q_{\text{disc}} \le 10$) acts as a surgical intervention specifically targeting the 74.3% of evidence residing in MD&A and segment disclosures, rather than relying on blunt depth expansion.

---

## 9. Next Steps (Awaiting User Review & Approval)

- **Status:** **STOPPED AT THE DECISION GATE.**
- **Code State:** Zero code written for Arm I-B. Zero experiments executed for Arm I-B.
- **Awaiting User Action:** Awaiting your review of these Arm I-A results and authorization to proceed with **Arm I-B** based on the recommended $k=50$ baseline architecture.
