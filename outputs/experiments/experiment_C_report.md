# FinSage Experiment C: Full-Universe Company Intent Detection & Boosting Report

**Document File:** `outputs/experiments/experiment_C_report.md`  
**Execution Timestamp:** November 02, 2026  
**Governing Standard:** Controlled Empirical RAG Ablation & Synthesis  
**Reference Paper:** *“FinSage: A Multi-aspect RAG System for Financial Filings Question Answering”* (Wang et al., arXiv:2504.14493, ACM CIKM 2025)  
**Workspace Root:** `C:\Users\user\Downloads\finsage-main`  
**Evaluation Baseline:** Experiment AB Combined ($K_{\text{pool}}=50$, BM25 $k=50$, Dense FAISS $k=50$, Statement Routing Fix)

---

## Executive Summary

Experiment C evaluates the impact of expanding **company intent detection and boosting** from the historical hardcoded 6 companies to the **full 32-company universe** of the FinanceBench benchmark (84 SEC filings, 32 public companies, 9 GICS sectors).

All other retrieval and reranking parameters were kept **strictly identical to Experiment AB**:
- Candidate pool size: $K_{\text{pool}} = 50$
- BM25 and Dense FAISS depths: $k = 50$
- RRF constant: $k = 60$
- Statement routing regex: active (covering registered debt and primary financial statements)
- Reranker weights: statement boost $+4.5$, penalty $-3.0$, company boost $+1.5$, period boost $+1.5$, table boost $+0.5$
- Cross-encoder model: `BAAI/bge-reranker-base` with 512-character input length unchanged

### Headline Findings
1. **MRR Target Exceeded ($\text{MRR} \ge 0.340$):** Mean Reciprocal Rank jumped from **0.3224 in Experiment AB to 0.3513 in Experiment C (+0.0289 gain)**, surpassing the target hypothesis threshold. Over the original baseline (0.2926), the cumulative gain is **+0.0587 (+20.1% relative surge)**.
2. **One-Quarter of Benchmark Now Ranked #1:** Top-tier retrieval precision hit an all-time peak: **Recall@1 reached 25.33% (38/150 questions)**, an increase of **+4.00 pp (+6 net questions) over Experiment AB** and **+7.33 pp (+11 net questions) over baseline**.
3. **Zero Ranking Regressions vs. Experiment AB:** Across all 150 benchmark questions, **13 questions improved in rank and 0 questions regressed** (137 remained unchanged).
4. **Substantial Contamination Reduction:** Cross-company chunk contamination in Top-3 retrieved results dropped from **35.33% (53/150) down to 23.33% (35/150)** at the question level (**-12.00 pp reduction**), and from **18.67% (84/450) down to 10.89% (49/450)** at the chunk level (**-7.78 pp reduction**).
5. **Key Case Progressions:**
   - **American Express Registered Debt (`financebench_id_00476`):** Baseline #999 $\to$ Exp AB #3 $\to$ **Exp C Rank #1**.
   - **3M Registered Debt (`financebench_id_00941`):** Baseline #4 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (Preserved).
   - **Ulta Beauty Registered Debt (`financebench_id_00746`):** Baseline #1 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (Preserved).
   - **American Express Card Retention (`financebench_id_01981`):** Baseline #999 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (Preserved).
   - **Boeing Metrics (`financebench_id_10285` & `financebench_id_00585`):** Both advanced to **Rank #1**.
6. **Target Hypotheses Assessment:**
   - **Hypothesis 1 (MRR $\ge 0.340$):** **MET & EXCEEDED.** Measured **0.3513** (+0.0113 above target).
   - **Hypothesis 2 (Recall@10 $\ge 60.0\%$):** **NARROWLY MISSED.** Measured **57.33% (86/150 questions)**, an improvement of **+1.33 pp (+2 net questions) over AB**, falling short of the 60.0% target by 2.67 pp due to the multi-page ratio ceiling and 512-char cross-encoder truncation identified in the failure analysis.

---

## 1. Safety, Preflight Verification & Isolation

1. **Pre-Execution Artifact Verification:**
   - Prior baseline and experiment artifacts were verified intact and preserved:
     - Baseline: `outputs/phase2/` (Scorecard & Checkpoint)
     - Experiment A: `outputs/experiments/experiment_A_pool60/`
     - Experiment B: `outputs/experiments/experiment_B_routing/`
     - Experiment AB: `outputs/experiments/experiment_AB_combined/`
2. **Dedicated Output Isolation:**
   - All Experiment C outputs were isolated strictly to:
     `outputs/experiments/experiment_C_company_intent/`
     - Scorecard: `outputs/experiments/experiment_C_company_intent/evaluation/phase2_full_scorecard.json`
     - Checkpoint: `outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl`
     - Official Markdown Report: `outputs/experiments/experiment_C_company_intent/reports/PHASE2_OFFICIAL_RESULTS.md`
3. **Fresh Execution Audit:**
   - The output directory was confirmed non-existent prior to benchmark launch. The benchmark evaluated all 150 questions sequentially from scratch in 32.48 seconds with 0 skips, 0 exceptions, and 0 stale checkpoint reads.

---

## 2. Implementation: Full-Universe Company Intent Detection

### 2.1 The Single Controlled Variable
In [`src/retrieval/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/phase2_reranker.py), `detect_query_company()` was updated from the hardcoded 6 companies to the full 32-company universe discovered from `data/financebench/questions.jsonl` metadata:

```python
FINANCEBENCH_COMPANIES = {
    "3M": [r"\b3m\b"],
    "ACTIVISIONBLIZZARD": [r"\bactivision(\s+blizzard)?\b"],
    "ADOBE": [r"\badobe\b"],
    "AES": [r"\baes(\s+corp(oration)?)?\b"],
    "AMAZON": [r"\bamazon\b"],
    "AMCOR": [r"\bamcor\b"],
    "AMD": [r"\bamd\b", r"\badvanced\s+micro\s+devices\b"],
    "AMERICANEXPRESS": [r"\bamerican\s+express\b", r"\bamex\b"],
    "AMERICANWATERWORKS": [r"\bamerican\s+water(\s+works)?\b"],
    "BESTBUY": [r"\bbest\s+buy\b"],
    "BLOCK": [r"\bblock\s*\(\s*formerly\b", r"\bblock\s+inc\.?\b", r"\bsquare\s+inc\.?\b", r"\bblock['’]s\b"],
    "BOEING": [r"\bboeing\b"],
    "COCACOLA": [r"\bcoca[\s-]*cola\b", r"\bcoke\b"],
    "CORNING": [r"\bcorning\b"],
    "COSTCO": [r"\bcostco\b"],
    "CVSHEALTH": [r"\bcvs(\s+health)?\b"],
    "FOOTLOCKER": [r"\bfoot\s+locker\b"],
    "GENERALMILLS": [r"\bgeneral\s+mills\b"],
    "JOHNSON": [r"\bjohnson\s+(&|and)\s+johnson\b", r"\bjnj\b"],
    "JPMORGAN": [r"\bjpmorgan(\s+chase)?\b", r"\bjp\s*morgan\b", r"\bjpm\b"],
    "KRAFTHEINZ": [r"\bkraft\s+heinz\b", r"\bkraft\b"],
    "LOCKHEEDMARTIN": [r"\blockheed(\s+martin)?\b"],
    "MGMRESORTS": [r"\bmgm(\s+resorts)?\b"],
    "MICROSOFT": [r"\bmicrosoft\b", r"\bmsft\b"],
    "NETFLIX": [r"\bnetflix\b"],
    "NIKE": [r"\bnike\b"],
    "PAYPAL": [r"\bpaypal\b"],
    "PEPSICO": [r"\bpepsico\b", r"\bpepsi\b"],
    "PFIZER": [r"\bpfizer\b"],
    "ULTABEAUTY": [r"\bulta(\s*beauty)?\b"],
    "VERIZON": [r"\bverizon\b"],
    "WALMART": [r"\bwalmart\b", r"\bwal-mart\b"],
}
```

### 2.2 Disambiguation & Safeguards Against False Positives
To prevent false-positive collisions:
1. **No bare acronyms for non-benchmark companies:** Acronyms like `CAT`, `GM`, and `WM` were omitted. General Mills only matches `r"\bgeneral\s+mills\b"`.
2. **Disambiguated Block regex:** Avoided bare `\bblock\b` (which falsely matched sentences like "block on credit card"). Required explicit context: `Block (formerly...`, `Block's`, `Block Inc.`, or `Square Inc.`.
3. **Preflight Unit Test Suite:** [`tests/test_company_detection.py`](file:///c:/Users/user/Downloads/finsage-main/tests/test_company_detection.py) was implemented with:
   - 32 positive cases (1 per company) $\to$ **100% Passed**.
   - 16 alias and abbreviation cases (`AMEX`, `JnJ`, `JPM`, `Pepsi`, `Coke`, `MSFT`) $\to$ **100% Passed**.
   - 12 negative and ambiguous cases (avoiding CAT, GM, WM, bare block, generic finance queries) $\to$ **100% Passed**.
   - 4 cross-company collision checks (Amcor vs AMD, Amex vs American Water Works) $\to$ **100% Passed**.
   - Intent routing unit test suite [`tests/test_intent_routing.py`](file:///c:/Users/user/Downloads/finsage-main/tests/test_intent_routing.py) $\to$ **11/11 Passed**.

---

## 3. Comprehensive 5-Way Benchmark Comparison ($N=150$)

| Evaluation Metric | Baseline | Exp A (Pool 60) | Exp B (Routing) | Exp AB (Combined) | **Exp C (Company Intent)** | **Delta vs Exp AB** | **Delta vs Baseline** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall@1** | 18.00% (27/150) | 20.00% (30/150) | 19.33% (29/150) | 21.33% (32/150) | **25.33% (38/150)** | **+4.00 pp** | **+7.33 pp** |
| **Recall@3** | 37.33% (56/150) | 39.33% (59/150) | 38.67% (58/150) | 40.67% (61/150) | **42.00% (63/150)** | **+1.33 pp** | **+4.67 pp** |
| **Recall@5** | 45.33% (68/150) | 45.33% (68/150) | 46.00% (69/150) | 46.00% (69/150) | **48.00% (72/150)** | **+2.00 pp** | **+2.67 pp** |
| **Recall@10** | 52.00% (78/150) | 56.67% (85/150) | 52.67% (79/150) | 56.00% (84/150) | **57.33% (86/150)** | **+1.33 pp** | **+5.33 pp** |
| **MRR (Mean Reciprocal Rank)** | 0.2926 | 0.3110 | 0.3042 | 0.3224 | **0.3513** | **+0.0289** | **+0.0587** |
| **Candidate-Pool Recall** | 52.00%* | **71.33% (107/150)** | 59.33% (89/150) | 68.67% (103/150) | **68.67% (103/150)** | **+0.00 pp** | **+16.67 pp** |
| **Cross-Company Contamination (Q-Level)** | ~40.0% | ~42.0% | ~38.0% | 35.33% (53/150) | **23.33% (35/150)** | **-12.00 pp** | **-16.67 pp** |
| **Cross-Company Contamination (Chunk-Level)**| ~22.0% | ~24.0% | ~20.0% | 18.67% (84/450) | **10.89% (49/450)** | **-7.78 pp** | **-11.11 pp** |
| **Financials Sector MRR** | 0.0962 | 0.1239 | 0.1731 | 0.1923 | **0.2500** | **+0.0577** | **+0.1538** |
| **Batch Retrieval Runtime** | 43.24 s | 37.38 s | 23.75 s | 30.73 s | **32.48 s** | **+1.75 s** | **-10.76 s** |

*\*Note: Baseline candidate pool recall represents post-reranking Top-10 lower bound, as pre-reranking pools were not logged in historical baseline.*

---

## 4. Cross-Company Contamination Analysis

To evaluate contamination rigorously, exact definitions and consistent denominators were enforced:

### Definitions:
- **Target Company:** The canonical corporation associated with the question (`questions.jsonl`).
- **Retrieved Chunk Company:** The corporation determined from the retrieved chunk's `doc_name` via `extract_chunk_company_and_period()`.
- **Contaminated Chunk:** Any retrieved chunk whose corporation does not match the question's target corporation.
- **Question-Level Contamination Rate:** Fraction of questions where $\ge 1$ chunk in the Top-3 retrieved list comes from the wrong company.
  - Denominator: All 150 FinanceBench questions.
- **Chunk-Level Contamination Rate:** Total contaminated chunks divided by all Top-3 retrieved chunk slots.
  - Denominator: $150 \times 3 = 450$ chunk slots.

### Results:
| Metric Dimension | Experiment AB | Experiment C | Absolute Change | Relative Improvement |
|---|:---:|:---:|:---:|:---:|
| **Question-Level Contamination** | 35.33% (53/150) | **23.33% (35/150)** | **-12.00 pp** | **-34.0% relative reduction** |
| **Chunk-Level Contamination** | 18.67% (84/450) | **10.89% (49/450)** | **-7.78 pp** | **-41.7% relative reduction** |

By applying the $+1.5$ company boost to all 32 companies, true company chunks successfully overtook cross-company distractors in dozens of queries, cutting chunk contamination nearly in half.

---

## 5. Per-Question Movement & Deep Dive

### 5.1 Question Classification Across Runs
- **Exp C vs Exp AB:** **13 Improved**, **0 Regressed**, **137 Unchanged**.
- **Exp C vs Baseline:** **25 Improved**, 21 Regressed in relative rank, 104 Unchanged.

### 5.2 The 13 Questions Improved in Experiment C vs Experiment AB
All 13 questions that shifted moved upward in rank, with 6 jumping directly to **Rank #1**:

| # | Question ID | Target Company | GICS Sector | Exp AB Rank | Exp C Rank | MRR Gain ($\Delta \text{RR}$) | Query Topic |
|:---:|---|---|---|:---:|:---:|:---:|---|
| 1 | `financebench_id_10285` | **Boeing** | Industrials | #4 | **#1** | **+0.7500** | Financial metric calculation |
| 2 | `financebench_id_00476` | **American Express** | Financials | #3 | **#1** | **+0.6667** | Registered debt securities |
| 3 | `financebench_id_07661` | **Block** | Info Tech | #3 | **#1** | **+0.6667** | FY2020 cash flow from operations |
| 4 | `financebench_id_00585` | **Boeing** | Industrials | #2 | **#1** | **+0.5000** | Effective tax rate FY22 vs FY21 |
| 5 | `financebench_id_10136` | **General Mills** | Consumer Staples | #2 | **#1** | **+0.5000** | Income statement metric calculation |
| 6 | `financebench_id_04458` | **Netflix** | Comm Services | #2 | **#1** | **+0.5000** | Financial metric calculation |
| 7 | `financebench_id_00407` | **MGM Resorts** | Consumer Discr | #4 | **#2** | **+0.2500** | Short-term debt investments |
| 8 | `financebench_id_00724` | **Pfizer** | Health Care | #999 | **#9** | **+0.1111** | Q2 2023 geographic revenue drop |
| 9 | `financebench_id_00678` | **Boeing** | Industrials | #7 | **#4** | **+0.1071** | Gross margin profile trend |
| 10 | `financebench_id_06272` | **Coca-Cola** | Consumer Staples | #999 | **#10** | **+0.1000** | FY2022 dividend payout ratio |
| 11 | `financebench_id_00720` | **American Express** | Financials | #6 | **#4** | **+0.0833** | FY2022 gross margin change |
| 12 | `financebench_id_01328` | **PepsiCo** | Consumer Staples | #10 | **#6** | **+0.0667** | Restructuring costs in income stmt |
| 13 | `financebench_id_01487` | **Johnson & Johnson** | Health Care | #6 | **#5** | **+0.0333** | Q2 2023 net earnings % of sales |

### 5.3 Tracking the 19 Category B Candidate-Pool Demotions
In the failure analysis of Experiment AB, 19 questions were identified where gold evidence reached the candidate pool ($K_{\text{pool}}=50$) but was demoted to Rank $> 10$ by the reranker.
- **Result in Experiment C:** **2 out of 19 questions** successfully recovered into the Top 10:
  - `financebench_id_00724` (Pfizer): climbed from #999 to **Rank #9**.
  - `financebench_id_06272` (Coca-Cola): climbed from #999 to **Rank #10**.
- **Remaining Bottleneck for Other 17 Questions:** For the remaining 17 questions, company boosting alone was insufficient because:
  1. Competitor chunks from the *same* company (e.g. prior fiscal years or narrative MD&A) also received the company boost.
  2. The 512-character input truncation prevented the cross-encoder from seeing quantitative evidence located deeper in the chunk.

### 5.4 Regression Check on Key Previously Solved Queries
All critical queries improved or stabilized in Experiment AB were verified in Experiment C:
- **3M Registered Debt (`financebench_id_00941`):** Base #4 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (0 regression).
- **American Express Registered Debt (`financebench_id_00476`):** Base #999 $\to$ Exp AB #3 $\to$ **Exp C Rank #1** (+2 ranks gained).
- **Ulta Beauty Registered Debt (`financebench_id_00746`):** Base #1 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (0 regression).
- **American Express Card Retention (`financebench_id_01981`):** Base #999 $\to$ Exp AB #1 $\to$ **Exp C Rank #1** (0 regression).
- **3M Net PPNE (`financebench_id_04672`):** Base #999 $\to$ Exp AB #3 $\to$ **Exp C Rank #3** (0 regression).
- **3M Liquidity Profile (`financebench_id_00807`):** Base #4 $\to$ Exp AB #4 $\to$ **Exp C Rank #4** (0 regression).

---

## 6. Sector Breakdown (Experiment AB vs Experiment C)

| Sector | Question Count | Exp AB MRR | Exp C MRR | Delta MRR | Exp AB Recall@10 | Exp C Recall@10 | Delta Recall@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Financials** | 13 | 0.1923 | **0.2500** | **+0.0577** | 30.8% | 30.8% | +0.0 pp |
| **Industrials** | 19 | 0.2487 | **0.3202** | **+0.0714** | 57.9% | 57.9% | +0.0 pp |
| **Information Technology** | 22 | 0.3914 | **0.4217** | **+0.0303** | 54.5% | 54.5% | +0.0 pp |
| **Communication Services** | 9 | 0.2407 | **0.2963** | **+0.0556** | 44.4% | 44.4% | +0.0 pp |
| **Consumer Staples** | 23 | 0.5152 | **0.5442** | **+0.0290** | 73.9% | **78.3%** | **+4.3 pp** |
| **Health Care** | 19 | 0.2618 | **0.2694** | **+0.0076** | 57.9% | **63.2%** | **+5.3 pp** |
| **Consumer Discretionary** | 30 | 0.3850 | **0.3934** | **+0.0083** | 66.7% | 66.7% | +0.0 pp |
| **Materials** | 9 | 0.0716 | **0.0716** | +0.0000 | 33.3% | 33.3% | +0.0 pp |
| **Utilities** | 6 | 0.2222 | **0.2222** | +0.0000 | 33.3% | 33.3% | +0.0 pp |

The largest MRR surges occurred in **Industrials (+0.0714)**, **Financials (+0.0577)**, and **Communication Services (+0.0556)**, where Boeing, American Express, Block, and Netflix chunks were boosted over generic competitors.

---

## 7. Target Hypothesis Evaluation

| Target Hypothesis | Target Threshold | Measured Outcome | Absolute Margin | Empirical Verdict |
|---|:---:|:---:|:---:|:---:|
| **Hypothesis 1 (MRR)** | $\text{MRR} \ge 0.340$ | **0.3513** | **+0.0113 above target** | ✅ **MET & EXCEEDED** (+0.0289 over AB, +0.0587 over Baseline) |
| **Hypothesis 2 (Recall@10)** | $\text{Recall@10} \ge 60.0\%$ | **57.33%** | -2.67 pp below target | ⚠️ **Narrowly Missed** (+1.33 pp over AB, +5.33 pp over Baseline) |

### Engineering Verdict:
Experiment C represents a major breakthrough in retrieval precision. By addressing the company-detection blind spot with a single controlled variable, MRR increased to **0.3513**, and Recall@1 reached **25.33%** (more than 1 in 4 questions ranked gold evidence at #1).

The remaining gap to 60% Recall@10 is driven by two orthogonal constraints identified in our failure analysis:
1. **Multi-page financial statements (Category E):** Single-chunk retrieval cannot bridge the gap for cross-statement ratio calculations.
2. **512-character cross-encoder truncation:** Truncation obscures numerical tables in candidate chunks. Extending input length to 1,024 characters is the natural next step.

---

## 8. Artifact Locations & Verification Checklist

- [x] **Benchmark Completeness:** All 150 questions evaluated (0 skips, 0 errors, 0 missing records).
- [x] **Zero Regressions:** 13 questions improved over AB, 0 regressed.
- [x] **Output Isolation:** Saved strictly in dedicated directory `outputs/experiments/experiment_C_company_intent/`.
- [x] **Reproducible Test Suite:** `tests/test_company_detection.py` and `tests/test_intent_routing.py` verified 100% passing.

### Physical Artifact Paths:
- **Experiment C Scorecard:** [`outputs/experiments/experiment_C_company_intent/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/evaluation/phase2_full_scorecard.json)
- **Experiment C Checkpoint:** [`outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl)
- **Experiment C Official Results Report:** [`outputs/experiments/experiment_C_company_intent/reports/PHASE2_OFFICIAL_RESULTS.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_company_intent/reports/PHASE2_OFFICIAL_RESULTS.md)
- **Comprehensive Evaluation Report:** [`outputs/experiments/experiment_C_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_C_report.md)
