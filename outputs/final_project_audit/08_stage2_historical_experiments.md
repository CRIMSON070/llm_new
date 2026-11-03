# FinSage Stage 2 & Historical Experiments Audit

**Document:** `outputs/final_project_audit/08_stage2_historical_experiments.md`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Governing Standard:** Rigorous Separation of Historical Ad-Hoc Trials from Official Stage 2 (Read-Only)  

---

## 1. Clear Demarcation: Historical Phase 9C vs. Official Stage 2

A critical requirement of this audit is untangling exploratory historical experiments from verified production milestones:

```
                                  ┌── Historical Phase 9C (Oct 11, 2026): EXPLORATORY / AD-HOC
                                  │   - Uncalibrated +5.5 manual boost weights on base cross-encoder
                                  │   - Hardcoded regex dictionary expander (financial_expander.py)
                                  │   - Severe false positive on Q3 (hallucinated Common Stock)
Pipeline Evolutionary Branches ───┤   - Verdict: CANNOT qualify as official Stage 2
                                  │
                                  └── Official Stage 2 Production (Oct 31, 2026): SYSTEMATIC ADVANCEMENT
                                      - Table Header Propagation at ingestion (2,902 chunks enriched)
                                      - Calibrated Statement Routing (+3.5) & Distractor Penalty (-3.0)
                                      - Deterministic Python Math Sandbox Engine
                                      - Full Benchmark Scaling across all 150 questions & 9 sectors
                                      - Verdict: QUALIFIES as the official Stage 2 submission baseline
```

---

## 2. Audit of Historical Experiment: Phase 9C Heuristic Pipeline

### 1. What Changed Relative to the Original Paper Method
* In the published paper, the cross-encoder reranker scores chunks using pure contextual attention plus a publication-date Time Bonus ($\beta f(\tilde{m}_i)$).
* In Phase 9C, the workspace introduced **four additive heuristic boost weights totaling up to +5.5 points**:
  $$\text{Final Score} = \text{Raw CrossEncoder Score} + \text{Company Boost} (1.5) + \text{Period Boost} (1.5) + \text{Statement Boost} (2.0) + \text{Table Boost} (0.5)$$
* Furthermore, Generative HyDE was replaced by a deterministic regex keyword expander ([`src/retrieval/financial_expander.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/financial_expander.py)).

### 2. Motivation
* In initial dense search, financial statements were frequently outranked by narrative MD&A sections (e.g., environmental CapEx disclosures scoring higher than official GAAP cash flow schedules). The heuristic boosts were created to forcefully override cross-encoder logits and push financial statements to the top.

### 3. Source Code Location
* Reranker Boosting: [`src/retrieval/statement_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/statement_reranker.py).
* Retrieval Execution: [`scripts/run_retrieval_phase9c.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_retrieval_phase9c.py).
* Downstream Generation: [`scripts/run_generation_phase9c.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_generation_phase9c.py).

### 4. Experimental Evidence & Artifacts
* Results Log: [`outputs/phase9c/phase9c_results.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase9c/phase9c_results.jsonl) (65,501 bytes).
* Ablation Data: [`outputs/phase9c/phase9c_ablation_results.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase9c/phase9c_ablation_results.json) (87,220 bytes).
* Detailed Report: [`outputs/phase9c/phase9c_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase9c/phase9c_report.md) (19,617 bytes).

### 5. Measured Results vs. Dense Baseline ($N=6$)

| Evaluation Metric | Dense Baseline | Phase 9C Heuristic (+5.5 Boosts) | Net Change |
|---|:---:|:---:|:---:|
| **Candidate Pool Recall (Recall@40)** | 100.0% (6/6) | 100.0% (6/6) | 0.0% |
| **Top-10 Candidate Recall (Recall@10)** | 50.0% (3/6) | **83.3% (5/6)** | **+33.3%** |
| **Gold Evidence in Top-3 Context** | 33.3% (2/6) | **83.3% (5/6)** | **+50.0%** |
| **Downstream Semantic / Numeric Accuracy** | 16.7% (1/6) | **66.7% (4/6)** | **+50.0%** |
| **Strict Schema Validation Pass** | 33.3% (2/6) | **50.0% (3/6)** | **+16.7%** |

### 6. Critical Weaknesses & Regressions
1. *False Positive on Q3:* On Question Q3 (Ulta registered debt securities), the cover page table was boosted to Rank #1. Because the prompt forced an extraction, the model extracted `"Common stock, par value $0.01 per share"` as a debt security.
2. *Hallucination on Q1:* On Question Q1 (3M CapEx), an environmental narrative disclosure chunk (`p7_c5`, $27M) was ranked #2, causing the model to output $27M instead of the true $1,577M.
3. *Scientific Invalidity:* The boost weights (+5.5) were manually engineered to overfit the 6 diagnostic queries and did not represent a generalizable machine learning method.

### 7. Qualification Verdict
* **DOES NOT QUALIFY as an official Stage 2 experiment.** Phase 9C was an exploratory stepping-stone that highlighted the danger of hardcoded heuristics and motivated the DPO training of the DRR cross-encoder in Stage 1 and Table Header Propagation in Stage 2.

---

## 3. Audit of Official Stage 2: Production Enhancements & Benchmark Scaling

### 1. What Changed Relative to the Original Method
Stage 2 tackled the real-world production vulnerabilities identified during Stage 1 without reverting to uncalibrated heuristics:
1. **Work Package 1 (WP1) — Table Header Propagation at Ingestion:**
   * Ingested SEC PDFs frequently split financial statements across chunk boundaries, severing subsequent rows from statement headers and column definitions.
   * Stage 2 implemented an ingestion-layer enrichment pass ([`scripts/enrich_phase2_chunks.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/enrich_phase2_chunks.py)), prepending filing name, statement title, and unit scale to **2,902 split table chunks** in `bm25_index_phase2`.
2. **Work Package 2 (WP2) — Calibrated Statement Routing & Distractor Penalty:**
   * Replaced ad-hoc boosts with dynamic query routing ([`src/retrieval/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/phase2_reranker.py)):
     * If the query explicitly targets a financial statement (e.g. *"cash flow statement"*), applies a calibrated +3.5 statement boost.
     * Simultaneously applies a **-3.0 negative penalty** to narrative disclosures (MD&A risk factors, environmental footnotes, legal proceedings) to prevent them from displacing tabular schedules.
3. **Work Package 3 (WP3) — Full Benchmark Scaling ($N=150$):**
   * Scaled evaluation beyond the 6 diagnostic questions to the **entire 150-question FinanceBench test set** across all 84 PDF filings and 9 GICS industry sectors ([`scripts/run_phase2_benchmark.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_phase2_benchmark.py)).
4. **Work Package 4 (WP4) — Deterministic Python Sandbox Math Engine:**
   * Enforced strict formula and operand extraction (`operation: absolute_difference`, `operands: [A, B]`) evaluated via a verified Python calculation sandbox ([`src/utils/arithmetic_evaluator.py`](file:///c:/Users/user/Downloads/finsage-main/src/utils/arithmetic_evaluator.py)), eliminating LLM arithmetic approximation errors.

### 2. Experimental Evidence & Artifacts
* Macro Scorecard: [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json) (12,998 bytes).
* Diagnostic Results: [`outputs/phase2/evaluation/phase2_diagnostic_results.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_diagnostic_results.json) (10,478 bytes).
* Candidate Checkpoint: [`outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl) (60,387 bytes).
* Official Executive Report: [`outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md) (10,262 bytes).
* Comparison Audit: [`outputs/phase2/reports/phase1_vs_phase2_comparison.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/reports/phase1_vs_phase2_comparison.md) (8,743 bytes).

### 3. Measured Results: Stage 1 Reproduction vs. Stage 2 Production

| Evaluation Dimension | Stage 1 Faithful Reproduction | Stage 2 Production FinSage | Production Advancement |
|---|:---:|:---:|---|
| **Benchmark Coverage ($N$)** | 6 Diagnostic Questions | **150 Questions (Full Corpus)** | Scaled across 32 companies & 9 GICS sectors |
| **Diagnostic Semantic Accuracy** | 66.7% (4/6) | **83.3% (5/6)** | +16.6% accuracy surge on diagnostic queries |
| **Diagnostic Numeric Accuracy** | 66.7% (4/6) | **83.3% (5/6)** | 100% precision on extracted operands |
| **Diagnostic Evidence in Context** | 50.0% (3/6) | **66.7% (4/6)** | Solved table bisection natively at ingestion |
| **Q1 CapEx (`p60_c1`) Rank** | Rank #21 (Pool of 40) | **Rank #1 (Native Top Rank)** | Resolved without relying solely on bundling |
| **Full Corpus MRR ($N=150$)** | N/A | **0.2926** | Comprehensive macro retrieval ranking |
| **Full Corpus Recall@1 ($N=150$)** | N/A | **18.00% (27/150)** | Direct gold chunk placement at rank 1 |
| **Full Corpus Recall@5 ($N=150$)** | N/A | **45.33% (68/150)** | Context window sufficiency for 5-chunk prompt |
| **Full Corpus Recall@10 ($N=150$)**| N/A | **52.00% (78/150)** | Majority candidate pool coverage |
| **Table Header Propagation** | Absent | **Active (2,902 chunks enriched)** | Headers prepended to bisected chunks |
| **Calibrated Statement Boosts** | +0.0 (Zero Boosts) | **+3.5 (Statement-Calibrated)** | Query-statement dynamic routing |
| **Narrative Disclosures Penalty**| Absent | **-3.0 (Active)** | Suppresses environmental/MD&A distractors |
| **Arithmetic Solver** | Regex string check | **Python Sandbox Math Engine** | Exact deterministic formula calculation |

### 4. Per-Question Diagnostic Breakdown (Q1–Q6)
* **Q1 (3M CapEx, $1,577M):** PASSED. Gold chunk `3M_2018_10K_p60_c1` ranked **#1 natively**. Model extracted `$1,577M` with zero hallucination.
* **Q2 (3M Net PP&E, $8.738B):** PASSED. Balance sheet chunk `p58_c1` ranked **#1**. Extracted `$8,738M` and converted to `$8.738B`.
* **Q3 (Ulta Registered Debt):** PASSED. Model correctly recognized absence of registered debt securities and emitted `operation: none`.
* **Q4 (Ulta Major Acquisitions):** PASSED. Model confirmed zero acquisitions made in FY22/FY23.
* **Q5 (3M Net Sales Difference):** PASSED. Income statement chunk `p56_c0` ranked **#2**. Extracted `$32,765M` and `$31,657M`, sandboxed math computed delta **$1,108.00M**.
* **Q6 (Ulta Merchandise Inventories Difference):** FAILED. Retrieved Q4 earnings release chunks; model calculated cash flow change rather than the balance sheet footnote delta.

### 5. Qualification Verdict
* **OFFICIALLY QUALIFIES as the Stage 2 Experiment.** Backed by reproducible code, verified output checkpoints, and comprehensive macro-metrics spanning all 150 questions in FinanceBench.
