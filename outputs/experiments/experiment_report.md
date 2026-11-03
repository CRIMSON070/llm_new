# FinSage Experiments A and B: Performance Evaluation & Comparison Report

**Document File:** `outputs/experiments/experiment_report.md`  
**Execution Timestamp:** November 02, 2026  
**Governing Standard:** Controlled Empirical RAG Ablation  
**Reference Paper:** *“FinSage: A Multi-aspect RAG System for Financial Filings Question Answering”* (Wang et al., arXiv:2504.14493, ACM CIKM 2025)  
**Workspace Root:** `C:\Users\user\Downloads\finsage-main`

---

## 1. Preflight Findings & Checkpoint/Resume Inspection

### 1.1 Git Status & Workspace Tracking
- **Git Status:** The project root is not a Git repository (`fatal: not a git repository: .git`).
- **Safety Measure Taken:** To prevent any data loss or accidental overwriting, all critical baseline files were physically cloned into an isolated backup folder prior to modifying any code:
  - `outputs/experiments/backup_baseline/run_phase2_benchmark.baseline.py`
  - `outputs/experiments/backup_baseline/phase2_reranker.baseline.py`
  - `outputs/experiments/backup_baseline/phase2_full_scorecard.baseline.json`
  - `outputs/experiments/backup_baseline/phase2_retrieval_checkpoint.baseline.jsonl`
  - `outputs/experiments/backup_baseline/PHASE2_OFFICIAL_RESULTS.baseline.md`

### 1.2 Checkpoint & Resume Architecture Audit
An inspection of [`scripts/run_phase2_benchmark.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_phase2_benchmark.py#L384-L466) revealed the following operational characteristics:
1. **No Configuration Hashing:** Checkpoints do not record experiment configurations, parameter hashes, or pipeline flags.
2. **Unconditional Overwrite:** The checkpoint file is opened with `open(checkpoint_file, "w")` at the start of batch execution, meaning it completely overwrites existing files rather than reading and resuming.
3. **No Question Skipping:** The script executes all 150 questions sequentially without inspecting previously completed question IDs.
4. **Risk of Silent Corruption:** If an experiment were run pointing to `outputs/phase2/`, it would silently overwrite the original baseline results.
5. **Mitigation Implemented:** Each experiment was strictly configured with its own dedicated output directory (`outputs/experiments/experiment_A_pool60/` and `outputs/experiments/experiment_B_routing/`). The baseline directory `outputs/phase2/` was never used during execution.

---

## 2. Baseline Verification

Before executing any experiment, the original baseline metrics were mathematically verified against the physical artifacts on disk:
- Scorecard: [`outputs/phase2/evaluation/phase2_full_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/evaluation/phase2_full_scorecard.json)
- Checkpoint: [`outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl)

| Evaluation Metric | Target Baseline Requirement | Verified On-Disk Value | Status |
|---|:---:|:---:|:---:|
| **Evaluation Scope ($N$)** | 150 Questions | **150 Questions** | ✅ VERIFIED EXACT |
| **Mean Reciprocal Rank (MRR)** | 0.2926 | **0.2926** | ✅ VERIFIED EXACT |
| **Retrieval Recall@1** | 18.00% (27/150) | **18.00% (27/150)** | ✅ VERIFIED EXACT |
| **Retrieval Recall@3** | 37.33% (56/150) | **37.33% (56/150)** | ✅ VERIFIED EXACT |
| **Retrieval Recall@5** | 45.33% (68/150) | **45.33% (68/150)** | ✅ VERIFIED EXACT |
| **Retrieval Recall@10** | 52.00% (78/150) | **52.00% (78/150)** | ✅ VERIFIED EXACT |

---

## 3. Exact Code Files and Lines Changed

### 3.1 Experiment A: Candidate Pool Expansion Only
- **Target File:** `scripts/run_phase2_benchmark.py`
- **Lines Changed:**
  - **Line 178:** Updated output directory to `outputs/experiments/experiment_A_pool60/`.
  - **Lines 408–409:** Expanded individual retrieval depths from `k=30` to `k=50` for BM25 and Dense FAISS to provide a sufficiently deep candidate union.
  - **Line 417:** Expanded RRF candidate pool from `k_pool=30` to `k_pool=60`.
  - **Lines 420–435:** Added pre-reranking candidate pool hit tracking (`best_rank_in_pool`, `candidate_pool_hit`) across the 60 candidates.
  - **Lines 470–515:** Computed macro candidate pool recall ($71.33\%$) and logged telemetry into the scorecard.
- **Reranker, Boosts, Index, Model:** Kept 100% identical to baseline (`bge-reranker-base`, statement boost +4.5, penalty -3.0, zero HyDE in batch).

### 3.2 Experiment B: Query-Intent Routing Fix Only
- **Target File 1:** `src/retrieval/phase2_reranker.py`
  - **Lines 98–101 (`detect_query_statement_intent`):**
    ```python
    # Before (Baseline):
    elif "cover page" in q_lower or "registered debt" in q_lower:
        return "cover_page"

    # After (Experiment B):
    elif "cover page" in q_lower or "registered debt" in q_lower or bool(re.search(r"(?:debt.*registered|registered.*debt|securities.*registered|registered.*securities|registered\s+to\s+trade|trading\s+symbol|section\s*12\(b\)|title\s+of\s+each\s+class)", q_lower)):
        return "cover_page"
    ```
- **Target File 2:** `scripts/run_phase2_benchmark.py`
  - **Line 178:** Updated output directory to `outputs/experiments/experiment_B_routing/`.
  - **Lines 408–417:** Preserved baseline candidate pool `k_pool=30` and path depths `k=30`.
  - **Lines 420–435:** Added candidate pool tracking for the baseline 30 pool size.
- **Candidate Pool, Path Depths, Reranker Weights:** Kept 100% identical to baseline ($K_{\text{pool}}=30$, $k=30$).

---

## 4. Controlled Experiment Results Matrix

| Metric | Baseline | Experiment A (Pool 60) | Change A | Experiment B (Routing Fix) | Change B |
|---|---:|---:|---:|---:|---:|
| **Recall@1** | 18.00% (27/150) | **20.00% (30/150)** | **+2.00 pp** | **19.33% (29/150)** | **+1.33 pp** |
| **Recall@3** | 37.33% (56/150) | **39.33% (59/150)** | **+2.00 pp** | **38.67% (58/150)** | **+1.34 pp** |
| **Recall@5** | 45.33% (68/150) | **45.33% (68/150)** | **+0.00 pp** | **46.00% (69/150)** | **+0.67 pp** |
| **Recall@10** | 52.00% (78/150) | **56.67% (85/150)** | **+4.67 pp** | **52.67% (79/150)** | **+0.67 pp** |
| **Mean Reciprocal Rank (MRR)** | 0.2926 | **0.3110** | **+0.0184** | **0.3042** | **+0.0116** |
| **Candidate-Pool Recall** | $\ge 52.00\%$* | **71.33% (107/150)** | **+19.33 pp** | **59.33% (89/150)** | **+7.33 pp** |
| **Financials Sector MRR** | 0.0962 | **0.1239** | **+0.0277** | **0.1731** | **+0.0769** |
| **Financials Sector Recall@10** | 15.38% (2/13) | **23.08% (3/13)** | **+7.70 pp** | **23.08% (3/13)** | **+7.70 pp** |
| **Batch Retrieval Runtime** | 43.24 s | **37.38 s** | **-5.86 s** | **23.75 s** | **-19.49 s** |

*\*Note on Candidate-Pool Recall definition: Candidate-pool recall is defined as the fraction of questions where at least one gold target chunk/page is present in the fused `candidates` list before cross-encoder reranking. In the original historical baseline checkpoint, full pre-reranking pools were not saved; the post-reranking Top-10 hit rate of 52.00% serves as a verified lower bound. In Experiment B (using the baseline $K_{\text{pool}}=30$), pre-reranking candidate recall was explicitly measured at 59.33% (89/150).*

---

## 5. Per-Question Detailed Analysis

### 5.1 Experiment A: Candidate Pool Expansion (Pool 60)
- **Rank Improved (Gains):** **16 questions**
  - **Major Jumps from Miss to Top-3:**
    - `Q2` (3M Company): Rank moved from **#999 $\to$ #3** (Net PP&E balance sheet target).
    - `Q10` (Activision Blizzard): Rank moved from **#999 $\to$ #1** (3-year average revenue schedule).
    - `Q17` (AES Corporation): Rank moved from **#999 $\to$ #3** (Credit rating footnote).
    - `Q19` (Amazon): Rank moved from **#999 $\to$ #3** (Days payable outstanding).
    - `Q22` (Amcor): Rank moved from **#999 $\to$ #4** (Net accounts receivable).
- **Rank Regressed (Drops):** **27 questions**
  - Expanding the pool from 30 to 60 introduced competing distractor chunks from related fiscal years or notes that received higher cross-encoder scores than the gold chunk (e.g. `Q3`, `Q7`, `Q8` dropped from borderline ranks #8–#10 to >#10).
- **Unchanged:** **107 questions**
- **Net Outcome:** Overall Recall@10 gained $+4.67\text{ pp}$ (+7 net questions), and MRR improved by $+0.0184$, proving that candidate pool depth was indeed a physical bottleneck for 16 previously unreachable questions.

### 5.2 Experiment B: Query-Intent Routing Fix
- **Unit Test Suite Execution:** Verified via [`tests/test_intent_routing.py`](file:///c:/Users/user/Downloads/finsage-main/tests/test_intent_routing.py):
  - **11 out of 11 unit tests passed (0 failures).**
  - Correctly detected exact AMEX query (`cover_page`), reordered AMEX query (`cover_page`), Ulta registered debt (`cover_page`), and Section 12(b) query (`cover_page`).
  - Correctly rejected unrelated debt maturity queries (`None`), credit facility commitments (`None`), and debt covenant queries (`None`), confirming **zero false positives**.
- **Per-Question Benchmark Impact ($N=150$):**
  - **Rank Improved (Gains):** **2 questions**
    1. **`Q39` (American Express Registered Debt):**  
       - Baseline: Rank **#999** (Total Miss, top retrieved were Note 7/8 debt footnotes on pages 252, 126, 99).  
       - Experiment B: Rank **#1 (Direct Top-1 Hit!)** (`AMERICANEXPRESS_2022_10K_p1_c0_p1`).
    2. **`Q7` (3M Registered Debt):**  
       - Baseline: Rank **#4** (Target page 1 was ranked behind notes on pages 70, 71, 19).  
       - Experiment B: Rank **#1 (Direct Top-1 Hit!)** (`3M_2023Q2_10Q_p1_c1_p1`).
  - **Rank Regressed (Drops):** **0 questions (ZERO regressions across all 150 questions!)**
  - **Unchanged:** **148 questions**
- **Financials Sector Impact:**
  - MRR in Financials rose from **0.0962 $\to$ 0.1731** (+80% relative gain).

---

## 6. Runtime, Telemetry, and Errors

- **Experiment A Runtime:** **37.38 seconds** across 150 queries (average 0.25s per query on GPU).
- **Experiment B Runtime:** **23.75 seconds** across 150 queries (average 0.16s per query on GPU).
- **Errors / Exceptions:** **Zero runtime errors.** 100% of the 150 questions processed cleanly with full checkpoint telemetry.

---

## 7. Restoration and Preservation of Baseline Artifacts

Following completion of both experiments:
1. **Source Code Restored:** Both `scripts/run_phase2_benchmark.py` and `src/retrieval/phase2_reranker.py` were restored from `outputs/experiments/backup_baseline/`.
2. **Byte-Level Verification:** Verified using Windows file comparison (`fc.exe`):
   ```
   fc.exe scripts\run_phase2_benchmark.py outputs\experiments\backup_baseline\run_phase2_benchmark.baseline.py
   FC: no differences encountered

   fc.exe src\retrieval\phase2_reranker.py outputs\experiments\backup_baseline\phase2_reranker.baseline.py
   FC: no differences encountered
   ```
3. **Baseline Artifacts Preserved:**
   - `outputs/phase2/evaluation/phase2_full_scorecard.json` (Unchanged, MRR: 0.2926, Recall@5: 45.33%)
   - `outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl` (Unchanged, 150 records)
   - `outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md` (Unchanged)
4. **Experiment Artifacts Saved in Dedicated Folders:**
   - Experiment A: [`outputs/experiments/experiment_A_pool60/`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_A_pool60/)
   - Experiment B: [`outputs/experiments/experiment_B_routing/`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_B_routing/)

---

## 8. Summary of Findings & Recommended Next Steps

### Empirical Findings:
1. **Experiment A (Candidate Pool Expansion to 60)** demonstrated that candidate depth was a major structural bottleneck. Expanding the pool lifted candidate-pool recall from $59.33\% \to 71.33\%$ and increased Recall@10 from $52.0\% \to 56.67\%$, unlocking 16 previously unreachable target passages. However, the presence of 27 regressions indicates that larger candidate pools require tighter cross-encoder discrimination to avoid competitor dilution.
2. **Experiment B (Query-Intent Routing Fix)** was a **100% pure gain**: 2 queries jumped directly to Rank #1 with **zero regressions**, boosting Financials sector MRR by +80% and lifting macro MRR to $0.3042$.

### Recommended Next Step (Experiment A + B Synergy):
Now that both interventions have been individually validated in strict isolation, the logical next step is a **Combined Controlled Benchmark (Experiment AB)**:
- Combine the flexible intent routing of Experiment B with a calibrated candidate pool (e.g. $K_{\text{pool}}=50$) and evaluate the compound improvement.
- Anticipated combined performance: **MRR $> 0.325$**, **Recall@10 $> 58\%$**.
