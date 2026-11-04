# Experiment L.7 — Controlled Context Window Stitching Evaluation Report

**Document Status:** Final  
**Date:** 2026-11-04  
**Benchmark Reference:** [stage3_heldout_questions_v2.jsonl](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl)  
**Execution Manifest:** [execution_manifest.json](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/execution_manifest.json)  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (CUDA), PyTorch, BGE-M3 (dense), BM25 (sparse), BGE-Reranker-Base (cross-encoder)  

---

## 1. Executive Summary & Core Objective

Experiment L.7 evaluates whether **Context Window Stitching**—enriching retrieved candidate chunks with immediately adjacent neighboring chunks from the same filing prior to cross-encoder reranking—improves retrieval precision and recall over the frozen **Experiment L.2 (Section-Aware Grounded K65)** baseline.

The evaluation was conducted on the physical-page reconciled **Stage 3 Benchmark V2**, isolating the primary verified cohort ($N=65$) from the secondary ambiguous cohort ($N=35$), under dual evaluation standards:
1. **Strict Direct Evidence**: Operative answer-bearing passages only.
2. **Comprehensive Evidence**: Direct evidence plus contextual continuation table headers.

### Definitive Conclusion
**Does Context Window Stitching genuinely improve retrieval over Experiment L.2?**

**Yes, for Top-10 Retrieval Coverage, with a slight tradeoff in top-1 ranking concentration:**
1. **Recall@10 increases significantly by +4.61 percentage points** on the primary verified cohort ($N=65$) under Strict Direct Evidence (climbing from **41.54% to 46.15%**, reaching 30/65 hits, a net gain of +3 queries). Under Comprehensive Evidence, Recall@10 increases by **+6.15 percentage points** (climbing from **44.62% to 50.77%**, reaching 33/65 hits, a net gain of +4 queries).
2. **5 critical questions recovered into Top-10 from `NO_HIT`:**
   - `stage3_heldout_0003` (Apple FY24 operating margin): **NO_HIT $\rightarrow$ Rank 1**.
   - `stage3_heldout_0025` (Broadcom Apple customer concentration): **NO_HIT $\rightarrow$ Rank 6**.
   - `stage3_heldout_0040` (Goldman Sachs wealth management net revenues): **NO_HIT $\rightarrow$ Rank 7**.
   - `stage3_heldout_0050` (Wells Fargo home lending revenue trend): **NO_HIT $\rightarrow$ Rank 4**.
   - `stage3_heldout_0083` (Merck Keytruda patent expiration): **NO_HIT $\rightarrow$ Rank 9**.
3. **Only 2 questions regressed out of Top-10:**
   - `stage3_heldout_0013` (NVIDIA gaming segment revenue): Rank 7 $\rightarrow$ NO_HIT.
   - `stage3_heldout_0024` (Broadcom semiconductor solutions revenue): Rank 8 $\rightarrow$ NO_HIT.
4. **Candidate Pool Invariance:** Candidate pool recall remains **100% identical at 72.31%** (Strict) and **73.85%** (Comprehensive). No artificial pool inflation occurred.
5. **MRR Tradeoff:** Mean Reciprocal Rank slightly shifted from **0.1999 to 0.1958** (-0.0041) under Strict Evidence, as expanding context to 1,200 characters slightly diffused cross-encoder attention across dense numbers, nudging a few Rank 1 hits to Rank 2 or 3 while lifting previously missed chunks into the top 10.

---

## 2. Experimental Controls and Implementation Architecture

All upstream retrieval components were frozen to guarantee isolated measurement:
- **Corpus:** Frozen `heldout_grounded_chunks.jsonl` (SHA-256: `ea3b2ae3...`).
- **Indexes:** Frozen `indexes_grounded_exp_l/` (FAISS dense + BM25 sparse).
- **Candidate Pool:** Frozen Section-Aware Grounded K=65 ($K_{pool}=65$, RRF $k=60$, Narrative Table Cap $Q=4$, Item 1 Boost $+0.05$).
- **Reranker:** `BAAI/bge-reranker-base` (FP16 on CUDA, max 512 tokens).

### Stitcher Specifications (`context_window_stitcher.py`)
- **Filing Isolation:** Predecessor and successor chunks are selected strictly from the same filing document (`doc_name`). Zero cross-filing boundary crossings.
- **Candidate Identity Preservation:** Evaluation identity remains strictly the retrieved candidate's original `chunk_id`. Neighboring chunks are NOT counted as independent hits.
- **Context Budget:** Maximum 1,200 characters. Original candidate text is prioritized. Preceding context (e.g. table header lead-in) and succeeding context (continuation) each receive up to half of remaining budget.
- **Deduplication:** Redundant corporate context headers are stripped from neighbor chunks. Word boundary snapping prevents truncated words.

---

## 3. Controlled A/B Evaluation Results

### Primary Verified Cohort ($N=65$)

#### 1. Strict Direct Evidence Standard (Operative Evidence Only)

| Metric | Arm A: Frozen L.2 (No Stitching) | Arm B: Exp L.7 (With Stitching) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall** | **72.31%** (47/65) | **72.31%** (47/65) | **+0.00%** | Invariant (Identical Pool) |
| **Recall@1** | 12.31% (8) | 10.77% (7) | -1.54% (-1 hit) | -12.5% |
| **Recall@3** | 23.08% (15) | 21.54% (14) | -1.54% (-1 hit) | -6.7% |
| **Recall@5** | 33.85% (22) | 32.31% (21) | -1.54% (-1 hit) | -4.5% |
| **Recall@10** | **41.54%** (27) | **46.15%** (30) | **+4.61%** (+3 net hits) | **+11.1%** |
| **Precision@5** | 6.77% | 6.46% | -0.31% | -4.6% |
| **Precision@10** | 4.15% | 4.62% | +0.47% | +11.3% |
| **Mean Reciprocal Rank (MRR)** | **0.1999** | **0.1958** | **-0.0041** | -2.1% |
| **Average Latency (sec)** | **8.35s** | **11.24s** | +2.89s | +34.6% |
| **P95 Latency (sec)** | 10.18s | 13.48s | +3.30s | +32.4% |

#### 2. Comprehensive Evidence Standard (Direct Evidence + Contextual Headers)

| Metric | Arm A: Frozen L.2 (No Stitching) | Arm B: Exp L.7 (With Stitching) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall** | **73.85%** (48/65) | **73.85%** (48/65) | **+0.00%** | Invariant (Identical Pool) |
| **Recall@1** | 13.85% (9) | 12.31% (8) | -1.54% (-1 hit) | -11.1% |
| **Recall@3** | 26.15% (17) | 26.15% (17) | +0.00% | Neutral |
| **Recall@5** | 36.92% (24) | 36.92% (24) | +0.00% | Neutral |
| **Recall@10** | **44.62%** (29) | **50.77%** (33) | **+6.15%** (+4 net hits) | **+13.8%** |
| **Precision@5** | 7.38% | 7.38% | +0.00% | Neutral |
| **Precision@10** | 4.46% | 5.08% | +0.62% | +13.9% |
| **Mean Reciprocal Rank (MRR)** | **0.2238** | **0.2211** | **-0.0027** | -1.2% |

---

### Secondary Cohort ($N=35$ Ambiguous Questions) & Full Benchmark ($N=100$)

In compliance with benchmark integrity guidelines:
- Secondary ambiguous questions remain evaluated with empty gold targets (`[]`). Both arms record 0.0% recall on this cohort.
- **Full Benchmark Aggregate ($N=100$)**:
  - **Strict Pool Recall:** Arm A = 47.00% $\rightarrow$ Arm B = **47.00%** (Identical).
  - **Strict Recall@10:** Arm A = 27.00% (27 hits) $\rightarrow$ Arm B = **30.00%** (30 hits) (**+3.00 pp**).
  - **Strict MRR:** Arm A = 0.1299 $\rightarrow$ Arm B = 0.1272.
  - **Comprehensive Recall@10:** Arm A = 29.00% (29 hits) $\rightarrow$ Arm B = **33.00%** (33 hits) (**+4.00 pp**).
  - **Comprehensive MRR:** Arm A = 0.1454 $\rightarrow$ Arm B = 0.1437.

---

## 4. Forensic Analysis of Key Anchor Questions

| Question ID | Question Summary | Original Candidate ID | Neighboring Chunks Stitched | Arm A Rank | Arm B Rank | Stitching Impact |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`stage3_heldout_0025`** | Broadcom Apple customer concentration | `AVGO_2023_10K_p4_c0` | `p3_c1` (predecessor) + `p5_c0` (successor) | **NO_HIT** | **Rank 6** | **Major Win.** In L.6, this query was the sole grounding regression (dropped to NO_HIT). Stitching adjacent context on Page 3/5 allowed the reranker to recover the customer concentration risk at **Rank 6**. |
| **`stage3_heldout_0075`** | Eli Lilly chronic weight regulatory milestone | `LLY_2023_10K_p92_c0` | `p91_c0` (predecessor) + `p92_c1` (successor) | **Rank 3** | **Rank 5** | **Retained in Top 5.** Original candidate retains high prominence. Slight rank shift due to inclusion of surrounding pipeline rows. |
| **`stage3_heldout_0074`** | Eli Lilly incretin manufacturing capacity | `LLY_2024_10K_p75_c0` | `p74_c0` (predecessor) + `p75_c1` (successor) | **Rank 1** | **Rank 1** | **Perfect Stability.** Operative incretin bottleneck text remains locked at **Rank 1**. |
| **`stage3_heldout_0099`** | Caterpillar dealer machine inventory | `CAT_2024_10K_p72_c2` | `p72_c1` (predecessor) + `p72_c3` (successor) | **Rank 2** | **Rank 2** | **Perfect Stability.** Dealer inventory impact passage remains locked at **Rank 2**. |
| **`stage3_heldout_0100`** | Caterpillar Energy & Trans. segment profit | `CAT_2023_10K_p94_c1` | `p94_c0` (predecessor) + `p94_c2` (successor) | **Rank 1** | **Rank 1** | **Perfect Stability.** Segment profit drivers remain locked at **Rank 1**. |
| **`stage3_heldout_0014`** | NVIDIA China GPU export controls | `NVDA_2024_10K_p21_c2` | `p21_c1` (predecessor) + `p21_c3` (successor) | **Rank 5** | **Rank 8** | **Retained in Top 10.** Export control passage remains inside Top 10 (Rank 8). Surrounding licensing text slightly adjusted score relative to competing chunks. |

---

## 5. Artifacts and Reproducibility

All generated artifacts have been verified and saved:
- **Module Implementation:** [`outputs/experiments/experiment_l7_context_window_stitching/context_window_stitcher.py`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/context_window_stitcher.py)
- **Benchmark Runner:** [`scripts/run_controlled_l7_benchmark.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_controlled_l7_benchmark.py)
- **Evaluation Scorecard:** [`outputs/experiments/experiment_l7_context_window_stitching/l7_scorecard.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/l7_scorecard.json)
- **Question-Level Results:** [`outputs/experiments/experiment_l7_context_window_stitching/l7_question_level_results.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/l7_question_level_results.csv)
- **Comparison Summary:** [`outputs/experiments/experiment_l7_context_window_stitching/l7_comparison.csv`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/l7_comparison.csv)
- **Execution Manifest:** [`outputs/experiments/experiment_l7_context_window_stitching/execution_manifest.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_l7_context_window_stitching/execution_manifest.json)
