# FinSage Stage 3: Frozen Evaluation Protocol

**Target Milestone:** Stage 3 — Independent Held-Out Validation  
**Target Path:** `outputs/experiments/experiment_stage3_heldout_validation/frozen_evaluation_protocol.md`  
**Governing Standard:** Pre-Registered, Double-Arm Comparative Evaluation  
**Date:** November 2026  

---

## 1. Protocol Objectives and Safeguards

To prevent post-hoc bias, metric cherry-picking, and data leakage, this evaluation protocol pre-registers the exact benchmark execution parameters, evaluation metrics, and statistical hypothesis tests for Stage 3 independent validation.

### 1.1 Non-Negotiable Safeguards
1. **Zero Parameter Tuning:** Neither the baseline (Experiment F) nor the candidate system (Experiment H2.1) may have its hyperparameters, fusion weights, reranker boosts, or statement quotas adjusted based on held-out results.
2. **Identical Query Inputs:** Both arms must evaluate exactly the same 100 questions from `outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl`.
3. **Isolated Index Storage:** All vector embeddings and BM25 postings must be read strictly from `data/heldout_stage3/indexes/`. Under no circumstances may development indexes (`data/financebench/indexes/`) or development chunks (`data/financebench/processed/`) be accessed during held-out evaluation.
4. **Gated Benchmark Execution:** The evaluation runner will **not** be executed until the complete corpus has been ingested and indexed, PDF page-level ground truth has been verified, and the user provides explicit authorization.

---

## 2. Frozen Evaluation Arm Specifications

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FROZEN EVALUATION ARMS COMPARISON                               │
├──────────────────────────────┬────────────────────────────┬────────────────────────────┤
│ Configuration Dimension      │ Arm 1: Reference (Exp F)   │ Arm 2: Candidate (H2.1)    │
├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Sparse Retrieval (BM25)      │ $k = 50$                   │ $k = 50$                   │
│ Dense Retrieval (BGE-M3)     │ $k = 50$                   │ $k = 50$                   │
│ Metadata Retrieval (Item)    │ $k = 3$ sections           │ $k = 3$ sections           │
│ Fusion Algorithm             │ Reciprocal Rank Fusion (60)│ Reciprocal Rank Fusion (60)│
│ Candidate Pool Size          │ $K_{\text{pool}} = 50$     │ $K_{\text{pool}} = 50$     │
│ In-Period Temporal Filter    │ Active (Year & Quarter)    │ Active (Year & Quarter)    │
│ Statement Routing Quota      │ **DISABLED (0 slots)**     │ **ENABLED ($Q \le 6$)**    │
│ RRF Floor Reserved           │ 50 slots (100%)            │ $\ge 44$ slots ($\ge 88\%$)│
│ Company Disambiguation       │ N/A                        │ Dynamic entity resolver    │
│ Statement Chunk Validation   │ N/A                        │ Primary statement filters  │
│ Reranker Model               │ `BAAI/bge-reranker-base`   │ `BAAI/bge-reranker-base`   │
│ Statement Boost              │ $0.0$                      │ $+4.5$                     │
│ Disclosure Penalty           │ $0.0$                      │ $-3.0$                     │
│ Company Match Boost          │ $+1.5$                     │ $+1.5$                     │
│ Period Match Boost           │ $+1.5$                     │ $+1.5$                     │
│ Table Structure Boost        │ $+0.5$                     │ $+0.5$                     │
└──────────────────────────────┴────────────────────────────┴────────────────────────────┘
```

---

## 3. Formal Metric Definitions

For a benchmark evaluation set of $N = 100$ independent queries $Q = \{q_1, \dots, q_N\}$:

### 3.1 Candidate Pool Recall ($K_{\text{pool}} = 50$)
Measures whether the gold evidence chunk is present anywhere within the 50 candidate chunks supplied to the reranker:
$$\text{Pool Recall} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Gold Chunk} \in \mathcal{C}_{i, 50})$$

### 3.2 Top-$K$ Retrieval Recall ($\text{Recall}@K$)
Measures whether the gold evidence chunk appears in the top $K$ reranked results presented to the generator:
$$\text{Recall}@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{rank}(\text{Gold Chunk}) \le K)$$
Reported at $K \in \{1, 3, 5, 10\}$.

### 3.3 Mean Reciprocal Rank (MRR)
Measures the reciprocal rank of the first relevant gold evidence chunk:
$$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank of the gold evidence chunk for query $q_i$, or $\infty$ if not retrieved in the top 10 ($\frac{1}{\infty} = 0$).

### 3.4 Narrative Invariance Rate
Verifies that qualitative and MD&A queries suffer zero candidate displacement from statement routing:
$$\text{Invariance Rate} = \frac{1}{N_{\text{narrative}}} \sum_{j=1}^{N_{\text{narrative}}} \mathbb{I}(\mathcal{C}_{j, 50}^{\text{H2.1}} \equiv \mathcal{C}_{j, 50}^{\text{Exp F}})$$
Expected value: **100.0%**.

---

## 4. Statistical Hypothesis Testing Protocol

To satisfy scientific rigor and avoid claiming unverified statistical superiority:

### 4.1 McNemar's Test for Paired Binary Transitions
To determine whether changes in Candidate Pool Recall or Recall@10 between Arm 1 and Arm 2 are statistically significant:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
Where:
* $b$ = number of queries missed by Arm 1 but recovered by Arm 2 (positive transitions).
* $c$ = number of queries retrieved by Arm 1 but missed by Arm 2 (regressions).

**Significance Threshold:** $\chi^2 > 3.841$ with $p < 0.05$ (1 degree of freedom).

### 4.2 Paired Wilcoxon Signed-Rank Test for MRR
To evaluate whether continuous ranking improvements across the paired query distribution are statistically distinguishable from random noise:
* Null Hypothesis ($H_0$): Median difference in reciprocal ranks is zero.
* Alternative Hypothesis ($H_1$): Median difference is strictly positive.
* Threshold: $p < 0.05$ (two-tailed).

---

## 5. Execution Runner Architecture: `run_heldout_stage3_benchmark.py`

The evaluation runner will be implemented in `scripts/run_heldout_stage3_benchmark.py`.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             ISOLATED EVALUATION RUNNER (scripts/run_heldout_stage3_benchmark.py)       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  Inputs:                                                                               │
│  - Questions: `outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl`
│  - Dense Index: `data/heldout_stage3/indexes/faiss_bge_m3/`                           │
│  - BM25 Index: `data/heldout_stage3/indexes/bm25/`                                     │
│  - Chunks: `data/heldout_stage3/processed/heldout_enriched_chunks.jsonl`               │
│                                                                                        │
│  Execution Pipeline:                                                                   │
│  1. Assert SHA-256 baseline hashes of protected modules.                              │
│  2. Load held-out index into memory.                                                  │
│  3. For each query:                                                                    │
│     a. Run Arm 1 (Frozen Exp F): Retrieve BM25 (50), Dense (50), Metadata (3) ->      │
│        RRF (60) -> Pool (50) -> Reranker -> Record scores & ranks.                    │
│     b. Run Arm 2 (Frozen H2.1): Same RRF -> Detect statement query -> Strict entity &  │
│        period match -> Inject <= 6 primary statement chunks -> Reranker with calibrated│
│        boosts -> Record scores & ranks.                                               │
│  4. Save question-level comparisons to `heldout_question_level_comparison.csv`.        │
│  5. Save metrics and statistical tests to `heldout_metrics.json`.                      │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
