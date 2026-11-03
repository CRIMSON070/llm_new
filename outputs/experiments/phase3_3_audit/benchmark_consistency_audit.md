# FinSage Phase 3.3 — Benchmark Metric & Historical Baseline Audit

**Audit Date:** 2026-11-03  
**Audit Status:** Complete (100% Read-Only, Zero Code Changes, Zero Benchmark Executions)  
**Report Artifact:** `outputs/experiments/phase3_3_audit/benchmark_consistency_audit.md`

---

## 1. Confirmed Reasons for Discrepancies

### Question 1: FinanceBench Baseline Recall@40 (60.0% vs. Historical Stage 1 75.33%)
- **Reason 1: Different Retrieval Architectures (VERIFIED)**  
  - **Historical Stage 1** (`scripts/run_faithful_stage1_benchmark.py` lines 16–22, 326–333) utilized **4 distinct retrieval paths** with `MultiPathRetriever`:
    1. BM25 Sparse Lexical ($k=40$)
    2. BGE-M3 Dense FAISS ($k=40$)
    3. Heading-Summary Metadata FAISS ($k_{\text{sections}}=4$)
    4. Generative HyDE via Qwen2.5-7B (3 hypothetical passages, $k_{\text{hypo}}=20$, totaling 60 dense searches)
  - **Phase 3.2 Baseline** (`scripts/run_phase3_2_controlled_benchmark.py` lines 132–138, `src/retrieval/hybrid_retriever.py` lines 284–309) utilized `HybridRetriever`, which contains **only 2 paths**: base FAISS ($k=20$) and base BM25 ($k=20$). It omits both the Metadata FAISS index and the Qwen2.5-7B HyDE generator.
- **Reason 2: Pre-Reranker vs. Post-Reranker Evaluation for Recall@5 (VERIFIED)**  
  - Historical Stage 1 Recall@5 (65.33%) was computed **after cross-encoder reranking** (`BAAI/bge-reranker-v2-gemma` + DPO adapter).
  - Phase 3.2 Recall@5 (28.67% baseline, 36.00% Phase 3) was computed on the **raw un-reranked initial candidate pool** directly from RRF.

---

### Question 2: Stage 3 Held-Out Baseline Recall@40 (25.0% vs. Historical Stage 3 30.0%)
- **Reason 1: Candidate Pool Depth K=40 vs. K=50 (VERIFIED)**  
  - **Historical Stage 3** (`scripts/run_heldout_stage3_benchmark.py` lines 586–588, 661–668) evaluated candidate pool depth **$K=50$** (with initial path depth $k=50$). Exactly 5 ground-truth evidence hits were retrieved at ranks 41–50.
  - **Phase 3.2 Baseline** evaluated candidate pool depth **$K=40$** (with initial path depth $k=20$). Truncating the pool at $K=40$ drops those 5 rank 41–50 hits ($25/100 = 25.0\%$).
- **Reason 2: Cross-Encoder Reranking in Historical MRR and Recall@5 (VERIFIED)**  
  - In `scripts/run_heldout_stage3_benchmark.py` (lines 671–689), Recall@1, 3, 5, 10 and MRR (0.1050) were evaluated **after cross-encoder reranking** using `Phase2Reranker`.
  - In Phase 3.2, MRR (0.0756) and Recall@5 (11.0%) were measured on the **un-reranked candidate pool** prior to the cross-encoder.

---

## 2. Internal Fairness of Phase 3.2 (Config A vs. Config B)

- **Evaluation Logic & Criteria Identity: 100% IDENTICAL (VERIFIED)**  
  Within `scripts/run_phase3_2_controlled_benchmark.py`:
  - **Questions Evaluated:** Exactly the same 150 FinanceBench and 100 Stage 3 questions.
  - **Candidate Pool Depth:** Exactly $K=40$ for both configurations.
  - **Relevance Criteria:** Bit-for-bit identical (`chunk.doc_name == gold_doc and chunk.page_number in gold_pages`).
  - **Metrics Computation:** Single shared function (`compute_retrieval_metrics`).
  - **Indexes and Corpus:** Shared in-memory FAISS and BM25 objects.
- **Fairness Verdict:**  
  Phase 3.2 is an **entirely fair and strictly controlled head-to-head comparison** between Baseline and Phase 3 candidate retrieval.

---

## 3. Reliability & Presentation Guidance

- **Are Phase 3.2 Results Unreliable?**  
  **NO.** Phase 3.2 results are completely reliable for what they measure: **candidate retrieval pool quality ($K=40$) prior to cross-encoder reranking**.
- **Can Phase 3.2 Be Directly Compared Against Historical 75.33% / 30.0% Figures?**  
  **NO.** That would be comparing two fundamentally different setups:
  - *Historical Stage 1:* 4-Path Retriever (including HyDE + Metadata FAISS) + Cross-Encoder Reranker.
  - *Phase 3.2:* 2-Path / Query-Intelligence Retriever without HyDE or Cross-Encoder.
- **Safe Reporting Recommendation:**  
  Report Phase 3.2 as an **ablation of candidate retrieval intelligence** ($K=40$ pool), highlighting that Phase 3 achieves **+8.00 pp Recall@10** and **+6.00 pp Pool Recall** over the baseline 2-path retriever without requiring expensive HyDE LLM calls.
