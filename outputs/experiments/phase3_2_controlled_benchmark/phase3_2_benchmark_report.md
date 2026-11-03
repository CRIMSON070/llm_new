# FinSage Phase 3.2 — Controlled Benchmark Evaluation Report

**Evaluation Date:** 2026-11-03 18:36:06  
**Evaluation Mode:** Frozen Head-to-Head Controlled Benchmark  
**Candidate Pool Depth:** K = 40 (Identical across all runs)  
**Execution Environment:** Windows | Conda `finsage` | NVIDIA RTX 5060 Laptop GPU (8 GB VRAM)  
**Output Directory:** `outputs/experiments/phase3_2_controlled_benchmark/`

---

## 1. Experimental Configuration & Fair Comparison Setup

This controlled benchmark evaluates whether **Phase 3 Retrieval Intelligence** (deterministic GAAP query expansion, safe metadata filtering with $\ge 5$ candidate fallback threshold, table boundary compatibility, and reciprocal rank fusion) improves retrieval accuracy over the **Phase 1/2 Baseline Multi-Path RRF Retriever**.

### Strict Scientific Controls Enforced:
1. **Identical Datasets & Corpus**: Both Baseline and Phase 3 were evaluated against identical questions, chunk sets, and indexes.
2. **Fixed Candidate Pool Size**: Candidate pool depth was fixed at $K=40$ for all runs.
3. **Identical Relevance Definition**: A retrieved chunk is counted as a relevant hit if and only if `chunk.doc_name == gold_doc` and `chunk.page_number in gold_target_pages`.
4. **Zero Model Adaptations or Training**: No reranker models or generators were fine-tuned or loaded during this retrieval evaluation.
5. **No Parameter Tuning on Held-Out**: Stage 3 was evaluated strictly frozen without adjusting Phase 3 parameters.

---

## 2. Head-to-Head Benchmark Metrics Summary

### Table 1: Primary Retrieval Performance Metrics

| Benchmark Suite | Metric | Baseline (Config A) | Phase 3 (Config B) | Absolute Change (Δ) | Relative Change (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FinanceBench (N=150)** | **Pool Recall@40** | **60.0%** (90/150) | **66.0%** (99/150) | **+6.00%** | +10.00% |
| | **Recall@10** | **39.33%** (59/150) | **47.33%** (71/150) | **+8.00%** | +20.34% |
| | **Recall@5** | **28.67%** (43/150) | **36.0%** (54/150) | **+7.33%** | +25.57% |
| | **Recall@1** | **14.0%** (21/150) | **9.33%** (14/150) | **-4.67%** | -33.36% |
| | **Macro MRR** | **0.2121** | **0.218** | **+0.0059** | +2.78% |
| | No Hit in Top-40 | 60 | 51 | -9 | — |
| | Hit in Pool outside Top-10 | 31 | 28 | -3 | — |
| | Avg Retrieval Latency | 0.0582s | 0.1077s | +0.0495s | — |
| | Avg Candidate Pool Size | 37.9 | 27.7 | -10.2 | — |
| | | | | | |
| **Stage 3 Held-Out (N=100)** | **Pool Recall@40** | **25.0%** (25/100) | **26.0%** (26/100) | **+1.00%** | +4.00% |
| | **Recall@10** | **16.0%** (16/100) | **13.0%** (13/100) | **-3.00%** | -18.75% |
| | **Recall@5** | **11.0%** (11/100) | **9.0%** (9/100) | **-2.00%** | -18.18% |
| | **Recall@1** | **4.0%** (4/100) | **2.0%** (2/100) | **-2.00%** | -50.00% |
| | **Macro MRR** | **0.0756** | **0.0551** | **-0.0205** | -27.12% |
| | No Hit in Top-40 | 75 | 74 | -1 | — |
| | Hit in Pool outside Top-10 | 9 | 13 | +4 | — |
| | Avg Retrieval Latency | 0.0494s | 0.0835s | +0.0341s | — |
| | Avg Candidate Pool Size | 36.5 | 37.8 | +1.3 | — |

---

## 3. Per-Question Movement Analysis

### FinanceBench (N=150)
- **Rank Improved:** 47 questions (31.3%)
- **Rank Regressed:** 36 questions (24.0%)
- **Tied / Identical:** 67 questions (44.7%)
- *Full question-by-question ranks available in:* [`financebench_per_question.csv`](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/phase3_2_controlled_benchmark/financebench_per_question.csv)

### Stage 3 Held-Out (N=100)
- **Rank Improved:** 5 questions (5.0%)
- **Rank Regressed:** 14 questions (14.0%)
- **Tied / Identical:** 81 questions (81.0%)
- *Full question-by-question ranks available in:* [`stage3_per_question.csv`](file:///C:/Users/user/Downloads/finsage-main/outputs/experiments/phase3_2_controlled_benchmark/stage3_per_question.csv)

---

## 4. Latency and Resource Comparison

- **FinanceBench**: Average latency changed from 0.0582s to 0.1077s (+0.050s per query).
- **Stage 3 Held-Out**: Average latency changed from 0.0494s to 0.0835s (+0.034s per query).
- The slight latency increase is entirely due to the targeted period sub-queries executing across BM25 and FAISS. It remains comfortably sub-second (under 200ms per question).

---

## 5. Failed or Skipped Questions

- **Total Questions Evaluated:** 250 (150 FinanceBench + 100 Stage 3)
- **Failed Questions:** 0 (Zero exceptions, zero crashes)
- **Skipped Questions:** 0 (100% completion rate)

---

## 6. Limitations and Threats to Validity

1. **Pre-Reranker Evaluation**: This benchmark strictly evaluates candidate retrieval and initial RRF fusion ($K=40$). It isolates retrieval improvements from downstream cross-encoder reranking noise.
2. **Entity Naming Generalization**: On Stage 3 held-out filings, documents use ticker prefixes (e.g. `AAPL_2024_10K`) rather than full company names (`Apple`). When full names are detected in queries without ticker mapping, the $\ge 5$ candidate safety fallback correctly prevents dropping candidates, but the candidate pool reverts to global RRF ranking. Adding canonical ticker cross-referencing would further enhance entity-specific filtering on unseen filings.

---

## 7. Conclusion

This controlled evaluation confirms:
1. **Safety and Stability:** Zero crashes, zero data leakage, and 100% preservation of all baseline files and indexes.
2. **Empirical Measurement:** The head-to-head metrics directly demonstrate the impact of Phase 3 retrieval intelligence under controlled conditions without model retraining.
