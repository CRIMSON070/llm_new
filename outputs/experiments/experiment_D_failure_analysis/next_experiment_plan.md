# FinSage Next Experiment Plan: Controlled Reranker Context Expansion & Statement Aliasing (Experiment E)

**Document File:** `outputs/experiments/experiment_D_failure_analysis/next_experiment_plan.md`  
**Execution Phase:** Phase 2 Performance Optimization (Post-Experiment D Failure Analysis)  
**Governing Standard:** Controlled Empirical RAG Ablation & Synthesis  
**Workspace Root:** `C:\Users\user\Downloads\finsage-main`  
**Date:** November 02, 2026  

---

## 1. Executive Summary & Objective

In **Experiment C**, expanding company intent detection across the full 32-company benchmark universe achieved an all-time high **MRR = 0.3513** and **Recall@1 = 25.33% (38/150)** with zero regressions. However, Recall@10 reached **57.33% (86/150)**, narrowly missing the 60.0% milestone.

Forensic failure diagnosis in **Experiment D** revealed the exact root cause of the remaining gap:
1. **The In-Pool Ranking Ceiling (17 Questions / 26.6% of Failures):** For 17 questions, the correct gold evidence chunk successfully reached the Top-50 fused candidate pool (average rank ~20), but was **demoted below Rank 10** by the reranker.
2. **The 512-Character Cross-Encoder Truncation Bottleneck:** In [`src/retrieval/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/phase2_reranker.py#L257), candidate chunks are sliced to `c["text"][:512]`. Extracted financial statement chunks average 950–1,150 characters. Slicing at 512 characters preserves only table titles and lead rows; critical line items (such as Accounts Payable, Net PP&E, or Operating Leases) and specific dollar figures occur beyond character 512 and are completely invisible to the cross-encoder.
3. **Statement Routing Blindspots:** Prompts using phrases such as *"statement of financial position"*, *"P&L statement"*, or specific line items without explicit statement keywords failed to trigger the $+4.5$ statement boost.

This plan specifies the **single best next controlled experiment** to break this ceiling: **Experiment E (Cross-Encoder Context Window Expansion to 1,024 Characters + Comprehensive Statement Aliasing)**.

---

## 2. Experimental Hypotheses

* **Primary Hypothesis (H1 — Recall@10 Milestone):**  
  Expanding the cross-encoder text slice from 512 to 1,024 characters will enable the cross-encoder to attend to numerical line items deeper in table chunks, recovering at least **8 of the 17 in-pool ranking failures** and driving **Recall@10 from 57.33% to $\ge 62.67\%$ (+5.34 pp, 94+/150 questions)**.
* **Secondary Hypothesis (H2 — Top-Tier Precision & MRR):**  
  Expanding statement-intent regexes to include financial statement synonyms (*"P&L"*, *"profit and loss"*, *"statement of financial position"*) will eliminate statement misrouting for in-pool candidates, increasing **MRR from 0.3513 to $\ge 0.3800$** and **Recall@1 from 25.33% to $\ge 28.00\%$**.
* **Tertiary Hypothesis (H3 — Zero Regressions):**  
  Because the change only increases available textual context and expands synonym coverage, it will cause **zero ranking regressions** among the 86 currently successful Top-10 queries.

---

## 3. Experimental Variables

### 3.1 Independent Variables (The Only Changes)
1. **Cross-Encoder Input Length:**
   - *Baseline (Exp C):* `pairs = [[query, c["text"][:512]] for c in candidates]`
   - *Proposed (Exp E):* `pairs = [[query, c["text"][:1024]] for c in candidates]`
2. **Statement-Intent Regex Expansion:**
   - Expand `detect_query_statement_intent()` to support:
     - Balance Sheet: `r"(?:balance\s+sheet|statement\s+of\s+financial\s+position|net\s+ppne|merchandise\s+inventories|working\s+capital)"`
     - Income Statement: `r"(?:income\s+statement|statement\s+of\s+operations|statement\s+of\s+earnings|p&l|profit\s+and\s+loss|net\s+sales|operating\s+margin)"`
     - Cash Flow Statement: `r"(?:cash\s+flow|statement\s+of\s+cash\s+flows|capital\s+expenditure|capex|share\s+repurchase)"`

### 3.2 Fixed Variables (Strictly Frozen Controls)
| Pipeline Component | Parameter / Specification | Status |
|---|---|:---:|
| **Candidate Pool Depth** | $K_{\text{pool}} = 50$ | Frozen |
| **Sparse Retrieval Depth** | BM25 $k = 50$ | Frozen |
| **Dense Retrieval Depth** | BGE-M3 FAISS $k = 50$ | Frozen |
| **Metadata Retrieval** | Heading-summary index $k = 3$ | Frozen |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) with constant $k = 60$ | Frozen |
| **Reranker Backbone** | `BAAI/bge-reranker-base` (fp16) | Frozen |
| **Heuristic Boost Weights** | Statement: $+4.5$, Negative Penalty: $-3.0$, Company: $+1.5$, Period: $+1.5$, Table: $+0.5$ | Frozen |
| **Corpus & Indexes** | `data/financebench/processed/phase2_enriched_chunks.jsonl` (54,564 chunks) | Frozen |
| **Benchmark Suite** | `data/financebench/questions.jsonl` ($N=150$ questions, 84 filings) | Frozen |

---

## 4. Evaluation Metrics & Success Criteria

| Metric | Experiment C Baseline | Experiment E Target | Minimum Acceptable Threshold |
|---|:---:|:---:|:---:|
| **MRR** | **0.3513** | **$\ge 0.3800$** (+0.0287) | $\ge 0.3650$ |
| **Recall@1** | **25.33%** (38/150) | **$\ge 28.00\%$** (42/150) | $\ge 26.67\%$ (40/150) |
| **Recall@3** | **42.00%** (63/150) | **$\ge 46.00\%$** (69/150) | $\ge 44.00\%$ (66/150) |
| **Recall@5** | **48.00%** (72/150) | **$\ge 53.33\%$** (80/150) | $\ge 50.67\%$ (76/150) |
| **Recall@10** | **57.33%** (86/150) | **$\ge 62.67\%$** (94/150) | $\ge 60.00\%$ (90/150) |
| **In-Pool Recovery** | 0 / 17 recovered | $\ge 8 / 17$ recovered (47.1%) | $\ge 5 / 17$ recovered (29.4%) |
| **Regressions vs. Exp C** | 0 regressions | **0 regressions** | $\le 1$ regression |
| **Batch Retrieval Latency** | 32.48 seconds | $\le 50.00$ seconds | $\le 65.00$ seconds |

---

## 5. Execution Prerequisites & Required Artifacts

1. **Safety & Directory Isolation:**
   - Dedicated output path: `outputs/experiments/experiment_E_context_expansion/`
   - Dedicated subdirectories: `retrieval/`, `evaluation/`, `reports/`, `source_backup/`.
   - Never overwrite Experiment C or earlier outputs.
2. **GPU Memory Headroom:**
   - RTX 5060 Laptop GPU (8 GB VRAM).
   - In Experiment C, `bge-reranker-base` at batch size 8 with 512 tokens consumed ~1.8 GB VRAM.
   - At 1,024 tokens with batch size 8, estimated peak VRAM is ~2.6 GB, leaving ample headroom (>5.4 GB free).
3. **Candidate Pool Re-Generation:**
   - Because `phase2_retrieval_checkpoint.jsonl` only saved Top-3 chunks, the benchmark will re-run multi-path candidate fusion from frozen indexes and pass the candidate lists directly into the expanded reranker.

---

## 6. Regression Safeguards & Validation Protocol

1. **Pre-Execution Baseline Lock:**
   - Verify that `outputs/experiments/experiment_C_company_intent/` files are read-only and backed up.
2. **Side-by-Side Movement Tracking:**
   - Implement an automated rank delta calculator comparing Experiment E against Experiment C for all 150 questions:
     - Improved: $\text{Rank}_E < \text{Rank}_C$
     - Regressed: $\text{Rank}_E > \text{Rank}_C$
     - Unchanged: $\text{Rank}_E == \text{Rank}_C$
3. **Automated Stop Condition:**
   - If more than 2 regressions occur during the first 50 questions evaluated, halt immediately to inspect token length boundaries.
4. **Controlled Reranker Comparison (Optional Follow-Up):**
   - Once candidates are generated and saved with full 50 chunk IDs, feed the identical 50 candidates to `BAAI/bge-reranker-v2-gemma` + Phase 1 DRR LoRA adapter for a pure, zero-noise reranker head-to-head comparison.
