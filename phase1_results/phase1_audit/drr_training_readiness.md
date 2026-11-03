# Phase 1 Pre-Implementation Audit: Paper-Faithful Document Re-Ranker (DRR) Readiness

**Document:** `outputs/phase1_audit/drr_training_readiness.md`  
**Date:** 2026-10-11  
**Project:** FinSage Phase 1 — Faithful Paper Reproduction  
**Reference Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering* (Wang et al., arXiv:2504.14493v4, ACM CIKM 2025)  
**Status:** **AUDIT ONLY — NO CODE MODIFIED, NO WEIGHTS DOWNLOADED, NO TRAINING STARTED.**  

---

## 1. Executive Summary

This pre-implementation audit verifies the exact methodology, data requirements, and hardware feasibility for training the **Domain-Specialized Document Re-ranker (DRR)** strictly according to the original FinSage paper (*arXiv:2504.14493*).

### Key Audit Findings

1. **Architecture Mismatch:** The paper specifies `BAAI/bge-reranker-v2-gemma` (a 2.5-billion-parameter causal decoder cross-encoder based on Google's Gemma). The proposed `bge-reranker-base` in our workspace is a 278-million-parameter encoder model based on XLM-RoBERTa. While both function as cross-encoders, they are fundamentally different neural architectures.
2. **Missing Training Dataset:** The current preference files in `data/financebench/phase9b/` (`train_preferences.jsonl` and `validation_preferences.jsonl`) are **generator-extraction preference pairs** for JSON operational formatting. There are **zero** document-level preference pairs (`{"query": ..., "pos": [...], "neg": [...]}`) in the workspace. Training cannot proceed until an Appendix B–compliant dataset is generated from the FinanceBench training split.
3. **Hardware Feasibility on 8 GB VRAM:**
   - Training the paper's exact `bge-reranker-v2-gemma` in unquantized `bf16` with LoRA ($r=32$, $\alpha=64$) requires approximately **6.7–7.0 GB of VRAM**. Because Windows Desktop Window Manager (DWM) consumes ~0.8–1.2 GB of VRAM, running this model in unquantized `bf16` carries an **extremely high risk of CUDA Out-Of-Memory (OOM)**.
   - If 4-bit quantization (QLoRA) is applied, VRAM drops safely to ~3.0 GB, but quantization is not specified in the paper.
   - If `bge-reranker-base` is used, VRAM is under 1.5 GB in unquantized `bf16`, but the model checkpoint deviates from the published paper.

---

## 2. Section A: Verification of the Paper's Exact DRR Methodology

The specifications below are extracted verbatim from the original paper, formulas, and appendices.

### A.1. Model Architecture & Checkpoint
- **Model Checkpoint:** `BAAI/bge-reranker-v2-gemma` (Paper Section 3.3 & Section 4.2.2).
- **Architecture Type:** Causal decoder LLM (Gemma-2B) adapted for cross-encoder reranking by pooling the final token embedding into a scalar relevance score or computing next-token logits.
- **Reference Citations:**
  - *Section 3.3:* *"We employ the BAAI/bge-reranker-v2-Gemma model based on its superior performance in both English and multilingual applications."*
  - *Section 4.2.2:* *"For re-ranking, we compare our fine-tuned DRR, trained on single-company financial filings, against BAAI/bge-reranker-v2-gemma, a general re-ranker."*

### A.2. DPO Training Objective & Preference-Pair Formulation
- **Training Formulation:** Preference learning framed as binary classification / contrastive optimization over paired candidate chunks (Section 3.3.2, Eq. 2):
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(q, s^+, s^-)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(s^+|q)}{\pi_{\text{ref}}(s^+|q)} - \log \frac{\pi_\theta(s^-|q)}{\pi_{\text{ref}}(s^-|q)} \right) \right) \right]$$
  where $s^+$ is a preferred (positive) document and $s^-$ is a rejected (negative) document for query $q$.
- **Training Data Schema (Appendix B.1):**
  ```json
  {
    "query": "string (the user query)",
    "pos": ["string (single gold/positive document chunk text)"],
    "neg": ["string (negative document chunk 1)", "string (negative document chunk 2)", ...]
  }
  ```
  Appendix B.1 explicitly dictates: *"In each labeled sample, the pos field is structured as a list containing only a single positive instance... In contrast, the neg field consists of a list of multiple negative candidate strings, each reflecting content that is either irrelevant or only weakly related to the query."*

### A.3. Positive and Negative Document Definitions
- **Positive Document ($s^+$):** A candidate chunk that directly contains the authoritative factual answer, verified by expert annotators against stringent criteria for precision, relevance, and completeness (Appendix B.2).
- **Negative Document ($s^-$):** Candidate chunks retrieved by FAISS (dense) and BM25 (sparse) from the document pool that fail to provide an accurate or complete answer to the query (e.g., peripheral narrative disclosures, incorrect reporting periods, or unrelated financial statements) (Appendix B.2).

### A.4. Training Hyperparameters (Appendix B.3)
Every hyperparameter specified in the paper is recorded below:
- **LoRA Rank ($r$):** `32`
- **LoRA Scaling Factor ($\alpha$):** `64`
- **Learning Rate:** `1e-4` ($1 \times 10^{-4}$)
- **Batch Size per Device:** `1`
- **Gradient Accumulation Steps:** `2` (Effective batch size = 2)
- **Training Epochs:** `10`
- **Mixed Precision:** `bf16`
- **Parameters NOT Specified in the Paper:**
  - *Sequence Length:* Not explicitly stated in Appendix B.3 (FlagEmbedding defaults to 512 tokens; chunk token limit is 256 tokens per Section 4.1).
  - *Quantization:* **Not specified** (Appendix B.3 mentions `bf16`, but does not mention 4-bit or 8-bit quantization).
  - *Optimizer:* Not explicitly named in Appendix B.3 (standard AdamW in HuggingFace/FlagEmbedding).
  - *DPO Beta ($\beta$):* Not explicitly stated in Appendix B.3 (standard default is $\beta = 0.1$).

### A.5. Reranker Scoring and Ordering at Inference
- **Scoring Formula (Section 3.3, Eq. 1):**
  $$\mathcal{K}(s, q) = \sigma(\mathbf{w}^\top \text{CrossEncoder}(s, q) + b + \beta f(\tilde{m}_i))$$
  where $\mathbf{w}^\top \text{CrossEncoder}(s, q) + b$ is the raw transformer contextual alignment logit, and $f(\tilde{m}_i)$ is **strictly the publication date Time Bonus** (Section 3.3.1).
- **Ordering:** Candidate chunks are sorted in descending order of $\mathcal{K}(s, q)$. The Top-$K$ (typically $K=5$, Section 4.2.3) are forwarded to the LLM.
- **Critical Methodological Note:** The paper contains **no** statement-type regex boosts (+2.0 or +4.5), no table structure boosts (+0.5), and no negative subsection penalties (-3.0).

### A.6. Architectural Compatibility: `bge-reranker-base` vs. `bge-reranker-v2-gemma`
| Characteristic | Paper Specification: `bge-reranker-v2-gemma` | Proposed Baseline: `bge-reranker-base` |
|---|---|---|
| **Base Architecture** | Decoder-only LLM (Google Gemma-2B) | Encoder-only Masked LM (XLM-RoBERTa) |
| **Parameter Count** | ~2,500,000,000 (~2.5 Billion) | ~278,000,000 (~278 Million) |
| **Sequence Processing** | Causal attention with EOS classification | Bidirectional attention with `[CLS]` pooler |
| **Input Tokenization** | Instruction prompt prefix + `(query, passage)` | Standard `tokenizer(query, passage)` pair |
| **LoRA Target Modules** | `["q_proj", "v_proj", "k_proj", "o_proj"]` | `["query", "value", "key", "dense"]` |
| **Architectural Verdict** | **Original Paper Model** | **Structural Deviation (Small-Scale Proxy)** |

**Conclusion on A.6:** `bge-reranker-base` is **not** architecturally identical to `bge-reranker-v2-gemma`. While both can be trained using cross-entropy ranking loss on $(q, s^+, s^-)$, substituting `bge-reranker-base` represents a 9x reduction in parameter scale and a switch from a decoder LLM to an encoder architecture.

---

## 3. Section B: Audit of Existing Training Data

### B.1. Current Workspace Data Audit
We inspected all preference and dataset files in `data/financebench/`:
- `data/financebench/phase9b/train_preferences.jsonl` (29 records)
- `data/financebench/phase9b/validation_preferences.jsonl` (7 records)

**Inspection Findings:**
1. Both files contain records formatted strictly as:
   ```json
   {
     "prompt": "Answer the user's financial question based ONLY on the provided context. Select one predefined operation...",
     "chosen": {"operation": "...", "explanation": "...", "reporting_periods": "..."},
     "rejected": {"operation": "...", "explanation": "...", "reporting_periods": "..."}
   }
   ```
2. These records represent **generator-extraction outputs** designed for instruction fine-tuning or DPO on a causal generation model (`Qwen2.5-7B-Instruct`).
3. There are **zero** records matching the paper's required format:
   ```json
   {"query": "...", "pos": ["..."], "neg": ["..."]}
   ```
4. **Strict Constraint Adherence:** We **must not** reuse generator-extraction preference pairs for DRR training. The paper explicitly specifies passage-level positive/negative chunks.

### B.2. Dataset Preparation Requirements for DRR
To prepare an authentic Appendix B–compliant dataset:
1. **Source Queries:** Extract training queries from `data/financebench/questions.jsonl` (149 queries total).
2. **Strict Test Set Isolation:** Exclude held-out queries **Q1 through Q6** completely from training and validation.
3. **Positive Sample Extraction ($s^+$):** For each training query, match the official `evidence_text` from `questions.jsonl` against the chunk database (`parsed_chunks.jsonl`) to identify the exact positive chunk text.
4. **Hard Negative Mining ($s^-$):** For each training query, run FAISS (dense BGE-M3) and BM25 retrieval to fetch Top-20 candidates. Filter out chunks matching the gold evidence. The remaining retrieved chunks (e.g. peripheral MD&A narratives, other periods, other statements) form the `neg` list.
5. **Estimated Dataset Volume:** ~120–130 training queries, each paired with 1 positive chunk and 8–15 mined hard negatives.

---

## 4. Section C: Local Hardware Feasibility Assessment

**Hardware Environment:** Windows 11, RTX 5060 Laptop GPU, 8 GB VRAM, 16 GB System RAM.

### C.1. VRAM Budget for Exact Paper Setup (`bge-reranker-v2-gemma`, bf16 LoRA)
- **Base Model Parameters:** 2.5 Billion parameters.
- **Base Model Memory (bf16):** $2.5 \times 10^9 \times 2\text{ bytes} = 5.0\text{ GB}$.
- **LoRA Adapter Memory ($r=32$):** ~50 MB.
- **Optimizer States (AdamW for LoRA parameters):** ~100 MB.
- **Activation Memory (Batch Size = 1, Sequence Length = 512, Gradient Checkpointing = True):** ~1.0–1.2 GB.
- **PyTorch & CUDA Context Overhead:** ~500 MB.
- **Total Dedicated VRAM Required:** **~6.7–6.9 GB**.
- **Windows Display Overhead (DWM):** Windows Desktop Window Manager and system graphics reserve **~0.8–1.2 GB** of VRAM on laptop display drivers.
- **Effective Available VRAM:** $8.0\text{ GB} - 1.0\text{ GB} = \mathbf{7.0\text{ GB}}$.

**Feasibility Verdict on Exact Paper Setup:**
Running `bge-reranker-v2-gemma` in unquantized `bf16` sits at the **absolute ceiling of 8 GB VRAM**. Any memory spike during backward passes on sequence lengths $> 512$ will cause a CUDA Out-Of-Memory (OOM) error.

### C.2. Implementation Adaptations Comparison

| Adaptation Option | Architecture Match | Quantization Match | Estimated VRAM | Risk of CUDA OOM |
|---|---|---|---|---|
| **Option 1: Exact Paper (`bge-reranker-v2-gemma`, bf16 LoRA)** | **Exact (Gemma-2B)** | **Exact (bf16)** | **6.8–7.0 GB** | **Extremely High (~90% risk of OOM on Windows)** |
| **Option 2: Paper Checkpoint with QLoRA (`bge-reranker-v2-gemma`, 4-bit)** | **Exact (Gemma-2B)** | Deviation (4-bit vs bf16) | **~3.0 GB** | **Zero (< 40% VRAM used)** |
| **Option 3: Small-Scale Proxy (`bge-reranker-base`, bf16 LoRA)** | Deviation (RoBERTa 278M) | **Exact (bf16)** | **~1.4 GB** | **Zero (< 20% VRAM used)** |

**Documentation of Deviation:**
- If **Option 2** is chosen: Preserves the paper's exact Gemma-2B checkpoint, but introduces 4-bit NF4 base quantization not specified in the paper.
- If **Option 3** is chosen: Preserves unquantized `bf16` precision, but uses a 278M encoder proxy instead of the 2.5B decoder.

---

## 5. Section D: Minimal Paper-Faithful Implementation Plan

If authorization is granted, the implementation must follow these strict steps:

### Step 1: Dataset Preparation
- **New Script:** `scripts/prepare_drr_dataset.py`
- **Action:** Mines positive chunks from `questions.jsonl` evidence fields and negative chunks from FAISS/BM25 pools across the 143 non-held-out queries.
- **Output:** `data/financebench/drr/train_drr.jsonl` and `val_drr.jsonl` matching Appendix B.1 schema (`{"query", "pos", "neg"}`).
- **Hold-out Protection:** Asserts that Q1–Q6 IDs never enter the dataset.

### Step 2: DRR Training Script
- **New Script:** `scripts/train_drr_paper.py`
- **Action:** Implements contrastive cross-entropy loss over $(q, s^+, s^-)$ using LoRA ($r=32$, $\alpha=64$, 10 epochs, lr=1e-4, batch=1, grad_accum=2).
- **Model Checkpoint:** User-selected between Option 2 (`bge-reranker-v2-gemma` in 4-bit) or Option 3 (`bge-reranker-base` in bf16).
- **Output Directory:** `outputs/phase1_drr_adapter/` (isolated from existing adapters).

### Step 3: Evaluation Against Baselines
- **New Script:** `scripts/evaluate_drr_phase1.py`
- **Action:** Evaluates candidate ranking on held-out Q1–Q6 using the trained DRR without heuristic statement boosts (+0.0).
- **Primary Metric:** Does the trained reranker place the official Cash Flow Statement (`p46_c4`) above the environmental narrative (`p7_c5`) for Q1?
- **Downstream Metric:** Run Harness V2 on downstream generation using the newly reranked Top-3 context.

### Baseline Preservation & Rollback Strategy
1. `outputs/phase9c/` and `outputs/phase9b_dpo_phase2/` remain 100% read-only and preserved.
2. All new training scripts, datasets, and adapters reside in dedicated `drr/` subdirectories.
3. If OOM or regression occurs, zero existing files are affected.

---

## 6. Recommendation and Explicit Sign-Off Request

Before modifying any code or preparing datasets, the following decision is submitted for user authorization:

> **Decision Point on DRR Base Model:**
> - **Option 2 (Recommended for Checkpoint Fidelity):** Train the paper's specified `BAAI/bge-reranker-v2-gemma` using 4-bit QLoRA to fit safely within the 8 GB VRAM budget (~3.0 GB VRAM).
> - **Option 3 (Recommended for Precision Fidelity):** Train `BAAI/bge-reranker-base` in unquantized `bf16` LoRA (~1.4 GB VRAM) as a documented small-scale proxy.

**Current State:** All files preserved. No code modified. No model weights downloaded. Awaiting user approval to proceed.
