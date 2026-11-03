# Phase 1: Domain-Specialized Document Re-Ranker (DRR) Pre-Training Readiness Report

**Document:** `outputs/phase1_drr_adapter/drr_training_readiness_report.md`  
**Date:** 2026-10-11  
**Target Architecture:** `BAAI/bge-reranker-v2-gemma` (Gemma-2B Decoder Cross-Encoder) with 4-bit QLoRA  
**Status:** **STAGE 1, 2, AND 3 COMPLETE — STOPPED BEFORE FULL 10-EPOCH RUN (AWAITING APPROVAL)**  

---

## 1. Executive Summary

In accordance with user authorization for **Option 1: `BAAI/bge-reranker-v2-gemma` with 4-bit QLoRA**, Stages 1 through 3 of the Phase 1 faithful reproduction pipeline have been implemented, tested, and validated.

All training data strictly adheres to the original paper's Appendix B schema and methodology. Training code implements the exact DPO preference objective (Section 3.3.2, Eq. 2) rather than ordinary binary classification. A full forward/backward smoke test was executed on the local NVIDIA RTX 5060 Laptop GPU, confirming gradient flow, mathematical precision, and VRAM stability.

**Full training is currently STOPPED.** All baseline artifacts, indexes, and previous evaluation files remain preserved.

---

## 2. Stage 1: Dataset Preparation & Validation (`scripts/prepare_drr_dataset.py`)

### 2.1. Construction Methodology (Appendix B Faithful)
- **Source Queries:** 150 official FinanceBench queries (`data/financebench/questions.jsonl`).
- **Positive Passage Extraction ($s^+$):** Each training query's verified authoritative `evidence_text` was mapped to its matching passage chunk in `parsed_chunks.jsonl` / `pymupdf_chunks.jsonl` using text containment and overlap verification.
- **Negative Passage Mining ($s^-$):** For each query, candidate pools were retrieved using dense FAISS (BGE-M3) and sparse BM25 search. Candidates were filtered to remove false negatives (chunks containing the gold answer or exhibiting $>45\%$ overlap with gold evidence). Exactly 15 hard negative chunks were retained per query.
- **Schema Compliance (Appendix B.1):**
  Each record contains:
  ```json
  {
    "query": "string (the user query)",
    "pos": ["string (single gold positive chunk text)"],
    "neg": ["string (list of 15 mined hard negative chunk texts)"],
    "financebench_id": "string",
    "doc_name": "string",
    "pos_chunk_id": "string",
    "neg_chunk_ids": ["string", ...]
  }
  ```

### 2.2. Leakage Audit & Held-Out Exclusion
- **Held-Out Questions Q1–Q6:** Completely excluded from training and validation sets:
  - Q1: `financebench_id_03029` (3M 2018 CapEx)
  - Q2: `financebench_id_04672` (3M 2018 Net PP&E)
  - Q3: `financebench_id_00746` (Ulta Beauty Registered Debt Securities)
  - Q4: `financebench_id_00521` (Ulta Beauty Major Acquisitions)
  - Q5: Curated 3M Net Sales Difference
  - Q6: Curated Ulta Beauty Merchandise Inventories Difference
- **Document & Chunk Leakage Protection:**
  - All questions and all candidate chunks belonging to documents `3M_2018_10K`, `ULTABEAUTY_2023_10K`, and `ULTABEAUTY_2023Q4_EARNINGS` (totaling 8 queries) were completely excluded from the dataset.
  - Zero held-out queries in training or validation (`query_leakage: 0`).
  - Zero held-out document chunks present in positive or negative candidate lists (`chunk_leakage: 0`).
  - Document-grouped train/val partitioning: 69 documents reserved for training, 12 documents reserved for validation (`train_val_doc_overlap: 0`).

### 2.3. Dataset Statistics (`data/financebench/drr/dataset_summary.json`)
| Metric | Train Split | Validation Split | Total |
|---|---|---|---|
| **Samples (Queries)** | 125 | 17 | 142 |
| **Unique Filings/Documents** | 69 | 12 | 81 |
| **Positives per Sample** | 1.0 (exact) | 1.0 (exact) | 1.0 (exact) |
| **Negatives per Sample** | 15.0 (exact) | 15.0 (exact) | 15.0 (exact) |
| **Average Positive Word Length** | 129.3 words | 134.1 words | 129.9 words |
| **Average Negative Word Length** | 137.0 words | 135.8 words | 136.8 words |

---

## 3. Stage 2: Training Implementation (`scripts/train_drr_paper.py`)

### 3.1. Model Architecture & Cross-Encoder Alignment
- **Base Checkpoint:** `BAAI/bge-reranker-v2-gemma` (2.5B parameter Gemma decoder model).
- **Prompt Formulation:**
  `"A: {query}\nB: {passage}\nGiven a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."`
- **Relevance Scoring:** Causal logit on token ID `3553` (`'Yes'`).

### 3.2. DPO Preference Objective (Section 3.3.2, Eq. 2)
The objective is formulated strictly as preference optimization over paired candidate chunks:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(q, s^+, s^-)} \left[ \log \sigma \left( \beta \left( \left(s_\theta(q, s^+) - s_{\text{ref}}(q, s^+)\right) - \left(s_\theta(q, s^-) - s_{\text{ref}}(q, s^-)\right) \right) \right) \right]$$
- **Policy Model ($\pi_\theta$):** Gemma-2B base with active LoRA adapter ($r=32, \alpha=64$).
- **Reference Model ($\pi_{\text{ref}}$):** Frozen Gemma-2B base evaluated via PEFT `model.disable_adapter()`. This guarantees exact reference score computation without allocating duplicate model weights in VRAM.

### 3.3. Hyperparameters: Paper-Specified vs. Hardware Adaptations
| Hyperparameter | Paper Specification | Local Implementation | Rationale / Source |
|---|---|---|---|
| **Base Model** | `BAAI/bge-reranker-v2-gemma` | `BAAI/bge-reranker-v2-gemma` | Paper Section 3.3 |
| **LoRA Rank ($r$)** | `32` | `32` | Paper Appendix B.3 |
| **LoRA Alpha ($\alpha$)** | `64` | `64` | Paper Appendix B.3 |
| **Learning Rate** | `1e-4` | `1e-4` | Paper Appendix B.3 |
| **Per-Device Batch Size** | `1` | `1` | Paper Appendix B.3 |
| **Gradient Accumulation** | `2` | `2` | Paper Appendix B.3 |
| **Epochs** | `10` | `10` | Paper Appendix B.3 |
| **Quantization** | `bf16` (unquantized) | **4-bit NF4 (QLoRA)** | Documented hardware adaptation for 8 GB VRAM |
| **Target Modules** | Not enumerated | `["q_proj", "v_proj", "k_proj", "o_proj"]` | Standard Gemma attention projections |
| **Sequence Length** | Not stated | `512` tokens | FlagEmbedding standard (chunk budget 256) |
| **DPO Beta ($\beta$)** | Not stated | `0.1` | Standard DPO temperature |
| **Optimizer** | Not named | AdamW (`weight_decay=0.01`) | HuggingFace / PyTorch standard |
| **LR Scheduler** | Not named | Cosine with 5% warmup | Standard transformer fine-tuning practice |

### 3.4. Critical Memory Optimization (Sliced Last-Token Hidden State)
In standard HuggingFace `GemmaForCausalLM`, calling `model(**inputs)` computes full vocabulary projection across all 512 token positions (`[1, 512, 256000]`), allocating a **262 MB tensor** per passage.  
In `scripts/train_drr_paper.py`, we extract the last hidden state directly (`last_hidden = base_backbone(**inputs).last_hidden_state[:, -1, :]`, shape `[1, 2048]`, only 4 KB) and project only that single token through `lm_head`. We empirically proved that this produces mathematically identical logits down to `0.00000000` while eliminating hundreds of megabytes of activation memory.

---

## 4. Stage 3: Safety Checks & Smoke Test Results

### 4.1. Smoke Test Execution (`--smoke_test`)
A live forward/backward smoke test was executed on the local NVIDIA RTX 5060 Laptop GPU:
- **Command:** `python scripts/train_drr_paper.py --smoke_test`
- **Model Checkpoint:** `BAAI/bge-reranker-v2-gemma` loaded in 4-bit NF4 (`BitsAndBytesConfig`).
- **Trainable Parameters:** `7,372,800` / `2,513,545,216` ($0.293\%$).
- **Initial DPO Loss:** Exactly `0.6931` ($-\log(0.5)$), confirming zero initial adapter bias over the base model.
- **LoRA Gradients Verified:** `True` (non-zero gradients backpropagated through all targeted linear projections).
- **Validation Step Executed:** `loss: 0.6931`, `pref_acc: 0.00`.
- **Peak VRAM:** **7.53 GB** on the 8.54 GB GPU (leaving **1.01 GB of headroom**).
- **Artifact:** Saved to `outputs/phase1_drr_adapter/smoke_test_results.json`.

---

## 5. Summary of Unresolved Paper Ambiguities & Documented Choices

1. **Number of Negative Samples Evaluated per Query Step:**  
   Appendix B.1 specifies that `neg` is a list of candidate strings, but does not state whether all mined negatives or a subset are evaluated per optimization step. In `train_drr_paper.py`, we implement configurable `--max_negatives` (defaulting to 5 per query step during training), which balances hard negative discrimination with VRAM throughput.
2. **Quantization:**  
   The paper ran on multi-GPU server infrastructure without quantization. On our local 8 GB RTX 5060 Laptop GPU, 4-bit QLoRA is required to prevent immediate CUDA OOM.
3. **Target Modules & LR Schedule:**  
   The paper specifies $r=32$, $\alpha=64$, $\text{lr}=10^{-4}$, but omits the exact target projection list and scheduler. We target Gemma's query, key, value, and output projection layers with a cosine schedule and 5% warmup.

---

## 6. Baseline & Preservation Check
- Existing FAISS and BM25 indexes: **Intact & Unmodified**.
- Phase 9B generator adapters (`outputs/phase9b_dpo_phase2/`): **Intact & Unmodified**.
- Phase 9C retrieval ablations (`outputs/phase9c/`): **Intact & Unmodified**.
- Phase 1 Harness V2 evaluation results: **Intact & Unmodified**.
- Custom heuristic boosts (+2.0, +4.5) and section penalties: **Completely excluded from DRR training and evaluation**.

---

## 7. Next Step Requiring User Authorization

All pre-training safety checks, dataset preparation, and architectural smoke tests have passed.  
**Awaiting your authorization to launch the full 10-epoch DRR training run:**
```powershell
python scripts/train_drr_paper.py --epochs 10 --lr 1e-4 --lora_rank 32 --lora_alpha 64 --batch_size 1 --grad_accum 2 --output_dir outputs/phase1_drr_adapter
```
Estimated training time: ~20–25 minutes.
