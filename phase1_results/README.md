# FinSage Phase 1 Complete Archive & Preservation Package

This directory contains the complete, frozen archive of **FinSage Phase 1 (Faithful Paper Reproduction of arXiv:2504.14493, ACM CIKM 2025)**.

## Package Contents

### 1. Model Checkpoints & Adapters (`phase1_drr_adapter/`)
- `adapter_model.safetensors` (29.5 MB): Fine-tuned 4-bit NF4 QLoRA adapter on `BAAI/bge-reranker-v2-gemma` ($r=32, \alpha=64$, 10 epochs, DPO loss).
- `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`
- `training_history.json`, `drr_training_metadata.json`

### 2. Evaluation Results & Scorecards
- `phase1_complete_reproduction/`: 4-Path MPR + Chunk Bundling full evaluation results and reports.
- `phase1_chunk_bundling/`: Adjacency cosine expansion diagnostics ($\tau=0.70$) and assembled contexts.
- `phase1_drr_evaluation/`: Zero-boost retrieval ranking (MRR 0.611) and Qwen2.5-7B downstream generation.
- `phase1_audit/`: Gap analyses, alignment audits, and Harness V2 multi-dimensional scorecard.
- `phase1_consolidated_metrics.json`: Consolidated master metrics across all runs.
- `PHASE1_OFFICIAL_RESULTS.md`: Official benchmark scorecard and question-by-question breakdown.

### 3. Reproduction Code (`reproduction_scripts/`)
- `run_complete_phase1_reproduction.py`: Executes the full 4-Path MPR pipeline.
- `evaluate_phase1_drr.py`: Computes ranking metrics (MRR, Recall@K) under zero heuristic boosts (+0.0).
- `run_generation_phase1_drr.py`: Generates answers using Qwen2.5-7B-Instruct with reranked Top-3 context.
- `evaluate_phase1_chunk_bundling.py`: Evaluates table bisection resolution.
- `train_drr_paper.py`: 10-epoch DPO training script for the Gemma reranker.
- `evaluate_harness_v2.py`: Offline decoupled evaluation suite.

### 4. Raw Logs (`evaluation_logs/`)
- All terminal output logs from Phase 1 training and evaluation runs.

---
*Archived on October 30, 2026. Standalone compressed file: `phase1_results.zip` (31.37 MB).*
