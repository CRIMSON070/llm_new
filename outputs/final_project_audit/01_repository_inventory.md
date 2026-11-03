# FinSage Repository Inventory & File Registry

**Document:** `outputs/final_project_audit/01_repository_inventory.md`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Governing Standard:** Comprehensive Repository Inspection (Read-Only)  
**Excluded Directories:** `.git`, `.venv`, `.vscode`, `__pycache__`, `mineru_env`  

---

## 1. Inventory Summary & Functional Categorization

The FinSage codebase contains **41 Python scripts and modules**, **2 configuration files**, **84 PDF filings**, **5 indexed stores (FAISS and BM25)**, **3 trained LoRA adapter directories**, and **over 30 structured evaluation reports and JSON/JSONL output artifacts**.

### Functional Breakdown of Workspace Assets

| Functional Role | Key Files / Directories | Count | Primary Purpose |
|---|---|:---:|---|
| **Primary Entrypoints** | `app.py`, `src/server.py` | 2 | Web UI dashboard and REST API service |
| **Pipeline Training Scripts** | `scripts/train_drr_paper.py`, `scripts/run_dpo_phase2.py`, `scripts/run_dpo_pilot.py` | 3 | DPO fine-tuning for DRR reranker and generator |
| **Data Preparation Scripts** | `scripts/download_source_pdfs.py`, `scripts/preprocess_documents.py`, `scripts/prepare_drr_dataset.py`, `scripts/enrich_phase2_chunks.py` | 6 | SEC PDF download, PyMuPDF chunking, DPO pair generation, table header propagation |
| **Indexing Scripts** | `scripts/build_vector_index.py`, `scripts/build_bm25_index.py`, `scripts/build_heading_summary_index.py` | 3 | Construction of BGE-M3 FAISS, bm25s, and metadata FAISS indices |
| **Retrieval Modules** | `src/retrieval/*.py`, `src/utils/bm25Retriever.py`, `src/utils/faissRetriever.py` | 10 | 4-path MPR retrieval, chunk bundling, and reranking |
| **Evaluation & Diagnostic Scripts** | `scripts/evaluate_phase1_drr.py`, `scripts/run_generation_phase1_drr.py`, `scripts/evaluate_phase1_chunk_bundling.py`, `scripts/run_complete_phase1_reproduction.py`, `scripts/run_phase2_benchmark.py` | 14 | Retrieval evaluation, downstream generation, Harness V2 verification |
| **Configuration Files** | `config/production.yaml`, `config/example.yaml` | 2 | Model identifiers, file paths, retrieval top-k parameters |
| **Dataset Stores** | `data/financebench/pdfs/`, `data/financebench/processed/`, `data/financebench/questions.jsonl` | 3 | 84 PDFs, 54,564 parsed chunks, 150 ground-truth questions |
| **Model Artifacts** | `outputs/phase1_drr_adapter/`, `outputs/phase9b_dpo_phase2/`, `outputs/phase9b_dpo_pilot/` | 3 | Trained LoRA safetensors checkpoints |
| **Evaluation Outputs & Reports** | `outputs/phase1_audit/`, `outputs/phase1_drr_evaluation/`, `outputs/phase1_chunk_bundling/`, `outputs/phase1_complete_reproduction/`, `outputs/phase2/`, `outputs/phase9c/` | 6 dirs | Empirical metrics, scorecards, per-question logs, markdown reports |

---

## 2. Comprehensive File Inventory Table

| Repository-Relative Path | File Type | Functional Role | Pipeline Stage | Necessary for Reproduction? | Evidence Status in Codebase |
|---|:---:|---|:---:|:---:|---|
| **Root & Entrypoint Files** | | | | | |
| `app.py` | Python (`.py`) | Interactive Streamlit Dashboard (1,776 lines); supports Mode toggling | Presentation / UI | Optional (UI) | **Implemented & Verified** (Running on `:8501`) |
| `README.md` | Markdown (`.md`) | Original repository README containing paper tables and API instructions | Documentation | No | **Implemented & Verified** (UTF-16LE, 21.5 KB) |
| `README_dashboard.md` | Markdown (`.md`) | Guide for running the Streamlit dashboard | Documentation | No | **Implemented & Verified** |
| `requirements_dashboard.txt`| Text (`.txt`) | Dependencies for Streamlit, Plotly, Pandas | Environment | Yes (UI) | **Implemented & Verified** |
| `FINSAGE_PROJECT_AUDIT.md` | Markdown (`.md`) | Prior technical audit (Oct 26, 2026) documenting Phase 1 baseline | Documentation | No | **Implemented & Verified** (63.5 KB) |
| `PHASE1_OFFICIAL_RESULTS.md`| Markdown (`.md`)| Formal archive of Phase 1 benchmark results and Q1–Q6 scores | Documentation | No | **Implemented & Verified** (10.1 KB) |
| `PHASE2_PLAN.md` | Markdown (`.md`) | Work package roadmap for Stage 2 production improvements | Planning | No | **Implemented & Verified** (4.6 KB) |
| **Configuration Files** | | | | | |
| `config/production.yaml` | YAML (`.yaml`) | Production runtime settings: paths, embedding/reranker models, token limits | All Stages | **YES** | **Implemented & Verified** |
| `config/example.yaml` | YAML (`.yaml`) | Template configuration file | All Stages | No | **Implemented & Verified** |
| **Core Retrieval & Model Code (`src/retrieval/`)** | | | | | |
| `src/retrieval/multi_path_retriever.py` | Python (`.py`) | 4-Path MPR Coordinator; queries BM25, Dense, Metadata, HyDE; applies RRF | Stage 1 / MPR | **YES** | **Implemented & Verified** (204 lines) |
| `src/retrieval/chunk_bundler.py` | Python (`.py`) | Semantic adjacency chunk bundling ($\tau_{\text{exp}}=0.70, \mathcal{K}=5$) | Stage 1 / MPR | **YES** | **Implemented & Verified** (130 lines) |
| `src/retrieval/hyde_generator.py` | Python (`.py`) | Generative HyDE; generates 3 hypothetical passages with Qwen2.5-7B | Stage 1 / MPR | **YES** | **Implemented & Verified** (96 lines) |
| `src/retrieval/hyde_retriever.py` | Python (`.py`) | Embeds HyDE passages and retrieves FAISS candidates | Stage 1 / MPR | **YES** | **Implemented & Verified** |
| `src/retrieval/metadata_retriever.py` | Python (`.py`) | Queries 2,678-section heading-summary FAISS index | Stage 1 / MPR | **YES** | **Implemented & Verified** |
| `src/retrieval/statement_reranker.py`| Python (`.py`) | Cross-encoder reranker wrapper; supports zero boosts and Phase 9C boosts | Stage 1 / DRR | **YES** | **Implemented & Verified** |
| `src/retrieval/financial_expander.py`| Python (`.py`) | Legacy regex keyword expander used in Phase 9C | Historical / 9C | No (Historical) | **Implemented & Verified** |
| `src/retrieval/phase2_reranker.py` | Python (`.py`) | Stage 2 reranker with calibrated statement routing and distractor penalty | Stage 2 | **YES (Stage 2)** | **Implemented & Verified** |
| **Utility Modules (`src/utils/`)** | | | | | |
| `src/utils/bm25Retriever.py` | Python (`.py`) | Lexical search wrapper utilizing `bm25s` library | Stage 1 / BM25 | **YES** | **Implemented & Verified** |
| `src/utils/faissRetriever.py` | Python (`.py`) | Dense vector store interface for LangChain FAISS | Stage 1 / FAISS | **YES** | **Implemented & Verified** |
| `src/utils/arithmetic_evaluator.py` | Python (`.py`) | Deterministic Python calculation sandbox for financial formulas | Stage 2 / Eval | **YES (Stage 2)** | **Implemented & Verified** |
| `src/utils/ensembleRetriever.py` | Python (`.py`) | Legacy Lotus retrieval manager from original paper repository | Paper Legacy | No (Lotus only) | **Implemented & Verified** |
| `src/utils/vllmChatService.py` | Python (`.py`) | Async LLM chat generation service via OpenAI-compatible endpoint | Serving | Optional | **Implemented & Verified** |
| `src/utils/vllmManager.py` | Python (`.py`) | Process manager for launching and monitoring vLLM instances | Serving | Optional | **Implemented & Verified** |
| `src/utils/profiler.py` | Python (`.py`) | Timing profiler for latency benchmarking | Telemetry | No | **Implemented & Verified** |
| `src/server.py` | Python (`.py`) | FastAPI / Gunicorn REST backend (`/api_chat`, `/health`) | Serving | Optional | **Implemented & Verified** |
| **Pipeline Scripts (`scripts/`)** | | | | | |
| `scripts/download_source_pdfs.py` | Python (`.py`) | Downloads 84 SEC filing PDFs based on `manifest.json` | Data Prep | **YES (Setup)** | **Implemented & Verified** |
| `scripts/preprocess_documents.py` | Python (`.py`) | PyMuPDF text & table extraction into `parsed_chunks.jsonl` | Ingestion (FFP) | **YES (Setup)** | **Implemented & Verified** |
| `scripts/build_bm25_index.py` | Python (`.py`) | Builds `bm25s` index over 54,564 chunks ($k_1=1.5, b=0.75$) | Indexing | **YES (Setup)** | **Implemented & Verified** |
| `scripts/build_vector_index.py` | Python (`.py`) | Builds BGE-M3 FAISS index (54,564 vectors, 1024-dim, L2-normalized) | Indexing | **YES (Setup)** | **Implemented & Verified** |
| `scripts/build_heading_summary_index.py`| Python (`.py`)| Builds metadata FAISS index over 2,678 filing sections | Indexing | **YES (Setup)** | **Implemented & Verified** |
| `scripts/prepare_drr_dataset.py` | Python (`.py`) | Mines hard triplets and formats DPO train/val pairs with zero leakage | DRR Prep | **YES (Training)**| **Implemented & Verified** |
| `scripts/train_drr_paper.py` | Python (`.py`) | Trains Gemma cross-encoder via 4-bit NF4 DPO LoRA (10 epochs) | Stage 1 / DRR | **YES (Training)**| **Implemented & Verified** (27.1 KB) |
| `scripts/evaluate_phase1_drr.py` | Python (`.py`) | Reranker evaluation harness on Q1–Q6 under zero boosts (+0.0) | Stage 1 / Eval | **YES** | **Implemented & Verified** (22.4 KB) |
| `scripts/run_generation_phase1_drr.py`| Python (`.py`)| End-to-end downstream generation with Qwen2.5-7B on Q1–Q6 | Stage 1 / Eval | **YES** | **Implemented & Verified** (25.3 KB) |
| `scripts/evaluate_phase1_chunk_bundling.py`| Python (`.py`)| Evaluates bundling on Q1–Q6 ($\tau=0.85$ vs. $\tau=0.70$) | Stage 1 / Eval | **YES** | **Implemented & Verified** (19.4 KB) |
| `scripts/generate_q1_bundled.py` | Python (`.py`) | Generates answer for Q1 using bundled Page 60 context | Stage 1 / Eval | **YES** | **Implemented & Verified** (3.8 KB) |
| `scripts/run_complete_phase1_reproduction.py`| Python (`.py`)| Executes full 4-Path MPR + Bundling + DRR + Generation pipeline | Stage 1 / Eval | **YES** | **Implemented & Verified** (18.5 KB) |
| `scripts/enrich_phase2_chunks.py`| Python (`.py`) | Implements Stage 2 Table Header Propagation across 2,902 chunks | Stage 2 / Ingest| **YES (Stage 2)** | **Implemented & Verified** (6.8 KB) |
| `scripts/run_phase2_benchmark.py` | Python (`.py`) | Executes full $N=150$ benchmark evaluation across all 84 filings | Stage 2 / Eval | **YES (Stage 2)** | **Implemented & Verified** (27.8 KB) |
| `scripts/evaluate_harness_v2.py` | Python (`.py`) | Decoupled 4-dimension deterministic evaluation harness | Evaluation | **YES** | **Implemented & Verified** (28.1 KB) |
| `scripts/run_dpo_phase2.py` | Python (`.py`) | 3-epoch DPO training script for Generator LLM (historical) | Historical / 9B | No (Historical) | **Implemented & Verified** |
| `scripts/run_dpo_pilot.py` | Python (`.py`) | 1-epoch DPO pilot training script for Generator LLM (historical) | Historical / 9B | No (Historical) | **Implemented & Verified** |
| `scripts/run_retrieval_phase9c.py`| Python (`.py`)| Historical retrieval script with +5.5 heuristic boosts | Historical / 9C | No (Historical) | **Implemented & Verified** |
| `scripts/run_generation_phase9c.py`| Python (`.py`)| Historical downstream generation script for Phase 9C | Historical / 9C | No (Historical) | **Implemented & Verified** |
| `scripts/pilot_multimodal.py` | Python (`.py`) | Pilot script testing MinerU layout parsing on 1 filing | Ingestion Pilot | No (Pilot only) | **Implemented & Verified** |
| **Data Artifacts (`data/financebench/`)** | | | | | |
| `data/financebench/pdfs/` | PDF directory | 84 original SEC regulatory filings (10-K, 10-Q, 8-K, Earnings) | Corpus Data | **YES** | **Verified Present (84 files)** |
| `data/financebench/manifest.json` | JSON (`.json`) | Download manifest for all 84 PDF filings | Corpus Data | **YES** | **Verified Complete (84 entries)** |
| `data/financebench/questions.jsonl` | JSONL (`.jsonl`)| 150 ground-truth questions with gold citations and answers | Ground Truth | **YES** | **Verified Complete (150 rows)** |
| `data/financebench/processed/parsed_chunks.jsonl`| JSONL (`.jsonl`)| 54,564 text & table chunks extracted via PyMuPDF (57.1 MB) | Chunk Corpus | **YES** | **Verified Present** |
| `data/financebench/faiss_index/` | FAISS Index | BGE-M3 dense chunk vector index (54,564 vectors, 223 MB) | Search Index | **YES** | **Verified Present** |
| `data/financebench/bm25_index/` | BM25 Index | `bm25s` sparse lexical index files (7 files) | Search Index | **YES** | **Verified Present** |
| `data/financebench/metadata_faiss_index/`| FAISS Index | Section heading-summary index (2,678 vectors, 10.9 MB) | Search Index | **YES** | **Verified Present** |
| `data/financebench/drr/` | Directory | DRR training (`train_drr.jsonl`, 125 pairs) and val (`val_drr.jsonl`, 17 pairs) | DRR Training | **YES** | **Verified Present & Audited** |
| `data/financebench/phase9b/` | Directory | Generator DPO preference pairs (28 train, 8 val) | Historical / 9B | No (Historical) | **Verified Present** |
| **Trained Model Checkpoints (`outputs/`)** | | | | | |
| `outputs/phase1_drr_adapter/adapter_model.safetensors`| SafeTensors | **Official Stage 1 DRR Adapter** (`bge-reranker-v2-gemma` 4-bit LoRA, 29.5 MB) | Stage 1 / DRR | **YES** | **Trained & Verified Active** |
| `outputs/phase9b_dpo_phase2/adapter_model.safetensors`| SafeTensors | 3-epoch Generator DPO adapter (`Qwen2.5-7B-Instruct`, 5.06 MB) | Historical / 9B | No (Historical) | **Preserved Archive** |
| `outputs/phase9b_dpo_pilot/adapter_model.safetensors` | SafeTensors | 1-epoch Generator DPO adapter (`Qwen2.5-7B-Instruct`, 5.06 MB) | Historical / 9B | No (Historical) | **Preserved Archive** |
| **Official Evaluation Reports & Data (`outputs/`)** | | | | | |
| `outputs/phase1_audit/final_reproduction_completion_audit.md` | Markdown | Comprehensive audit of DRR training and zero-boost evaluation | Stage 1 Audit | No | **Verified Present (26.8 KB)** |
| `outputs/phase1_audit/phase1_reproduction_gap_analysis.md` | Markdown | Gap analysis mapping codebase against paper sections | Stage 1 Audit | No | **Verified Present (16.4 KB)** |
| `outputs/phase1_audit/evaluation_harness_v2/` | Directory | Harness V2 per-question results, summary metrics, and report | Stage 1 Audit | No | **Verified Present** |
| `outputs/phase1_chunk_bundling/bundling_diagnostics.json` | JSON | Diagnostic trace of chunk similarity and expansion for Q1–Q6 | Stage 1 Eval | No | **Verified Present** |
| `outputs/phase1_chunk_bundling/q1_generation_result.json` | JSON | Q1 generation result ($1,577M extracted via bundling) | Stage 1 Eval | No | **Verified Present** |
| `outputs/phase1_complete_reproduction/phase1_full_reproduction_results.json`| JSON | Full 4-path reproduction results across Q1–Q6 | Stage 1 Eval | No | **Verified Present (26.1 KB)** |
| `outputs/phase1_drr_evaluation/summary_metrics.json` | JSON | MRR and Recall@K comparison for base vs. trained DRR | Stage 1 Eval | No | **Verified Present** |
| `outputs/phase2/reports/PHASE2_OFFICIAL_RESULTS.md` | Markdown | Official Stage 2 report with full $N=150$ FinanceBench results | Stage 2 Results | No | **Verified Present (10.3 KB)** |
| `outputs/phase2/evaluation/phase2_full_scorecard.json` | JSON | Macro scorecards across 150 questions and 9 GICS sectors | Stage 2 Results | No | **Verified Present (13.0 KB)** |
| `outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl` | JSONL | Full candidate rankings for all 150 questions in Stage 2 | Stage 2 Results | No | **Verified Present (60.4 KB)** |
| `outputs/phase9c/phase9c_results.jsonl` | JSONL | Historical Phase 9C per-question results (+5.5 boosts) | Historical / 9C | No (Historical) | **Verified Present (65.5 KB)** |
| `outputs/phase9c/phase9c_report.md` | Markdown | Historical Phase 9C analysis report | Historical / 9C | No (Historical) | **Verified Present (19.6 KB)** |

---

## 3. Minimal Reproduction Package

To reproduce the project's experiments on a fresh machine with an NVIDIA GPU (8 GB+ VRAM), the minimal required files are:

1. **Environment Setup:** `requirements_dashboard.txt` + `config/production.yaml`.
2. **Data Stores:** `data/financebench/manifest.json`, `data/financebench/questions.jsonl`, `data/financebench/processed/parsed_chunks.jsonl`.
3. **Pre-built Indices:** `data/financebench/faiss_index/`, `data/financebench/bm25_index/`, `data/financebench/metadata_faiss_index/`.
4. **Trained Weights:** `outputs/phase1_drr_adapter/adapter_model.safetensors` and `adapter_config.json`.
5. **Codebase:** Entire `src/` directory.
6. **Execution Harnesses:** `scripts/evaluate_phase1_drr.py`, `scripts/evaluate_phase1_chunk_bundling.py`, `scripts/run_generation_phase1_drr.py`, `scripts/run_phase2_benchmark.py`, and `app.py`.
