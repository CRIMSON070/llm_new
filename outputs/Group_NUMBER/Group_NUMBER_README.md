# FinSage: Replication & Benchmark Execution Guide

**Course Project:** FinSage Financial RAG System Reproduction (Stage 1) & Production Enhancements (Stage 2)  
**Group Number:** Group [NUMBER]  
**Team Members:** [Name (Student ID), Name (Student ID), ...]  
**Course & Instructor:** [Course / instructor, optional]  
**Governing Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering* (Wang et al., arXiv:2504.14493, ACM CIKM 2025)  

---

## 1. System Requirements & Environment Setup

This project is optimized to run on consumer hardware (tested on an 8GB NVIDIA GeForce RTX 5060 Laptop GPU under Windows 11).

### 1.1 Prerequisites
* Python `3.10.x` (recommended: Anaconda / Miniconda)
* NVIDIA GPU with CUDA support (Compute Capability 7.0+ recommended)
* Git

### 1.2 Installation Steps
```bash
# 1. Clone repository and navigate to root
cd c:\Users\user\Downloads\finsage-main

# 2. Activate Conda environment
conda activate finsage

# 3. Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 peft==0.13.2 bitsandbytes==0.44.1 accelerate==1.0.1
pip install faiss-cpu bm25s pymupdf pymupdf4llm streamlit plotly pandas reportlab python-pptx
```

---

## 2. Running Stage 1: Faithful Paper Reproduction

Stage 1 reproduces the published core mechanics of *arXiv:2504.14493* (Multi-Path RRF, Chunk Bundling, and DPO Cross-Encoder Re-ranking) under **strictly zero heuristic boosts (+0.0)**.

### 2.1 Step 1: Preprocess Documents & Build Baseline Indexes
```bash
# Ingest 84 SEC PDFs into 54,564 text & markdown table chunks
python scripts/preprocess_documents.py

# Build sparse BM25 index
python scripts/build_bm25_index.py

# Build dense FAISS vector index using BGE-M3
python scripts/build_dense_index.py

# Build section heading metadata index
python scripts/build_heading_summary_index.py
```
*Expected Output:* `data/financebench/processed/parsed_chunks.jsonl` (54,564 chunks) and FAISS index at `data/financebench/faiss_index/index.faiss`.

### 2.2 Step 2: Run Full 150-Question Stage 1 Benchmark
```bash
python scripts/run_faithful_stage1_benchmark.py
```
*Expected Output:*
* Scorecard saved to: `outputs/reproduction_stage1_faithful/scorecard_stage1_full.json`
* Benchmark Report: `outputs/reproduction_stage1_faithful/stage1_reproduction_report.md`
* Key Metrics:
  * Multi-Path RRF Pool Recall@40: **75.33%**
  * Base Gemma Cross-Encoder MRR: **0.3503** (Recall@5 = 45.33%)
  * DPO Trained Cross-Encoder MRR: **0.5379** (Recall@5 = 65.33%)

### 2.3 Step 3: Run Diagnostic End-to-End Suite (Q1–Q6)
```bash
python scripts/run_complete_phase1_reproduction.py
```
*Expected Output:* Evaluation metrics across 6 diagnostic cases demonstrating CapEx table recovery under Chunk Bundling ($\tau=0.70$) with 83.3% semantic accuracy.

---

## 3. Running Stage 2: Production Enhancements

Stage 2 deploys our five proposed engineering innovations: Table Header Propagation, Calibrated Statement Routing & Penalties, Deterministic Python Math Sandbox, Coreference Resolution, and Forward Context Stitching.

### 3.1 Step 1: Ingestion Table Header Enrichment (WP1)
```bash
# Enrich 2,902 continuation chunks with document titles and column headers
python scripts/enrich_phase2_chunks.py
```
*Expected Output:* `data/financebench/bm25_index_phase2/` populated with 2,902 enriched chunks.

### 3.2 Step 2: Run Full 150-Question Stage 2 Benchmark
```bash
python scripts/run_phase2_benchmark.py
```
*Expected Output:*
* Scorecard saved to: `outputs/phase2/evaluation/phase2_full_scorecard.json`
* Checkpoint log: `outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl` (150 lines)
* Key Metrics:
  * Candidate Pool Recall: **86.00%** (+10.67 pp gain over Stage 1)
  * Macro MRR: **0.2926** (evaluated with sub-second batch latency of 43.24 seconds total)
  * Diagnostic Accuracy: **83.33%** with deterministic Python math sandbox

### 3.3 Step 3: Run Stage 2 Coreference Benchmark (Phase 4)
```bash
python scripts/run_coreference_benchmark.py
```
*Expected Output:* `outputs/phase4_coreference_benchmark_report.md` showing +6.00 pp Recall@5 gain and recovery of *Mounjaro* from Rank 999 to Rank 6.

---

## 4. Launching the Interactive Web Dashboard

To interactively test queries, inspect candidate pools, and examine bundling traces:

```bash
# Launch Streamlit dashboard on local port 8501
streamlit run app.py --server.port 8501 --server.headless true
```
* Access URL: `http://localhost:8501`
* Features:
  * **💬 Ask:** Select diagnostic benchmarks (Q1–Q6, Stage 3 V2) or submit custom queries with live L.9 retrieval and 4-bit Qwen2.5-7B generation.
  * **🎯 Retrieval Evidence:** Interactive cross-encoder reranking diagnostic and semantic chunk bundling traces.
  * **📊 Benchmarks:** Toggle between Phase 1 Academic Reproduction, Phase 2 Enhanced Production, and Phase 3 Controlled benchmarks.
  * **📋 Reproduction Notes:** Detailed breakdown of deviations from the paper and recorded hardware telemetry.
  * **📁 Corpus Explorer:** Browse indexed 10-K filings and chunk metadata.
  * **Theme Switcher:** Seamless Auto, Dark, and Light mode toggling.

---
*Maintained by Group [NUMBER]. All source code and evaluation checkpoints are certified reproducible.*
