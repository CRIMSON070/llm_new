# FinSage Financial RAG Dashboard — Phase 1 Reproduction

An executive, modern, interactive Streamlit dashboard for the **FinSage** financial RAG project, implementing and validating the Phase 1 methodology of *arXiv:2504.14493* (CIKM 2025).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you are in the configured Conda environment (`finsage`):
```bash
conda activate finsage
```

### 2. Install Dashboard Dependencies
Install the required dashboard dependencies from `requirements_dashboard.txt`:
```bash
pip install -r requirements_dashboard.txt
```

### 3. Launch the Dashboard
Run the Streamlit application from the repository root:
```bash
streamlit run app.py
```
Or with custom port and host:
```bash
streamlit run app.py --server.port 8501 --server.headless false
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧭 Dashboard Sections & Modules

### 1. 🏠 Overview
- **Project Objective & Architecture:** Detailed breakdown of Multi-Path Retrieval (BGE-M3 Dense + BM25 Sparse + Financial Query Expansion), Chunk Bundling ($\tau_{\text{exp}}$, $\mathcal{K}=5$), Domain-Specialized Document Re-Ranker (BGE-Reranker-v2-Gemma with 4-bit QLoRA, zero heuristic boosts), and downstream generation.
- **Key Metrics:** Corpus chunk count (54,565), indexed SEC filings (84), diagnostic test size (6 held-out cases), and benchmark semantic accuracy (83.3%).
- **Diagnostic Benchmark Scope Notice:** Prominently notes that displayed metrics reflect the 6-question held-out diagnostic suite (Q1–Q6) isolating critical retrieval failure modes, not the full 180-question FinanceBench evaluation.
- **Component Status Matrix:** Real-time visibility into active indices, model weights, and ingestion pipelines.

### 2. 💬 Financial Q&A
- **Diagnostic Benchmark Quick-Select:** One-click presets for held-out questions Q1 through Q6 with pre-configured target filings and gold standards.
- **Document Scope Filtering:** Select between global multi-document search across all 84 filings or focus retrieval on specific 10-K / 10-Q reports.
- **Interactive Pipeline Controls:**
  - Candidate pool size ($k_{\text{pool}}$: 10–60, default 40).
  - Final context capacity ($k_{\text{final}}$: 1–10, default 5).
  - Chunk Bundling toggle: Threshold $\tau_{\text{exp}}$ slider (default 0.70 adapted, 0.85 paper reported) and maximum bundle size $\mathcal{K}$.
  - Cross-Encoder Reranker selection (Phase 1 DRR Adapter, Base BGE-Reranker baseline, Base Gemma baseline, or RRF Fusion).
  - Downstream local LLM generation toggle (Qwen2.5-7B-Instruct).
- **Context & Evidence Cards:** Displays retrieved chunks with chunk IDs, document names, page numbers, and statement categories.
- **Semantic Adjacency Trace:** Explores step-by-step cosine similarity transitions between adjacent chunks during bundling.

### 3. 🎯 Retrieval & Reranking
- **Cross-Encoder Comparison:** Compares Phase 1 Trained DRR Adapter against Base BGE-base and Base Gemma baselines under zero heuristic boosts.
- **Interactive Plotly Visualizations:** MRR, Recall@3, Recall@5, and Recall@10.
- **Per-Question Evidence Rank Matrix:** Detailed inspection of gold chunk ranks across diagnostic questions Q1–Q6.
- **Case Analysis on Q1 (3M CapEx Bisection):** Visual side-by-side inspection of chunk `p60_c0` (Operating Activities) vs chunk `p60_c1` (Investing Activities & CapEx $1,577M) and cosine similarity threshold audit table.

### 4. 📝 Answer Evaluation
- **Phase 1 Downstream Accuracies:** Evaluates local Qwen2.5-7B generated answers using Phase 1 DRR Top-3 context (Zero Boosts).
- **Multi-Dimensional Metrics:** Semantic Answer Accuracy (83.3%), Numeric & Unit Accuracy (83.3%), Strict Schema Compliance (66.7%), Evidence Retrieval Accuracy (33.3%).
- **Per-Question Diagnostic Table:** Breakdown of gold vs generated answers and verification status across all 6 diagnostic cases.
- **Decoupling Insights:** Explains the distinction between programmatic JSON syntax formatting and true factual/numeric accuracy in financial NLP.

### 5. 💻 Hardware & Telemetry
- **Hardware Telemetry:** Real-time and recorded GPU device, peak VRAM usage (0.84 GB Base, 2.26 GB DRR Gemma), and latency breakdown.
- **Memory Safety & Quantization:** Details on 4-bit BitsAndBytes NF4 quantization and Streamlit resource caching for 8GB VRAM cards.
- **Index Architecture:** 223MB FAISS vector store (54,565 chunks) and BM25S lexical index status.

### 6. ⚠️ Reproduction Gaps
- **Documented Architectural Discrepancies:** Clear comparison between arXiv:2504.14493 paper specification and workspace implementation (Evaluation scale, FFP preprocessing, Generative HyDE vs regex expander, model scale, and evaluation standard).
- **Roadmap to Full Reproduction:** Concrete milestones to scale from 6 diagnostic cases to the full 180 FinanceBench dataset.
- **Project Boundary:** Phase 1 reproduction focus; Phase 2 improvement proposals remain paused.

### 7. 📁 Documents Catalog
- **Document Catalog:** Search, sort, and filter the 84 indexed SEC filings (3M, Ulta Beauty, Microsoft, PepsiCo, Walmart, etc.).
- **Chunk Inspector:** Browse individual chunk records, page assignments, and raw extracted text.
- **PDF Upload:** Clearly labeled as **Pending Integration** (explaining that offline batch parsing exists in `scripts/preprocess_documents.py`, but real-time incremental vector index injection is not yet connected).

---

## ⚙️ Feature Availability & Integration Status

| Feature | Status | Notes |
|---|:---:|---|
| **FAISS Dense Retrieval** | **Working** | Local BGE-M3 embeddings on CUDA (54,565 chunks) |
| **BM25 Sparse Retrieval** | **Working** | Local BM25S lexical index |
| **Reciprocal Rank Fusion (RRF)** | **Working** | Combines dense and sparse candidates ($k_{\text{pool}}=40$) |
| **Chunk Bundling Module** | **Working** | Full document adjacency and cosine expansion ($\tau=0.70/0.85$, $K=5$) |
| **Phase 1 DRR Adapter** | **Working** | Loaded from `outputs/phase1_drr_adapter` (Zero Boosts) |
| **Base Reranker Baselines** | **Working** | Base BGE-base and Base Gemma zero-boost baselines |
| **Benchmark Metric Visualization** | **Working** | Dynamic rendering from verified Phase 1 evaluation JSONs |
| **Local Qwen2.5-7B Generation** | **Working** | 4-bit QLoRA inference via BitsAndBytes |
| **Live PDF Ingestion API** | **Pending Integration** | Labeled in UI; offline batch scripts exist in `scripts/` |
| **Phase 2 Experimental Features** | **Paused** | Out of scope per Phase 1 strict reproduction guidelines |

---

## 🔒 Memory Safety & Performance Notes

- Local models (BGE-M3, BGE-Reranker, Qwen2.5-7B) are loaded on demand and cached using `@st.cache_resource` to avoid memory fragmentation.
- To prevent GPU out-of-memory issues on 8GB VRAM cards (e.g. RTX 5060 Laptop GPU), Q&A allows users to inspect retrieved contexts and precomputed benchmark generations before triggering full 7B parameter inference.
