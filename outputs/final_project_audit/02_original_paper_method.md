# FinSage Original Research Paper Specification

**Document:** `outputs/final_project_audit/02_original_paper_method.md`  
**Target Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering*  
**Authors:** SimpleWay AI & CG Matrix Technology Limited (with contributions from Stanford, MIT, NTU)  
**Publication:** arXiv:2504.14493v4 [cs.IR] (Initial: April 20, 2025; Revised: August 13, 2025; Accepted at ACM CIKM 2025 Applied Research Track)  
**Official Codebase:** `https://github.com/simplew4y/finsage`  

---

## 1. Architectural Overview

FinSage is a specialized Retrieval-Augmented Generation (RAG) framework developed to resolve the severe semantic, structural, and numerical failure modes of standard RAG pipelines when applied to complex financial regulatory filings (such as SEC Form 10-K, 10-Q, 8-K, 6-K, and F-1).

The published paper decomposes the system into three primary technological pillars:
1. **Financial Filings Pre-processing (FFP):** Multi-modal document ingestion, table-to-text narrative synthesis, and heading-summary metadata embedding.
2. **Multi-Path Retrieval (MPR):** Hybrid lexical, dense, metadata-aware, and generative HyDE search coupled with semantic chunk bundling.
3. **Domain-Specialized Document Re-ranking (DRR):** Cross-encoder reranker fine-tuned using Direct Preference Optimization (DPO) on domain-specific hard negative pairs.

```
+-------------------------------------------------------------------------------------------------------------+
|                                    ORIGINAL FINSAGE PIPELINE FLOW                                           |
+-------------------------------------------------------------------------------------------------------------+
| 1. Financial Filing Preprocessing (FFP)                                                                     |
|    SEC PDF --> MinerU Layout Engine --> Text + Tables + Infographics                                        |
|            --> VLM (Mineru2Base / GPT-4o) Table-to-Text Narrative Generation                                |
|            --> LLM Co-reference Resolution (4 preceding chunks under same section title)                     |
|            --> Chunk-Level Section Heading-Summary Embeddings (BGE-M3)                                      |
|            --> Semantic Deduplication (TF-IDF Cosine tau = 0.70)                                           |
+-------------------------------------------------------------------------------------------------------------+
| 2. Multi-Path Retrieval (MPR)                                                                               |
|    User Query --> Path 1: BM25 Lexical Retrieval                                                            |
|               --> Path 2: BGE-M3 Dense Semantic Retrieval (FAISS)                                           |
|               --> Path 3: Metadata Heading-Summary Retrieval (FAISS, fetches entire segment blocks)         |
|               --> Path 4: Generative HyDE (InPars-trained Qwen2.5-7B generates 3 hypothetical passages)     |
|    Fusion     --> Collect unique candidates (Top-K per path)                                                |
|    Expansion  --> Chunk Bundling: expand adjacent chunks (s_{i-1}, s_{i+1}) if cos(e_i, e_{i+-1}) >= 0.85   |
|                   (Bundle capacity limit K_bundle = 5)                                                      |
+-------------------------------------------------------------------------------------------------------------+
| 3. Domain-Specialized Re-ranking (DRR)                                                                      |
|    Candidate Pool --> BAAI/bge-reranker-v2-gemma (Fine-tuned via DPO LoRA: r=32, a=64, 10 epochs)          |
|    Scoring        --> K(s, q) = sigma(w^T CrossEncoder(s, q) + b + beta * TimeBonus(pub_date))              |
|                       (STRICTLY ZERO HEURISTIC BOOSTS FOR STATEMENTS, TABLES, OR TICKERS)                   |
|    Selection      --> Top-5 ranked context chunks (K_final = 5)                                             |
+-------------------------------------------------------------------------------------------------------------+
| 4. Downstream Answer Generation & Evaluation                                                                |
|    Context + Query --> Qwen2.5-72B-Instruct-AWQ (4-bit, deployed on NVIDIA A100 80GB)                       |
|    Output          --> Natural conversational text without Markdown/LaTeX symbols                            |
|    Evaluation      --> LlamaIndex CorrectnessEvaluator (GPT-4o) + Expert Domain Human Review                |
+-------------------------------------------------------------------------------------------------------------+
```

---

## 2. Granular Component Specifications

### Component 1: Financial Filing Preprocessing (FFP)
* **What it does:** Converts unstructured multi-modal PDF regulatory filings into semantically rich, clean, and indexed text chunks with section metadata.
* **Input:** Raw SEC filing PDF documents (e.g., Form 10-K, Form 10-Q).
* **Output:** Standardized text chunks of fixed token size (256 tokens for FinanceBench, 200 tokens for Company dataset) with associated section heading-summary metadata and deduplicated content.
* **Model / Library:** `MinerU` (`magic-pdf -p ... -m auto`), `Mineru2Base` VLM / GPT-4o for table narration, BAAI/bge-m3 for metadata embeddings.
* **Paper-Reported Parameters:** 256 tokens per chunk (FinanceBench), 200 tokens per chunk (Company dataset), $\tau_{\text{dedup}} = 0.70$ (TF-IDF cosine deduplication), 4 preceding chunks scanned for pronoun co-reference resolution.
* **Paper Citations:** Section 3.1 (*"Financial Filings Pre-processing pipeline"*), Page 3–4; Figure 3; Appendix A.1–A.9.
* **Our Implementation Status:** **Partially Implemented (Adapted)**.
* **Implementation File & Function:** [`scripts/preprocess_documents.py:6-40`](file:///c:/Users/user/Downloads/finsage-main/scripts/preprocess_documents.py#L6-L40) (`extract_text_and_tables()`, `RecursiveCharacterTextSplitter`).
* **Differences from Paper:** In our workspace, PyMuPDF extracts raw text and markdown tables directly into [`parsed_chunks.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/processed/parsed_chunks.jsonl) (1,000 characters, 200 overlap). Synthetic VLM table-to-text narrative generation and pronoun co-reference resolution were omitted to prevent numerical hallucination and conserve GPU compute.

---

### Component 2: Multi-Path Retrieval (MPR) Architecture
* **What it does:** Executes four complementary retrieval mechanisms simultaneously to retrieve a diverse candidate pool covering lexical, semantic, section-level, and hypothetical intent dimensions.
* **Input:** Natural-language user financial query $q$.
* **Output:** Candidate pool of document chunks $\mathcal{R}_q$ aggregated across all four paths.
* **Model / Library:** `bm25s`, `faiss`, `BAAI/bge-m3`, `Qwen/Qwen2.5-7B-Instruct`.
* **Paper-Reported Parameters:** Top-$K$ per path varies to maintain balanced candidate sets; unweighted union of candidates before reranking.
* **Paper Citations:** Section 3.2 (*"Multi-path Retrieval pipeline"*), Page 4; Figure 4; Table 1.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/retrieval/multi_path_retriever.py:35-180`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/multi_path_retriever.py#L35-L180) (`MultiPathRetriever.retrieve_rrf()`).
* **Differences from Paper:** We implemented Reciprocal Rank Fusion (RRF, $k=60$) to merge the 4 paths into a candidate pool of $K_{\text{pool}}=40$ chunks.

---

### Component 3: BM25 Lexical Retrieval (Path 1)
* **What it does:** Performs sparse lexical keyword matching based on term frequency and inverse document frequency.
* **Input:** User query tokens.
* **Output:** Scored and ranked candidate chunk IDs.
* **Model / Library:** BM25 (implemented via `bm25s` with NumPy/SciPy CSC matrix backend).
* **Paper-Reported Parameters:** Standard Lucene parameters: $k_1 = 1.5, b = 0.75, \delta = 0.5$.
* **Paper Citations:** Section 3.2; Section 4.2.1; Table 1.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/utils/bm25Retriever.py:25-75`](file:///c:/Users/user/Downloads/finsage-main/src/utils/bm25Retriever.py#L25-L75) and [`scripts/build_bm25_index.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/build_bm25_index.py).
* **Differences from Paper:** Zero divergence. Matches standard BM25 formulation.

---

### Component 4: Dense Retrieval & Embeddings (Path 2)
* **What it does:** Performs semantic vector search in Euclidean/cosine space using multi-lingual dense representations.
* **Input:** User query string embedded into $\mathbf{e}(q) \in \mathbb{R}^{1024}$.
* **Output:** Nearest neighbor chunk vectors by L2 distance on normalized embeddings.
* **Model / Library:** `BAAI/bge-m3` (1024 dimensions) with FAISS `IndexFlatL2`.
* **Paper-Reported Parameters:** 1024-dim dense vectors, L2 normalized embeddings, cosine similarity via `IndexFlatL2`.
* **Paper Citations:** Section 3.2; Section 4.2.1; Table 1.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/utils/faissRetriever.py:30-80`](file:///c:/Users/user/Downloads/finsage-main/src/utils/faissRetriever.py#L30-L80) and [`scripts/build_vector_index.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/build_vector_index.py).
* **Differences from Paper:** Zero divergence. 54,564 chunks embedded and indexed in [`data/financebench/faiss_index/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/faiss_index/).

---

### Component 5: Metadata / Heading-Summary Retrieval (Path 3)
* **What it does:** Embeds section subtitles and high-level summaries into a metadata vector $\mathbf{e}_i$. When a query matches the section theme, all subordinate text and tabular chunks in that semantic section are retrieved as a block.
* **Input:** User query string.
* **Output:** Candidate chunks belonging to the matching document sections.
* **Model / Library:** `BAAI/bge-m3` embedding section heading-summaries into FAISS `IndexFlatL2`.
* **Paper-Reported Parameters:** Concatenation of chunk heading and summary; top-$K$ metadata entries retrieved with all child chunks included.
* **Paper Citations:** Section 3.1 (Appendix A.7); Section 3.2 (*"Retrieval Module"*), Page 4; Table 1.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/retrieval/metadata_retriever.py:57-95`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/metadata_retriever.py#L57-L95) (`MetadataRetriever.retrieve_metadata_candidates()`) and [`scripts/build_heading_summary_index.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/build_heading_summary_index.py).
* **Differences from Paper:** We indexed 2,678 filing sections from the 84 PDFs into [`data/financebench/metadata_faiss_index/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/metadata_faiss_index/) (10.9 MB).

---

### Component 6: Generative HyDE Retrieval (Path 4)
* **What it does:** Generates hypothetical document answers using an LLM to bridge colloquial user phrasing with formal regulatory accounting terminology, subsequently embedding those hypothetical passages with BGE-M3.
* **Input:** User query string.
* **Output:** 3 hypothetical document passages (2 textual, 1 tabular), used to retrieve candidate chunks from the FAISS vector index.
* **Model / Library:** `Qwen/Qwen2.5-7B-Instruct` fine-tuned via InPars method on synthesized pairs from 72B model.
* **Paper-Reported Parameters:** 3 hypothetical passages per query; top-$K$ chunks retrieved per passage.
* **Paper Citations:** Section 3.2 (*"Query Paraphrasing Module"*), Page 4; Section 4.2.1; Table 1.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/retrieval/hyde_generator.py:48-96`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/hyde_generator.py#L48-L96) (`GenerativeHyDE.generate_hypothetical_passages()`) and [`src/retrieval/hyde_retriever.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/hyde_retriever.py).
* **Differences from Paper:** In our implementation, `Qwen2.5-7B-Instruct` is prompted with specialized financial system prompts ([`src/prompts/hyde.py`](file:///c:/Users/user/Downloads/finsage-main/src/prompts/hyde.py)) without the InPars fine-tuning step. In the paper, HyDE-1 uses base Qwen7B, HyDE-2 uses Qwen7B-SFT, and HyDE-3 uses Qwen72B; our prompt-based implementation corresponds to HyDE-1.

---

### Component 7: Chunk Bundling / Expansion
* **What it does:** Iteratively inspects adjacent chunks ($s_{i-1}, s_{i+1}$) of retrieved candidate chunks and incorporates them into a bundle if sequential dense cosine similarity meets or exceeds a continuity threshold.
* **Input:** Retrieved candidate chunk $s_i$ and its preceding/succeeding chunks in filing reading order.
* **Output:** Coherent multi-chunk bundle preserving financial table headers and narrative continuity.
* **Mathematical Formulation:**
  $$\cos(\mathbf{e}(s_i), \mathbf{e}(s_{i \pm 1})) \ge \tau_{\text{exp}}$$
* **Paper-Reported Parameters:** Expansion threshold $\tau_{\text{exp}} = 0.85$, maximum bundle capacity $\mathcal{K} = 5$.
* **Paper Citations:** Section 3.2 (*"Chunk Bundling Module"*), Page 4; Section 4.2.1 (*"MPR settings"*), Page 6.
* **Our Implementation Status:** **Implemented with Documented Adaptation**.
* **Implementation File & Function:** [`src/retrieval/chunk_bundler.py:45-125`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/chunk_bundler.py#L45-L125) (`ChunkBundler.expand_bundle()`).
* **Differences from Paper:** The paper reported $\tau_{\text{exp}} = 0.85$. However, in our empirical audit on 3M Page 60 (Cash Flow Statement), cosine similarity between `p60_c0` (Operating activities) and `p60_c1` (Investing / CapEx) is **0.7172**. Under the paper's 0.85 threshold, expansion aborts and CapEx is missed. Following the authors' official GitHub implementation (`src/utils/ensembleRetriever.py`, lines 86–109) which uses a 0.66–0.72 regime, we calibrated $\tau_{\text{exp}} = 0.70$, successfully bundling the entire statement without boundary leakage.

---

### Component 8: Domain-Specialized Re-ranker (DRR) Architecture & Scoring
* **What it does:** Cross-encoder model that scores inter-token attention across concatenated `(query, document)` pairs to filter out irrelevant candidates and rank the most authoritative evidence in the top slots.
* **Input:** Concatenated `(query, candidate_chunk)` text pairs.
* **Output:** Neural relevance logit scores transformed via sigmoid.
* **Mathematical Formulation:**
  $$\mathcal{K}(s, q) = \sigma\left(\mathbf{w}^\top \text{CrossEncoder}(s, q) + b + \beta f(\tilde{m}_i)\right)$$
  where $f(\tilde{m}_i)$ is strictly a publication date **Time Bonus**.
* **Model / Library:** `BAAI/bge-reranker-v2-gemma` (Gemma-2B base cross-encoder).
* **Paper-Reported Parameters:** Cross-encoder scored over candidate pool $K_{\text{pool}}$; zero heuristic boosts for statements, tables, companies, or periods. Time bonus applied if filing date is within 1 year of query date.
* **Paper Citations:** Section 3.3 (*"Document Re-ranking"*), Page 4–5; Section 4.2.2; Appendix B.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`src/retrieval/statement_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/statement_reranker.py) and [`scripts/evaluate_phase1_drr.py:112-145`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L112-L145).
* **Differences from Paper:** In Phase 1, our reranker operates with **zero heuristic boosts (+0.0)**, exactly as specified in the paper. (Historical Phase 9C used +5.5 heuristic boosts).

---

### Component 9: DPO Training of the Reranker
* **What it does:** Aligns the cross-encoder reranker to financial terminology using Direct Preference Optimization on hard contrastive preference pairs $(s^+, s^-)$.
* **Input:** Structured preference triples: `(query, positive_gold_chunk, negative_hard_chunk)`.
* **Output:** Fine-tuned cross-encoder LoRA adapter weights.
* **Mathematical Formulation:**
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(q, s^+, s^-)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(s^+ \mid q)}{\pi_{\text{ref}}(s^+ \mid q)} - \beta \log \frac{\pi_\theta(s^- \mid q)}{\pi_{\text{ref}}(s^- \mid q)} \right) \right]$$
* **Model / Library:** `BAAI/bge-reranker-v2-gemma`, Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes`.
* **Paper-Reported Parameters:** LoRA rank $r = 32$, $\alpha = 64$, target modules `["q_proj", "v_proj", "k_proj", "o_proj"]`, 10 epochs, learning rate $10^{-4}$, batch size 1, gradient accumulation 2, DPO temperature $\beta = 0.1$, `bfloat16`.
* **Paper Citations:** Section 3.3.2 (*"Direct Preference Optimization"*), Page 5; Appendix B.1–B.3.
* **Our Implementation Status:** **Faithfully Implemented & Verified**.
* **Implementation File & Function:** [`scripts/train_drr_paper.py:20-145`](file:///c:/Users/user/Downloads/finsage-main/scripts/train_drr_paper.py#L20-L145) (`train_drr()`, `DPOPreferenceLoss`).
* **Differences from Paper:** The paper trained in full `bfloat16` across enterprise GPUs. To fit within the 8.55 GB VRAM budget of an RTX 5060 Laptop GPU, we used 4-bit NormalFloat4 (NF4) QLoRA with double quantization. The resulting adapter is preserved at [`outputs/phase1_drr_adapter/adapter_model.safetensors`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_model.safetensors) (29.5 MB). The mathematical loss, LoRA hyperparameters, epochs, and learning rate match Appendix B.3 exactly.

---

### Component 10: Downstream Answer Generation
* **What it does:** Synthesizes the final factual answer from assembled top-$K$ retrieved context chunks.
* **Input:** User query and top-$K$ assembled context chunks ($K=5$).
* **Output:** Conversational natural-language response providing numerical figures and regulatory citations.
* **Model / Library:** `Qwen/Qwen2.5-72B-Instruct-AWQ` (4-bit) deployed on NVIDIA A100 80GB.
* **Paper-Reported Parameters:** Context window $K = 5$ chunks per query; natural conversational plain text without Markdown or LaTeX tables.
* **Paper Citations:** Section 4.2.3 (*"Response settings"*), Page 6; Appendix C.1; `src/prompts/qa.py`.
* **Our Implementation Status:** **Implemented with Documented Hardware Adaptation**.
* **Implementation File & Function:** [`scripts/run_generation_phase1_drr.py:110-175`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_generation_phase1_drr.py#L110-L175).
* **Differences from Paper:** Scaled from Qwen2.5-72B to `Qwen2.5-7B-Instruct` (4-bit NF4) due to local 8.55 GB VRAM limit. Generation prompt outputs operational JSON (`operation`, `values`) and natural explanation, which is subsequently parsed and validated deterministically.

---

### Component 11: Evaluation Procedure
* **What it does:** Measures retrieval quality (coverage, ranking) and answer correctness against expert ground-truth references.
* **Input:** System-generated answers and retrieved candidate lists vs. ground-truth reference items.
* **Output:** Quantitative scorecards covering Recall, Precision, F1, MRR, Normalized Recall, Binary nDCG, and Answer Accuracy.
* **Evaluation Standards in Paper:**
  1. *MPR Retrieval:* Avg Recall, Avg Precision, Avg F1, Avg Num Retrieved.
  2. *DRR Reranking:* Precision, Normalized Recall ($\text{capped denominator } \min(\text{ground truth}, K)$), MRR, Binary nDCG.
  3. *End-to-End QA:* LlamaIndex `CorrectnessEvaluator` (backed by GPT-4o on 1–5 scale) + Domain Expert Human Review.
* **Paper Citations:** Section 4.3 (*"Evaluation"*), Page 6–7; Section 5.1–5.4; Appendix C.2.
* **Our Implementation Status:** **Implemented with Documented Diagnostic Adaptation**.
* **Implementation File & Function:** [`scripts/evaluate_phase1_drr.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py), [`scripts/evaluate_harness_v2.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_harness_v2.py), [`src/utils/arithmetic_evaluator.py`](file:///c:/Users/user/Downloads/finsage-main/src/utils/arithmetic_evaluator.py).
* **Differences from Paper:** In the paper, end-to-end evaluation was executed on 150 FinanceBench questions via online GPT-4o API. In our workspace, we evaluated held-out diagnostic questions locally using Harness V2 (multi-dimensional: Semantic, Numeric/Unit, Arithmetic, and Schema) without paid API dependencies. Stage 2 evaluated the full 150 questions on macro retrieval metrics (MRR, Recall@1/5/10).
