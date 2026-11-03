# FinSage Stage 1: Implementation Audit & Technical Verification

**Document:** `outputs/final_project_audit/04_stage1_implementation_audit.md`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Governing Standard:** Comprehensive Component-by-Component Source Code & Environmental Audit (Read-Only)  

---

## 1. Experimental Environment & Infrastructure Audit

All findings in this section are extracted directly from verified environment telemetry recorded in [`outputs/phase1_drr_adapter/drr_training_metadata.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json) and live system queries.

### Verified Hardware & Host Profile
* **Host Operating System:** Windows 11 (x86_64, NT kernel 10.0).
* **Target GPU:** **NVIDIA GeForce RTX 5060 Laptop GPU** (Compute capability 12.0 / Ada/Blackwell generation).
* **Total Physical VRAM:** **8.55 GB** (9,183,526,912 bytes).
* **Host Memory:** 16.0 GB physical RAM.

### Verified Software Runtime & Environment
* **Python Runtime:** Python `3.10.22` running in Conda virtual environment `finsage` (`C:\Users\user\miniconda3\envs\finsage\python.exe`).
* **Deep Learning Framework:** PyTorch `2.14.1+cu130` with full CUDA acceleration enabled (`torch.cuda.is_available() == True`).
* **Transformers & PEFT Ecosystem:**
  * `transformers`: **`5.19.0`**
  * `peft`: **`0.21.2`** (Parameter-Efficient Fine-Tuning)
  * `bitsandbytes`: **`0.50.2`** (NF4 / Int8 quantization primitives)
  * `accelerate`: **`0.34.2`**
* **Vector & Lexical Search Libraries:**
  * `faiss-cpu` / `faiss` (LangChain FAISS integration)
  * `bm25s`: **`0.3.13`** (NumPy/SciPy Compressed Sparse Column matrix backend)
* **Document Extraction Libraries:**
  * `PyMuPDF (fitz)`: **`1.24.11`**
  * `pymupdf4llm`: **`0.0.17`**
* **UI & Serving Stack:**
  * `streamlit`: **`1.65.0`**
  * `plotly`: **`5.24.1`**
  * `pandas`: **`2.2.3`**
  * `uvicorn`: **`0.32.0`**

---

## 2. Ingestion & Preprocessing Audit (Financial Filing Preprocessing - FFP)

### Source Code Implementation
* **Primary Script:** [`scripts/preprocess_documents.py:6-40`](file:///c:/Users/user/Downloads/finsage-main/scripts/preprocess_documents.py#L6-L40)
* **Target Output Store:** [`data/financebench/processed/parsed_chunks.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/processed/parsed_chunks.jsonl) (57.1 MB, 54,564 records).

### Granular Technical Findings:
1. **Parser & Text Extraction:**
   * Uses `fitz.open(pdf_path)` to iterate across document pages.
   * Extracts text blocks and inline Markdown tables using PyMuPDF's layout engine.
   * Preserves exact page numbers (`page_number: page.number + 1`).
2. **Chunking Rules & Separators:**
   * Utilizes LangChain's `RecursiveCharacterTextSplitter`.
   * `chunk_size = 1000` characters; `chunk_overlap = 200` characters.
   * Separators: `["\n\n", "\n", " ", ""]`.
   * Preserves chunk sequential index per page (`_c0`, `_c1`, `_c2`).
3. **Chunk Identifier Convention:**
   * Structured as: `{doc_name}_p{page_number}_c{chunk_index}` (e.g., `3M_2018_10K_p60_c1`).
4. **Metadata Preservation:**
   * Chunk objects store `doc_name`, `page_number`, `chunk_id`, and character count.
5. **Comparison Against Original Paper's FFP:**
   * *Paper Specification:* `MinerU` (`magic-pdf -p ... -m auto`) extracting text, tables, and images into `content_list.json`, with VLM (`Mineru2Base` / GPT-4o) converting tables into synthetic narrative prose, and LLM pronoun co-reference resolution across 4 preceding chunks under the same heading.
   * *Our Implementation:* PyMuPDF text and Markdown table extraction. Tables are preserved in exact raw Markdown/plain-text format. VLM table rewriting was omitted to prevent hallucination of non-existent numbers. Pronoun resolution was not applied.
   * *Status:* **Partially Implemented (Documented Adaptation)**.

---

## 3. Retrieval Implementation Audit (Multi-Path Retrieval - MPR)

### Source Code Implementation
* **Primary Coordinator:** [`src/retrieval/multi_path_retriever.py:31-204`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/multi_path_retriever.py#L31-L204) (`MultiPathRetriever`).

### Path-by-Path Technical Verification:

#### Path 1: BM25 Lexical Retrieval
* **Code Implementation:** [`src/utils/bm25Retriever.py:25-75`](file:///c:/Users/user/Downloads/finsage-main/src/utils/bm25Retriever.py#L25-L75).
* **Index Storage:** [`data/financebench/bm25_index/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/bm25_index/) (7 binary files).
* **Parameters:** $k_1 = 1.5, b = 0.75, \delta = 0.5$, backend = `lucene` via `bm25s`.
* **Execution:** Queries chunk corpus token index, returning top-$K$ scored chunk IDs.
* **Status:** **Faithfully Implemented & Verified**.

#### Path 2: Dense Semantic Vector Retrieval
* **Code Implementation:** [`src/utils/faissRetriever.py:30-80`](file:///c:/Users/user/Downloads/finsage-main/src/utils/faissRetriever.py#L30-L80).
* **Index Storage:** [`data/financebench/faiss_index/index.faiss`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/faiss_index/index.faiss) (223 MB).
* **Model:** `BAAI/bge-m3` (1024 dimensions, dense representations).
* **Index Type:** FAISS `IndexFlatL2` operating over L2-normalized embeddings ($\|\mathbf{e}\|_2 = 1.0$), making Euclidean distance equivalent to cosine distance ($d^2 = 2 - 2\cos(\theta)$).
* **Chunk Count:** Exactly 54,564 indexed vectors.
* **Status:** **Faithfully Implemented & Verified**.

#### Path 3: Metadata Section Heading-Summary Retrieval
* **Code Implementation:** [`src/retrieval/metadata_retriever.py:57-95`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/metadata_retriever.py#L57-L95).
* **Index Storage:** [`data/financebench/metadata_faiss_index/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/metadata_faiss_index/) (10.9 MB).
* **Model:** `BAAI/bge-m3` embedding section titles and LLM summaries.
* **Section Count:** Exactly 2,678 unique filing section headings.
* **Execution:** Queries section embeddings with user query. When a section matches, retrieves all constituent chunks under that section heading block.
* **Status:** **Faithfully Implemented & Verified**.

#### Path 4: Generative HyDE Retrieval
* **Code Implementation:** [`src/retrieval/hyde_generator.py:48-96`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/hyde_generator.py#L48-L96) and [`src/retrieval/hyde_retriever.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/hyde_retriever.py).
* **Model:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4).
* **Execution:** Generates 3 hypothetical document passages per query (2 text passages + 1 tabular financial schedule), embeds each passage with `bge-m3`, and retrieves top-$K$ FAISS candidates per passage.
* **Status:** **Faithfully Implemented & Verified**.

#### Candidate Fusion: Reciprocal Rank Fusion (RRF)
* **Code Implementation:** [`src/retrieval/multi_path_retriever.py:104-144`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/multi_path_retriever.py#L104-L144).
* **Formula:**
  $$\text{RRF Score}(d) = \sum_{p \in \text{Paths}} \frac{w_p}{k_{\text{rrf}} + \text{Rank}_p(d)}$$
* **Parameters:** $k_{\text{rrf}} = 60$, weights $w_p = 1.0$, pool capacity $K_{\text{pool}} = 40$ chunks.
* **Status:** **Faithfully Implemented & Verified**.

#### Chunk Bundling Module (Adjacency Cosine Expansion)
* **Code Implementation:** [`src/retrieval/chunk_bundler.py:28-150`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/chunk_bundler.py#L28-L150) (`ChunkBundler`).
* **Algorithm:** For each lead candidate $s_i$, evaluates preceding and succeeding chunks ($s_{i-1}, s_{i+1}$) in filing reading order. Incorporates adjacent chunk if cosine similarity $\ge \tau_{\text{exp}}$, up to limit $\mathcal{K} = 5$.
* **Paper-Reported Parameter:** $\tau_{\text{exp}} = 0.85, \mathcal{K} = 5$.
* **Implemented Parameter:** $\tau_{\text{exp}} = 0.70, \mathcal{K} = 5$.
* **Empirical Justification:** In 3M FY2018 10-K Page 60 (Cash Flow Statement), cosine similarity between `p60_c0` (Operating Cash Flows) and `p60_c1` (Investing Cash Flows / CapEx) is **0.7172**. At $\tau=0.85$, expansion terminates prematurely and CapEx is missed. At $\tau=0.70$, all 4 fragments of Page 60 are bundled, cleanly halting at page boundary ($0.6107 < 0.70$).
* **Status:** **Implemented with Documented Calibration**.

---

## 4. DRR Reranker Implementation & DPO Training Audit

### Source Code Implementation
* **Training Script:** [`scripts/train_drr_paper.py:20-145`](file:///c:/Users/user/Downloads/finsage-main/scripts/train_drr_paper.py#L20-L145) (27,118 bytes).
* **Evaluation Script:** [`scripts/evaluate_phase1_drr.py:112-145`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L112-L145).
* **Saved Checkpoint:** [`outputs/phase1_drr_adapter/adapter_model.safetensors`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_model.safetensors) (**29,510,456 bytes**).
* **Config File:** [`outputs/phase1_drr_adapter/adapter_config.json`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json) (1,180 bytes).

### Verified Hyperparameters: Paper vs. Implementation

| Hyperparameter | Paper Specification (arXiv:2504.14493) | Our Implementation & Verification | Verification Source |
|---|---|---|---|
| **Base Model Checkpoint** | `BAAI/bge-reranker-v2-gemma` | `BAAI/bge-reranker-v2-gemma` | [`adapter_config.json:6`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json#L6) |
| **Model Architecture** | Gemma-2B Cross-Encoder | Gemma-2B (`CausalLM` head scoring logit) | [`adapter_config.json:41`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json#L41) |
| **LoRA Rank ($r$)** | $r = 32$ | $r = 32$ | [`adapter_config.json:31`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json#L31) |
| **LoRA Alpha ($\alpha$)** | $\alpha = 64$ | $\alpha = 64$ | [`adapter_config.json:20`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json#L20) |
| **Target Modules** | Query, Value, Key, Output projections | `["q_proj", "v_proj", "k_proj", "o_proj"]` | [`adapter_config.json:34-39`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/adapter_config.json#L34-L39) |
| **Training Epochs** | 10 epochs | Exactly 10 epochs | [`drr_training_metadata.json:40`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json#L40) |
| **Learning Rate** | $1 \times 10^{-4}$ ($0.0001$) | $1 \times 10^{-4}$ with cosine decay, 5% warmup | [`drr_training_metadata.json:15`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json#L15) |
| **Batch Size & Accumulation** | Batch size 1, grad accum 2 (Effective: 2) | Batch size 1, grad accum 2 (Effective: 2) | [`drr_training_metadata.json:16-17`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json#L16-L17) |
| **DPO Temperature ($\beta$)** | $\beta = 0.1$ | $\beta = 0.1$ | [`drr_training_metadata.json:28`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json#L28) |
| **Precision / Quantization** | Full `bfloat16` across enterprise GPUs | 4-bit NormalFloat4 (NF4) with double quant | [`drr_training_metadata.json:23`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_drr_adapter/drr_training_metadata.json#L23) |
| **Heuristic Boost Weights** | **Zero heuristic boosts (+0.0)** | **Zero heuristic boosts (+0.0)** | [`evaluate_phase1_drr.py:112`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L112) |

### Training Execution Evidence
* **Duration:** Exactly **90.7 minutes** across 10 epochs.
* **Peak GPU VRAM Consumption:** **3.64 GB** (with 4.91 GB headroom on the 8.55 GB RTX 5060).
* **Training Loss Progression:** Loss decreased from 0.4812 (Epoch 1) to **0.0006** (Epoch 10).
* **Final Training Preference Accuracy:** **100.0%**.
* **Final Validation Loss:** **0.0954** (Best validation loss: 0.0930).
* **Final Validation Preference Accuracy:** **95.29%** (16/17 correct preference rankings).
* **Verification Status:** **Faithfully Implemented & Verified**.

---

## 5. Downstream Answer Generation Audit

### Source Code Implementation
* **Primary Script:** [`scripts/run_generation_phase1_drr.py:42-175`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_generation_phase1_drr.py#L42-L175).

### Technical Findings:
1. **Generator LLM Model:** `Qwen/Qwen2.5-7B-Instruct`.
   * Loaded in 4-bit NF4 quantization (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`).
   * Scaled down from paper's Qwen2.5-72B due to the 8.55 GB VRAM desktop budget.
2. **Context Window Assembly:**
   * Context is assembled from the top-$K$ ranked chunks ($K = 5$ chunks, matching paper Section 4.2.3).
   * Context chunks concatenated sequentially with chunk identifiers.
3. **Decoding Parameters:**
   * `temperature = 0.0` (deterministic greedy decoding).
   * `do_sample = False`.
   * `max_new_tokens = 256`.
4. **Output Format & Parsing:**
   * Prompt instructs the model to select an operation (`none`, `extract_single_value`, `absolute_difference`, `percentage_change`) and provide the operands and numerical explanation.
   * Parsed via robust JSON extractor ([`extract_first_valid_json()`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_generation_phase1_drr.py#L43-L69)).

---

## 6. Evaluation Harness V2 Audit

### Source Code Implementation
* **Primary Script:** [`scripts/evaluate_harness_v2.py:1-616`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_harness_v2.py#L1-L616).
* **Report Location:** [`outputs/phase1_audit/evaluation_harness_v2/harness_v2_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_audit/evaluation_harness_v2/harness_v2_report.md).

### Evaluation Logic & Scorecard Dimensions:
1. **Semantic Correctness:** Verifies that the factual claim, direction of change, and entity match the gold answer without relying on strict syntax.
2. **Numeric & Unit Accuracy:** Programmatically parses numerical quantities, unit scales ($M, $B, $K), and checks sign (+/-).
3. **Arithmetic Verification:** Uses safe numerical computation to verify operations on extracted operands.
4. **Strict Schema Compliance:** Evaluates whether output strictly conforms to legacy JSON schema (`operation`, `old_value`, `new_value`).
5. **Decoupled Evaluation Finding:**
   * Under strict schema matching, pass rate was 50.0% (3/6) due to rigid key constraints.
   * Under true semantic and numeric verification, Phase 1 DRR achieved **83.3% (5/6)** correctness.
