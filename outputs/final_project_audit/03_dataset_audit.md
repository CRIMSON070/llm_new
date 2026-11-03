# FinSage Dataset Audit & Corpus Inspection

**Document:** `outputs/final_project_audit/03_dataset_audit.md`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Governing Standard:** Comprehensive Empirical Dataset & Manifest Audit (Read-Only)  

---

## 1. Overview of Datasets in the FinSage Project

The FinSage project involves two primary datasets across the published research paper and the local implementation, alongside two derived alignment datasets:

1. **FinanceBench (Patronus AI):** An open-source benchmark of professionally annotated financial questions over public SEC corporate regulatory filings.
2. **Lotus Technology Company Dataset:** An in-depth single-company dataset curated from SEC 6-K and F-1 filings of Lotus Technology Inc. (Nasdaq: LOT).
3. **DRR Preference Dataset (`data/financebench/drr/`):** 142 hard contrastive triplet samples for cross-encoder DPO re-ranker training.
4. **Phase 9B Generator Preference Dataset (`data/financebench/phase9b/`):** 36 preference pairs from early exploratory generator fine-tuning.

---

## 2. Dataset 1: Patronus AI FinanceBench

### Metadata & Provenance
* **Official Name:** FinanceBench: A Benchmark for Financial Question Answering over Open-Book Regulatory Filings.
* **Source Organization:** Patronus AI.
* **Source Repository / URL:** [Hugging Face (`PatronusAI/financebench`)](https://huggingface.co/datasets/PatronusAI/financebench) / [GitHub (`patronus-ai/financebench`)](https://github.com/patronus-ai/financebench).
* **Paper Reference:** Islam et al., *“FinanceBench: A New Benchmark for Evaluating Financial Question Answering in Large Language Models”*, arXiv:2311.11944 (2023).

### Role in Original Paper vs. Local Workspace
* **Purpose in Original Paper:** Used as the primary open-source cross-company benchmark to evaluate end-to-end question answering (Table 2). The paper processed 83 financial documents and evaluated 149 questions using fixed 256-token chunks.
* **Purpose in Our Implementation:** Serves as the primary corpus for document ingestion, indexing (BM25, BGE-M3 Dense, Metadata Section Index), diagnostic evaluation (Q1–Q6 in Stage 1), and full benchmark scaling ($N=150$ in Stage 2).

### Physical Inventory & Verified File Counts
* **Manifest File:** [`data/financebench/manifest.json`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/manifest.json) (10,562 bytes).
  * Contains exactly **84 document entries**, all marked with `"status": "downloaded"`.
* **Raw Document Directory:** [`data/financebench/pdfs/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/pdfs/).
  * Contains exactly **84 verified `.pdf` files** (100% parity with manifest).
* **Question Ground Truth File:** [`data/financebench/questions.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/questions.jsonl) (962,867 bytes).
  * Contains exactly **150 annotated questions** (`financebench_id_00001` through `financebench_id_10231`).
  * 0 missing fields across `financebench_id`, `company`, `doc_name`, `question`, `answer`, and `evidence`.
* **Parsed Chunk Store:** [`data/financebench/processed/parsed_chunks.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/processed/parsed_chunks.jsonl) (57,132,868 bytes).
  * Contains exactly **54,564 text & table chunks** across all 84 PDF documents.

### Corporate Scope & Filing Types
* **Unique Corporations:** **32 major publicly traded corporations**:
  * 3M, Adobe, Advanced Micro Devices (AMD), Amazon, American Express, American Water Works, Apple, Best Buy, Boeing, Coca-Cola, Costco, CVS Health, General Mills, Home Depot, Intel, Johnson & Johnson, JPMorgan Chase, Lockheed Martin, Microsoft, Netflix, Nike, Northrop Grumman, PayPal, PepsiCo, Pfizer, Procter & Gamble, Qualcomm, Salesforce, Ulta Beauty, Verizon, Walmart, Walt Disney.
* **Industry Representation:** **9 GICS Sectors**:
  * Consumer Discretionary (30 questions)
  * Consumer Staples (23 questions)
  * Information Technology (22 questions)
  * Industrials (19 questions)
  * Health Care (19 questions)
  * Financials (13 questions)
  * Communication Services (9 questions)
  * Materials (9 questions)
  * Utilities (6 questions)
* **Document Types Across Corpus:**
  * **Form 10-K (Annual Reports):** Associated with 112 questions.
  * **Form 10-Q (Quarterly Reports):** Associated with 15 questions.
  * **Earnings Releases (8-K / Press Releases):** Associated with 14 questions.
  * **Form 8-K (Current Reports):** Associated with 9 questions.
* **Reporting Periods Covered:** Fiscal years 2015 through 2023.

### Ground-Truth Schema & Evidence Structure
Each record in `questions.jsonl` follows this validated structure:
```json
{
  "financebench_id": "financebench_id_03029",
  "company": "3M",
  "doc_name": "3M_2018_10K",
  "question_type": "metrics-generated",
  "question_reasoning": "Need to locate CapEx under investing cash flows...",
  "domain_question_num": 1,
  "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
  "answer": "$1577.00",
  "justification": "From 3M 2018 10-K, page 60: Purchases of property, plant and equipment was $1,577 million.",
  "dataset_subset_label": "FB_test",
  "evidence": [
    {
      "evidence_page_num": 60,
      "evidence_text": "Purchases of property, plant and equipment ... (1,577)"
    }
  ]
}
```

### Preprocessing Applied to FinanceBench
1. **PyMuPDF Extraction:** Ingested via `fitz` into plain text and markdown tables ([`scripts/preprocess_documents.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/preprocess_documents.py)).
2. **Chunking:** `RecursiveCharacterTextSplitter` configured with `chunk_size=1000` characters, `chunk_overlap=200`, separators `["\n\n", "\n", " ", ""]`.
3. **Chunk Naming Convention:** `{doc_name}_p{page_number}_c{chunk_index}` (e.g., `3M_2018_10K_p60_c1`).
4. **Metadata Section Indexing:** 2,678 sections extracted, summarized, and indexed into [`data/financebench/metadata_faiss_index/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/metadata_faiss_index/).
5. **Stage 2 Enrichment:** 2,902 split table chunks enriched with prepended document and table titles in `bm25_index_phase2`.

---

## 3. Dataset 2: Lotus Technology Company Dataset

### Metadata & Provenance
* **Official Name:** In-depth Single Company QA Dataset (Lotus Technology Inc.).
* **Filing Source:** SEC Form 6-K and Form F-1 regulatory filings filed with the U.S. Securities and Exchange Commission (SEC CIK: 0001968853).
* **Filing Count:** 12 SEC filings.
* **Language:** English source filings; questions and answers curated in Chinese.
* **Local Location:** [`src/test/test_questions/`](file:///c:/Users/user/Downloads/finsage-main/src/test/test_questions/).

### Role in Original Paper vs. Local Workspace
* **Purpose in Original Paper:**
  * Used in Section 5.1 (Table 1) to evaluate the Multi-Path Retrieval (MPR) ablation (FAISS, BM25, Metadata, HyDE).
  * Used in Section 5.2 (Figure 5) to evaluate the Document Re-ranker (DRR) precision, normalized recall, MRR, and binary nDCG.
  * Used in Section 5.3 (Table 2) to evaluate end-to-end question answering (85.33% LLM accuracy, 88.00% manual accuracy).
* **Purpose in Our Implementation:**
  * Inherited directly from the authors' official repository (`simplew4y/finsage`).
  * Used to audit the original paper's baseline code (`src/utils/ensembleRetriever.py`, `src/utils/vllmChatService.py`).

### Dataset Statistics & Structure
* **Question Count:** Exactly **75 curated questions** (`src/test/test_questions/question75.md`).
* **Relevant Chunks per Question:** An average of **8.7 ground-truth chunks** per question.
* **Domain Topics Covered:**
  * Business, Products, and Competition (35.2% of queries): vehicle models (Eletre, Emeya), pricing, distribution networks.
  * Financial Performance and Ratios (28.0% of queries): gross margins, revenue, R&D expenses, deliveries.
  * Corporate Governance and Capital Structure: shareholder structure, Geely affiliation, ADR share ratios.
  * Regulatory and Trade Policies (2.5% of queries): EU and US tariff exposure and applicable customs rates.
* **Batch Partitions:** Preserved in `question_batch1.md` through `question_batch5.md`.

---

## 4. Dataset 3: Domain Re-Ranker (DRR) Training Data

### Verification of Local Files
* **Training Data:** [`data/financebench/drr/train_drr.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/drr/train_drr.jsonl) (**125 records**).
* **Validation Data:** [`data/financebench/drr/val_drr.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/drr/val_drr.jsonl) (**17 records**).
* **Validation Summary:** [`data/financebench/drr/dataset_summary.json`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/drr/dataset_summary.json).

### Record Structure & Positive/Negative Architecture
Each JSONL record contains:
* `query`: Natural-language financial question.
* `pos`: Array containing exactly 1 positive (ground-truth authoritative evidence) passage.
* `neg`: Array containing exactly **15 hard mined negative passages** retrieved via initial dense search from unrelated sections of the same or different filings.
* `financebench_id`: Originating question identifier.
* `doc_name`: Filing identifier.
* `pos_chunk_id`: Chunk identifier of the positive passage.
* `neg_chunk_ids`: List of 15 chunk identifiers for the negative passages.

### Rigorous Leakage Audit Findings
To prevent artificial inflation of evaluation metrics, [`scripts/prepare_drr_dataset.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/prepare_drr_dataset.py) enforced strict data segregation:

1. **Held-Out Questions Excluded (Zero Query Leakage):**
   * The 4 diagnostic benchmark questions were hardcoded in `HELD_OUT_QUESTION_IDS` and completely removed:
     * `financebench_id_03029` (Q1: 3M CapEx)
     * `financebench_id_04672` (Q2: 3M Net PP&E)
     * `financebench_id_00746` (Q3: Ulta Registered Debt)
     * `financebench_id_00521` (Q4: Ulta Acquisitions)
2. **Held-Out Documents Excluded (Zero Document Leakage):**
   * The source filings corresponding to diagnostic evaluation were placed in `HELD_OUT_DOCS` and forbidden from appearing in training or validation:
     * `3M_2018_10K`
     * `ULTABEAUTY_2023_10K`
     * `ULTABEAUTY_2023Q4_EARNINGS`
3. **Independent Verification from `dataset_summary.json`:**
   * `query_leakage_detected`: **`false`**
   * `held_out_doc_chunks_in_train`: **`0`**
   * `held_out_doc_chunks_in_val`: **`0`**
   * `train_val_doc_overlap`: **`0`**
   * `train_unique_docs`: **`69`**
   * `val_unique_docs`: **`12`**
   * `total_samples`: **`142`** ($125 + 17$)

---

## 5. Dataset 4: Phase 9B Generator DPO Preference Data

### Verification of Files
* **Directory:** [`data/financebench/phase9b/`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/phase9b/).
* **Train File:** `train_preferences.jsonl` (**28 preference pairs**).
* **Validation File:** `validation_preferences.jsonl` (**8 preference pairs**).
* **Total Pairs:** **36 pairs**.

### Structure & Target
* **Format:** Standard DPO prompt-chosen-rejected format (`prompt`, `chosen`, `rejected`).
* **Intended Target:** Fine-tuning generator token preferences on `Qwen2.5-7B-Instruct`.
* **Historical Finding:** While the model learned the formatting preferences (validation accuracy 83.3%), this fine-tuning failed to improve factual QA accuracy because the errors were retrieval-bound (missing CapEx chunks), proving why DPO belongs on the reranker rather than the generator.

---

## 6. Paper vs. Local Dataset Size Summary

| Dataset Dimension | Original Paper Specification | Our Local Repository Reality | Audit Notes & Classification |
|---|:---:|:---:|---|
| **FinanceBench Documents** | 83 documents | **84 PDF filings** | Fully matched (1 extra filing downloaded for coverage). |
| **FinanceBench Questions** | 149 questions evaluated | **150 questions in store** | 100% of benchmark questions available locally. |
| **Stage 1 Evaluated Questions**| 149 questions (via GPT-4o) | **6 held-out questions (Q1–Q6)**| Diagnostic slice targeting 3M and Ulta Beauty. |
| **Stage 2 Evaluated Questions**| N/A | **150 questions (Full Corpus)** | Macro benchmark scaled across all 9 sectors. |
| **Lotus Company Filings** | 12 filings | 12 filings (Codebase legacy) | Preserved from official repo. |
| **Lotus Company Questions** | 75 questions | 75 questions (`question75.md`) | Verified present. |
| **DRR Training Samples** | Single-company filings | 125 train, 17 val ($N=142$) | Hard triplets mined from 81 SEC filings (zero leakage). |
| **Parsed Corpus Chunks** | Fixed 256 tokens | **54,564 text/table chunks** | Preserved in `parsed_chunks.jsonl` (57.1 MB). |
