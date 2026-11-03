# Stage 3 Independent Validation Dataset Construction Plan

**Project:** FinSage Financial RAG Pipeline  
**Target Milestone:** Stage 3 — Independent Held-Out Validation  
**Target Directory:** `outputs/experiments/experiment_stage3_heldout_validation/`  
**Isolated Data Path:** `data/heldout_stage3/`  
**Governing Objective:** Establish a defensible, publication-grade independent validation benchmark to test whether Experiment H2.1 (Correctness Fixes Only) generalizes to unseen companies, SEC filings, and financial questions.

---

## 1. Executive Summary & Rationale

Throughout Phase 1 and Phase 2 development (Experiments C, D, E1, F, G, H1, H2, and H2.1), all empirical optimization was performed against a development benchmark consisting of:
* **150 FinanceBench questions**
* **84 SEC filings**
* **32 corporate entities**
* **54,564 enriched chunks**

While Experiment H2.1 achieved strong development results:
* **Candidate Pool Recall (k=50):** 86.00% (129/150)
* **Recall@1:** 32.67% (49/150)
* **Recall@3:** 51.33% (77/150)
* **Recall@5:** 60.00% (90/150)
* **Recall@10:** 72.00% (108/150)
* **MRR:** 0.4453
* **System Regressions:** 0

These metrics represent **in-sample / development benchmark performance**, not independent generalization. In particular:
1. The company name normalizer (`COMPANY_MAP`) in `scripts/run_controlled_comparison_h2_1.py` contains hardcoded entries specifically matching the 32 development companies.
2. The statement-heading recognition rules were refined using failure cases identified from the 150 benchmark questions (e.g. `financebench_id_10130` on Corning, `00723` on Pfizer).
3. Re-evaluating on any subset of the 150 questions (e.g. Q1–Q6 or an 80/20 train/test split) represents post-hoc data leakage.

Therefore, Stage 3 establishes a true, out-of-distribution independent held-out benchmark consisting of **20 completely unseen companies**, **40 new annual 10-K filings** (FY2023 & FY2024), and **100 independently constructed financial questions**.

---

## 2. Benchmark Architecture & Target Universe

### 2.1 Corporate Entity Universe (20 Unseen Companies)
To ensure broad generalization across varying financial structures, reporting styles, and disclosure complexities, the candidate universe spans 7 major GICS economic sectors with zero overlap with the 32 development companies:

| GICS Sector | Selected Companies | Ticker | CIK | Fiscal Year-End | Reporting Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Information Technology** | Apple Inc. | `AAPL` | `0000320193` | September | Hardware & services revenue; complex deferred revenue |
| | NVIDIA Corporation | `NVDA` | `0001045810` | January | Hyper-growth compute hardware; gross margin expansion |
| | Cisco Systems, Inc. | `CSCO` | `0000858877` | July | Enterprise networking, recurring software subscriptions |
| | Broadcom Inc. | `AVGO` | `0001730168` | October/November | Semiconductor & VMware acquisition consolidation |
| **Communication Services** | Alphabet Inc. | `GOOGL` | `0001652044` | December | Google Services / Cloud segments, massive AI CapEx |
| | Meta Platforms, Inc. | `META` | `0001326801` | December | Family of Apps vs Reality Labs, large share buybacks |
| **Financials** | Bank of America Corp | `BAC` | `0000070858` | December | Net interest income, provision for credit losses, deposits |
| | Goldman Sachs Group | `GS` | `0000886982` | December | Global Banking & Markets, Asset Management, trading rev |
| | Morgan Stanley | `MS` | `0000895421` | December | Wealth Management, Institutional Securities, AUM fees |
| | Wells Fargo & Company | `WFC` | `0000072971` | December | Retail banking, commercial lending, asset cap disclosures |
| **Consumer Discretionary** | The Home Depot, Inc. | `HD` | `0000354950` | January/February | Same-store sales, merchandise inventories, vendor rebates |
| | Target Corporation | `TGT` | `0000027419` | January/February | Retail gross margin, inventory shrinkage disclosures |
| | Starbucks Corporation | `SBUX` | `0000829224` | September/October | Company-operated vs licensed stores, stored value card |
| **Consumer Staples** | Procter & Gamble | `PG` | `0000080424` | June | Organic sales growth, commodity cost impacts, dividends |
| **Health Care** | Eli Lilly and Company | `LLY` | `0000059478` | December | Mounjaro/Zepbound sales surge, R&D expense capitalization |
| | UnitedHealth Group | `UNH` | `0000731766` | December | Medical loss ratio (MLR), Optum vs UnitedHealthcare |
| | Merck & Co., Inc. | `MRK` | `0000310158` | December | Keytruda concentration risk, Gardasil international growth |
| | AbbVie Inc. | `ABBV` | `0001551152` | December | Humira biosimilar erosion, Skyrizi/Rinvoq replacement |
| **Energy** | Exxon Mobil Corp | `XOM` | `0000034088` | December | Upstream/Downstream earnings, Pioneer Natural Resources |
| **Industrials** | Caterpillar Inc. | `CAT` | `0000018230` | December | Construction/Resource/Energy segments, dealer inventory |

### 2.2 SEC Filing Selection (40 Annual 10-K Filings)
Each company is represented by exactly two consecutive annual Form 10-K filings:
* **FY2023 Form 10-K** (providing baseline historical comparative periods)
* **FY2024 Form 10-K** (providing the most recent completed fiscal year disclosures)

All 40 filings are officially verified from SEC EDGAR with validated CIKs, accession numbers, filing dates, and direct EDGAR URLs in `outputs/experiments/experiment_stage3_heldout_validation/candidate_companies_and_filings.csv`.

---

## 3. Question Taxonomy & Dataset Stratification

The independent benchmark consists of **100 questions** structured across three distinct operational archetypes to stress-test all components of the retrieval pipeline:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3 BENCHMARK QUESTION STRATIFICATION                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   1. Single-Statement Financial Queries (40% — 40 Questions)                    │
│      - Balance Sheet: Cash & Equivalents, Total Assets, Long-Term Debt, Inventory│
│      - Income Statement: Revenue, Cost of Goods Sold, Gross Profit, SG&A, Net Inc│
│      - Cash Flow Statement: Operating Cash Flow, CapEx, Dividends, Repurchases   │
│      - Primary Test: Statement classification & primary table routing            │
│                                                                                  │
│   2. Multi-Statement & Ratio Queries (20% — 20 Questions)                        │
│      - Profitability Ratios: Operating Margin, Net Profit Margin, ROA, ROE       │
│      - Liquidity & Leverage: Current Ratio, Quick Ratio, Debt-to-Equity          │
│      - Efficiency & Cash Conversion: Cash Conversion Cycle, Capex/Cash Flow Ratio│
│      - Primary Test: Cross-statement candidate retrieval and balance retention   │
│                                                                                  │
│   3. Qualitative & MD&A Queries (40% — 40 Questions)                             │
│      - Segment Performance & Operational Drivers (Item 7 MD&A)                   │
│      - Risk Factors & Contingencies (Item 1A Risk Factors)                       │
│      - Strategic Agendas, M&A Integration, Supply Chain Constraints              │
│      - Primary Test: Narrative semantic retrieval without statement corruption   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. End-to-End Implementation Workflow

The dataset construction and evaluation pipeline proceeds through five isolated stages:

### Step 1: Document Acquisition & Ingestion
* Ingest the 40 verified Form 10-K filings from SEC EDGAR.
* Save raw source files exclusively in `data/heldout_stage3/raw_filings/`.
* Parse files using the standardized PyMuPDF parser (`fitz`) maintaining identical table extraction logic used in Phase 2 development.

### Step 2: Isolated Chunking & Enrichment
* Extract text and tabular blocks using the established chunking parameters:
  * Chunk size: ~500 tokens with 100-token sliding window overlap.
  * Metadata preservation: filing doc_name, company, ticker, fiscal year, form type, 1-indexed physical PDF page number.
  * Table enrichment: Prepend markdown table column headers to tabular chunk contents.
* Save the resulting chunks in `data/heldout_stage3/processed/heldout_enriched_chunks.jsonl`.
* **Zero Leakage Guarantee:** Verify that no held-out chunks are written to `data/financebench/processed/` or combined with the 54,564 development chunks.

### Step 3: Isolated Index Generation
* **Dense Vector Index:**
  * Model: `BAAI/bge-m3` (identical to development).
  * Storage: `data/heldout_stage3/indexes/faiss_bge_m3/`.
* **Sparse BM25 Index:**
  * Engine: Memory-mapped BM25 index matching development parameters.
  * Storage: `data/heldout_stage3/indexes/bm25/`.
* Development indexes in `data/financebench/indexes/` remain write-protected and completely untouched.

### Step 4: Annotation & Expert Verification
* Annotators formulate 100 questions adhering strictly to `annotation_guidelines.md`.
* Each question specifies:
  * Canonical Question Text
  * Company & Ticker
  * Target Filing (`doc_name`)
  * Target Fiscal Period
  * Category / Intent Type (`single_statement`, `multi_statement`, `qualitative_narrative`)
  * Financial line items / concepts required
  * Gold evidence page number (1-indexed physical page in filing)
  * Exact extracted gold passage / table row
  * Verified numerical/textual answer
  * Verification status (`verified` vs `pending_expert_annotation`)

### Step 5: Frozen Comparative Evaluation
* Execute `scripts/run_heldout_stage3_benchmark.py` evaluating both frozen arms against the 100 independent questions:
  * **Arm 1 (Reference):** Frozen Experiment F baseline (`dense_k=50`, `bm25_k=50`, `pool_k=50`, period-aware promotion, reranker boosts).
  * **Arm 2 (Candidate):** Frozen Experiment H2.1 (`dense_k=50`, `bm25_k=50`, `pool_k=50`, dynamic company detection, strict statement qualification, $Q \le 6$, calibrated boosts).
* Calculate macro metrics, sector breakdowns, question-level transitions, and McNemar's test for statistical significance.

---

## 5. Risk Assessment & Safeguards

| Risk Factor | Impact | Mitigation Strategy |
| :--- | :---: | :--- |
| **Data Leakage into Dev Corpus** | High | Strict directory separation (`data/heldout_stage3/`); automated pre- and post-run hash verification on development assets. |
| **Hallucinated Gold Answers** | Critical | Explicit provenance requirement. Questions without completed primary PDF page verification are flagged `pending_expert_annotation`. |
| **Company Map Generalization Failure** | Moderate | H2.1 company resolver upgraded from static 32-entry dictionary to dynamic company name/ticker extraction from question text. |
| **Premature Benchmark Execution** | High | Benchmark execution strictly blocked until full annotation verification is signed off and user provides explicit authorization. |

---

## 6. Milestones & Delivery Checklist

- [x] **Phase A Audit:** Development corpus inventory and company overlap check completed.
- [x] **Candidate Selection:** 20 unseen companies and 40 SEC EDGAR Form 10-K filings identified and verified.
- [x] **Candidate Catalog:** `candidate_companies_and_filings.csv` generated with valid accession numbers and URLs.
- [x] **Provenance Audit:** `dataset_provenance_and_overlap_audit.md` generated documenting 0.0% overlap.
- [x] **Annotation Standard:** `annotation_guidelines.md` established.
- [x] **Independent Dataset:** `heldout_questions.jsonl` constructed covering 100 questions.
- [x] **Evaluation Protocol:** `frozen_evaluation_protocol.md` drafted.
- [x] **Safety Verification:** Baseline source hashes and prior checkpoints verified unchanged.
- [ ] **Physical Ingestion & Indexing:** Dependent on explicit user authorization.
- [ ] **Final Benchmark Execution:** Dependent on explicit user authorization.
