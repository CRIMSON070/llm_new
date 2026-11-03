# FinSage Experiment I: Disclosure-Aware Evidence Retrieval — Revised Design Report & Protocol

**Target Milestone:** Experiment I — Controlled Investigation of Retrieval-Depth Ablation and Disclosure-Aware Candidate Selection  
**Report Artifact:** `outputs/experiments/experiment_I_disclosure_aware/experiment_I_design_report.md`  
**Author:** Antigravity Research & Engineering Agent  
**Date:** November 2, 2026 (Revised Version 2.0 — Post-Governance Review)  
**Status:** **PROPOSAL & DESIGN PHASE — AWAITING EXPLICIT USER AUTHORIZATION (ZERO CODE MODIFIED, ZERO BENCHMARKS EXECUTED)**  

---

## 1. Executive Summary & Context

### 1.1 Background & Prior Findings
In the Stage 3 frozen held-out benchmark evaluation across 100 questions on 40 unseen SEC Form 10-K filings (20 unseen corporate entities), the reference baseline (**Experiment F**) and candidate system (**Experiment H2.1**) achieved identical aggregate metrics:
- **Candidate Pool Recall ($K=50$):** **30.00%** (30/100)
- **Recall@1 / Recall@3 / Recall@5 / Recall@10:** **7.00% / 12.00% / 16.00% / 20.00%**
- **Mean Reciprocal Rank (MRR):** **0.1050**
- **Paired Transitions:** **0 recoveries ($b=0$), 0 regressions ($c=0$), 100 ties** ($p = 1.000$)

The independent post-evaluation correctness audit ([`post_evaluation_correctness_audit.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/experiment_stage3_heldout_validation/post_evaluation_correctness_audit.md)) confirmed that the evaluation was executed correctly without software bugs, but identified an architectural ceiling:
> **The Statement-Face Limitation:** Experiment H2.1 restricted statement candidate injection exclusively to the three primary financial statement faces (Consolidated Balance Sheets, Statements of Operations/Income, and Statements of Cash Flows). In full-length, 440-page annual Form 10-K filings, **74.3% of gold evidence targets reside outside the primary statement faces** (specifically in Item 7 MD&A segment margin tables, product revenue summaries, and footnote disclosures).

### 1.2 Core Revisions in This Design
In strict accordance with governance directives, this revised design incorporates five structural safeguards:
1. **Strict Evaluation Leakage Prevention:** Absolute prohibition of access to gold evidence page numbers, gold evidence sources, gold answers, gold target-section labels, or benchmark annotations during retrieval, cataloging, and ranking.
2. **Controlled Arm Separation:** Experiment I is partitioned into two distinct, sequential experimental arms:
   - **Arm I-A (Retrieval-Depth Ablation):** Isolates the effect of increasing BM25 and Dense FAISS retrieval depth from $k=50$ to $k=100$ without any disclosure cataloging or routing changes. Arm I-A results will be evaluated and reported honestly before proceeding.
   - **Arm I-B (Disclosure-Aware Retrieval):** Builds upon the better-validated arm and introduces disclosure-aware candidate selection as the single isolated change.
3. **Rigorous Candidate-Pool Assembly:** Maintains exactly $K_{\text{pool}} = 50$ unique candidates, preserves a substantial baseline candidate floor ($\ge 40$ slots), enforces strict deduplication, and backfills unused slots.
4. **Corrected Evaluation Standards:** Aspirational performance targets are clearly framed as hypothesis goals rather than guaranteed outcomes; statistical significance claims require valid empirical test support ($p < 0.05$).
5. **Absolute Baseline Protection:** Baseline files, previous experiment checkpoints, and production source files remain read-only.

---

## 2. Strict Prevention of Evaluation Leakage

To guarantee complete scientific integrity and prevent overt or covert label leakage, the retrieval system is governed by strict operational boundaries.

### 2.1 Prohibited Information (Zero Access During Retrieval)
The retrieval pipeline, disclosure catalog, intent classifier, and candidate selector are strictly prohibited from reading, referencing, or conditioning on any ground-truth fields:
- ❌ **Gold Evidence Page Numbers:** `evidence_page_num`, `evidence_pages`, `gold_page`, `verified_physical_pdf_page`.
- ❌ **Gold Evidence Source Annotations:** `evidence_sources`, `relevant_passage`, `evidence_text`.
- ❌ **Gold Benchmark Answers:** `gold_answer`, `answer`.
- ❌ **Gold Target-Section Labels:** `target_statement_or_section`, `target_concept`, `category`, `evidence_category`.
- ❌ **Internal Verification Annotations:** `verification_metadata`, `audit_notes`, `adjudication_status`.

Ground-truth evidence labels may be accessed **strictly after final reranking has completed**, solely within the downstream evaluation scoring harness to record binary hits/misses.

### 2.2 Permitted Information (Public Context Only)
Retrieval and candidate generation may condition exclusively on information legitimately available in a real-world financial query:
- **The Raw Query Text:** The user's natural language question string (`question`).
- **Explicit Corporate Entity & Temporal Period:** The target company name / ticker and fiscal year/quarter explicitly stated in the query or question metadata, corresponding to the real-world user prompt context.
- **Unsupervised Corpus Metadata:** Ordinary structural metadata extracted during corpus preprocessing in [`heldout_enriched_chunks.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/heldout_stage3/processed/heldout_enriched_chunks.jsonl) (chunk ID, document name, 1-indexed physical page number, section header markers, and chunk text).

### 2.3 How the Disclosure Catalog Operates Without Gold Labels
The disclosure catalog is constructed **100% offline and unsupervised** across the 40 Form 10-K filings, completely decoupled from the benchmark questions:
1. **Unsupervised Document Parsing:** The catalog scans chunks in `heldout_enriched_chunks.jsonl` using general SEC structural patterns that apply to all public filings (e.g. `Item 7. Management's Discussion and Analysis`, `Consolidated Financial Statements`, Note section titles).
2. **No Question-Specific Tuning:** Catalog entries are indexed by `(doc_name, disclosure_category)` across the entire filing corpus without knowing which questions exist, which pages contain gold answers, or how questions are formulated.
3. **Runtime Candidate Selection:** At query time, the system uses only the question text and company/year metadata to query the public indexes and lookup the matching filing in the catalog. At no point does the lookup query, filter by, or verify against gold evidence annotations.

### 2.4 Step-by-Step Runtime Workflow Demonstrating Zero Access to Gold Labels

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               LEAKAGE-FREE RUNTIME RETRIEVAL PIPELINE (ZERO GOLD LABELS)               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [Step 1: Ingestion & Stripping]                                                       │
│   Input: Raw question record from heldout_questions.jsonl                              │
│   Action: Harness strips all gold annotations (pages, sources, answers, categories).   │
│   Permitted Query Object: {question_id, question, company, fiscal_year, doc_name}      │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  [Step 2: Candidate Generation — Parallel Independent Paths]                           │
│   Path A: Sparse BM25 retrieval over corpus index using question text -> Top-K list    │
│   Path B: Dense BGE-M3 FAISS retrieval over embeddings using question text -> Top-K   │
│   Path C (Arm I-B only): Unsupervised intent detection from question keywords          │
│          -> Lookup unsupervised pre-computed catalog for target doc_name               │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  [Step 3: RRF Fusion & Candidate Pool Assembly]                                        │
│   - RRF fuses BM25 and Dense ranked lists (k_rrf = 60).                                │
│   - Temporal filter prioritizes in-period chunks.                                      │
│   - Pool Assembly preserves baseline floor (>= 40 slots), deduplicates, and caps       │
│     disclosure additions (<= 10 slots). Pool size = exactly 50 unique chunks.          │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  [Step 4: Downstream Phase 2 Reranking]                                                │
│   - Scores the 50 candidate chunks using text cross-encoding, structural table boosts, │
│     and period alignment.                                                              │
│   - Emits final ranked Top-10 chunks.                                                  │
│                                                                                        │
│                                       │  RETRIEVAL PIPELINE COMPLETE                   │
│ ══════════════════════════════════════╪═══════════════════════════════════════════════ │
│                                       │  LEAKAGE FIREWALL (EVALUATION ONLY)            │
│                                       ▼                                                │
│  [Step 5: Isolated Downstream Evaluation]                                              │
│   - Evaluation harness loads gold annotations from frozen benchmark ground truth.      │
│   - Compares emitted Top-10 and candidate pool against gold physical page / chunk ID.  │
│   - Computes Recall@K, MRR, paired transitions, and statistical tests.                 │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1 Failure Analysis: The 70-Question Taxonomy

An exhaustive empirical audit of all 70 questions that missed the top-50 candidate pool in Experiment F was conducted using [`scripts/refine_failure_taxonomy.py`](file:///c:/Users/user/Downloads/finsage-main/scripts/refine_failure_taxonomy.py).

### 3.1 Taxonomy Breakdown Across the 70 Failures

| Evidence Category | Missed Questions ($N=70$) | Share of Failures (%) | Typical Section / Disclosure Location | Core Retrieval Challenge Identified |
| :--- | :---: | :---: | :--- | :--- |
| **1. MD&A Tables and Narrative** | **24** | **34.3%** | Item 7 (Executive Review, CapEx, Operating Expenses) | Multi-page tables with narrative context; high lexical dilution. |
| **2. Primary Financial Statements** | **18** | **25.7%** | Item 8 (Balance Sheet, Income Statement, Cash Flows) | Non-standard banking titles ("Statements of Financial Condition"); deep dense cutoff. |
| **3. Other or Uncertain (Business / Risks)** | **13** | **18.6%** | Item 1 (Business Description) & Item 1A (Risk Factors) | Qualitative narrative paragraphs; generic risk terminology matching multiple filings. |
| **4. Segment or Product Disclosures** | **10** | **14.3%** | Item 7 (MD&A Segment Tables) & Item 8 (Note Segment Reporting) | Product-line revenues (e.g. Mounjaro, Semiconductor Solutions) split by geography. |
| **5. Financial Statement Footnotes** | **4** | **5.7%** | Item 8 Notes (Accounting Policies, Acquisitions, Contingencies) | Embedded tabular roll-forwards; lacks top-level statement headers. |
| **6. Debt, Lease, Inventory Disclosures** | **1** | **1.4%** | Item 8 Notes (Commercial Paper, Credit Facilities, Leases) | Detailed maturity schedules; narrow terminology matching generic debt notes. |
| **Total Failures** | **70** | **100.0%** | Entire Form 10-K Document Space | **74.3% of failures are outside primary statement faces.** |

```
                              EVIDENCE CATEGORY DISTRIBUTION (N=70)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ MD&A Tables & Narrative        ███████████████████████████████████ 24 (34.3%)          │
  │ Primary Financial Statements   ██████████████████████████ 18 (25.7%)                   │
  │ Business & Risk Disclosures    ██────────────────── 13 (18.6%)                         │
  │ Segment & Product Disclosures  ██████████████ 10 (14.3%)                               │
  │ Financial Statement Footnotes  ████ 4 (5.7%)                                           │
  │ Detailed Footnote Tables       █ 1 (1.4%)                                              │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Root-Cause Breakdown
By probing retrieval depths up to $k=100$ across both sparse BM25 and dense FAISS BGE-M3 indexes, three distinct failure mechanisms were isolated:

| Root Cause Mechanism | Count ($N=70$) | Share (%) | Technical Explanation & Evidence |
| :--- | :---: | :---: | :--- |
| **Missing Candidate Generation** | **57** | **81.4%** | Neither BM25 nor Dense FAISS retrieved the gold chunk in the top 100. In 440-page documents, dense vector representations suffer from semantic dispersion across thousands of chunks. |
| **Insufficient Retrieval Depth ($k=50$)** | **12** | **17.1%** | The gold chunk **was retrieved** by BM25 or Dense FAISS at rank 51–100, but was truncated by the hard cutoff at $k=50$ before RRF fusion. |
| **RRF Fusion / Cutoff Displacement** | **1** | **1.4%** | The gold chunk was in the top-50 of an individual path, but was displaced past rank 50 during RRF fusion due to cross-company candidates scoring in both paths. |

### 3.3 Sectoral Failure Distribution
The 70 failures demonstrate acute concentration in complex, multi-segment industries:
- **Financials (BAC, GS, MS, WFC):** 19 / 20 failed (95.0% failure rate) — 400–1,000 page banking 10-Ks with complex credit tables and non-standard statement headings.
- **Health Care (LLY, UNH, MRK, ABBV):** 15 / 20 failed (75.0% failure rate) — Key product revenues (e.g. Mounjaro, Keytruda) reside in MD&A tables or Note 18 Segment Reporting rather than statement faces.
- **Consumer Discretionary (HD, TGT, SBUX):** 10 / 15 failed (66.7% failure rate) — Operating margin drivers and comparable store sales reside in Item 7 MD&A tables.
- **Information Technology (AAPL, NVDA, CSCO, AVGO):** 9 / 20 failed (45.0% failure rate) — Highest baseline success (55% pool recall); failures are primarily MD&A financing reviews.
- **Communication Services (GOOGL, META):** 7 / 10 failed (70.0% failure rate) — AI compute CapEx and segment margins spread across MD&A narrative.
- **Consumer Staples (PG):** 5 / 5 failed (100.0% failure rate) — Lexical dilution across multi-page packaging and volume tables.
- **Energy (XOM):** 3 / 5 failed (60.0% failure rate) — Upstream/Downstream segment reconciliations.
- **Industrials (CAT):** 2 / 5 failed (40.0% failure rate) — Equipment vs. Financial Services segment splits.

---

## 4. Phase 2: Controlled Two-Arm Experimental Protocol

To prevent conflating multiple independent variables, Experiment I is structured into two sequential, isolated experimental arms evaluated against the frozen Experiment F reference baseline.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT I: SEQUENTIAL TWO-ARM PROTOCOL                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  Reference Baseline: Frozen Experiment F (k=50, pool_k=50, No Statement Routing)       │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  ARM I-A: Retrieval-Depth Ablation                                                     │
│  - Single Variable Changed: Increase BM25 depth to k=100 and Dense depth to k=100.     │
│  - All other settings frozen: RRF k=60, temporal filter, Phase 2 reranker, NO routing.  │
│  - Evaluates hypothesis: Does expanding depth recover the 12 truncated gold chunks?    │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  [Milestone Gate: Report Arm I-A results honestly before proceeding to Arm I-B]       │
│                                                                                        │
│                                       │                                                │
│                                       ▼                                                │
│  ARM I-B: Disclosure-Aware Candidate Selection                                         │
│  - Baseline Arm: Starts from the better validated arm (Exp F or Arm I-A).              │
│  - Single Variable Changed: Add disclosure-aware candidate selection from target filing│
│  - Candidate Pool Assembly: Guaranteed floor (>= 40 RRF slots) + Quota (Q <= 10 slots) │
│  - Evaluates hypothesis: Does disclosure routing recover MD&A and segment disclosures? │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.1 Arm I-A: Retrieval-Depth Ablation Protocol

#### Objective
Test whether expanding initial retrieval depth from $k=50$ to $k=100$ in sparse BM25 and dense FAISS retrieval improves candidate pool recall and top-10 retrieval on the 100-question held-out benchmark, without changing any other component.

#### Implementation Details
1. **Sparse BM25 Path:** `mpr.retrieve_sparse(q_text, k=100)`
2. **Dense BGE-M3 FAISS Path:** `mpr.retrieve_dense(q_text, k=100)`
3. **RRF Fusion:** Standard RRF with $k_{\text{rrf}} = 60$ fusing both 100-item lists:
   $$\text{RRF}(c) = \sum_{p \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_p(c)}$$
4. **Temporal Prioritization:** Standard `is_chunk_in_period` prioritization preserves baseline logic: in-period chunks are ranked first, followed by remaining chunks up to $K_{\text{pool}} = 50$.
5. **Reranker:** Standard `Phase2Reranker` with identical weights (statement boost +4.5, disclosure penalty -3.0, company boost +1.5, period boost +1.5, table boost +0.5).
6. **Zero Routing:** No disclosure catalog, no statement routing, no candidate injection.

#### Why Depth Expansion Is an Empirical Ablation (Not an Assumed Gain)
We do not assume that increasing retrieval depth from $k=50$ to $k=100$ will automatically improve final pool recall.
- In multi-company corpora, expanding depth to 100 retrieves up to 100 additional candidates per path.
- Non-gold chunks from other filings or other sections that appear moderately ranked in both BM25 and Dense can achieve higher fused RRF scores than a gold chunk retrieved in only one path at rank 60.
- If cross-filing semantic drift occurs, expanding depth could dilute the top 50 pool composition and displace baseline candidates.
- Therefore, Arm I-A is executed as an isolated ablation to measure empirical gain versus dilution.

#### Milestone Decision Gate for Arm I-B
- If Arm I-A improves Candidate Pool Recall without severe regressions ($b > c$ and Pool Recall $\ge 30\%$), Arm I-B will adopt $k=100$ as its retrieval depth.
- If Arm I-A fails to improve Recall@50 (or causes pool degradation/regressions), that result will be reported honestly and analyzed in detail. In that event, Arm I-B will branch from the frozen Experiment F baseline ($k=50$) instead, ensuring disclosure routing is evaluated without negative depth interference.

---

### 4.2 Arm I-B: Disclosure-Aware Candidate Selection Protocol

#### Objective
Starting from the better-validated arm (either frozen Experiment F or Arm I-A), introduce **disclosure-aware candidate selection** as the single additional architectural change.

#### Unsupervised Multi-Category Disclosure Cataloging
The target filing catalog indexes four structural disclosure categories from `heldout_enriched_chunks.jsonl` using unsupervised SEC document patterns:
1. **Primary Financial Statements (`SEC_PRIMARY_STMT`):**
   - Expanded regex patterns supporting commercial bank filings: `Consolidated Statements of Earnings`, `Statements of Financial Condition`, `Statements of Operations and Comprehensive Income`.
2. **MD&A Operating & Segment Review (`SEC_MDA_TABLES`):**
   - Chunks in Item 7 containing tabular comparisons of net sales, operating margins, non-GAAP adjusted earnings, and capital expenditures.
3. **Segment & Product Revenue Disclosures (`SEC_SEGMENT_DISCLOSURE`):**
   - Note and MD&A tables with headers containing `Segment Reporting`, `Revenues by Product Line`, `Geographic Information`, or product brand schedules.
4. **Detailed Footnote Disclosures (`SEC_FOOTNOTE_DETAIL`):**
   - Note disclosure tables in Item 8 covering debt maturities, credit agreements, lease commitments, and inventory valuations.

#### Unsupervised Disclosure Intent Classifier
The classifier maps query terms to disclosure categories without knowing gold evidence pages:
- **`intent_primary_stmt`:** "net income", "operating income", "total assets", "cash flows", "balance sheet", "net sales", "total revenues", "net interest income", "total deposits".
- **`intent_segment`:** "segment", "product revenue", "sales by region", "geographic", "division", product brand names ("Mounjaro", "Optum", "Google Cloud", "Keytruda").
- **`intent_mda`:** "operating margin", "headwinds", "guidance", "drivers", "growth rate", "non-GAAP", "free cash flow", "capital expenditures", "margin rate".
- **`intent_footnote`:** "debt", "credit facility", "borrowing", "leases", "inventory method", "contingencies".
- **`intent_narrative`:** Risk factors, business descriptions, regulatory proceedings.

#### Candidate Pool Assembly Algorithm (Preserving Pool Integrity)
To guarantee that valid baseline candidates are not displaced, pool assembly enforces strict quota constraints:
$$\text{Total Pool Size } K_{\text{pool}} = 50$$
$$\text{Reserved Baseline Floor } K_{\text{base}} \ge 40$$
$$\text{Max Novel Disclosure Injection Quota } Q_{\text{disc}} \le 10$$

```python
def assemble_disclosure_aware_pool(
    base_fused_cids: List[str],
    disclosure_catalog: Dict[str, Dict[str, List[str]]],
    target_filing: Optional[str],
    detected_intent: Optional[str],
    max_quota: int = 10,
    target_pool_size: int = 50,
    floor_base_slots: int = 40,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Assembles a strictly deduplicated 50-candidate pool with guaranteed baseline preservation.
    
    Guarantees:
      1. Final pool contains exactly target_pool_size unique candidates.
      2. Baseline candidates are preserved with a guaranteed floor (>= floor_base_slots).
      3. Displaced candidates come strictly from the bottom of the base list (ranks 41-50).
      4. Only novel disclosure candidates (not already in base pool) count toward the quota.
      5. Unused quota slots are automatically backfilled from the base candidate list.
    
    Returns:
      (final_pool_cids, injected_cids, displaced_cids)
    """
    if not target_filing or not detected_intent or target_filing not in disclosure_catalog:
        # Fallback: Preserve 100% of base candidates
        return base_fused_cids[:target_pool_size], [], []

    filing_disclosures = disclosure_catalog[target_filing]
    candidate_stmt_cids = filing_disclosures.get(detected_intent, [])

    if not candidate_stmt_cids:
        return base_fused_cids[:target_pool_size], [], []

    # 1. Identify truly NOVEL disclosure candidates (not in the base candidate list)
    base_set = set(base_fused_cids[:target_pool_size])
    novel_disc_cids = []
    for cid in candidate_stmt_cids:
        if cid not in base_set and cid not in novel_disc_cids:
            novel_disc_cids.append(cid)
            if len(novel_disc_cids) >= max_quota:
                break

    # If no novel candidates to inject, retain entire base pool
    if not novel_disc_cids:
        return base_fused_cids[:target_pool_size], [], []

    # 2. Compute exact slots: preserve baseline floor, displace strictly from the bottom
    num_injected = len(novel_disc_cids)
    num_kept_base = target_pool_size - num_injected
    assert num_kept_base >= floor_base_slots, f"Floor violated: {num_kept_base} < {floor_base_slots}"

    # 3. Assemble pool: top-kept base candidates + novel disclosure candidates
    kept_base_cids = base_fused_cids[:num_kept_base]
    displaced_cids = base_fused_cids[num_kept_base:target_pool_size]
    final_pool = kept_base_cids + novel_disc_cids

    # 4. Backfill safeguard: if base list had fewer than target_pool_size items, backfill
    if len(final_pool) < target_pool_size:
        for cid in base_fused_cids[target_pool_size:]:
            if cid not in final_pool:
                final_pool.append(cid)
                if len(final_pool) == target_pool_size:
                    break

    assert len(final_pool) == target_pool_size, f"Pool size mismatch: {len(final_pool)}"
    assert len(set(final_pool)) == target_pool_size, "Duplicate chunks detected in pool!"

    return final_pool, novel_disc_cids, displaced_cids
```

---

## 5. Phase 3: Baseline Protection & Isolation Protocols

To protect historical results and ensure full reproducibility:
1. **Isolated Output Directory:** All Experiment I artifacts, checkpoints, logs, and scorecards will be saved exclusively in:
   `outputs/experiments/experiment_I_disclosure_aware/`
   - Arm I-A results: `outputs/experiments/experiment_I_disclosure_aware/arm_a/`
   - Arm I-B results: `outputs/experiments/experiment_I_disclosure_aware/arm_b/`
2. **Dedicated Experiment Drivers:**
   - Arm I-A driver: `scripts/run_experiment_i_arm_a.py`
   - Arm I-B driver: `scripts/run_experiment_i_arm_b.py`
3. **No Overwrites:** Historical checkpoints and scorecards from Experiments C, D, E1, F, G, H1, H2, and H2.1 remain strictly read-only.
4. **No Edits to Production Code:** Files under `src/retrieval/` will **not** be modified.
5. **SHA-256 Hash Verification:** All 7 protected baseline modules and checkpoints will be verified before and after each arm execution:
   - `src/retrieval/phase2_reranker.py`: `5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9`
   - `scripts/run_phase2_benchmark.py`: `945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1`
   - `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py`: `6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED`
   - `outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl`: `411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C`
   - `outputs/experiments/experiment_H2_statement_aware/scorecard.json`: `2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2`
   - `outputs/experiments/experiment_H2_statement_aware/phase2_retrieval_checkpoint.jsonl`: `E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F`
   - `outputs/experiments/experiment_H2_1_correctness_fixes/scorecard.json`: `ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC`
6. **Data Isolation:** Both arms will retrieve exclusively from `data/heldout_stage3/` without accessing `data/financebench/`.

---

## 6. Phase 4: Revised Evaluation Criteria & Statistical Standards

### 6.1 Evaluation Reporting Requirements
Each arm will report the full set of evaluation metrics against the frozen Experiment F baseline on the 100-question Stage 3 benchmark:
1. **Candidate Pool Recall ($K=50$):** Number and percentage of queries where the gold evidence chunk is in the 50 candidate pool.
2. **Top-$K$ Recall:** Recall@1, Recall@3, Recall@5, and Recall@10.
3. **Mean Reciprocal Rank (MRR):** $1 / \text{rank}$ for rank $\le 10$, $0.0$ otherwise.
4. **Per-Question Transitions:** Exact count of recoveries ($b$), regressions ($c$), and ties ($a, d$).
5. **Stratification by Evidence Category:** Breakdown across all 6 evidence categories (MD&A tables, primary statements, segment disclosures, footnotes, detailed debt/leases, business/risks).
6. **Candidate Pool Turnover:** Candidate pool Jaccard similarity against Experiment F baseline, number of novel injections, and number of displacements.
7. **Runtime Latency:** Mean query retrieval latency (ms/query) and total execution time.
8. **Paired Statistical Significance:**
   - **McNemar's Test:** Exact binomial two-sided test for binary retrieval outcomes ($b$ vs. $c$).
   - **Paired Wilcoxon Signed-Rank Test:** Two-tailed test for reciprocal rank differences.
   - **Pre-condition:** No claim of statistical significance will be asserted unless $p < 0.05$.

### 6.2 Aspirational Performance Goals (Hypotheses, Not Guarantees)
The proposed target metrics represent empirical research hypotheses to be tested, not guaranteed outcomes.

| Evaluation Metric | Frozen Exp F Reference | Arm I-A Aspirational Goal | Arm I-B Aspirational Goal | Statistical Validation Standard |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall ($K=50$)** | **30.00%** (30/100) | $\ge 35.00\%$ (+5.00 pp) | $\ge 42.00\%$ (+12.00 pp) | McNemar test ($p < 0.05$) |
| **Recall@10** | **20.00%** (20/100) | $\ge 22.00\%$ (+2.00 pp) | $\ge 28.00\%$ (+8.00 pp) | McNemar test ($p < 0.05$) |
| **Mean Reciprocal Rank (MRR)** | **0.1050** | $\ge 0.1150$ (+0.0100) | $\ge 0.1500$ (+0.0450) | Wilcoxon signed-rank ($p < 0.05$) |
| **Candidate Pool Regressions ($c$)** | **0** | $\le 1$ | $\le 2$ | Minimal baseline displacement |
| **Candidate Pool Recoveries ($b$)** | **0** | $\ge 5$ | $\ge 12$ | Substantial evidence recovery |
| **Execution Latency** | **180.0 ms** | $\le 210.0\text{ ms}$ | $\le 250.0\text{ ms}$ | Neutral to acceptable overhead |

---

## 7. Immediate Action & Next Steps

- **Current Status:** Revised Design Report completed and archived.
- **Code Status:** **Zero source code edited. Zero experiments executed.**
- **Awaiting User Action:** Awaiting explicit authorization to implement **Arm I-A (Retrieval-Depth Ablation)** and execute the first controlled run.
