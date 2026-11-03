# FinSage Phase 2: Production Enhancements & Benchmark Scaling
**Document File:** `PHASE2_PLAN.md`  
**Governing Standard:** Advanced Production RAG for Financial Filings  
**Prerequisite:** Phase 1 Faithful Paper Reproduction Archived in [`outputs/phase1_archive/`](file:///c:/Users/user/Downloads/finsage-main/outputs/phase1_archive/) and [`PHASE1_OFFICIAL_RESULTS.md`](file:///c:/Users/user/Downloads/finsage-main/PHASE1_OFFICIAL_RESULTS.md)

---

## 1. Phase 2 Mission and Objectives

While **Phase 1** strictly reproduced the core academic methodology of *arXiv:2504.14493* under zero heuristic boosts, **Phase 2** is dedicated to **overcoming practical production limitations** and **scaling benchmark coverage** across the full 84-filing corpus.

### Key Objectives:
1. **Solve Table Bisection at the Ingestion Layer:** Implement Table Header & Metadata Propagation in PDF preprocessing so bisected statement lines carry metadata tokens into BM25 and Dense FAISS indices natively.
2. **Deploy Calibrated Statement Boosts & Negative Penalties:** Implement calibrated statement boost weights (+2.0 to +4.5) and negative penalties (-3.0) for distracting narrative disclosures (resolving Q1 and Q6 retrieval precision).
3. **Scale Evaluation to Full FinanceBench ($N=150$):** Expand beyond the 6 held-out diagnostic questions to the full 150 annotated questions spanning 32 companies and 9 GICS sectors.
4. **Refine Multi-Step Arithmetic & Unit Normalization:** Enhance generator prompts with verified calculation extraction logic to eliminate arithmetic errors on financial delta queries.
5. **Interactive Phase 1 vs. Phase 2 Dashboard Comparison:** Enable dynamic toggling and side-by-side metric comparison in the Streamlit interface.

---

## 2. Work Packages (WPs)

### WP1: Preprocessing & Table Header Propagation
- **Problem:** Multi-page financial tables (like Page 60 of 3M 2018 10-K) split operating activities into `p60_c0` and investing activities/CapEx into `p60_c1`. `p60_c1` lacks company name, filing year, and statement title tokens, scoring 0 in sparse/dense retrieval.
- **Action:** Update preprocessing to detect table continuations and prepend document title, statement header, and column definitions to subsequent chunks.
- **Target:** Gold evidence `3M_2018_10K_p60_c1` ranks directly in Top-5 retrieval without relying solely on bundling.

### WP2: Targeted Negative Disclosures Penalty & Calibrated Boosts
- **Problem:** Reranker phrasing bias often favors lengthy MD&A paragraphs or environmental risk footnotes over concise tabular statements.
- **Action:** Implement query-context routing:
  - If query targets *"cash flow statement"*, apply a negative penalty (-3.0) to environmental disclosures and increase statement boost from +0.0 to +3.5.
- **Target:** Eliminate false positives on statement queries.

### WP3: Full-Scale Benchmark Evaluation ($N=150$)
- **Problem:** Phase 1 was evaluated only on 6 held-out diagnostic questions (Q1–Q6) covering 2 companies.
- **Action:** Build an automated batch evaluation harness with SQLite/JSONL checkpointing to evaluate all 150 questions in [`data/financebench/questions.jsonl`](file:///c:/Users/user/Downloads/finsage-main/data/financebench/questions.jsonl).
- **Target:** Compute macro-level benchmark metrics: MRR, Recall@1/3/5/10, Numeric Accuracy, and Harness V2 scorecards.

### WP4: Enhanced Multi-Step Arithmetic Extraction
- **Problem:** Subtractions on non-standard units (e.g., Ulta Beauty FY23 vs FY22 inventories in thousands) failed due to LLM arithmetic approximations.
- **Action:** Enforce strict deterministic formula extraction (`operation: absolute_difference`, `operands: [A, B]`) evaluated via Python sandbox math evaluator.
- **Target:** 100% precision on numeric difference questions where operands exist in retrieved context.

### WP5: Streamlit Executive Dashboard Updates
- **Problem:** Dashboard currently marks Phase 2 features as "Paused".
- **Action:** Add a mode selector ("Phase 1: Academic Paper Reproduction" vs "Phase 2: Enhanced Production FinSage") with side-by-side comparison tables.

---

## 3. Directory Layout for Phase 2 Outputs

All Phase 2 results and artifacts will be generated cleanly in:
```text
outputs/phase2/
├── retrieval/      # Phase 2 retrieval ablation pools and candidate rankings
├── evaluation/     # Phase 2 full benchmark evaluation JSONs & scorecards
└── reports/        # Executive Markdown reports & comparison audits
```

---

*Phase 1 results remain permanently preserved in `outputs/phase1_archive/` and `PHASE1_OFFICIAL_RESULTS.md`.*
