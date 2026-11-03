# FinSage Phase 1: Evaluation Harness V2 Correction Report

**Document:** `outputs/phase1_audit/evaluation_harness_v2/harness_v2_report.md`  
**Date:** 2026-10-11  
**Project:** FinSage Reproduction & Evaluation Harness Correction  
**Scope:** Narrowly scoped evaluation-harness correction removing artificial false negatives from string matching and output-schema assumptions without changing any retrieval, reranking, model weights, adapters, or datasets.  
**Preserved Artifacts:** `outputs/phase9c/phase9c_results.jsonl`, `outputs/phase9c/phase9c_ablation_results.json`, `outputs/phase9c/phase9c_report.md`  
**Corrected Results Directory:** `outputs/phase1_audit/evaluation_harness_v2/`  

---

## 1. Executive Summary

In response to the Phase 1 alignment audit, we implemented **Evaluation Harness V2** in `scripts/evaluate_harness_v2.py`. This script applies a decoupled, multi-dimensional evaluation suite to the exact downstream generation outputs produced during Phase 9C.

### Key Results

| Evaluation Dimension | Original Strict Harness (Phase 9C) | Harness V2 (Top-3 Context) | Harness V2 (Top-5 Context) |
|---|---|---|---|
| **Evidence Retrieval Success** | Not evaluated separately | **83.3% (5/6)** | **83.3% (5/6)** |
| **Reporting Period Correctness** | Not evaluated separately | **100.0% (6/6)** | **100.0% (6/6)** |
| **Arithmetic Exactness** | Collapsed into JSON schema | **83.3% (5/6)** | **100.0% (6/6)** |
| **Numeric & Unit Correctness** | 50.0% (3/6) (Q1, Q4, Q5) | **50.0% (3/6)** (Q2, Q4, Q5) | **50.0% (3/6)** (Q4, Q5, Q6) |
| **Semantic Answer Correctness** | Not evaluated separately | **50.0% (3/6)** (Q2, Q4, Q5) | **50.0% (3/6)** (Q4, Q5, Q6) |
| **Output-Format Compliance (Strict Schema)** | **50.0% (3/6)** (Q1, Q4, Q5) | **50.0% (3/6)** (Q1, Q4, Q5) | **33.3% (2/6)** (Q4, Q5) |

### Core Observations
1. **Reversal of False Negative in Q2:** In Q2 (3M Net PP&E), the model retrieved the official Balance Sheet table (`3M_2018_10K_p58_c1`) and extracted `$8,738 million`, converting it to `8.738 billion USD` as requested by the query. The original harness flagged this as `valid=False` because `"8.738"` did not appear literally in the table text. Harness V2 recognizes unit scaling ($8,738\text{M} = 8.738\text{B}$), validating numeric, arithmetic, and semantic correctness.
2. **Reversal of Format Rejection in Q6 (Top-5):** In Q6 (Ulta Inventory Difference), the model extracted both operands ($1,603,451k and $1,499,218k) and computed the exact expected difference ($104,233k). The original harness flagged this as `valid=False` because the model outputted the precalculated difference under `"value": 104233` rather than populating separate `"old_value"` and `"new_value"` keys. Harness V2 accepts the precomputed difference, verifies operand grounding, and confirms arithmetic exactness.
3. **Rigorous Handling of Q3:** In Q3 (Ulta Debt Securities), the model retrieved the Section 12(b) registered-securities table and explained in natural language that no debt securities exist. However, in its structured JSON output, it extracted `"Common stock, par value $0.01 per share"` under `extract_single_value`. Under strict project instructions (*"Do not mark an answer correct merely because the model's explanation sounds plausible"* and *"verify whether the question concerns securities registered under Section 12(b) or debt securities held as investments"*), Harness V2 marks Q3 as **Semantically Incorrect** because an equity security was returned for a debt security query, and the Section 12(b) table alone does not establish whether debt securities are held as investments.

---

## 2. Exact Validator Changes Implemented in Harness V2

The table below contrasts the original validation logic in `scripts/run_generation_phase9c.py` with the corrected logic in `scripts/evaluate_harness_v2.py`:

| Component | Original Phase 9C Logic (`run_generation_phase9c.py`) | Harness V2 Corrected Logic (`evaluate_harness_v2.py`) | Rationale & Impact |
|---|---|---|---|
| **String Grounding** | `is_value_in_context(val, context_text)`: Strictly checked if `str(val)` was a literal substring in the chunk text (with comma/decimal stripping). | `unit_aware_grounding(val, context_text, query_text)`: First attempts direct string matching; if numeric, detects context monetary scale (`in millions`, `in thousands`) and query requested scale (`in billions`). Recognizes standard financial scaling factors ($10^3, 10^6, 10^{-3}, 10^{-6}$). | Eliminates false negatives when an LLM correctly converts units (e.g. $8,738M $\rightarrow$ 8.738B) as explicitly demanded by the query. |
| **Difference Calculation** | `safe_calculate(op, old_val, new_val)`: Required both `old_value` and `new_value` to be populated in JSON. If the model emitted `"value": 104233`, flagged `"Missing required old_value or new_value"`. | `evaluate_difference_reporting(parsed, raw_output, context_text)`: Evaluates whether a difference is reported in `"value"`, `"result"`, or `"new_value"`. Searches for operands in `values` array or in raw explanation text. Verifies if operands are grounded in context and confirms $|op_1 - op_2| == \text{diff}$. | Permits valid financial answers where the LLM performed the subtraction step internally without violating the semantic meaning of the question. |
| **Evaluation Dimensions** | Single boolean `valid` flag that collapsed schema parsing, literal string grounding, arithmetic, and factual correctness into one metric. | **Six Decoupled Dimensions:**<br>1. Evidence Retrieval Success<br>2. Numeric & Unit Correctness<br>3. Reporting Period Correctness<br>4. Arithmetic Exactness<br>5. Semantic Answer Correctness<br>6. Output-Format Compliance | Prevents conflating prompt-formatting non-compliance with factual/financial reasoning failures. Preserves original strict schema pass rate. |
| **Non-Existence / Refusal Handling** | Handled only via `operation: "none"`. If an LLM extracted a related entity (e.g. Common Stock when debt was requested) while noting absence in text, marked `valid=True`. | Evaluates whether the structured extraction matches the requested asset class. Explicitly checks whether debt securities vs equity securities were returned. | Enforces strict truthfulness: extracting Common Stock for a debt query is rejected even if the surrounding text explanation sounds plausible. |

---

## 3. Per-Question Q1–Q6 Evaluation Comparison

### Question Q1: 3M FY2018 Capital Expenditures (Cash Flow Statement)
- **Query:** *"What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."*
- **Gold Answer:** `$1577.00` (Purchases of PP&E from Cash Flow Statement)
- **Top-3 Context Chunks:** `3M_2018_10K_p39_c1`, `3M_2018_10K_p7_c5`, `3M_2023Q2_10Q_p72_c3`
- **Model Output (Top-3):** `operation: "extract_single_value"`, `values: ["27"]`, `unit: "million USD"`
- **Evaluation Comparison:**
  - **Original Phase 9C:** `valid=True`, `errors=[]` (Paradoxical pass: `"27"` was literally in chunk `p7_c5`).
  - **Harness V2 Breakdown:**
    - Evidence Retrieval: **MISSING (FAILED)** (Official cash flow chunks `p46_c4` and `p60_c1` ranked #4 and #6).
    - Reporting Period: **CORRECT** (FY2018).
    - Arithmetic: **CORRECT** (No arithmetic involved).
    - Numeric & Unit: **INCORRECT** ($27M is environmental capital projects, not total cash flow CapEx of $1,577M).
    - Semantic Correctness: **INCORRECT**.
    - Strict Schema Compliance: **PASS**.
- **Root Cause & Finding:** Genuine retrieval ranking gap caused by cross-encoder phrasing bias (narrative *"capital projects"* logit +3.98 vs table *"Purchases of PP&E"* logit -2.77).

---

### Question Q2: 3M FY2018 Net PP&E (Balance Sheet)
- **Query:** *"Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions."*
- **Gold Answer:** `$8.70` (or $8.738 billion / $8,738 million)
- **Top-3 Context Chunks:** `3M_2018_10K_p44_c2`, `3M_2018_10K_p58_c1` (GOLD), `3M_2018_10K_p47_c4`
- **Model Output (Top-3):**
  ```json
  {
    "operation": "extract_single_value",
    "values": [8.738]
  }
  ```
  *Explanation:* *"The year-end FY2018 net property, plant, and equipment (PP&E) for 3M is $8.738 billion. Note: The value is extracted from the balance sheet provided in the context, specifically from the 'Property, plant and equipment — net' line item under the 'Total assets' section for the year 2018. The value is presented in millions, hence the conversion to billions for the final answer."*
- **Evaluation Comparison:**
  - **Original Phase 9C:** `valid=False`, `errors=["new_value '8.738' not in context"]`.
  - **Harness V2 Breakdown:**
    - Evidence Retrieval: **FOUND (SUCCESS)** (Gold chunk `p58_c1` at Rank 2).
    - Reporting Period: **CORRECT** (Year-end Dec 31, 2018).
    - Arithmetic: **CORRECT** ($8,738\text{M} / 1,000 = 8.738\text{B}$).
    - Numeric & Unit: **CORRECT** (Recognized factor-of-1,000 scaling between millions in table and billions in query).
    - Semantic Correctness: **CORRECT**.
    - Strict Schema Compliance: **FAIL** (Preserved original literal check).
- **Root Cause & Finding:** Pure evaluation-format problem in original harness. Downstream financial reasoning and extraction were 100% correct.
- **Top-5 Context Note:** In Top-5 context, the model was distracted by the `Total assets` line item ($36,500M) in `p58_c1`, outputting 36.5 billion (Semantic: INCORRECT). This confirms that Top-3 context prevents distraction.

---

### Question Q3: Ulta Beauty Registered Debt Securities (Cover Page)
- **Query:** *"Which debt securities are registered to trade on a national securities exchange under Ulta Beauty's name as of FY2023?"*
- **Gold Answer:** `"There are none"`
- **Top-3 Context Chunks:** `ULTABEAUTY_2023_10K_p1_c1` (GOLD), `ULTABEAUTY_2023_10K_p1_c0` (GOLD), `ULTABEAUTY_2023_10K_p27_c2`
- **Model Output (Top-3):**
  ```json
  {
    "operation": "extract_single_value",
    "values": ["Common stock, par value $0.01 per share"]
  }
  ```
  *Explanation:* *"The context indicates that Ulta Beauty has registered common stock... There are no debt securities mentioned in the provided context. Therefore, the response focuses on the registered equity security."*
- **Evaluation Comparison:**
  - **Original Phase 9C:** `valid=True`, `errors=[]`.
  - **Harness V2 Breakdown:**
    - Evidence Retrieval: **FOUND (SUCCESS)** (Official SEC cover page table at Rank 1).
    - Reporting Period: **CORRECT** (FY2023 Form 10-K).
    - Arithmetic: **CORRECT** (N/A).
    - Numeric & Unit: **INCORRECT** (Returned equity security for debt query).
    - Semantic Correctness: **INCORRECT**.
    - Strict Schema Compliance: **PASS**.
- **Root Cause & Finding:**
  - In strict compliance with instructions: Section 12(b) of the Exchange Act governs securities registered to trade on a national exchange. The Section 12(b) table lists only Common Stock, proving Ulta has zero exchange-registered debt securities under its name.
  - However, the Section 12(b) table **does not** report debt securities held as investments in Ulta's asset portfolio or unlisted private debt notes.
  - More critically: the model emitted `Common stock` in its extraction field instead of selecting `operation: "none"`. Returning an equity security for a debt query is factually incorrect.

---

### Question Q4: Ulta Beauty Major Acquisitions (FY2022 & FY2023)
- **Query:** *"What are major acquisitions that Ulta Beauty has done in FY2023 and FY2022?"*
- **Gold Answer:** `"Ulta Beauty did not make any acquisitions in FY2023 and FY2022."`
- **Top-3 Context Chunks:** `ULTABEAUTY_2023_10K_p33_c2`, `ULTABEAUTY_2023Q4_EARNINGS_p1_c2` (GOLD), `ULTABEAUTY_2023_10K_p8_c1`
- **Model Output (Top-3):**
  ```json
  {
    "operation": "none",
    "result": "No information about major acquisitions in FY2023 and FY2022 is provided in the given context."
  }
  ```
- **Evaluation Comparison:**
  - **Original Phase 9C:** `valid=True`, `errors=[]`.
  - **Harness V2 Breakdown:**
    - Evidence Retrieval: **FOUND (SUCCESS)**.
    - Reporting Period: **CORRECT** (FY2022 / FY2023).
    - Arithmetic: **CORRECT** (N/A).
    - Numeric & Unit: **CORRECT** (Proper null value).
    - Semantic Correctness: **CORRECT** (Proper assertion of non-existence).
    - Strict Schema Compliance: **PASS**.
- **Root Cause & Finding:** Consistent pass across both harnesses.

---

### Question Q5: 3M Net Sales Difference 2017 to 2018 (Income Statement)
- **Query:** *"Based on the statements of income, what is the absolute difference in net sales for 3M from 2017 to 2018? Also calculate the percentage change."*
- **Gold Answer:** `Net sales in 2018 - Net sales in 2017. Approx 3%` ($32,765M - $31,657M = $1,108M diff, ~3.5% pct change)
- **Top-3 Context Chunks:** `3M_2018_10K_p56_c0` (GOLD), `3M_2018_10K_p16_c0`, `3M_2018_10K_p26_c0`
- **Model Output (Top-3):**
  ```json
  {
    "operation": "absolute_difference",
    "values": [32765, 31657],
    "result": 1108
  }
  ```
  *Explanation:* Steps explicitly state 2018 net sales $32,765M, 2017 net sales $31,657M, diff = 1,108M, % change = 3.5%.
- **Evaluation Comparison:**
  - **Original Phase 9C:** `valid=True`, `errors=[]`.
  - **Harness V2 Breakdown:**
    - Evidence Retrieval: **FOUND (SUCCESS)** (Official Income Statement at Rank 1).
    - Reporting Period: **CORRECT** (2017 to 2018).
    - Arithmetic: **CORRECT** ($32,765 - 31,657 = 1,108$; $(1,108 / 31,657) \times 100 = 3.50\%$).
    - Numeric & Unit: **CORRECT** (Exact operand grounding).
    - Semantic Correctness: **CORRECT**.
    - Strict Schema Compliance: **PASS**.
- **Root Cause & Finding:** Flawless retrieval, extraction, arithmetic, and schema compliance across both harnesses.

---

### Question Q6: Ulta Beauty Merchandise Inventories Difference (FY2023 vs FY2022)
- **Query:** *"What is the difference between Ulta Beauty's merchandise inventories at the end of FY2023 and the end of FY2022?"*
- **Gold Answer:** `$1,603,451 - $1,499,218 = $104,233` (in thousands)
- **Top-3 Context Chunks:** `ULTABEAUTY_2023Q4_EARNINGS_p7_c0`, `ULTABEAUTY_2023_10K_p54_c0`, `ULTABEAUTY_2023_10K_p57_c0` (GOLD)
- **Top-5 Context Chunks:** Included above + `ULTABEAUTY_2023Q4_EARNINGS_p8_c0`, `ULTABEAUTY_2023_10K_p33_c2`
- **Model Output (Top-3):**
  ```json
  {
    "operation": "absolute_difference",
    "value": 104219
  }
  ```
- **Model Output (Top-5):**
  ```json
  {
    "operation": "absolute_difference",
    "value": 104233
  }
  ```
  *Explanation:* *"The absolute difference between Ulta Beauty's merchandise inventories at the end of fiscal year 2023 and the end of fiscal year 2022 is $104,233 thousand. This can be seen from the balance sheet data where merchandise inventories were $1,603,451 thousand in 2023 and $1,499,218 thousand in 2022. The difference is calculated as $1,603,451 - $1,499,218 = $104,233 thousand."*
- **Evaluation Comparison:**
  - **Top-3 Context:**
    - Original Phase 9C: `valid=False`, `errors=["Missing required old_value or new_value", "new_value '104219' not in context"]`.
    - Harness V2: Evidence Retrieval = **FOUND**; Reporting Period = **CORRECT**; Arithmetic = **INCORRECT** (Reported difference 104,219 is off by 14 from 104,233); Semantic Correctness = **INCORRECT**; Schema Compliance = **FAIL**.
  - **Top-5 Context:**
    - Original Phase 9C: `valid=False`, `errors=["Missing required old_value or new_value"]`.
    - Harness V2: Evidence Retrieval = **FOUND**; Reporting Period = **CORRECT**; Arithmetic = **CORRECT** ($1,603,451 - 1,499,218 = 104,233$ exact); Numeric & Unit = **CORRECT** (Operands grounded in balance sheet chunk `p57_c0`); Semantic Correctness = **CORRECT**; Schema Compliance = **FAIL** (Schema expected separated `old_value`/`new_value`).
- **Root Cause & Finding:** In Top-5 context, the model achieved perfect factual extraction and exact arithmetic ($104,233k). The original failure was 100% caused by the test harness requiring operands to be partitioned into specific JSON keys.

---

## 4. Multi-Dimensional Summary Metrics

```
========================================================================================
                         EVALUATION HARNESS V2 SCORECARD
========================================================================================
Question   Retrieval (Gold)   Reporting Period   Arithmetic   Numeric/Unit   Semantic Correct   Strict Schema
----------------------------------------------------------------------------------------
Q1 (Top-3)   MISSING (p46 #4)     FY2018 (Pass)    N/A (Pass)   $27M (Fail)      INCORRECT          PASS
Q2 (Top-3)   FOUND   (p58 #2)     FY2018 (Pass)    Exact (Pass) $8.738B (Pass)   CORRECT            FAIL (Format)
Q3 (Top-3)   FOUND   (p1  #1)     FY2023 (Pass)    N/A (Pass)   Equity (Fail)    INCORRECT          PASS
Q4 (Top-3)   FOUND   (p1  #2)     FY22/23 (Pass)   N/A (Pass)   Null (Pass)      CORRECT            PASS
Q5 (Top-3)   FOUND   (p56 #1)     2017/18 (Pass)   Exact (Pass) Exact (Pass)     CORRECT            PASS
Q6 (Top-3)   FOUND   (p57 #3)     FY22/23 (Pass)   -14 (Fail)   104219 (Fail)    INCORRECT          FAIL (Format)
----------------------------------------------------------------------------------------
Q6 (Top-5)   FOUND   (p57 #3)     FY22/23 (Pass)   Exact (Pass) $104,233k (Pass) CORRECT            FAIL (Format)
========================================================================================
```

### Overall Benchmark Accuracy Across Harnesses:
- **Original Strict Schema Pass Rate:**
  - Top-3 Context: **50.0% (3/6)** (Q1, Q4, Q5)
  - Top-5 Context: **33.3% (2/6)** (Q4, Q5)
- **Harness V2 Semantic / Numeric Correctness:**
  - Top-3 Context: **50.0% (3/6)** (Q2, Q4, Q5)
  - Top-5 Context: **50.0% (3/6)** (Q4, Q5, Q6)
  - **Union across Contexts (System Capability):** **66.7% (4/6)** (Q2, Q4, Q5, Q6)

---

## 5. Confirmation of Immutability and Environment Integrity

We confirm that throughout this evaluation harness correction:
1. **Model Weights & Adapters:** No weights were updated. `outputs/phase9b_dpo_phase2` remains completely untouched.
2. **Retrieval & Reranking:** No weights, statement boosts, table boosts, or company multipliers were modified. `statement_boost_weight` remains `2.0`.
3. **Query Expander:** `src/retrieval/financial_expander.py` remains unchanged. No new regex rules were added.
4. **Existing Results & Datasets:** `outputs/phase9c/phase9c_results.jsonl`, `phase9c_ablation_results.json`, and all chunk indexes (`parsed_chunks.jsonl`, `pymupdf_chunks.jsonl`, FAISS index) were preserved with 100% byte-for-byte integrity.
5. **New Deliverables Location:** All corrected evaluation artifacts are isolated in:
   `outputs/phase1_audit/evaluation_harness_v2/`

---

## 6. Remaining Reproduction Gaps

Following this harness correction, the following genuine reproduction gaps remain identified:

1. **Q1 CapEx Phrasing & Cross-Encoder Bias:**
   - The official cash flow statement (`3M_2018_10K_p46_c4` / `p60_c1`, $1,577M) ranks at #4 in Top-3 retrieval.
   - The cross-encoder strongly prefers narrative disclosures containing the exact words *"capital projects"* (logit +3.98) over official table rows titled *"Purchases of property, plant and equipment"* (logit -2.77).
   - In the paper, this is addressed via **HyDE query expansion** and **DPO negative sample contrastive fine-tuning**, not via statement boost heuristics.
2. **Q3 Structured Answer Discipline on Negative Queries:**
   - In Q3, the model correctly reasoned in prose that no debt securities exist, but populated Common Stock in the JSON value field under `extract_single_value`.
   - The model must learn to emit `operation: "none"` or null extraction when the target security class does not exist.
3. **Local Semantic Evaluation Limitation:**
   - The original paper used LlamaIndex's `CorrectnessEvaluator` backed by online OpenAI GPT-4o API (1–5 scale).
   - Because our environment operates locally without active OpenAI API keys, Harness V2 uses deterministic, multi-dimensional rule-based evaluations grounded against verified primary source evidence. We explicitly document this local limitation rather than claiming parity with online GPT-4o evaluation.

---

*Report authored and saved to `outputs/phase1_audit/evaluation_harness_v2/harness_v2_report.md`.*  
*Evaluation completed. Awaiting user approval before proceeding to any Phase 2 improvements.*
