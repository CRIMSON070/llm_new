# Period Representation and Mapping Audit: Experiment F

**Audit Date:** November 2, 2026  
**Auditor:** Antigravity AI Pair Programmer  
**Investigation Mode:** Controlled RAG Experiment Pre-Flight Audit  
**Scope:** 150 FinanceBench Questions across 84 SEC Filings  
**Target Output File:** `outputs/experiments/experiment_F_period_aware/period_mapping_audit.md`  

---

## 1. Executive Summary

As part of **Experiment F (Fiscal-Period-Aware Candidate Generation)**, an exhaustive audit was conducted across all 150 benchmark questions, gold evidence labels, and document identifiers in the FinSage repository (`C:\Users\user\Downloads\finsage-main`). 

In Experiment D, **18 Category C retrieval failures** were attributed to **Filing / Fiscal Period Confusion**—where the multi-path candidate generator retrieved chunks from incorrect filings of the target company (e.g., 2018 10-K vs. 2022 10-K, or Q4 Earnings Call transcripts vs. Annual 10-K reports). This audit establishes the exact structure of filing metadata, the linguistic patterns used in benchmark queries, the relationship between fiscal and calendar years, and critical ambiguity edge cases that must govern the Experiment F retrieval algorithm.

---

## 2. Filing Metadata & Document Representation

The benchmark corpus consists of **84 distinct SEC filings** across **32 companies** and **9 GICS sectors**.

### 2.1 Filename Schema
100% of filing document identifiers follow the deterministic schema:
$$\texttt{\{COMPANY\}\_\{YEAR\}[Q\{QUARTER\}]\_\{TYPE\}}$$

- **Company Token (`COMPANY`):** Alphanumeric string with optional underscores (e.g., `3M`, `ADOBE`, `JOHNSON_JOHNSON`, `Pfizer`, `WALMART`).
- **Year Token (`YEAR`):** 4-digit integer (`2015` to `2024`).
- **Quarter Token (`[Q{QUARTER}]`):** Present for quarterly filings and certain earnings calls (e.g., `Q1`, `Q2`, `Q4`; absent for annual 10-K filings).
- **Filing Type Token (`TYPE`):** `10K`, `10Q`, `EARNINGS`, or `8K_dated-...`.

### 2.2 Filing Type Breakdown Across the 84 Documents
- **10-K Annual Reports:** 61 filings (72.6%)
- **10-Q Quarterly Reports:** 10 filings (11.9%)
- **8-K Current Event Reports:** 7 filings (8.3%)
- **Earnings Call Transcripts:** 6 filings (7.1%)

### 2.3 Company Filing Multiplicity
- **Single-Filing Companies (6 companies):** `ACTIVISIONBLIZZARD` (1), `AES` (1), `AMERICANEXPRESS` (1), `COSTCO` (1), `KRAFTHEINZ` (1), `PAYPAL` (1). For these companies, filing confusion cannot occur.
- **Multi-Filing Companies (26 companies):** Contain 2 to 5 distinct filings across different years and document types (e.g., `ADOBE` has 4 filings: 2015, 2016, 2017, 2022; `3M` has 3 filings: 2018 10-K, 2022 10-K, 2023 Q2 10-Q; `AMCOR` has 5 filings; `PEPSICO` has 5 filings). All 18 period-confusion failures occurred exclusively within this group.

---

## 3. Query Period Intent Distribution ($N=150$)

Every question was parsed for temporal expressions using calibrated financial regex extractors:

| Query Period Classification | Question Count | Benchmark % ($N=150$) | Typical Query Syntax Examples |
| :--- | :---: | :---: | :--- |
| **Explicit Single Fiscal Year** | 81 | 54.0% | `"FY2022"`, `"FY 2018"`, `"in fiscal year 2021"`, `"as of FY 2023"` |
| **Multi-Year Range / Comparison** | 31 | 20.7% | `"FY2018 - FY2020 3 year average"`, `"between FY2021 and FY2022"`, `"from FY2015 to FY2016"` |
| **Plain 4-Digit Calendar Year** | 16 | 10.7% | `"in 2022"`, `"between 2022 and 2021"`, `"in 2019"` |
| **Explicit Quarter Specified** | 12 | 8.0% | `"Q2 of FY2023"`, `"2021 Q1"`, `"Q2'2023"`, `"in Q4 of FY 2023"` |
| **No Temporal Expression** | 10 | 6.7% | `"Is Boeing's business subject to cyclicality?"`, `"Does 3M maintain a stable trend of dividend distribution?"` |
| **Total** | **150** | **100.0%** | Comprehensive coverage |

---

## 4. Fiscal Year vs. Calendar Year Mapping

A crucial question is whether a company's fiscal year differs from the calendar year in its filing filename:

1. **Non-Calendar Fiscal Year Endings in Benchmark:**
   - **Walmart (`WALMART_2018_10K`):** Fiscal year ends January 31. The 10-K covering the fiscal year ended January 31, 2018 is designated `2018`. In the benchmark questions: *"what is the FY2018 capital expenditure amount for Walmart?"*. The benchmark query matches `2018`.
   - **Best Buy (`BESTBUY_2024Q2_10Q`):** Fiscal year 2024 ends January 2024. Q2 of FY2024 ended July 29, 2023. The benchmark query asks: *"Was there any drop in Cash & Cash equivalents between FY 2023 and Q2 of FY2024?"*. The filing is named `BESTBUY_2024Q2_10Q`.
   - **Nike (`NIKE_2023_10K`):** Fiscal year ends May 31. The 10-K covering the period ending May 31, 2023 is named `NIKE_2023_10K`. The benchmark query asks for `"FY2023"`.
   - **Adobe (`ADOBE_2022_10K`):** Fiscal year ends early December. Designated `2022`.
   - **Microsoft (`MICROSOFT_2023_10K`):** Fiscal year ends June 30. Designated `2023`.

2. **Core Finding on Filenames:**  
   In FinanceBench, SEC filing filenames strictly adhere to the **Fiscal Year** designation assigned by the SEC and company reporting conventions. When a financial question asks for `FY2022`, it unambiguously targets filings containing `2022` in their identifier.

---

## 5. Ambiguities and Edge Cases

The following critical edge cases were identified during pre-flight analysis. Any period-aware retrieval implementation must handle these cases gracefully to prevent retrieval regressions:

### 5.1 Edge Case 1: Multi-Year Calculations and Comparisons (31 Questions)
- **Example:** `financebench_id_07507`: *"What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016?"*
- **Target Filing:** `ADOBE_2016_10K`.
- **Mechanics:** In SEC filings, a 10-K report includes 3 consecutive years of Income Statements (e.g., the 2016 10-K contains data for 2016, 2015, and 2014).
- **Hazard:** If a naive filter detects `2015` and hard-filters to `ADOBE_2015_10K`, or detects both and drops `ADOBE_2016_10K`, retrieval fails.
- **Rule:** When multiple years are detected ($Y_1 < Y_2$), candidate generation must include filings from **all** detected years, with priority given to the latest filing year ($Y_{\max}$) which typically contains comparative tables.

### 5.2 Edge Case 2: Missing Historical Filing (`financebench_id_00702`)
- **Query:** *"Were there any potential events that are not in Pfizer's standard business operations that substantially increased net income in 2019?"*
- **Target Filing:** `PFIZER_2021_10K`.
- **Corpus Reality:** The corpus contains only `PFIZER_2021_10K` and `Pfizer_2023Q2_10Q`. There is no `PFIZER_2019_10K`. The 2021 10-K reports 2019 numbers in its 3-year comparative financial statements.
- **Hazard:** A hard period filter requiring `2019` in `doc_name` would yield **0 candidates**, causing a catastrophic retrieval failure.
- **Rule:** **Never apply a hard, zero-fallback filter.** If no filing matches the exact year, or if fewer than $K_{\text{pool}}$ candidates match, the system must seamlessly backfill from the unfiltered candidate pool.

### 5.3 Edge Case 3: Regex Word Boundary Hazard with Alphanumerics
- In standard Python regex, `\b(20\d{2})\b` fails on `"FY2024"` because `Y` and `2` are both word characters (`\w`), meaning no word boundary `\b` exists between them!
- **Rule:** Use `(?:FY\s*|Q[1-4]\s*(?:of\s*)?FY\s*|\b)(20\d{2})\b` to ensure robust extraction across both `"FY2024"` and `"FY 2024"`.

### 5.4 Edge Case 4: Queries Without Temporal Expressions (10 Questions)
- Questions inquiring about general corporate structure, cyclicality, or non-dated policies have no temporal tokens.
- **Rule:** If no period is detected with high confidence, the system must completely bypass period filtering and use standard multi-path RRF fusion.

---

## 6. Implementation Guardrails for Experiment F

Based on this audit, the isolated change for Experiment F will be implemented with the following invariants:
1. **Soft Preference / Priority Partitioning:** Matching period chunks are promoted to the top of the candidate pool ($K_{\text{pool}}=50$), while preserving non-period chunks as backfill.
2. **Zero Evaluation Leakage:** Period detection operates strictly on `query_text` and document metadata available in `chunk_store` (`doc_name`). Gold labels (`q["doc_name"]`, `q["evidence"]`) are completely absent from the retrieval loop.
3. **Multi-Period Preservation:** For ranges and comparisons, chunks from all detected years are treated as in-period.
4. **Guaranteed Pool Size ($K_{\text{pool}}=50$):** The candidate pool passed to the reranker will always contain exactly 50 candidates (or all available candidates if total $<50$).
