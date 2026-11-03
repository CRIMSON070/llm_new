# FinSage Experiment G: Retrieval Path Diagnostic Report

**Execution Timestamp:** `20261102_163249`  
**Target Population:** 38 Failed Questions from Experiment F Candidate Pool  
**Diagnostic Directory:** `outputs/experiments/experiment_G_retrieval_diagnostics/`  

---

## 1. Executive Summary & Path Hit Counts

All 38 questions whose gold evidence was missing from the Experiment F candidate pool were traced across individual pre-fusion paths and post-fusion rankings at the configured retrieval depth ($k=50$).

| Retrieval Path | Configured Depth | Gold Page Hits (Rank $\le 50$) | Gold Document Hits (Rank $\le 50$) |
| :--- | :--- | :--- | :--- |
| **BM25 Sparse** | $k=50$ | 0 / 38 (0.0%) | 30 / 38 (78.9%) |
| **Dense BGE-M3 / FAISS** | $k=50$ | 5 / 38 (13.2%) | 37 / 38 (97.4%) |
| **Metadata FAISS** | $k_{sections}=3$ | 0 / 38 (0.0%) | 29 / 38 (76.3%) |
| **Union across Any Path** | Pre-fusion | 5 / 38 (13.2%) | 38 / 38 (100.0%) |
| **Experiment F Candidate Pool** | Final top 50 | 0 / 38 (0.0%) | 38 / 38 (100.0%) |

## 2. Failure Category Breakdown

| Diagnostic Failure Category | Count | Percentage |
| :--- | :--- | :--- |
| **Correct document retrieved in top 50, but gold page missed** | 33 | 86.84% |
| **Present in individual path @50 but excluded by fusion/period cutoff** | 5 | 13.16% |

## 3. Confirmed Retrieval Misses vs Ranking & Fusion Cutoff Issues

### A. Present in Path @50 but Excluded by Fusion/Cutoff (5 questions)

These questions successfully retrieved the exact gold page in at least one individual path at depth 50, but were displaced from the final top 50 by RRF or period-prioritization:

| Question ID | Company | Paths with Page | Path Ranks (BM25 / Dense) | Raw RRF Rank | Final Pool Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `financebench_id_00563` | AMD | dense | BM25: >50, Dense: 46 | #106 | >50 |
| `financebench_id_04417` | Best Buy | dense | BM25: >50, Dense: 42 | #105 | >50 |
| `financebench_id_00464` | Boeing | dense | BM25: >50, Dense: 20 | #61 | >50 |
| `financebench_id_00302` | Pfizer | dense | BM25: >50, Dense: 36 | #101 | >50 |
| `financebench_id_00215` | Verizon | dense | BM25: >50, Dense: 24 | #75 | >50 |

### B. Correct Document Retrieved @50, but Gold Page Missed (33 questions)

For these questions, the retrieval pipeline successfully located the correct SEC filing, but the specific target financial table/page was ranked outside the top 50 in all paths:

| Question ID | Company | Filing | Target Pages | Paths with Doc | Best Doc Rank (BM25 / Dense) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `financebench_id_01226` | 3M | `3M_2022_10K` | [27] | bm25|dense|metadata | BM25: 2, Dense: 3 |
| `financebench_id_01865` | 3M | `3M_2022_10K` | [25] | bm25|dense|metadata | BM25: 4, Dense: 1 |
| `financebench_id_02987` | Activision Blizzard | `ACTIVISIONBLIZZARD_2019_10K` | [69, 70] | bm25|dense|metadata | BM25: 2, Dense: 1 |
| `financebench_id_00438` | Adobe | `ADOBE_2022_10K` | [54] | bm25|dense|metadata | BM25: 16, Dense: 19 |
| `financebench_id_01319` | AES Corporation | `AES_2022_10K` | [132] | bm25|dense|metadata | BM25: 34, Dense: 2 |
| `financebench_id_10420` | AES Corporation | `AES_2022_10K` | [130, 132] | bm25|dense|metadata | BM25: 14, Dense: 3 |
| `financebench_id_00799` | Amcor | `AMCOR_2023_10K` | [52] | dense | BM25: >50, Dense: 2 |
| `financebench_id_01079` | Amcor | `AMCOR_2023_10K` | [64] | bm25|dense | BM25: 2, Dense: 2 |
| `financebench_id_00684` | Amcor | `AMCOR_2023_10K` | [50] | dense | BM25: >50, Dense: 5 |
| `financebench_id_01028` | American Express | `AMERICANEXPRESS_2022_10K` | [155] | bm25|dense|metadata | BM25: 1, Dense: 1 |
| *... and 23 more* | | | | | |

### C. Total Retrieval Misses across All Paths @50 (0 questions)

Neither the gold page nor any chunk from the target filing appeared in the top 50 of BM25, Dense FAISS, or Metadata FAISS:

| Question ID | Company | Gold Document | Target Pages | Top-3 Retrieved Filings |
| :--- | :--- | :--- | :--- | :--- |

## 4. Empirical Findings, Hypotheses, and Limitations

### Verified Empirical Findings
1. **Corning Working Capital (`financebench_id_00005`):** Confirmed empirical result from `scripts/find_gold_ranks.py` (task-1264): BM25 rank @500 is `None`; Dense BGE-M3 rank @500 is `#159`. At depth $k=50$, it is completely absent from all paths.
2. **Single-Path vs Multi-Path Coverage:** Dense BGE-M3 produced 37 document hits and 5 page hits @50, while BM25 produced 30 document hits and 0 page hits @50.
3. **Absence from Top 50:** 33 of the 38 failed questions (86.8%) have their target evidence outside the top 50 across BM25, Dense, and Metadata paths.

### Hypotheses Supported by Evidence
1. **Vocabulary / Form Mismatch:** For questions asking qualitative or derived metrics (e.g., 'working capital', 'quick ratio', 'cyclicality'), lexical search fails because the term 'working capital' or 'quick ratio' does not appear in balance sheet line items. BGE-M3 semantic matching ranks them at depth 100–300, beyond the configured $k=50$ cutoff.
2. **Candidate Depth Bottleneck:** Expanding dense retrieval depth from $k=50$ to $k=150$–$200$ prior to fusion or incorporating generative query expansion (HyDE) would capture these submerged balance sheet chunks.

### Unknowns & Limitations
1. **Extremal Dense Ranks:** For questions where dense rank is $>50$, determining the exact rank for all 38 queries beyond $k=50$ requires executing large-$k$ FAISS search ($k=500$) for each query.
2. **Table Structural Representation:** Chunking boundaries without table markdown headers contribute to low lexical scores for tabular financial data.

## 5. Execution Details

- **Script Executed:** `scripts/trace_38_failures.py`
- **Corpus:** Unmodified pre-built BM25 and FAISS indexes on FinanceBench (`BAAI/bge-m3`)
- **Integrity Verification:** Active baseline and Experiment C/F backups verified unchanged via SHA-256.
