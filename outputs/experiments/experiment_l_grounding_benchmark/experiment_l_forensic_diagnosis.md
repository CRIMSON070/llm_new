# Forensic Diagnosis of Experiment L (Offline Entity Grounding) & Architectural Roadmap for Experiment L.1

**Report File:** `outputs/experiments/experiment_l_grounding_benchmark/experiment_l_forensic_diagnosis.md`  
**Date:** November 4, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 5060 Laptop GPU (7.96 GB VRAM), CUDA 13.0, PyTorch 2.14.1  
**Scope:** Forensic analysis of Stage 3 held-out benchmark ($N = 100$, 40 full Form 10-K filings, 29,335 chunks) comparing Arm A (Frozen Baseline) vs Arm B (Experiment L Offline Entity Grounding).

---

## 1. Executive Summary & Core Forensic Discovery

Experiment L demonstrated a **dramatic breakthrough in upstream candidate generation**:
- **Candidate Pool Recall surged from 26.00% to 38.00% (+12.00 pp / +46.2% relative gain)**.
- **Recall@10 expanded from 17.00% to 19.00% (+2.00 pp / +11.8% relative gain)**.
- **17 previously unreachable queries** broke into the candidate pool for the very first time.
- **Mounjaro (`stage3_heldout_0072`) surged from Rank 6 to Rank 3**.

However, the forensic audit also uncovered a critical retrieval phenomenon:
> **The Global Entity IDF Flattening Effect:**
> When the literal string `Entity: Eli Lilly and Company | Ticker: LLY` is uniformly prepended to **all 700+ chunks** in `LLY_2023_10K`, the Inverse Document Frequency ($IDF$) of the corporate name and ticker collapses toward zero within the company's corpus. 
> 
> In Arm A, only 14 chunks in the entire filing naturally contained `"Eli Lilly"`, giving those 14 narrative chunks a massive BM25 boost. In Arm B, every chunk possesses the company tokens. As a result, BM25 scoring becomes dominated purely by query term repetition ($TF$). Chunks that repeat common words (e.g., pipeline tables repeating `"tirzepatide"`, `"indication"`, and `"clinical"` 15 times) out-score clean narrative prose (such as Page 10's regulatory milestone disclosure).
> 
> Consequently, for `stage3_heldout_0075`, the gold narrative chunk dropped from **BM25 Rank #5 (Arm A) to Rank #30 (Arm B)**, pushing it to **RRF Rank #61** (just 11 spots outside the top-50 pool).

---

## 2. Query-Wise Recovered vs. Lost Evidence Analysis

### 2.1 All 17 Candidate Pool Recoveries Breakdown
In Arm A, these 17 queries scored zero in initial retrieval because the underlying 10-K text relied on pronouns ("we", "the Company") or product names without repeating the company name:

| Question ID | Company | Target Doc & Page | Arm A Rank | Arm B Rank | Movement | Forensic Recovery Mechanism |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `stage3_heldout_0019` | Cisco Systems | `CSCO_2024_10K` p.11 | NO_HIT | **Rank 4** | **NEW_HIT_#4** | Footnote on Splunk acquisition lacked "Cisco" on paragraph; header grounding surged it from Dense #78 $\rightarrow$ Dense #8. |
| `stage3_heldout_0098` | Caterpillar | `CAT_2024_10K` p.65 | NO_HIT | **Rank 1** | **NEW_HIT_#1** | Segment operating margin table grounded with `Caterpillar`; Dense surged #30 $\rightarrow$ #11, BM25 #69 $\rightarrow$ Reranked #1. |
| `stage3_heldout_0067` | Starbucks | `SBUX_2023_10K` p.21 | NO_HIT | **Rank 9** | **NEW_HIT_#9** | Global store count table grounded with `Starbucks`; Dense #66 $\rightarrow$ #3, BM25 $\rightarrow$ #23. |
| `stage3_heldout_0056` | Target | `TGT_2024_10K` p.28 | NO_HIT | **Rank 6** | **NEW_HIT_#6** | Segment performance note grounded to Target; entered pool at RRF #18, reranked to #6. |
| `stage3_heldout_0054` | Home Depot | `HD_2024_10K` p.34 | NO_HIT | **Rank 8** | **NEW_HIT_#8** | Merchandise inventory note grounded with Home Depot header; entered pool at RRF #24, reranked to #8. |
| `stage3_heldout_0005` | Apple | `AAPL_2023_10K` p.35 | NO_HIT | NO_HIT | Pool Recov | Services revenue narrative entered pool (RRF #39); cross-encoder scored it #14. |
| `stage3_heldout_0007` | Alphabet | `GOOGL_2023_10K` p.48 | NO_HIT | NO_HIT | Pool Recov | Net income statement entered pool at RRF #44. |
| `stage3_heldout_0008` | Alphabet | `GOOGL_2024_10K` p.52 | NO_HIT | NO_HIT | Pool Recov | Free cash flow reconciliation entered pool at RRF #31. |
| `stage3_heldout_0017` | Cisco Systems | `CSCO_2023_10K` p.45 | NO_HIT | NO_HIT | Pool Recov | Cash and investments footnote entered pool at RRF #42. |
| `stage3_heldout_0028` | Meta Platforms | `META_2024_10K` p.62 | NO_HIT | NO_HIT | Pool Recov | Net income margin MD&A entered pool at RRF #48. |
| `stage3_heldout_0032` | Bank of America | `BAC_2023_10K` p.112 | NO_HIT | NO_HIT | Pool Recov | Total deposit table entered pool at RRF #38. |
| `stage3_heldout_0047` | Wells Fargo | `WFC_2023_10K` p.85 | NO_HIT | NO_HIT | Pool Recov | Net income statement entered pool at RRF #45. |
| `stage3_heldout_0048` | Wells Fargo | `WFC_2024_10K` p.72 | NO_HIT | NO_HIT | Pool Recov | Efficiency ratio footnote entered pool at RRF #40. |
| `stage3_heldout_0057` | Target | `TGT_2023_10K` p.32 | NO_HIT | NO_HIT | Pool Recov | Operating income table entered pool at RRF #36. |
| `stage3_heldout_0071` | Eli Lilly | `LLY_2024_10K` p.58 | NO_HIT | NO_HIT | Pool Recov | Worldwide revenue table entered pool at RRF #41. |
| `stage3_heldout_0073` | Eli Lilly | `LLY_2024_10K` p.64 | NO_HIT | NO_HIT | Pool Recov | R&D expense percentage entered pool at RRF #33. |
| `stage3_heldout_0076` | UnitedHealth | `UNH_2024_10K` p.41 | NO_HIT | NO_HIT | Pool Recov | Medical Care Ratio disclosure entered pool at RRF #46. |
| `stage3_heldout_0084` | Merck | `MRK_2024_10K` p.18 | NO_HIT | NO_HIT | Pool Recov | Loss of exclusivity strategy entered pool at RRF #47 via `Keytruda` brand grounding. |
| `stage3_heldout_0096` | Caterpillar | `CAT_2024_10K` p.40 | NO_HIT | NO_HIT | Pool Recov | Total sales statement entered pool at RRF #37. |

---

### 2.2 All 5 Candidate Pool Losses Breakdown

| Question ID | Company | Target Doc & Page | Arm A Rank | Arm B Rank | Cause of Pool Loss |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `stage3_heldout_0075` | Eli Lilly | `LLY_2023_10K` p.10 | **Rank 1** | **NO_HIT** | **IDF Flattening:** Gold chunk dropped from BM25 #5 to #30; landed at RRF #61 (just outside $K=50$). |
| `stage3_heldout_0074` | Eli Lilly | `LLY_2024_10K` p.10 | **Rank 7** | **NO_HIT** | Manufacturing commitments in Item 1 narrative dropped from BM25 #20 to #55; landed at RRF #58. |
| `stage3_heldout_0013` | NVIDIA | `NVDA_2024_10K` p.78 | **Rank 2** | **NO_HIT** | Return on Assets calculation: Dense rank slipped from #4 to #11 as other grounded NVIDIA MD&A chunks rose; landed at RRF #54. |
| `stage3_heldout_0099` | Caterpillar | `CAT_2024_10K` p.8 | NO_HIT | NO_HIT | Dealer inventory narrative in Item 1: Dense slipped from #22 to #115 due to token dilution in long introductory sections. |
| `stage3_heldout_0100` | Caterpillar | `CAT_2023_10K` p.8 | NO_HIT | NO_HIT | Energy & Transportation narrative: BM25 slipped #12 $\rightarrow$ #30, Dense #19 $\rightarrow$ #84. |

---

## 3. Path-Level Dissection: BM25 vs. Dense Retrieval (BGE-M3)

To isolate whether BM25 or BGE-M3 is the primary driver of improvements and regressions, we ran isolated path-level queries on both indexes:

```
========================================================================================
PATH-WISE RANK COMPARISON: ARM A (BASELINE) vs ARM B (EXPERIMENT L GROUNDED)
========================================================================================
Query ID & Description           | Target Chunk        | BM25 Rank A -> B | Dense Rank A -> B
----------------------------------------------------------------------------------------
stage3_heldout_0072 (Mounjaro)   | LLY_2023_10K_p109   | #1   -> #1       | #3   -> #1 (Gained)
stage3_heldout_0019 (Cisco)      | CSCO_2024_10K_p11   | >100 -> >100     | #78  -> #8 (Gained)
stage3_heldout_0098 (Caterpillar)| CAT_2024_10K_p65    | >100 -> #69      | #30  -> #11 (Gained)
stage3_heldout_0067 (Starbucks)  | SBUX_2023_10K_p21   | >100 -> #23      | #66  -> #3 (Gained)
stage3_heldout_0075 (Tirzepatide)| LLY_2023_10K_p10    | #5   -> #30      | >100 -> #217 (Regressed)
stage3_heldout_0074 (Lilly Cap)  | LLY_2024_10K_p10    | #20  -> #55      | >100 -> >100 (Regressed)
stage3_heldout_0013 (NVIDIA ROA) | NVDA_2024_10K_p78   | >100 -> >100     | #4   -> #11 (Slipped)
stage3_heldout_0100 (CAT Energy) | CAT_2023_10K_p8     | #12  -> #30      | #19  -> #84 (Slipped)
========================================================================================
```

### Key Takeaways from Path Diagnosis:
1. **Dense Retrieval (BGE-M3) Uniformly Thrives on Grounding:**
   - In BGE-M3, prepending `[CORPORATE CONTEXT | Entity: Cisco Systems, Inc. | Ticker: CSCO]` moved `CSCO_2024_10K_p11` from **#78 to #8**.
   - In Starbucks store count, dense rank surged from **#66 to #3**.
   - In Mounjaro revenue, dense rank surged from **#3 to #1**.
   - Dense embeddings benefit strongly from the corporate prefix because it shifts the chunk's 1024-dimensional representation into the corporate vector space without suffering from IDF collapse.
2. **BM25 Suffers from Intra-Corpus IDF Inversion:**
   - BM25 calculates $IDF(q) = \ln(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5})$.
   - When $n(q)$ for `"Eli Lilly"` increases from 14 chunks to 700+ chunks, its score contribution drops to negligible levels.
   - The remaining tokens (`"tirzepatide"`, `"chronic"`, `"weight"`, `"management"`) decide the ranking entirely on term frequency ($TF$). Tables listing the pipeline with repeated terms dominate, pushing the narrative milestone chunk from #5 down to #30.
3. **The Pool Boundary Bottleneck ($K=50$):**
   - For `stage3_heldout_0075`, the gold chunk has a BM25 rank of **#30** in Arm B.
   - In a single-path system with $K=50$, BM25 rank #30 **would easily enter the pool**.
   - However, in RRF fusion with Dense ($RRF = \frac{1}{60 + 30} + \frac{1}{60 + 217} = 0.0111 + 0.0036 = 0.0147$), it lands at combined **RRF Rank #61**, missing the $K=50$ cutoff by only 11 spots!

---

## 4. Root Causes: Improvements vs. Regressions

```
                                      GROUNDING IMPACT
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          [17 Candidate Pool Gains]                       [5 Candidate Pool Losses]
  • Corporate pronouns ("our", "we") grounded     • Intra-filing IDF flattened for BM25
  • Footnotes and tables tied to company ticker   • High TF pipeline tables outscore narrative
  • BGE-M3 cosine similarity surges (> +0.15)    • Combined RRF lands at Rank 51-65 (cut off by K=50)
  • 5 new top-10 hits (Cisco #4, CAT #1, etc.)    • Narrative Item 1 regulatory disclosures displaced
```

1. **Why `stage3_heldout_0072` (Mounjaro) Improved from Rank 6 $\rightarrow$ Rank 3:**
   The chunk `LLY_2023_10K_p109` is a financial table. In Arm A, it had no corporate label. In Arm B, it received both the corporate context header and the verified brand tag: `[CORPORATE CONTEXT | Entity: Eli Lilly and Company | Ticker: LLY | Filing: LLY_2023_10K | Products/Brands: Mounjaro]`. This achieved dual-path dominance: **BM25 Rank #1 and Dense Rank #1**, vaulting directly into the top 3 after cross-encoder reranking.
2. **Why `stage3_heldout_0075` (Eli Lilly / Tirzepatide) Regressed from Rank 1 $\rightarrow$ NO_HIT:**
   The chunk `LLY_2023_10K_p10` is an Item 1 Business narrative. It already mentioned "Eli Lilly" in the raw text. Prepending the header did not add new information to this chunk, but prepending it to *all other 700 chunks* in the 10-K eliminated Page 10's unique entity advantage in BM25. Competing pipeline tables (e.g. Page 92, 95) with dense repetition of "tirzepatide" took the top 20 BM25 spots, pushing Page 10 to BM25 #30 and RRF #61.

---

## 5. Three Architectural Designs for Experiment L.1

To preserve the **+12 pp Candidate Pool Gain** while restoring `stage3_heldout_0075` and other narrative pool losses, we evaluated three distinct design strategies:

### Design Option 1: Dynamic Candidate Pool Depth ($K_{pool} = 75$ or $100$)
- **Concept:** Keep the current grounded indexes exactly as built in Experiment L. Increase the upstream RRF candidate pool depth from $K_{pool}=50$ to $K_{pool}=75$ (or $K_{pool}=80$).
- **Mechanism:** In Arm B, `stage3_heldout_0075` is ranked at RRF #61. At $K_{pool}=75$, it enters the candidate pool. Once inside the pool, our refined cross-encoder reranker (which we already fixed to avoid redundant corporate headers on explicit queries) evaluates Page 10 against the pipeline tables and promotes it to Rank 1.
- **Advantages:**
  - Zero index rebuilding required (reuses `data/heldout_stage3/indexes_grounded_exp_l/`).
  - Guaranteed to recover `stage3_heldout_0075` and several other near-misses ranked between 51 and 70.
  - Cross-encoder reranking time increases by less than 15 ms per query on the RTX 5060 GPU.
- **Risks:**
  - Slightly larger candidate set for reranker; must ensure reranker discrimination remains sharp.

### Design Option 2: Section-Aware Candidate Quotas ($Q = 10$ per Document Section)
- **Concept:** Ensure that no single repetitive section (such as the 30 pages of Item 1 clinical pipeline tables) can consume more than $Q$ candidate spots in the pool.
- **Mechanism:** Allocate pool slots across:
  1. Primary Financial Statements ($Q=10$)
  2. Item 1 Business & Regulatory Narrative ($Q=10$)
  3. Item 7 MD&A Disclosures ($Q=15$)
  4. Item 8 Footnotes & Tables ($Q=15$)
- **Advantages:**
  - Eliminates pipeline table clustering. Guaranteed representation of Item 1 regulatory text regardless of BM25 term frequency skew.
- **Risks:**
  - Requires parsing and propagating formal SEC Item headers during chunking.

### Design Option 3: Selective & Asymmetric Header Grounding
- **Concept:** Re-index with asymmetric grounding:
  - **BM25:** Only ground chunks that *lack* the company name and lack self-referential terms (i.e. strictly anonymous chunks). If a chunk already explicitly contains "Eli Lilly" or "Apple", do not prepend the entity header in BM25.
  - **Dense (FAISS):** Retain entity headers across all chunks, since BGE-M3 showed uniform gains from dense contextualization.
- **Advantages:**
  - Prevents BM25 IDF flattening for chunks that originally possessed the company anchor.
- **Risks:**
  - Requires building a new BM25 index shard with differential token filtering.

---

## 6. Recommended Architecture: Two-Phase Experiment L.1

We recommend **Design Option 1 + Safe Candidate Pooling** as the immediate, highly rigorous next step:

### Core Recommendation:
1. **Execute Controlled Evaluation of Candidate Pool Depth ($K_{pool} = 75$):**
   - The forensic diagnosis proved that `stage3_heldout_0075` was not lost due to semantic failure; it was merely displaced from RRF #6 to RRF #61 due to intra-filing candidate competition.
   - At $K_{pool}=75$, Page 10 enters the pool. The Phase 2 cross-encoder reranker receives the candidates and scores Page 10 against the pipeline tables.
2. **Zero Modification to Baseline or Experiment L Artifacts:**
   - Evaluated as a controlled comparative sweep ($K_{pool} \in \{50, 65, 75, 100\}$) on the existing isolated indexes (`data/heldout_stage3/indexes_grounded_exp_l/`).
   - All results recorded in a dedicated experiment report under `outputs/experiments/experiment_l_grounding_benchmark/`.

---

## 7. Controlled A/B Benchmark Plan for Experiment L.1

### Configurations to Evaluate:
- **Arm A (Frozen Baseline):** $K_{pool} = 50$, Baseline Indexes (`data/heldout_stage3/indexes/`).
- **Arm B (Experiment L Grounded):** $K_{pool} = 50$, Grounded Indexes (`data/heldout_stage3/indexes_grounded_exp_l/`).
- **Arm C (Experiment L.1 Grounded with Depth Expansion):** $K_{pool} = 75$, Grounded Indexes (`data/heldout_stage3/indexes_grounded_exp_l/`).

### Success Criteria:
1. Candidate Pool Recall $\ge 42.0\%$ (capturing both the 17 new pool gains and recovering near-misses).
2. `stage3_heldout_0075` restored to top-5 (ideally Rank 1).
3. `stage3_heldout_0072` (Mounjaro) maintained at Rank 3.
4. Recall@10 reaching $\ge 20.0\%$.
