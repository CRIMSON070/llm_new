# Experiment L.1: Controlled Candidate-Pool Depth Benchmark Report

**Benchmark Date:** November 4, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 5060 Laptop GPU (7.96 GB VRAM), CUDA 13.0, PyTorch 2.14.1  
**Scope:** Stage 3 Held-Out Benchmark ($N = 100$, 40 Form 10-K Filings, 29,335 Chunks)  
**Evaluated Configurations:**
- **Arm A:** Baseline Indexes, $K_{pool} = 50$ (Frozen Baseline)
- **Arm B:** Grounded Indexes, $K_{pool} = 50$ (Experiment L)
- **Arm C1:** Grounded Indexes, $K_{pool} = 65$
- **Arm C2:** Grounded Indexes, $K_{pool} = 75$
- **Arm C3:** Grounded Indexes, $K_{pool} = 100$

---

## 1. Executive Summary & Core Benchmark Findings

Experiment L.1 evaluated the exact empirical effect of varying candidate pool depth $K \in \{50, 65, 75, 100\}$ on the newly grounded Stage 3 corpus. All other variables (query-time coreference, BM25 retrieval, BGE-M3 dense embeddings, RRF fusion parameter $k=60$, and BGE-reranker-base FP16) were held strictly constant.

### Key Measured Results:
1. **Candidate Pool Growth Across Depth:**
   - $K=50$ (Baseline Arm A): **26.00%** (26/100)
   - $K=50$ (Grounded Arm B): **38.00%** (+12 pp, 38/100)
   - $K=65$ (Grounded Arm C1): **44.00%** (+18 pp over baseline, 44/100)
   - $K=75$ (Grounded Arm C2): **46.00%** (+20 pp over baseline, 46/100)
   - $K=100$ (Grounded Arm C3): **48.00%** (+22 pp over baseline, 48/100)
2. **Ranking Metric Peak at $K=65$ (Arm C1):**
   - **Recall@10 peaked at 20.00% (20 hits) at $K=65$**, up from 17.00% in Arm A and 19.00% in Arm B.
   - At $K=75$, Recall@10 slipped to 19.00% as additional distractor chunks diluted reranker scoring.
   - At $K=100$, Recall@10 slipped further to 18.00%, with Recall@5 dropping to 14.00% and MRR dropping to 0.0861.
3. **Latency and Compute Profile:**
   - Latency scaled linearly with pool depth: **0.2339s (K=50) $\rightarrow$ 0.2918s (K=65) $\rightarrow$ 0.3235s (K=75) $\rightarrow$ 0.3993s (K=100)**.
   - Peak GPU VRAM remained virtually identical across all grounded arms (**3,323.9 MB to 3,325.1 MB**), demonstrating that reranking up to 100 candidates fits easily within the 8 GB GPU budget.
4. **Forensic Discovery on Eli Lilly `0075`:**
   - Detailed RRF path decomposition revealed that when `k_pool=250` is evaluated across both paths, `LLY_2023_10K_p10_c0` lands at **RRF Rank #233** (with a BM25 rank of #123 within the multi-path retrieve call).
   - Because all 700+ chunks in `LLY_2023_10K` have the company context header, BM25 term weighting on `"Eli Lilly"` collapsed. Simply expanding $K$ from 50 to 100 **does not recover `0075`**, because 122 other Lilly pipeline table chunks with higher term frequency score ahead of it.

---

## 2. Multi-Arm Comparative Scorecard Matrix

| Metric | Arm A (Base K=50) | Arm B (Grounded K=50) | Arm C1 (Grounded K=65) | Arm C2 (Grounded K=75) | Arm C3 (Grounded K=100) | Best Configuration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Pool Recall** | 26.00% (26/100) | 38.00% (38/100) | 44.00% (44/100) | 46.00% (46/100) | **48.00%** (48/100) | **Arm C3 (+22.0 pp)** |
| **Recall@1** | 5.00% (5/100) | 5.00% (5/100) | 5.00% (5/100) | 5.00% (5/100) | 5.00% (5/100) | **Tied Across All** |
| **Recall@3** | **14.00%** (14/100) | 10.00% (10/100) | 10.00% (10/100) | 10.00% (10/100) | 10.00% (10/100) | Arm A |
| **Recall@5** | **15.00%** (15/100) | **15.00%** (15/100) | **15.00%** (15/100) | **15.00%** (15/100) | 14.00% (14/100) | **Arms A, B, C1, C2** |
| **Recall@10** | 17.00% (17/100) | 19.00% (19/100) | **20.00%** (20/100) | 19.00% (19/100) | 18.00% (18/100) | **Arm C1 (20 hits)** |
| **Precision@5** | 3.00% | 3.00% | 3.00% | 3.00% | 2.80% | **Arms A, B, C1, C2** |
| **Precision@10** | 1.70% | 1.90% | **2.00%** | 1.90% | 1.80% | **Arm C1 (2.00%)** |
| **MRR** | **0.0984** | 0.0886 | 0.0887 | 0.0877 | 0.0861 | Arm A |
| **Avg Latency (s)** | **0.2339 s** | 0.2455 s | 0.2918 s | 0.3235 s | 0.3993 s | Arm A (Fastest) |
| **P95 Latency (s)** | **0.2615 s** | 0.2816 s | 0.3407 s | 0.3729 s | 0.4561 s | Arm A |
| **Peak GPU VRAM (MB)**| **2,781.2 MB** | 3,323.9 MB | 3,323.9 MB | 3,323.9 MB | 3,325.1 MB | Within 8 GB Ceiling |
| **Total Top-10 Hits** | 17 | 19 | **20** | 19 | 18 | **Arm C1 (+3 hits)** |

---

## 3. Special Cases Forensic Deep Dive

The table below tracks the 6 critical audit queries across all 5 arms:

| Question ID & Focus | Arm A (Base K=50) | Arm B (Grounded K=50) | Arm C1 (Grounded K=65) | Arm C2 (Grounded K=75) | Arm C3 (Grounded K=100) | Forensic Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`stage3_heldout_0072`**<br>*Mounjaro Revenue* | Pool: True (RRF #2)<br>Final Rank: **#6** | Pool: True (RRF #1)<br>Final Rank: **#3** | Pool: True (RRF #1)<br>Final Rank: **#3** | Pool: True (RRF #1)<br>Final Rank: **#3** | Pool: True (RRF #1)<br>Final Rank: **#3** | **Unanimous Winner:** Product brand grounding promotes Mounjaro to Rank 3 consistently across all depths. |
| **`stage3_heldout_0075`**<br>*Eli Lilly Tirzepatide* | Pool: True (RRF #11)<br>Final Rank: **#1** | Pool: False (RRF #233)<br>Final Rank: NO_HIT | Pool: False (RRF #233)<br>Final Rank: NO_HIT | Pool: False (RRF #233)<br>Final Rank: NO_HIT | Pool: False (RRF #233)<br>Final Rank: NO_HIT | **IDF Flattening:** Gold chunk is at RRF #233 due to 120+ repetitive pipeline tables. Depth alone does not solve this. |
| **`stage3_heldout_0074`**<br>*Eli Lilly Mfg Capacity* | Pool: True (RRF #41)<br>Final Rank: **#7** | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Item 1 narrative displaced by grounded clinical tables across all K values. |
| **`stage3_heldout_0013`**<br>*NVIDIA ROA* | Pool: True (RRF #18)<br>Final Rank: **#2** | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: True (RRF #34)<br>Final Rank: **#8** | Pool: True (RRF #35)<br>Final Rank: **#8** | Pool: True (RRF #40)<br>Final Rank: NO_HIT | **Recovered at K=65 & 75:** Slipped out at K=50, successfully recovered into pool at K=65 and placed at Rank 8. Slipped out of top 10 at K=100. |
| **`stage3_heldout_0099`**<br>*Caterpillar Dealer Inventory*| Pool: True (RRF #43)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Long prose in introductory Item 1 dilutes BGE-M3 dense representation. |
| **`stage3_heldout_0100`**<br>*Caterpillar Energy Segment* | Pool: True (RRF #8)<br>Final Rank: NO_HIT | Pool: False (OUT)<br>Final Rank: NO_HIT | Pool: True (RRF #55)<br>Final Rank: NO_HIT | Pool: True (RRF #57)<br>Final Rank: NO_HIT | Pool: True (RRF #63)<br>Final Rank: NO_HIT | **Recovered at K=65 & 75:** Entered pool at RRF #55; requires segment statement ranking. |

---

## 4. Key Questions Recovered at $K=65$ and $K=75$

Expanding candidate pool depth to $K=65$ and $K=75$ brought **8 additional questions into the pool** that were missing at $K=50$:
1. **`stage3_heldout_0013` (NVIDIA ROA):** Entered pool at RRF #34 at $K=65$, converted to **Rank 8** in the top 10.
2. **`stage3_heldout_0016` (Cisco Total Revenue):** Entered pool at RRF #31 at $K=65$, converted to **Rank 9** in the top 10.
3. **`stage3_heldout_0026` (Meta Total Revenue):** Entered pool at $K=65$ (RRF #60).
4. **`stage3_heldout_0046` (Wells Fargo Credit Loss Provision):** Entered pool at $K=65$ (RRF #58).
5. **`stage3_heldout_0059` (Target Cost Reductions):** Entered pool at $K=75$ (RRF #71).
6. **`stage3_heldout_0065` (P&G Pricing/FX Drivers):** Entered pool at $K=65$ (RRF #54).
7. **`stage3_heldout_0081` (Merck Keytruda Worldwide Sales):** Entered pool at $K=75$ (RRF #72).
8. **`stage3_heldout_0100` (Caterpillar Energy Profit):** Entered pool at $K=65$ (RRF #55).

---

## 5. Reranker Saturation & Distractor Dilution Analysis

The benchmark revealed an important trade-off between **Candidate Pool Recall** and **Reranking Discrimination**:

```
Candidate Pool Recall:  K=50 (38%)  ──────>  K=65 (44%)  ──────>  K=75 (46%)  ──────>  K=100 (48%)
Recall@10 Hits:         K=50 (19 hits) ───>  K=65 (20 hits) ───>  K=75 (19 hits) ───>  K=100 (18 hits)
Latency (P95):          K=50 (0.28s)  ────>  K=65 (0.34s)  ────>  K=75 (0.37s)  ────>  K=100 (0.46s)
```

- When moving from $K=50$ to $K=65$, Recall@10 increases from **19 to 20 hits** because high-relevance chunks (such as NVIDIA ROA and Cisco Revenue) are successfully promoted into the top 10.
- When expanding to $K=75$ and $K=100$, Candidate Pool Recall continues to climb (up to 48%), but **Recall@10 degrades back down to 18 hits**. 
- **Cause:** Ingesting 75 to 100 candidates presents the cross-encoder with 25–50 additional distractors from the same 10-K filing that share corporate context tokens. The cross-encoder occasionally prefers a distractor chunk with superficial vocabulary overlap over the true target chunk.

---

## 6. Architectural Recommendation & Conclusion

### Measured Recommendation: Adopt $K_{pool} = 65$ (Arm C1)

Based strictly on empirical evidence:
1. **Arm C1 ($K_{pool}=65$) achieves the highest Recall@10 on Stage 3:** **20.00% (20 hits)**, a +3.0 pp gain (+17.6% relative) over the baseline (17 hits).
2. **Candidate Pool Recall reaches 44.00%:** A +18.0 pp gain (+69.2% relative) over the baseline (26.0%).
3. **Mounjaro is optimized at Rank 3** (improved from Rank 6).
4. **NVIDIA ROA (`0013`) is recovered into the top 10 at Rank 8.**
5. **Latency remains sub-300ms:** 0.2918s average, 0.3407s P95.
6. **Peak GPU VRAM is completely stable:** 3.32 GB (well within the 8 GB ceiling).

### Crucial Lesson for the Eli Lilly Issue (`0075`):
Candidate pool depth alone cannot solve the narrative loss in `0075` because the gold chunk sits at **RRF #233** due to 120+ clinical pipeline tables repeating product terms. 

To resolve `0075` while locking in the 44% pool recall of Arm C1, the optimal next architectural advancement is **Section-Aware Candidate Quotaing (or Item 1 Narrative Protection)**, which prevents tables from consuming all top spots in BM25.
