# Phase 1: Domain-Specialized Document Re-Ranker (DRR) Diagnostic Evaluation Report

> **Diagnostic Scope Notice:** Held-out test set size is $N=6$ (Q1–Q6). This represents a small diagnostic benchmark to evaluate ranking behavior, not a statistically conclusive large-scale evaluation.

## 1. Summary Metrics Comparison

| Configuration | MRR | Recall@3 | Recall@5 | Recall@10 | Recall@40 | Status / Paradigm |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **1. Base BGE-Reranker-base** | 0.4604 | 66.7% | 83.3% | 83.3% | 100.0% | Baseline (Zero Boosts) |
| **2. Base Gemma Reranker (Ablation)** | 0.5185 | 66.7% | 66.7% | 100.0% | 100.0% | Unadapted Gemma (Zero Boosts) |
| **3. Trained Phase 1 DRR Adapter** | **0.4200** | **33.3%** | **50.0%** | **83.3%** | 100.0% | **Paper-Faithful DPO (Zero Boosts)** |
| **4. Historical Phase 9C** | 0.5556 | 83.3% | 83.3% | 83.3% | 83.3% | Historical (GAAP Heuristic Boosts) |

## 2. Per-Question Diagnostic Ranking Table (Q1–Q6)

| Q# | Financial Focus | Gold Chunks | Base Rank | Base Gemma | Trained DRR Rank | Phase 9C Rank | $\Delta$ Rank (DRR vs Base) | DRR in Top-3? |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | What is the FY2018 capital expenditure a... | `3M_2018_10K_p46_c4, 3M_2018_10K_p49_c4, 3M_2018_10K_p60_c1` | #16 | #9 | **#37** | #None | -21 | NO |
| **Q2** | Assume that you are a public equities an... | `3M_2018_10K_p58_c1, 3M_2018_10K_p58_c0, 3M_2018_10K_mm_58_1` | #5 | #1 | **#1** | #2 | +4 | YES |
| **Q3** | Which debt securities are registered to ... | `ULTABEAUTY_2023_10K_p1_c1, ULTABEAUTY_2023_10K_p1_c0` | #1 | #2 | **#7** | #1 | -6 | NO |
| **Q4** | What are major acquisitions that Ulta Be... | `ULTABEAUTY_2023Q4_EARNINGS_p1_c2, ULTABEAUTY_2023_10K_p57_c0, ULTABEAUTY_2023Q4_EARNINGS_mm_1_1` | #2 | #1 | **#10** | #2 | -8 | NO |
| **Q5** | Based on the statements of income, what ... | `3M_2018_10K_p127_c1, 3M_2018_10K_p56_c0` | #2 | #6 | **#1** | #1 | +1 | YES |
| **Q6** | What is the difference between Ulta Beau... | `ULTABEAUTY_2023_10K_p57_c0, ULTABEAUTY_2023_10K_p41_c3` | #2 | #3 | **#4** | #3 | -2 | NO |

## 3. Top-3 Retrieved Chunks per Configuration

### Q1: What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.

- **Target Gold Evidence**: `['3M_2018_10K_p46_c4', '3M_2018_10K_p49_c4', '3M_2018_10K_p60_c1']`
- **Base BGE-Reranker Top-3**: `['3M_2018_10K_p7_c5', '3M_2018_10K_p39_c1', '3M_2023Q2_10Q_p72_c3']` (Gold Rank: #16)
- **Trained Phase 1 DRR Top-3**: `['3M_2018_10K_p60_c0', '3M_2023Q2_10Q_p72_c3', '3M_2022_10K_p39_c4']` (Gold Rank: **#37**)
- **Phase 9C Historical Top-3**: `['3M_2018_10K_p39_c1', '3M_2018_10K_p7_c5', '3M_2023Q2_10Q_p72_c3']` (Gold Rank: #None)

### Q2: Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.

- **Target Gold Evidence**: `['3M_2018_10K_p58_c1', '3M_2018_10K_p58_c0', '3M_2018_10K_mm_58_1']`
- **Base BGE-Reranker Top-3**: `['3M_2018_10K_p47_c4', '3M_2018_10K_p44_c2', '3M_2018_10K_p16_c1']` (Gold Rank: #5)
- **Trained Phase 1 DRR Top-3**: `['3M_2018_10K_p58_c1', '3M_2018_10K_p58_c0', '3M_2023Q2_10Q_p5_c0']` (Gold Rank: **#1**)
- **Phase 9C Historical Top-3**: `['3M_2018_10K_p44_c2', '3M_2018_10K_p58_c1', '3M_2018_10K_p47_c4']` (Gold Rank: #2)

### Q3: Which debt securities are registered to trade on a national securities exchange under Ulta Beauty's name as of FY2023?

- **Target Gold Evidence**: `['ULTABEAUTY_2023_10K_p1_c1', 'ULTABEAUTY_2023_10K_p1_c0']`
- **Base BGE-Reranker Top-3**: `['ULTABEAUTY_2023_10K_p1_c1', 'ULTABEAUTY_2023_10K_p1_c0', 'ULTABEAUTY_2023_10K_p27_c2']` (Gold Rank: #1)
- **Trained Phase 1 DRR Top-3**: `['3M_2023Q2_10Q_p1_c1', 'PEPSICO_2023_8K_dated-2023-05-30_p1_c1', 'MICROSOFT_2023_10K_p1_c0']` (Gold Rank: **#7**)
- **Phase 9C Historical Top-3**: `['ULTABEAUTY_2023_10K_p1_c1', 'ULTABEAUTY_2023_10K_p1_c0', 'ULTABEAUTY_2023_10K_p27_c2']` (Gold Rank: #1)

### Q4: What are major acquisitions that Ulta Beauty has done in FY2023 and FY2022?

- **Target Gold Evidence**: `['ULTABEAUTY_2023Q4_EARNINGS_p1_c2', 'ULTABEAUTY_2023_10K_p57_c0', 'ULTABEAUTY_2023Q4_EARNINGS_mm_1_1']`
- **Base BGE-Reranker Top-3**: `['ULTABEAUTY_2023_10K_p33_c2', 'ULTABEAUTY_2023Q4_EARNINGS_p1_c2', 'ULTABEAUTY_2023_10K_p8_c1']` (Gold Rank: #2)
- **Trained Phase 1 DRR Top-3**: `['ULTABEAUTY_2023_10K_p55_c0', 'ULTABEAUTY_2023_10K_p54_c0', 'ULTABEAUTY_2023_10K_p59_c0']` (Gold Rank: **#10**)
- **Phase 9C Historical Top-3**: `['ULTABEAUTY_2023_10K_p33_c2', 'ULTABEAUTY_2023Q4_EARNINGS_p1_c2', 'ULTABEAUTY_2023_10K_p8_c1']` (Gold Rank: #2)

### Q5: Based on the statements of income, what is the absolute difference in net sales for 3M from 2017 to 2018? Also calculate the percentage change.

- **Target Gold Evidence**: `['3M_2018_10K_p127_c1', '3M_2018_10K_p56_c0']`
- **Base BGE-Reranker Top-3**: `['3M_2018_10K_p16_c0', '3M_2018_10K_p56_c0', '3M_2018_10K_p26_c0']` (Gold Rank: #2)
- **Trained Phase 1 DRR Top-3**: `['3M_2018_10K_p56_c0', 'WALMART_2018_10K_p41_c1', '3M_2018_10K_p21_c2']` (Gold Rank: **#1**)
- **Phase 9C Historical Top-3**: `['3M_2018_10K_p56_c0', '3M_2018_10K_p16_c0', '3M_2018_10K_p26_c0']` (Gold Rank: #1)

### Q6: What is the difference between Ulta Beauty's merchandise inventories at the end of FY2023 and the end of FY2022?

- **Target Gold Evidence**: `['ULTABEAUTY_2023_10K_p57_c0', 'ULTABEAUTY_2023_10K_p41_c3']`
- **Base BGE-Reranker Top-3**: `['ULTABEAUTY_2023Q4_EARNINGS_p8_c0', 'ULTABEAUTY_2023_10K_p57_c0', 'ULTABEAUTY_2023Q4_EARNINGS_p7_c0']` (Gold Rank: #2)
- **Trained Phase 1 DRR Top-3**: `['ULTABEAUTY_2023Q4_EARNINGS_p7_c0', 'ULTABEAUTY_2023_10K_p54_c0', 'ULTABEAUTY_2023Q4_EARNINGS_p8_c0']` (Gold Rank: **#4**)
- **Phase 9C Historical Top-3**: `['ULTABEAUTY_2023Q4_EARNINGS_p7_c0', 'ULTABEAUTY_2023_10K_p54_c0', 'ULTABEAUTY_2023_10K_p57_c0']` (Gold Rank: #3)

## 4. Hardware & Latency Telemetry

- **Device**: NVIDIA GeForce RTX 5060 Laptop GPU
- **Adapter Active Verification**: True
- **Base Reranker Time**: 7.16s (Peak VRAM: 0.84 GB)
- **DRR Gemma Time**: 73.75s (Peak VRAM: 2.26 GB)
