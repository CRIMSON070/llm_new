# Experiment H2: Statement-Aware Candidate Selection Report

**Execution Timestamp:** `20261102_171803`  
**Benchmark Scale:** N=150 FinanceBench Questions across 84 SEC Filings  
**Isolated Directory:** `outputs/experiments/experiment_H2_statement_aware/`  

---

## 1. Executive Summary & Benchmark Matrix

| Configuration | Candidate Pool Recall | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (Exp F)** | **74.67%** (112/150) | 23.33% | 42.67% | 50.67% | 61.33% (92/150) | 0.3493 | 30.41s |
| **Experiment H2** | **79.33%** (119/150) | 25.33% | 47.33% | 55.33% | 65.33% (98/150) | 0.3796 | 29.99s |
| **Delta (H2 vs Ref)** | **+4.66%** (+7) | +2.00% | +4.66% | +4.66% | **+4.00%** (+6) | **+0.0303** | -0.42s |

## 2. Acceptance Criteria Evaluation

- [x] **Candidate Pool Recall $\ge 77.33\%$:** PASSED (79.33% vs threshold 77.33%)
- [x] **Recall@10 $\ge$ Reference (61.33%):** PASSED (65.33%)
- [x] **MRR $\ge$ Reference (0.3493):** PASSED (0.3796)
- [x] **Statement Failures Recovered $\ge 5$:** PASSED (8 recovered)
- [x] **Regressions on Successful Questions $\le 1$:** PASSED (1 regressed)
- [x] **Runtime $\le 35$s:** PASSED (29.99s)

## 3. Recovery of the 38 Previously Failed Questions

Experiment H2 recovered **8** of the 38 previous candidate pool misses into the top 50 pool:

| Question ID | Company | Filing | Intent | Exp F Pool Rank | H2 Pool Rank | H2 Final Reranked Rank |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `financebench_id_00438` | Adobe | `ADOBE_2022_10K` | income_statement | >50 | **#47** | **#3** |
| `financebench_id_00723` | American Express | `AMERICANEXPRESS_2022_10K` | income_statement | >50 | **#48** | **#6** |
| `financebench_id_04417` | Best Buy | `BESTBUY_2019_10K` | balance_sheet | >50 | **#45** | **#2** |
| `financebench_id_00005` | Corning | `CORNING_2022_10K` | balance_sheet | >50 | **#49** | **#1** |
| `financebench_id_05915` | CVS Health | `CVSHEALTH_2018_10K` | balance_sheet | >50 | **#49** | **#2** |
| `financebench_id_04103` | General Mills | `GENERALMILLS_2019_10K` | balance_sheet+income_statement | >50 | **#46** | **#1** |
| `financebench_id_00552` | Microsoft | `MICROSOFT_2023_10K` | balance_sheet | >50 | **#46** | **#3** |
| `financebench_id_00216` | Verizon | `VERIZON_2022_10K` | balance_sheet | >50 | **#50** | **#999** |

## 4. Regressions Analysis

The following 1 previously successful questions regressed:

| Question ID | Company | Exp F Pool Rank | H2 Pool Rank | Exp F Final Rank | H2 Final Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `financebench_id_10130` | Corning | #48 | >50 | #5 | #999 |

## 5. Sector-Level Breakdown

| GICS Sector | Count | Ref Pool Recall | H2 Pool Recall | Pool Delta | Ref Recall@10 | H2 Recall@10 | Delta R@10 | Ref MRR | H2 MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Communication Services** | 9 | 55.6% | 66.7% | +11.1% | 44.4% | 44.4% | +0.0% | 0.2333 | 0.2333 |
| **Consumer Discretionary** | 30 | 86.7% | 90.0% | +3.3% | 66.7% | 70.0% | +3.3% | 0.3893 | 0.4212 |
| **Consumer Staples** | 23 | 91.3% | 95.7% | +4.3% | 87.0% | 91.3% | +4.3% | 0.5453 | 0.6304 |
| **Financials** | 13 | 30.8% | 38.5% | +7.7% | 30.8% | 38.5% | +7.7% | 0.2462 | 0.2590 |
| **Health Care** | 19 | 73.7% | 79.0% | +5.3% | 63.2% | 68.4% | +5.3% | 0.2730 | 0.2993 |
| **Industrials** | 19 | 79.0% | 79.0% | +0.0% | 57.9% | 57.9% | +0.0% | 0.2930 | 0.2947 |
| **Information Technology** | 22 | 77.3% | 86.4% | +9.1% | 63.6% | 72.7% | +9.1% | 0.4278 | 0.4717 |
| **Materials** | 9 | 66.7% | 66.7% | +0.0% | 33.3% | 33.3% | +0.0% | 0.0611 | 0.0556 |
| **Utilities** | 6 | 66.7% | 66.7% | +0.0% | 66.7% | 66.7% | +0.0% | 0.3611 | 0.3611 |

