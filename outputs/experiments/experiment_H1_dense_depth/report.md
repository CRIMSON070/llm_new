# Experiment H1: Dense Retrieval Depth Ablation Report

**Execution Timestamp:** `20261102_164317`  
**Benchmark Scale:** N=150 FinanceBench Questions across 84 SEC Filings  
**Isolated Directory:** `outputs/experiments/experiment_H1_dense_depth/`  

---

## 1. Executive Summary & Benchmark Matrix

| Configuration | Dense Depth | Candidate Pool Size | Candidate Pool Recall | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (Exp F)** | 50 | 50 | **74.67%** (112/150) | 23.33% | 42.67% | 50.67% | 61.33% | 0.3493 | 30.77s |
| **Experiment H1a** | 100 | 50 | **73.33%** (110/150) | 23.33% | 41.33% | 50.67% | 60.00% | 0.3469 | 30.15s |
| **Experiment H1b** | 150 | 50 | **73.33%** (110/150) | 22.00% | 40.67% | 50.00% | 60.67% | 0.3366 | 30.43s |

## 2. Recovery on the 38 Experiment G Failed Questions

- **Recovered in H1a (depth=100):** 0 / 38 (0.0%)  
- **Recovered in H1b (depth=150):** 1 / 38 (2.6%)  

| Question ID | Company | Exp F Pool Rank | H1a Pool Rank | H1b Pool Rank | Final Reranked Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `financebench_id_04660` | Block | >50 | 999 | 31 | 7 |

## 3. Candidate Pool Turnover & Stability Analysis

- **Average Top-50 Jaccard Similarity (H1a vs Ref):** 89.57% (Average 3.0 new chunks per question)
- **Average Top-50 Jaccard Similarity (H1b vs Ref):** 86.47% (Average 4.1 new chunks per question)

## 4. Sector-Level Breakdown

| GICS Sector | Count | Exp F Pool Recall | H1b Pool Recall | Delta | Exp F Recall@10 | H1b Recall@10 | Exp F MRR | H1b MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Communication Services** | 9 | 55.6% | 55.6% | +0.0% | 44.4% | 33.3% | 0.2333 | 0.2222 |
| **Consumer Discretionary** | 30 | 86.7% | 83.3% | -3.3% | 66.7% | 70.0% | 0.3893 | 0.3931 |
| **Consumer Staples** | 23 | 91.3% | 91.3% | +0.0% | 87.0% | 87.0% | 0.5453 | 0.5435 |
| **Financials** | 13 | 30.8% | 23.1% | -7.7% | 30.8% | 23.1% | 0.2462 | 0.1692 |
| **Health Care** | 19 | 73.7% | 73.7% | +0.0% | 63.2% | 57.9% | 0.2730 | 0.2678 |
| **Industrials** | 19 | 79.0% | 79.0% | +0.0% | 57.9% | 63.2% | 0.2930 | 0.3013 |
| **Information Technology** | 22 | 77.3% | 77.3% | +0.0% | 63.6% | 63.6% | 0.4278 | 0.3850 |
| **Materials** | 9 | 66.7% | 66.7% | +0.0% | 33.3% | 33.3% | 0.0611 | 0.0611 |
| **Utilities** | 6 | 66.7% | 66.7% | +0.0% | 66.7% | 66.7% | 0.3611 | 0.3611 |

## 5. Summary & Conclusions

- Candidate Pool Recall changed by **-1.34%** (-2 net questions).
- Recall@10 changed by **-0.66%** (-1 net questions).
- MRR changed by **-0.0127**.
