# FinSage Original Research Paper Results

**Document:** `outputs/final_project_audit/05_paper_results.md`  
**Reference Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering* (Wang et al., arXiv:2504.14493, ACM CIKM 2025)  
**Official Repository Data:** `README.md` (lines 50–250) and arXiv paper Section 5  
**Governing Standard:** Exact Extraction of Published Experimental Figures (Zero Fabrication)  

---

## 1. Metric Definitions in Simple, Accessible Language

Before examining the paper's numerical tables, the table below provides plain-English definitions and mathematical formulations for each metric reported in the publication:

| Metric Name | Plain-English Explanation | Formal Mathematical Definition | Real-World Financial Implication |
|---|---|:---:|---|
| **Average Recall** | The proportion of all authoritative financial evidence chunks that the retriever successfully returned. | $\text{Recall} = \frac{|\mathcal{R} \cap \mathcal{G}|}{|\mathcal{G}|}$ | If Recall is low, the LLM is starved of evidence and forced to guess or hallucinate. |
| **Average Precision** | The proportion of retrieved chunks that were actually relevant (measuring noise resistance). | $\text{Precision} = \frac{|\mathcal{R} \cap \mathcal{G}|}{|\mathcal{R}|}$ | Low precision floods the LLM context window with distracting footnotes and wrong fiscal periods. |
| **Average F1 Score** | The harmonic balance between precision (cleanliness) and recall (completeness). | $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Identifies whether retrieval improvements genuinely help or simply dump massive text chunks. |
| **Average Number Retrieved** | The average count of document chunk fragments returned by the candidate stage. | $\text{Avg Num} = \frac{1}{N}\sum_{i=1}^N |\mathcal{R}_i|$ | Reflects token payload passed forward to the cross-encoder and generator. |
| **Mean Reciprocal Rank (MRR)** | Measures how close to the #1 position the first relevant chunk was ranked. | $\text{MRR} = \frac{1}{N}\sum_{i=1}^N \frac{1}{\text{rank}_i}$ | If the gold chunk is at Rank 1, MRR is 1.0; at Rank 2, it is 0.5; at Rank 10, it is 0.1. |
| **Normalized Recall** | Adjusted recall where the denominator is capped at $\min(\text{gold chunks}, K)$. | $\text{NormRecall} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{\min(|\mathcal{G}|, K)}$ | Prevents unfairly penalizing the system when $K < |\mathcal{G}|$ (e.g. 5 slots for 8 gold chunks). |
| **Binary nDCG** | Discounted cumulative gain rewarding systems that rank relevant chunks near the very top. | $\text{nDCG} = \frac{\text{DCG}}{\text{IDCG}}, \text{DCG} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i+1)}$ | A financial statement ranked at #1 gives far higher nDCG than one ranked at #5. |
| **LLM Accuracy** | The proportion of questions where GPT-4o judged the generated response factually correct. | $\text{Acc}_{\text{LLM}} = \frac{\sum \mathbb{I}(\text{GPT-4o Score} \ge \text{Pass})}{N}$ | Automated benchmark judge assessing factual alignment with ground truth. |
| **Manual Accuracy** | The proportion of answers verified as completely correct by expert human financial analysts. | $\text{Acc}_{\text{Manual}} = \frac{\sum \mathbb{I}(\text{Human Verdict} = \text{Correct})}{N}$ | The definitive real-world ground truth standard for high-stakes regulatory QA. |

---

## 2. Table 1: Multi-Path Retrieval (MPR) Performance Breakdown

* **Paper Reference:** Table 1, Section 5.1 (*"MPR Results"*), Page 6.
* **Evaluation Dataset:** In-depth Company Dataset (Lotus Technology Inc., 75 questions across 12 SEC filings, average 8.7 relevant chunks per question).
* **Experimental Configuration:** Candidates retrieved across single and multi-path configurations. Chunk bundling expansion threshold $\tau_{\text{exp}} = 0.85$, bundle limit $\mathcal{K} = 5$.

| Retrieval Category | Specific Method / Configuration | Avg Recall | Avg Precision | Avg F1 | Avg Num Retrieved |
|---|---|:---:|:---:|:---:|:---:|
| **FAISS Dense Retrieval** | FAISS (Baseline BGE-M3 Dense) | 0.8194 | 0.0941 | 0.1646 | 75.60 |
| | + Bundle Expansion (Exp, $\tau=0.85$) | 0.8103 | 0.0999 | 0.1733 | 70.25 |
| **BM25 Lexical Retrieval** | + BM25 Sparse Lexical | 0.8452 | 0.1147 | 0.1949 | 66.24 |
| | + Metadata Section Heading-Summary | 0.8573 | 0.1198 | 0.2037 | 64.87 |
| **HyDE Generative Variants** | HyDE-1: HyDE (`Qwen2.5-7B` Zero-Shot) | 0.8228 | 0.1076 | 0.1830 | 69.09 |
| | HyDE-2: HyDE (`Qwen2.5-7B-SFT` InPars Trained)| 0.8567 | 0.1072 | 0.1831 | 72.92 |
| | HyDE-3: HyDE (`Qwen2.5-72B` Large Zero-Shot) | 0.8323 | 0.1078 | 0.1844 | 69.77 |
| **FinSage (Full System)** | **+ BM25 + Metadata + HyDE-2 + Exp** | **0.9251** | **0.1272** | **0.2156** | **68.75** |

### Key Scientific Takeaways:
1. **The 92.51% Recall Milestone:** Combining all 4 paths (BM25, BGE-M3 dense, section metadata, and fine-tuned HyDE-2) achieves **92.51% average recall**, outperforming naive FAISS (81.94%) by +10.57 percentage points.
2. **Fine-Tuned 7B Beats 72B:** HyDE-2 (`Qwen2.5-7B-SFT`, Recall 0.8567) outperforms the 10x larger `Qwen2.5-72B` (Recall 0.8323), proving that domain-specific alignment outweighs raw model parameter count in financial retrieval.
3. **Chunk Bundling Cleans the Candidate Set:** Adding bundle expansion to FAISS reduced candidate count from 75.60 to 70.25 while raising precision from 0.0941 to 0.0999 and F1 from 0.1646 to 0.1733.

---

## 3. Section 5.2 & Figure 5: Domain-Specialized Re-ranker (DRR) Results

* **Paper Reference:** Section 5.2 (*"DRR Results"*), Figure 5, Page 6.
* **Evaluation Dataset:** In-depth Company Dataset (75 questions, Lotus Technology SEC filings).
* **Experimental Configuration:** Comparing the base general reranker (`BAAI/bge-reranker-v2-gemma`) against the fine-tuned FinSage DRR (DPO LoRA, 10 epochs) under Candidate Pool sizes $R=5$ ($4 \times 5 = 20$ chunks) and $R=10$ ($4 \times 10 = 40$ chunks).

### Candidate Pool Size $R = 5$ (Pool $\ge 20$ Chunks)

| Evaluated Reranker Model | Top-$K$ Evaluated | Precision | Normalized Recall | MRR | Binary nDCG |
|---|:---:|:---:|:---:|:---:|:---:|
| **Base BGE Reranker** (`bge-v2-gemma`) | Top-5 | 0.5913 | 0.6043 | 0.6570 | 0.7540 |
| **FinSage Trained DRR** (DPO Adapter) | **Top-5** | **0.7324** | **0.7456** | **0.7078** | **0.8124** |
| **Base BGE Reranker** (`bge-v2-gemma`) | Top-10 | 0.3771 | 0.6881 | 0.3854 | 0.6011 |
| **FinSage Trained DRR** (DPO Adapter) | **Top-10** | **0.4424** | **0.7783** | **0.3338** | **0.6097** |

### Candidate Pool Size $R = 10$ (Pool $\ge 40$ Chunks)

| Evaluated Reranker Model | Top-$K$ Evaluated | Precision | Normalized Recall | MRR | Binary nDCG |
|---|:---:|:---:|:---:|:---:|:---:|
| **Base BGE Reranker** (`bge-v2-gemma`) | Top-5 | 0.6028 | 0.6130 | 0.6540 | 0.7638 |
| **FinSage Trained DRR** (DPO Adapter) | **Top-5** | **0.7878** | **0.7910** | **0.7545** | **0.8533** |
| **Base BGE Reranker** (`bge-v2-gemma`) | Top-10 | 0.4133 | 0.5985 | 0.4533 | 0.6285 |
| **FinSage Trained DRR** (DPO Adapter) | **Top-10** | **0.5657** | **0.8196** | **0.5155** | **0.6958** |

### Key Scientific Takeaways:
1. **Dramatic Precision & Ranking Surge:** Under the $R=10$ candidate pool, fine-tuning DRR via DPO boosts Top-5 precision from **60.28% to 78.78%** (+18.5%), normalized recall from **61.30% to 79.10%** (+17.8%), MRR from **0.6540 to 0.7545** (+10.1%), and Binary nDCG from **0.7638 to 0.8533** (+8.95%).
2. **Context Window Sufficiency:** Notice that as $K$ increases from 5 to 10, precision drops sharply (from 78.8% to 56.6% on $R=10$), confirming that **Top-5 chunks provide the optimal signal-to-noise ratio** for downstream generation.

---

## 4. Table 2: Performance of End-To-End QA (Published Benchmark)

* **Paper Reference:** Table 2, Section 5.3 (*"Response Results"*), Page 7.
* **Evaluation Evaluators:**
  * **LLM Accuracy:** Scored via LlamaIndex `CorrectnessEvaluator` backed by GPT-4o.
  * **Manual Accuracy:** Scored by expert financial analysts reviewing factual correctness, completeness, and reasoning.
* **Evaluated Benchmarks:** Full open-source **FinanceBench** (149 questions, 83 documents) and custom **Company Dataset** (75 questions, 12 filings).

| Target Evaluation Dataset | Benchmark System / Baseline | Generator LLM | LLM Evaluator Accuracy | Manual Human Accuracy |
|---|---|---|:---:|:---:|
| **FinanceBench** | Islam et al. (2023) [Shared Vector Store] | Llama 2 / GPT-4 | — | 19.00% (0.1900) |
| **FinanceBench** | Jimeno-Yepes et al. (2024) [Base256] | GPT-4 | 32.62% (0.3262) | 36.88% (0.3688) |
| **FinanceBench** | Setty et al. (2024) [Reranker Baseline] | GPT-4 + Reranker | 25.60% (0.2560) | — |
| **FinanceBench** | **FinSage (Published System)** | **Qwen2.5-72B-AWQ** | **49.66% (0.4966)** | **57.05% (0.5705)** |
| **Company Dataset** | **FinSage (Published System)** | **Qwen2.5-72B-AWQ** | **85.33% (0.8533)** | **88.00% (0.8800)** |

### Key Scientific Takeaways:
1. **Outperforming Leading Baselines on FinanceBench:** FinSage achieved **49.66% automated LLM accuracy** and **57.05% manual human accuracy** on FinanceBench, outperforming Jimeno-Yepes et al. by **+17.04%** in LLM accuracy and **+20.17%** in human accuracy.
2. **Company Dataset Mastery:** On the single-company dataset where filing structure was deeply modeled, FinSage achieved **85.33% LLM accuracy** and **88.00% human accuracy**.
3. **The Academic Context of 49.66%:** Notice that even state-of-the-art FinSage with a 72B parameter model achieved ~50% on FinanceBench. FinanceBench is an exceptionally challenging open-book financial QA benchmark requiring complex mathematical deductions across dense regulatory tables.

---

## 5. Section 5.4 & Table 3/4: Comparison with Graph-Based RAG Frameworks

* **Paper Reference:** Section 5.4, Table 3 & 4; Appendix F (*"Comparison with Other RAG Solutions"*).
* **Baselines Evaluated:** GraphRAG (Microsoft) and LightRAG.
* **Evaluation Metrics:** End-to-end response latency (seconds) and Faithful Evaluation Score (1 to 5 scale judged by GPT-4o).

### Response Latency Comparison (Seconds)

| RAG Framework | Mean Latency (s) | Median Latency (s) | Min Latency (s) | Max Latency (s) |
|---|:---:|:---:|:---:|:---:|
| **GraphRAG** | 16.90s | 15.27s | 9.86s | 40.67s |
| **LightRAG** | 12.16s | 8.84s | 3.41s | 859.51s |
| **FinSage** | 19.34s | 18.57s | 8.57s | 40.02s |

### Faithful Evaluation Quality Scorecard (1.0 to 5.0 Scale)

| RAG Framework | Questions Evaluated | Mean Score | Median Score | Min Score | Max Score | Pass Rate (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GraphRAG** | 71 (4 timeouts) | 3.46 | 3.50 | 2.0 | 5.0 | 42.50% |
| **LightRAG** | 75 | 2.45 | 2.00 | 1.0 | 5.0 | 13.67% |
| **FinSage** | 75 | **4.31** | **5.00** | **1.0** | **5.0** | **82.67%** |

### Key Scientific Takeaways:
1. **Graph RAG Fails in Financial Filing Compliance:** LightRAG achieved only a 13.67% pass rate, while GraphRAG achieved 42.50%. Constructing knowledge graphs over financial filings introduces entity hallucination and loses row/column numerical alignments.
2. **FinSage Wins on Factual Fidelity:** FinSage achieved an **82.67% pass rate** (mean score 4.31 / 5.0, median 5.0 / 5.0). While slightly slower (19.34s vs 12.16s), in financial compliance, accuracy and numerical fidelity are non-negotiable.

---

## 6. Section 5.6: Retrieval Latency Breakdown

* **Paper Reference:** Section 5.6 (*"Retrieval Latency"*), `README.md` (lines 235–245).

| Retrieval Component | Average Latency (Seconds) | Percentage of Retrieval Time |
|---|:---:|:---:|
| **FAISS Dense Retrieval** | 0.057s | 47.1% |
| **Metadata Section Retrieval (FAISS)** | 0.050s | 41.3% |
| **BM25 Lexical Retrieval** | 0.014s | 11.6% |
| **Total Parallel Retrieval Time** | **0.121s** | **100.0%** |
