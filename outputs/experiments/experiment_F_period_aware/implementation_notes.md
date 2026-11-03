# Implementation Notes: Experiment F — Fiscal-Period-Aware Candidate Generation

**Implementation Date:** November 2, 2026  
**Author:** Antigravity AI Pair Programmer  
**Experiment Branch:** Experiment F (`outputs/experiments/experiment_F_period_aware/`)  
**Reference Configuration:** Experiment C (`outputs/experiments/experiment_C_company_intent/`)  

---

## 1. Algorithmic Architecture & Formulation

Experiment F implements **Fiscal-Period-Aware Candidate Generation** to solve filing/period confusion without modifying embedding models, BM25 indices, or reranker weights.

### 1.1 Mathematical Formulation
Let $Q$ be the user query string. Multi-path retrieval generates candidate lists across three paths:
- Path 1 (BM25 sparse): $R_{\text{bm25}} = \{(c, \text{score}_{\text{bm25}})\}_{i=1}^{k=50}$
- Path 2 (Dense FAISS): $R_{\text{dense}} = \{(c, \text{score}_{\text{dense}})\}_{i=1}^{k=50}$
- Path 3 (Metadata): $R_{\text{meta}} = \{(c, \text{score}_{\text{meta}})\}_{i=1}^{k_{\text{meta}}}$

Under standard Reciprocal Rank Fusion (RRF with $k=60$):
$$\text{RRF}(c) = \sum_{p \in \{\text{bm25}, \text{dense}, \text{meta}\}} \frac{1}{60 + \text{rank}_p(c)}$$

Let $\mathcal{C}_{\text{all}} = \text{UniqueCandidates}(R_{\text{bm25}} \cup R_{\text{dense}} \cup R_{\text{meta}})$ be the set of unique retrieved chunks ($|\mathcal{C}_{\text{all}}| \approx 100\text{--}125$).

In standard Experiment C, the candidate pool $\mathcal{P}$ is formed by truncating to the top 50 strictly by raw RRF score:
$$\mathcal{P}_{\text{ExpC}} = \text{TopK}_{50}(\mathcal{C}_{\text{all}}, \text{key}=\text{RRF})$$

In **Experiment F**, candidate pool selection is made **fiscal-period-aware**:
1. **Query Temporal Intent Extraction:** $\mathcal{T}(Q) = \{Y_{\text{years}}, Q_{\text{quarters}}\}$ is extracted using deterministic financial regexes.
2. **Chunk Metadata Classification:** For each candidate chunk $c \in \mathcal{C}_{\text{all}}$, its document identifier $D(c)$ is parsed into $(Y_c, Q_c, T_c)$ where $Y_c$ is the filing fiscal year and $Q_c$ is the filing quarter.
3. **Partitioning:**
   $$\mathcal{C}_{\text{in-period}} = \{c \in \mathcal{C}_{\text{all}} \mid \text{MatchPeriod}((Y_c, Q_c), \mathcal{T}(Q))\}$$
   $$\mathcal{C}_{\text{other}} = \mathcal{C}_{\text{all}} \setminus \mathcal{C}_{\text{in-period}}$$
   Both subsets maintain internal descending order by $\text{RRF}(c)$.
4. **Candidate Pool Assembly with Guaranteed Fallback:**
   $$\mathcal{P}_{\text{ExpF}} = (\mathcal{C}_{\text{in-period}} \mathbin{\Vert} \mathcal{C}_{\text{other}})[:50]$$
   - If $|\mathcal{C}_{\text{in-period}}| \ge 50$, the top 50 in-period candidates form the pool.
   - If $|\mathcal{C}_{\text{in-period}}| < 50$ (e.g. 28 chunks), all 28 in-period chunks are placed first, and the remaining 22 slots are backfilled from $\mathcal{C}_{\text{other}}$ by RRF score.
   - If $\mathcal{T}(Q) = \emptyset$ (no period detected) or $|\mathcal{C}_{\text{in-period}}| = 0$, $\mathcal{P}_{\text{ExpF}} = \mathcal{P}_{\text{ExpC}}$ identically.

---

## 2. Temporal Intent Extraction Regexes

The extraction logic operates solely on the text of the query:

```python
FY_PATTERN = re.compile(
    r"(?:FY\s*|fiscal\s*year\s*)(\d{4})|(\d{4})\s*fiscal\s*year",
    re.IGNORECASE
)
YEAR_PATTERN = re.compile(
    r"(?:FY\s*|Q[1-4]\s*(?:of\s*)?FY\s*|\b)(20\d{2})\b",
    re.IGNORECASE
)
QUARTER_PATTERN = re.compile(
    r"\bQ([1-4])\b(?:\s*(?:of\s*)?(?:FY\s*)?(\d{4}))?",
    re.IGNORECASE
)
RANGE_PATTERN = re.compile(
    r"(?:FY\s*)?(\d{4})\s*(?:-|to|and)\s*(?:FY\s*)?(\d{4})",
    re.IGNORECASE
)
```

### Multi-Year Expansion
For multi-year comparative queries (e.g., *"FY2018 - FY2020 3 year average"* or *"between FY2021 and FY2022"*):
- Both boundary years $Y_1$ and $Y_2$ and all intermediate years $Y_1 \le y \le Y_2$ are included in $Y_{\text{years}}$.
- Chunks matching any year in the range are considered in-period.

---

## 3. Strict Verification of Zero Evaluation Leakage

| Candidate Generation Parameter | Information Source | Evaluation Ground Truth Used? |
| :--- | :--- | :---: |
| Query Period Extraction | User Query String `q["question"]` | **NO** |
| Chunk Period Metadata | Chunk document identifier `c["doc_name"]` from chunk database | **NO** |
| Target Document Name | Ground truth `q["doc_name"]` | **NEVER USED** |
| Target Page Numbers | Ground truth `q["evidence"]` / `q["target_pages"]` | **NEVER USED** |
| Candidate Ranks & Scores | Unsupervised BM25 & BGE-M3 FAISS cosine similarity | **NO** |

The candidate generator receives only `(query_text, path_results, chunk_store, k_pool)`. No field from the ground truth annotation is ever accessible to the retrieval or fusion functions.

---

## 4. Isolation and Reproducibility Protocol

To comply with the requirement that active repository baseline files remain untouched:
1. **Self-Contained Runner:** A dedicated runner `scripts/run_experiment_F.py` executes both:
   - **Branch A (Reference Run):** Exact reproduction of Experiment C using identical parameters saved in `outputs/experiments/experiment_F_period_aware/reference/`.
   - **Branch B (Experiment F Run):** Exact reproduction with the isolated fiscal-period-aware candidate generation change saved in `outputs/experiments/experiment_F_period_aware/experiment/`.
2. **Zero Modification to Baseline Source Tree:** Active `src/retrieval/phase2_reranker.py` and `scripts/run_phase2_benchmark.py` are preserved byte-for-byte. SHA-256 checksums are recorded before and after execution.
