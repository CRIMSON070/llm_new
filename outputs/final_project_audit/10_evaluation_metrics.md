# FinSage Evaluation Methodology & Metrics Audit

**Document:** `outputs/final_project_audit/10_evaluation_metrics.md`  
**Workspace Root:** `c:\Users\user\Downloads\finsage-main`  
**Governing Standard:** Comprehensive Mathematical & Algorithmic Audit of Evaluator Modules (Read-Only)  

---

## 1. Overview of Evaluation Frameworks in Codebase

The FinSage workspace implements two distinct evaluation suites:
1. **Retrieval & Reranking Evaluator (`evaluate_phase1_drr.py`, `run_phase2_benchmark.py`):** Measures ranking quality, evidence coverage, and retrieval depth against ground-truth chunk citations.
2. **Harness V2 Downstream Evaluator (`evaluate_harness_v2.py`, `arithmetic_evaluator.py`):** Programmatically decouples semantic accuracy, numeric/unit precision, arithmetic operations, and legacy JSON schema compliance.

---

## 2. Granular Metric Specifications

### 1. Mean Reciprocal Rank (MRR)
* **What it Measures:** The efficiency and speed with which the retriever places the **first authoritative evidence chunk** at the top of the candidate list.
* **Mathematical Definition:**
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i^*}$$
  where $\text{rank}_i^*$ is the 1-based index of the highest-ranked gold evidence chunk for query $i$. If no gold evidence chunk appears in the evaluated candidate pool, $\frac{1}{\text{rank}_i^*} = 0$.
* **Evaluation Level:** Evaluated per question, averaged across all $N$ benchmark queries.
* **Denominator:** $N$ (the total number of queries evaluated). Missing evidence is assigned reciprocal rank $0.0$.
* **Code Implementation:** [`scripts/evaluate_phase1_drr.py:58-69`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L58-L69) and [`scripts/run_phase2_benchmark.py:82-94`](file:///c:/Users/user/Downloads/finsage-main/scripts/run_phase2_benchmark.py#L82-L94):
  ```python
  def compute_mrr(ranks: List[Optional[int]]) -> float:
      reciprocal_ranks = [1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks]
      return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0
  ```
* **Alignment with Paper:** Fully compliant with standard IR definition and paper Section 4.3.2.

---

### 2. Retrieval Recall@K (Recall at Rank K)
* **What it Measures:** The proportion of queries for which at least one authoritative gold evidence chunk is captured within the top-$K$ retrieved slots.
* **Mathematical Definition:**
  $$\text{Recall@}K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \exists c \in \mathcal{R}_i^{(K)} \text{ s.t. } c \in \mathcal{G}_i \right)$$
  where $\mathcal{R}_i^{(K)}$ is the set of top-$K$ retrieved chunks, and $\mathcal{G}_i$ is the set of gold evidence chunks for query $i$.
* **Evaluated Cutoffs:** $K \in \{1, 3, 5, 10, 40\}$.
* **Evaluation Level:** Binary per query, averaged across $N$ queries.
* **Denominator:** $N$ (total queries).
* **Code Implementation:** [`scripts/evaluate_phase1_drr.py:71-85`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_phase1_drr.py#L71-L85):
  ```python
  def compute_recall_at_k(ranks: List[Optional[int]], k: int) -> float:
      hits = sum(1 for r in ranks if r is not None and r <= k)
      return round(hits / len(ranks), 4) if ranks else 0.0
  ```
* **Alignment with Paper:** Compliant with paper Section 4.3.1. Note that on FinanceBench each question has 1–3 gold evidence chunks; capturing any valid gold chunk constitutes a retrieval hit.

---

### 3. Normalized Recall (Paper Specific Metric)
* **What it Measures:** The proportion of retrieved ground-truth chunks bounded by the maximum capacity of the top-$K$ retrieval window.
* **Mathematical Definition (Paper Eq. 3, Section 4.3.2):**
  $$\text{Normalized Recall@}K = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{\min(|\mathcal{G}|, K)}$$
* **Why the Denominator is Capped:** If a complex query has 8 gold chunks ($|\mathcal{G}|=8$), but the re-ranker is constrained to return only $K=5$ chunks, standard recall is mathematically capped at $\frac{5}{8} = 62.5\%$, falsely penalizing an optimal re-ranker. Capping the denominator at $\min(8, 5) = 5$ restores the metric to an achievable 100%.
* **Evaluation Level:** Per question, averaged across all queries.
* **Code Implementation:** Formulated in original paper's `ensembleRetriever.py` and audited in [`FINSAGE_PROJECT_AUDIT.md:330-344`](file:///c:/Users/user/Downloads/finsage-main/FINSAGE_PROJECT_AUDIT.md#L330-L344).
* **Alignment with Paper:** Exact paper formulation (Section 4.3.2).

---

### 4. Binary Normalized Discounted Cumulative Gain (nDCG@K)
* **What it Measures:** Evaluates the ranking quality of retrieved candidate sets by applying a logarithmic discount factor based on chunk rank position.
* **Mathematical Definition:**
  $$\text{DCG@}K = \sum_{i=1}^K \frac{\mathbb{I}(s_i \in \mathcal{G})}{\log_2(i + 1)}, \quad \text{IDCG@}K = \sum_{i=1}^{\min(|\mathcal{G}|, K)} \frac{1}{\log_2(i + 1)}, \quad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
* **Evaluation Level:** Per question, averaged across $N$ queries.
* **Alignment with Paper:** Exact paper formulation (Section 4.3.2).

---

### 5. Semantic Answer Accuracy (Harness V2)
* **What it Measures:** Factual correctness of the generated answer compared to the reference ground truth, evaluating semantic intent, direction of change, and corporate entities while forgiving harmless syntactic formatting variations.
* **Calculation Rule:**
  1. For single-value queries: Verifies that the numerical quantity and entity match ground truth.
  2. For delta queries: Verifies that the direction of change (increase/decrease) and absolute delta match.
  3. For negative queries: Verifies that the answer properly affirms absence of corporate events or debt without false hallucination.
* **Evaluation Level:** Evaluated per generated answer.
* **Denominator:** $N=6$ (on diagnostic suite).
* **Code Implementation:** [`scripts/evaluate_harness_v2.py:120-210`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_harness_v2.py#L120-L210).
* **Alignment with Paper:** Local deterministic surrogate for LlamaIndex `CorrectnessEvaluator` (GPT-4o), ensuring 100% reproducible offline verification.

---

### 6. Numeric & Unit Accuracy (Harness V2)
* **What it Measures:** Precision of numerical extraction, unit scale normalization (e.g. converting USD millions to USD billions), and algebraic sign (+/-).
* **Calculation Rule:**
  * Extracts numerical tokens from generated text via regex: `r'[-+]?\d*\.?\d+'`.
  * Multiplies value by extracted unit multiplier:
    * `"billion"` or `"B"` $\implies 10^9$
    * `"million"` or `"M"` $\implies 10^6$
    * `"thousand"` or `"K"` $\implies 10^3$
  * Compares normalized value against ground truth within a 0.5% tolerance band:
    $$\left| \frac{V_{\text{norm}} - V_{\text{gold}}}{V_{\text{gold}}} \right| \le 0.005$$
* **Evaluation Level:** Per generated answer.
* **Code Implementation:** [`scripts/evaluate_harness_v2.py:45-95`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_harness_v2.py#L45-L95) (`unit_aware_grounding()`).
* **Alignment with Paper:** Eliminates false negatives from unit conversions (e.g. recognizing $8,738M = 8.738B$ on Q2).

---

### 7. Strict Schema Compliance (Legacy Diagnostic Metric)
* **What it Measures:** Whether the model's output strictly adheres to the rigid Phase 9C JSON structure:
  ```json
  {
    "operation": "extract_single_value | absolute_difference | percentage_change | none",
    "old_value": float or null,
    "new_value": float or null,
    "unit": string
  }
  ```
* **Calculation Rule:**
  * If output is not valid JSON $\implies$ **Fail**.
  * If `operation` is invalid $\implies$ **Fail**.
  * If `values` is used instead of separate `old_value` and `new_value` keys $\implies$ **Fail** (even if the precomputed arithmetic is 100% correct).
* **Code Implementation:** [`scripts/evaluate_harness_v2.py:73-115`](file:///c:/Users/user/Downloads/finsage-main/scripts/evaluate_harness_v2.py#L73-L115) (`evaluate_original_strict_compliance()`).
* **Critical Finding:** This metric is a **local artifact of Phase 9C development**, NOT an official paper metric. It generated artificial failures for valid natural language responses (e.g. Q6).

---

### 8. Deterministic Arithmetic Engine (Stage 2)
* **What it Measures:** Deterministic execution of mathematical operations on extracted operands within a safe Python execution sandbox.
* **Operations Handled:**
  * `extract_single_value`: Returns $V_0$.
  * `absolute_difference`: Computes $|V_0 - V_1|$ or $V_1 - V_0$.
  * `percentage_change`: Computes $\frac{V_1 - V_0}{V_0} \times 100\%$.
* **Code Implementation:** [`src/utils/arithmetic_evaluator.py:20-75`](file:///c:/Users/user/Downloads/finsage-main/src/utils/arithmetic_evaluator.py#L20-L75):
  ```python
  def evaluate_arithmetic(operation: str, operands: List[float], unit: str) -> Dict[str, Any]:
      if operation == "absolute_difference" and len(operands) >= 2:
          val = abs(operands[0] - operands[1])
          return {"success": True, "operation": operation, "value": val, "formatted": f"{val:,.2f}", "unit": unit}
  ```
* **Alignment with Paper:** Resolves the LLM arithmetic failure mode highlighted in paper Section 5.3.

---

## 3. Summary Scorecard of Implemented Metrics

| Metric Identifier | Implemented In | Level | Denominator | Paper vs. Local Classification |
|---|---|:---:|:---:|---|
| **MRR** | `evaluate_phase1_drr.py`, `run_phase2_benchmark.py` | Query | $N$ | **Paper Official Metric** |
| **Recall@K ($K=1,3,5,10,40$)**| `evaluate_phase1_drr.py`, `run_phase2_benchmark.py` | Query | $N$ | **Paper Official Metric** |
| **Normalized Recall@K** | `ensembleRetriever.py` | Query | $\min(\|\mathcal{G}\|, K)$ | **Paper Official Metric** |
| **Binary nDCG@K** | `FINSAGE_PROJECT_AUDIT.md` (audit) | Query | Ideal DCG | **Paper Official Metric** |
| **Semantic Accuracy** | `evaluate_harness_v2.py` | Answer | $N=6$ | **Local Offline Evaluation Surrogate** |
| **Numeric & Unit Accuracy** | `evaluate_harness_v2.py` | Answer | $N=6$ | **Local Offline Evaluation Metric** |
| **Strict Schema Compliance** | `evaluate_harness_v2.py` | Answer | $N=6$ | **Local Legacy Diagnostic Metric** |
| **Deterministic Math Eval** | `arithmetic_evaluator.py` | Operation | $N$ | **Stage 2 Production Engine** |
