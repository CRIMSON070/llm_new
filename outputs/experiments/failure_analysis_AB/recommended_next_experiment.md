# FinSage Phase 2: Recommended Next Experiment Proposal

**Document File:** `outputs/experiments/failure_analysis_AB/recommended_next_experiment.md`  
**Execution Timestamp:** November 02, 2026  
**Status:** PROPOSED ONLY (Read-Only Analysis; Pipeline Unchanged)  
**Proposed Identifier:** Experiment C — Full-Universe Company Intent Detection & Boosting  
**Target Codebase:** `C:\Users\user\Downloads\finsage-main`

---

## 1. Context & Forensic Motivation

In the Experiment AB failure diagnosis ([`outputs/experiments/failure_analysis_AB/failure_analysis_report.md`](file:///c:/Users/user/Downloads/finsage-main/outputs/experiments/failure_analysis_AB/failure_analysis_report.md)), the single most widespread and easily remediable failure mode was identified:

> **40.9% of all retrieval failures (27 out of 66 questions)** suffered from **cross-company chunk contamination**, where chunks from completely unrelated corporations (e.g., Best Buy, Pfizer, Walmart) outranked gold chunks from the target corporation (e.g., Boeing, American Express, AMD).

### The Root Cause:
In [`src/retrieval/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/phase2_reranker.py#L153-L168), `detect_query_company()` only matches **6 hardcoded companies**:
- `3M`
- `ULTABEAUTY`
- `ADOBE`
- `AMAZON`
- `MICROSOFT`
- `JPMORGAN`

For the remaining **26 public companies in FinanceBench (81.3% of the dataset)**, `detect_query_company()` returns `None`. Consequently:
1. The designed $+1.5$ `company_boost_weight` is **never applied** to relevant chunks for 26 companies.
2. In contrast, generic financial prose from any corporation that scores highly under BGE semantic matching (e.g. Best Buy discussing operating margins) enters the Top 10 with no penalty or disadvantage.
3. In **19 questions (Category B)**, the gold chunk reached the candidate pool ($K_{\text{pool}}=50$), but was pushed down below Rank 10 by competitor chunks from other corporations.

---

## 2. Core Hypothesis

> **Hypothesis:** Expanding `detect_query_company()` from 6 hardcoded entities to cover all 32 public companies in FinanceBench will:
> 1. Eliminate cross-company chunk contamination in the reranker, reducing the contamination rate from **40.9% to < 5.0%**.
> 2. Apply the $+1.5$ company boost to true company chunks across all 32 companies, promoting a significant fraction of the **19 Category B pool-hit demotions** into the final Top 10.
> 3. Elevate benchmark Mean Reciprocal Rank (MRR) beyond the previously missed **0.325** threshold to **$\ge 0.340$**, and push Recall@10 from **56.00% to $\ge 60.0\%$**.

---

## 3. The Single Variable to Change

**Only one variable will be modified:** The company detection dictionary in [`src/retrieval/phase2_reranker.py`](file:///c:/Users/user/Downloads/finsage-main/src/retrieval/phase2_reranker.py#L153-L168).

### Exact Proposed Code Change
Replace the hardcoded 6-company if-statement in `Phase2Reranker.detect_query_company()` with a comprehensive mapping matching all 32 FinanceBench companies:

```python
# In src/retrieval/phase2_reranker.py

FINANCEBENCH_COMPANIES = {
    "3M": [r"\b3m\b"],
    "ADOBE": [r"\badobe\b"],
    "AES": [r"\baes(\s+corp(oration)?)?\b"],
    "AMAZON": [r"\bamazon\b"],
    "AMCOR": [r"\bamcor\b"],
    "AMD": [r"\bamd\b", r"\badvanced\s+micro\s+devices\b"],
    "AMERICANEXPRESS": [r"\bamerican\s+express\b", r"\bamex\b"],
    "AMERICANWATERWORKS": [r"\bamerican\s+water(\s+works)?\b"],
    "BESTBUY": [r"\bbest\s+buy\b"],
    "BOEING": [r"\bboeing\b"],
    "CATERPILLAR": [r"\bcaterpillar\b", r"\bcat\b"],
    "COCACOLA": [r"\bcoca[\s-]*cola\b", r"\bcoke\b"],
    "CORNING": [r"\bcorning\b"],
    "CVSHEALTH": [r"\bcvs(\s+health)?\b"],
    "GENERALMILLS": [r"\bgeneral\s+mills\b"],
    "GENERALMOTORS": [r"\bgeneral\s+motors\b", r"\bgm\b"],
    "HOMEDEPOT": [r"\bhome\s+depot\b"],
    "JOHNSON_JOHNSON": [r"\bjohnson\s+(&|and)\s+johnson\b", r"\bjnj\b"],
    "JPMORGAN": [r"\bjpmorgan(\s+chase)?\b", r"\bjp\s*morgan\b"],
    "KRAFTHEINZ": [r"\bkraft\s+heinz\b"],
    "LOCKHEEDMARTIN": [r"\blockheed(\s+martin)?\b"],
    "MGMRESORTS": [r"\bmgm(\s+resorts)?\b"],
    "MICROSOFT": [r"\bmicrosoft\b", r"\bmsft\b"],
    "NETFLIX": [r"\bnetflix\b"],
    "NIKE": [r"\bnike\b"],
    "PEPSICO": [r"\bpepsico\b", r"\bpepsi\b"],
    "PFIZER": [r"\bpfizer\b"],
    "SALESFORCE": [r"\bsalesforce\b"],
    "ULTABEAUTY": [r"\bulta(\s*beauty)?\b"],
    "VERIZON": [r"\bverizon\b"],
    "WALMART": [r"\bwalmart\b", r"\bwal-mart\b"],
    "WASTEMANAGEMENT": [r"\bwaste\s+management\b", r"\bwm\b"],
}

def detect_query_company(self, query: str) -> Optional[str]:
    q_lower = query.lower()
    for canonical_name, patterns in self.FINANCEBENCH_COMPANIES.items():
        for p in patterns:
            if re.search(p, q_lower):
                return canonical_name
    return None
```

### Parameters Kept 100% Fixed (No Change)
- Candidate pool size: $K_{\text{pool}} = 50$ (Experiment AB setting).
- Retrieval depths: BM25 $k = 50$, Dense FAISS $k = 50$ (Experiment AB setting).
- Query-intent routing regex: Preserved (Experiment B/AB setting).
- Statement boost weight: $+4.5$.
- Company boost weight: $+1.5$ (existing parameter, now active for all 32 companies instead of just 6).
- Negative disclosure penalty: $-3.0$.
- Period boost weight: $+1.5$.
- Table boost weight: $+0.5$.
- Cross-encoder model: `BAAI/bge-reranker-base`.
- Dense embedding model: `BAAI/bge-m3`.
- HyDE batch status: Bypassed.

---

## 4. Controlled Evaluation Plan & Metrics to Compare

When executed, Experiment C should run across the identical 150 FinanceBench questions and be compared directly against Baseline, Experiment A, Experiment B, and Experiment AB.

### Metrics to Track
1. **MRR (Mean Reciprocal Rank)**
2. **Recall@1, Recall@3, Recall@5, and Recall@10**
3. **Cross-Company Chunk Contamination Rate** in Top-3 retrieved chunks (Target: decrease from 40.9% to $< 5.0\%$)
4. **Category B Recovery Rate:** Count of the 19 Category B questions whose rank improves from $> 10$ into the Top 10.
5. **Sector MRR:** Particularly for Industrials (Boeing), Information Technology (AMD, Corning), and Financials (American Express, JPMorgan).

### Comparison Benchmark Matrix
| Configuration | $K_{\text{pool}}$ | Intent Routing | Company Boost Scope | Baseline MRR | Expected Exp C MRR |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 30 | Original | 6 Companies | 0.2926 | — |
| **Experiment A** | 60 | Original | 6 Companies | 0.3110 | — |
| **Experiment B** | 30 | Regex Fix | 6 Companies | 0.3042 | — |
| **Experiment AB** | 50 | Regex Fix | 6 Companies | 0.3224 | — |
| **Experiment C (Proposed)** | 50 | Regex Fix | **All 32 Companies** | — | **$\ge 0.340$** |

---

## 5. Formal Success Criteria

Experiment C will be deemed empirically successful if and only if:
1. **MRR Milestone:** Achieves $\text{MRR} \ge 0.340$ (an improvement of $\ge +0.0176$ over Experiment AB).
2. **Top-10 Recall Milestone:** Achieves $\text{Recall@10} \ge 60.0\%$ ($\ge 90/150$ questions retrieved in Top 10, an improvement of $\ge +4.00$ pp).
3. **Cross-Company Contamination:** Drops below $5.0\%$ of retrieved questions (from 40.9% in Experiment AB).
4. **Preservation of Past Gains:** No regression on the 3M, Ulta, or American Express registered-debt queries successfully solved in Experiment AB.

---

## 6. Execution Safeguards

- **Isolation:** Must be saved to a dedicated output directory: `outputs/experiments/experiment_C_company_intent/`.
- **Preflight Unit Tests:** A new unit test suite `tests/test_company_detection.py` must be written to verify that all 32 company regexes fire accurately on sample benchmark queries with zero false-positive collisions.
- **Production Baseline Preservation:** Restored baseline files must remain untouched until the user explicitly requests execution of Experiment C.
