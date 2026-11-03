# Configuration Verification: Active Source vs. Experiment C Recorded Configuration

**Audit Date:** November 2, 2026  
**Auditor:** Antigravity AI Pair Programmer  
**Investigation Mode:** Strict Read-Only Forensic Analysis  

---

## 1. Executive Summary: Restoration Status

> [!WARNING]
> **Experiment C is NOT currently restored in the active workspace source tree.**  
> The active workspace files (`src/retrieval/phase2_reranker.py` and `scripts/run_phase2_benchmark.py`) reflect the **clean baseline configuration** restored after Experiment C's run. Experiment C's active source code and configuration were preserved in `outputs/experiments/experiment_C_company_intent/source_backup/`.

In accordance with strict safety rules (Rule 1 & 5: Read-Only, no modification or overwriting of files), the active files were NOT modified or overwritten. All evaluations, per-question analyses, and metrics reported in Experiment D are extracted directly from the verified Experiment C run artifacts preserved in:
- Checkpoint: `outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl`
- Source Backup: `outputs/experiments/experiment_C_company_intent/source_backup/`
- Report & Scorecard: `outputs/experiments/experiment_C_report.md`

---

## 2. Configuration Comparison Matrix

| Component / Parameter | Active Workspace Source | Experiment C Source Backup | Experiment C Report / Checkpoint | Discrepancy Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Workspace Code Path** | `src/retrieval/phase2_reranker.py` | `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py` | `outputs/experiments/experiment_C_report.md` | Active code is baseline; Exp C code is preserved in backup. |
| **Company Detection Coverage** | **6 hardcoded companies** (`3M`, `Apple`, `Amazon`, `Microsoft`, `Alphabet`, `Google`) | **32 canonical benchmark companies** with aliases (`FINANCEBENCH_COMPANIES`) | 32 canonical benchmark companies | **Discrepancy:** Active file lacks expanded detection for 26 benchmark companies (e.g. Adobe, Boeing, Pfizer, Walmart, etc.). |
| **Company Match Boost** | `+1.5` | `+1.5` | `+1.5` | **Match:** Boost magnitude identical. |
| **Wrong Company Penalty** | `-10.0` | `-10.0` | `-10.0` | **Match:** Penalty magnitude identical. |
| **Statement Intent Boost** | `+4.5` | `+4.5` | `+4.5` | **Match:** Routing regex and boost identical. |
| **Reranker Context Window** | **512 characters** (`chunk["text"][:512]`) | **512 characters** (`chunk["text"][:512]`) | **512 characters** | **Match:** Confirmed 512 characters in both active code and Exp C. |
| **BM25 Retrieval Depth ($k_{\text{sparse}}$)** | `30` | `50` | `50` | **Discrepancy:** Active default is 30; Exp C used 50. |
| **Dense Retrieval Depth ($k_{\text{dense}}$)** | `30` | `50` | `50` | **Discrepancy:** Active default is 30; Exp C used 50. |
| **Candidate Pool Size ($K_{\text{pool}}$)** | `40` | `50` | `50` (Prompt states 60) | **Discrepancy:** Active default is 40. Exp C code and run artifacts used $K_{\text{pool}} = 50$. (See Note below regarding Prompt vs. Artifacts). |
| **RRF Parameter ($k$)** | `60` | `60` | `60` | **Match:** RRF smoothing constant is 60. |
| **Cross-Encoder Model** | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-base` | **Match:** Identical reranker model. |
| **Dense Embedding Model** | `BAAI/bge-m3` | `BAAI/bge-m3` | `BAAI/bge-m3` | **Match:** Identical embedding model. |

> [!NOTE]
> **Candidate Pool Size Verification ($K_{\text{pool}}$):**  
> The user request notes "Candidate pool size: 60" under Experiment C summary. However, direct forensic inspection of:
> 1. `outputs/experiments/experiment_C_company_intent/source_backup/run_phase2_benchmark.py` (Line 38: `DEFAULT_K_POOL = 50`)
> 2. `outputs/experiments/experiment_C_report.md` ("Candidate pool size: 50, BM25/dense depth: 50")
> 3. `phase2_retrieval_checkpoint.jsonl` (candidate pool contains chunks ranked up to index 50)  
> confirms that Experiment C was executed with $K_{\text{pool}} = 50$ (derived from Experiment AB's combination of $k_{\text{sparse}}=50, k_{\text{dense}}=50, K_{\text{pool}}=50$).

---

## 3. Detailed Source Code Audits

### 3.1 Reranker Context Truncation
In both `src/retrieval/phase2_reranker.py` (Line 137) and `outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py` (Line 183):
```python
pairs = [(query, chunk["text"][:512]) for chunk in chunks]
```
The reranker context window is confirmed to be **exactly 512 characters**.

### 3.2 Scoring Boosts and Penalties
In both active source and Experiment C backup:
```python
# Statement intent boost
if detected_statement and chunk_statement == detected_statement:
    score += self.statement_boost  # +4.5

# Company matching boost / penalty
if detected_company:
    if chunk_company == detected_company:
        score += self.company_boost    # +1.5
    elif chunk_company is not None:
        score -= self.company_penalty  # -10.0
```
All boost and penalty constants are verified unchanged.

### 3.3 Company Detection Mapping
- In `src/retrieval/phase2_reranker.py`:
  ```python
  companies = {
      "3m": "3M", "apple": "APPLE", "amazon": "AMAZON",
      "microsoft": "MICROSOFT", "alphabet": "ALPHABET", "google": "ALPHABET",
  }
  ```
- In Experiment C backup (`outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py`):
  Comprehensive `FINANCEBENCH_COMPANIES` mapping covering all 32 benchmark companies:
  `3M`, `ACTIVISIONBLIZZARD`, `ADOBE`, `AES`, `AMAZON`, `AMD`, `AMGEN`, `BESTBUY`, `BLOCK`, `BOEING`, `BOOKING`, `COCACOLA`, `CORNING`, `CVS`, `EBAY`, `GENERALMILLS`, `GENERALMOTORS`, `JOHNSONJOHNSON`, `JPMORGAN`, `LOCKHEEDMARTIN`, `MICROSOFT`, `NETFLIX`, `NIKE`, `PAYPAL`, `PEPSICO`, `PFIZER`, `PG`, `SALESFORCE`, `VERIZON`, `WALMART`, `ORACLE`, `INTEL`.

---

## 4. Conclusion

- The active repository is in a safe, restored baseline state.
- Experiment C's recorded configuration has been verified from its source backup and execution artifacts.
- The downstream analysis in Experiment D evaluates the exact behavior of Experiment C as executed.
