# Experiment L.5: Ground Truth Correction and Benchmark V2 Report

**Report Date:** November 4, 2026  
**Investigated Artifact:** Stage 3 Held-Out Benchmark Suite ($N = 100$ questions across 40 Form 10-K Filings)  
**Output Directory:** `outputs/experiments/experiment_l5_ground_truth_v2/`  
**Benchmark V2 Dataset:** `outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl`  
**Audit Database:** `outputs/experiments/experiment_l5_ground_truth_v2/ground_truth_v2_audit.json`  
**SHA-256 Manifest:** `outputs/experiments/experiment_l5_ground_truth_v2/sha256_manifest.json`

---

## 1. Executive Summary & Verification Metrics

Experiment L.5 established a rigorously verified, versioned ground-truth benchmark dataset (**Stage 3 Benchmark V2**) while keeping the original Benchmark V1 completely frozen and untouched.

Every question, gold answer, and evidence citation was audited against the primary SEC Form 10-K filings using PyMuPDF physical layout reconciliation.

### Benchmark V2 Verification Breakdown ($N=100$):
- **CONFIDENTLY VERIFIED:** **65 questions (65.0%)**
  - **`ORIGINAL_VALID` (33 questions):** The original annotated physical page and passage directly and completely answer the question (e.g. single-page Financial Statements, Statements of Operations, Cash Flows).
  - **`ORIGINAL_BOUNDARY_SPLIT` (3 questions):** Tabular or narrative content spans a page break; adjacent continuation page added as direct evidence while preserving the table header.
  - **`ORIGINAL_DISPLACED` (29 questions):** Original annotation referenced introductory Item 1 boilerplate; true operative evidence in MD&A or Footnotes was located, verified, and mapped as direct evidence while retaining the original reference.
- **PARTIALLY VERIFIED / AMBIGUOUS:** **35 questions (35.0%)**
  - Text contains indirect narrative descriptions or composite disclosures where a single distinct operative sentence cannot be isolated with $>90\%$ certainty.
  - In accordance with experimental constraints, these 35 questions are **preserved with their original annotations labeled as `unverified_original`** rather than inventing speculative corrections.
- **UNRESOLVED / MISSING:** **0 questions (0.0%)**

---

## 2. Immutability & Preservation of Original Benchmark V1

The original benchmark file remained completely untouched throughout the audit:
- **Original V1 Benchmark Path:** `outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl`
- **Original V1 SHA-256 Hash:** `1b75376bfe803a85b600a66fb3f9609e7c9097bb96875b0a3c407fa7883f41e2` (**Verified 100% Unchanged**)
- **Zero In-Place Overwrites:** All new artifacts, metadata, and versioned JSONL records are saved exclusively under `outputs/experiments/experiment_l5_ground_truth_v2/`.

---

## 3. Systematic Before & After Audit of Anchor Questions

The table below details the before and after evidence mappings for key benchmark anchors:

| Question ID & Focus | Original V1 Annotation | V2 Classification | Benchmark V2 Evidence Mapping | Forensic Validation Summary |
| :--- | :--- | :---: | :--- | :--- |
| **`stage3_heldout_0075`**<br>*Eli Lilly Regulatory Milestone* | Page 10 (`p10_c0`)<br>*"Products Certain Indications Basaglar..."* | `BOUNDARY_AND_PIPELINE_DUAL` | **Page 10:** Table Header<br>**Page 11 (`p11_c1`):** *Zepbound chronic weight management*<br>**Page 92 (`p92_c1`):** *Tirzepatide Obesity Approved in U.S. in 2023* | Physical PDF audit confirmed that Page 10 contains zero mentions of tirzepatide. Page 11 defines Zepbound, and Page 92 explicitly confirms 2023 FDA approval. |
| **`stage3_heldout_0074`**<br>*Eli Lilly Manufacturing Commitments* | Page 10 (`p10_c0`)<br>*"Item 1.Business Eli Lilly and Company..."* | `ORIGINAL_DISPLACED` | **Page 10:** Historical Reference<br>**Page 46 (`p46_c1`):** *Facilities in NC, WI, Ireland, Germany, Indiana*<br>**Page 75 (`p75_c0`):** *Incretin production exceeded by demand* | Page 10 is boilerplate founding history. Physical Page 46 specifies multi-site facility expansions; Page 75 details incretin capacity bottlenecks. |
| **`stage3_heldout_0072`**<br>*Mounjaro Revenue* | Page 109 (`p109_c0`)<br>*"Product Revenue Table..."* | `ORIGINAL_VALID` | **Page 109 (`p109_c0`):** *Mounjaro FY23 Net Revenue Table* | Directly valid. Page 109 provides the definitive product revenue breakdown. |
| **`stage3_heldout_0013`**<br>*NVIDIA ROA* | Page 78 (`p78_c0`)<br>*Consolidated Statements of Income & Balance Sheets* | `ORIGINAL_VALID` | **Page 78 (`p78_c0`):** *Statements of Income & Balance Sheet Data* | Directly valid. Page 78 contains the official multi-statement financials required to compute ROA. |
| **`stage3_heldout_0016`**<br>*Cisco Total Revenue* | Page 81 (`p81_c0`)<br>*Consolidated Statements of Operations* | `ORIGINAL_VALID` | **Page 81 (`p81_c0`):** *Statement of Operations ($53,803M)* | Directly valid. Physical Page 81 contains the Statement of Operations. |
| **`stage3_heldout_0099`**<br>*Caterpillar Dealer Inventory* | Page 8 (`p8_c0`)<br>*"Originally organized as Caterpillar Tractor Co. in 1925..."* | `ORIGINAL_DISPLACED` | **Page 8:** Historical Reference<br>**Page 72 (`p72_c0`):** *Machine dealer inventory liquidation impact on sales volume* | Page 8 is Item 1 founding history. Page 72 contains the MD&A narrative explaining that machine dealer inventory reductions unfavorably impacted volume. |
| **`stage3_heldout_0100`**<br>*Caterpillar Energy Segment Profit* | Page 8 (`p8_c0`)<br>*"With 2023 sales and revenues of $67.060 billion..."* | `ORIGINAL_DISPLACED` | **Page 8:** Historical Reference<br>**Page 93 (`p93_c0`):** *Energy & Transportation Sales*<br>**Page 94 (`p94_c0`):** *Reciprocating engines & profit growth* | Page 8 is Item 1 overview. Pages 93–94 contain the segment profit analysis ($4,936M, +49% growth). |
| **`stage3_heldout_0014`**<br>*NVIDIA Export Restrictions* | Page 5 (`p5_c0`)<br>*Item 1 Opening Header* | `ORIGINAL_DISPLACED` | **Page 5:** Historical Reference<br>**Page 21 (`p21_c0`):** *China export control regulations and licensing requirements* | Operative regulatory disclosure located on Page 21. |
| **`stage3_heldout_0025`**<br>*Broadcom Apple Concentration* | Page 4 (`p4_c0`)<br>*Table of Contents* | `ORIGINAL_DISPLACED` | **Page 4:** Historical Reference<br>**Page 25 (`p25_c0`):** *Customer concentration: Apple accounted for 20% of net revenue* | Operative customer concentration disclosure located on Page 25. |

---

## 4. Benchmark V2 Dataset Schema & Structure

In `stage3_heldout_questions_v2.jsonl`, all original metadata fields are preserved with 100% fidelity:
```json
{
  "question_id": "stage3_heldout_0075",
  "question": "What regulatory milestone did Eli Lilly achieve in late 2023 for tirzepatide in the chronic weight management indication?",
  "company": "Eli Lilly and Company",
  "ticker": "LLY",
  "cik": "0000059478",
  "gics_sector": "Health Care",
  "doc_name": "LLY_2023_10K",
  "form_type": "10-K",
  "fiscal_year": 2023,
  "question_type": "qualitative_narrative",
  "target_statement_or_section": "Item 1. Business / Note 3 Product Approvals",
  "gold_answer": "Lilly received US FDA approval in November 2023 for tirzepatide under the brand name Zepbound for chronic weight management in adults with obesity or overweight.",
  "benchmark_version": "2.0",
  "evidence_sources": [
    {
      "doc_name": "LLY_2023_10K",
      "evidence_page_num": 10,
      "target_section": "Item 1. Business / Products Table Header",
      "evidence_type": "contextual_table_header",
      "matching_chunk_ids": ["LLY_2023_10K_p10_c0", "LLY_2023_10K_p10_c1"],
      "relevant_passage": "Products Certain Indications Diabetes, Obesity and Other Cardiometabolic products"
    },
    {
      "doc_name": "LLY_2023_10K",
      "evidence_page_num": 11,
      "target_section": "Item 1. Business / Product Approval Definition",
      "evidence_type": "direct_evidence",
      "matching_chunk_ids": ["LLY_2023_10K_p11_c0", "LLY_2023_10K_p11_c1"],
      "relevant_passage": "Zepbound: For the treatment of adults with obesity or overweight with weight-related comorbidities..."
    },
    {
      "doc_name": "LLY_2023_10K",
      "evidence_page_num": 92,
      "target_section": "Item 7. MD&A - Executive Overview / Late-Stage Pipeline Table",
      "evidence_type": "direct_evidence",
      "matching_chunk_ids": ["LLY_2023_10K_p92_c0", "LLY_2023_10K_p92_c1"],
      "relevant_passage": "Tirzepatide (Mounjaro, Zepbound): Obesity — Approved in the U.S. and the EU in 2023."
    }
  ],
  "verification_metadata": {
    "benchmark_v2_status": "VERIFIED",
    "original_annotation_classification": "ORIGINAL_BOUNDARY_AND_PIPELINE_DUAL",
    "verified_direct_pages": [11, 92],
    "contextual_pages": [10],
    "verification_method": "PyMuPDF physical PDF page layout and tabular reconciliation against primary SEC Form 10-K filings"
  }
}
```

---

## 5. SHA-256 Checksum Manifest

All files are cryptographically registered in `sha256_manifest.json`:

| File | Relative Workspace Path | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :---: |
| **Original Benchmark V1** | `outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl` | `1b75376bfe803a85b600a66fb3f9609e7c9097bb96875b0a3c407fa7883f41e2` | **FROZEN_UNCHANGED** |
| **Benchmark V2 Questions** | `outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl` | `d26aac2fe7a7e269f8260dd2ceeae2d147eabba97018dd1e2453a2b0e1598530` | **CREATED_VERSION_2** |
| **V2 Audit Database** | `outputs/experiments/experiment_l5_ground_truth_v2/ground_truth_v2_audit.json` | `0b82af730550fe47eb37fb82b59877891766305abc007eff5c6613c91a1d3e66` | **VALIDATED** |

---

## 6. Readiness Assessment for Controlled Re-Evaluation

**Is Benchmark V2 Ready for Evaluation?**
**Yes.** 
1. **Direct Evidence Cleared:** 65 of 100 questions possess verified, unambiguous direct evidence passages confirmed against the physical PDF.
2. **Backward Compatibility:** All 100 questions retain their original question text, company metadata, and gold answers. 
3. **Dual Metric Evaluation:** Re-evaluating the system under Benchmark V2 allows reporting two distinct recall standards:
   - **Strict Direct Evidence Recall:** Matching chunks with `evidence_type == "direct_evidence"`.
   - **Comprehensive Evidence Recall:** Matching chunks with either `direct_evidence` or `contextual_table_header`.
4. **Zero Disruption to Existing Indexes:** No FAISS or BM25 index needs to be rebuilt, ensuring that future comparisons against Arm A, Arm B, and Arm C will be scientifically controlled and strictly comparable.
