"""
scripts/refine_failure_taxonomy.py
==================================
Refines the taxonomy classification of the 70 candidate pool failures
strictly into the 6 categories required by the prompt:
1. Primary financial statements
2. MD&A tables and narrative
3. Segment or product disclosures
4. Financial statement footnotes
5. Debt, lease, inventory, or other detailed disclosures
6. Other or uncertain
"""

import os
import json
import csv
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation")
DIAG_CSV = os.path.join(OUT_DIR, "failure_70_questions_diagnostic.csv")
QUESTIONS_JSONL = os.path.join(OUT_DIR, "heldout_questions.jsonl")

def main():
    with open(DIAG_CSV, "r", encoding="utf-8") as f:
        failures = list(csv.DictReader(f))

    with open(QUESTIONS_JSONL, "r", encoding="utf-8") as f:
        questions = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    refined_records = []

    for r in failures:
        qid = r["question_id"]
        q_data = questions[qid]
        sec = q_data.get("target_statement_or_section", "")
        concept = q_data.get("target_concept", "")
        ev = q_data["evidence_sources"][0]
        passage = ev.get("relevant_passage", "")
        sec_lower = sec.lower()
        concept_lower = concept.lower()

        # Classify into exact 6 prompt categories
        if any(k in sec_lower for k in ["consolidated balance sheet", "consolidated statement of operations", "consolidated statement of income", "consolidated statement of cash flow", "statements of operations", "balance sheet", "statements of cash flow", "consolidated statements of earnings", "statements of consolidated income"]):
            if "note" in sec_lower or "footnote" in sec_lower:
                category = "Financial statement footnotes"
            else:
                category = "Primary financial statements"

        elif any(k in sec_lower for k in ["segment", "product revenue", "geographic", "sales by"]) or any(k in concept_lower for k in ["segment", "product revenue", "geographic"]):
            category = "Segment or product disclosures"

        elif any(k in sec_lower for k in ["debt", "credit facility", "borrowing", "lease", "inventor", "tax", "goodwill", "contingenc", "pension", "stockholders' equity note", "fair value"]):
            category = "Debt, lease, inventory, or other detailed disclosures"

        elif any(k in sec_lower for k in ["note", "footnote", "notes to consolidated"]):
            category = "Financial statement footnotes"

        elif any(k in sec_lower for k in ["item 7", "management's discussion", "md&a"]):
            category = "MD&A tables and narrative"

        elif any(k in sec_lower for k in ["item 1a", "risk factor", "item 1.", "item 1 -", "business"]):
            category = "Other or uncertain" # Item 1 / 1A business & risks

        else:
            category = "Other or uncertain"

        # Check retrieval-path breakdown
        bm25_rank = int(r["bm25_rank_in_100"])
        dense_rank = int(r["dense_rank_in_100"])

        # Determine detailed root cause
        if bm25_rank > 100 and dense_rank > 100:
            root_cause_type = "Missing Candidate Generation (neither BM25 nor Dense retrieved gold page in top 100)"
        elif (bm25_rank <= 50 or dense_rank <= 50):
            root_cause_type = "RRF Fusion / Cutoff Failure (gold chunk hit top-50 in single path, but pushed past rank 50 by temporal/cross-company scoring)"
        elif (bm25_rank <= 100 or dense_rank <= 100):
            root_cause_type = "Insufficient Path Depth (gold chunk present at rank 51-100; missed by depth k=50)"
        else:
            root_cause_type = "Other Failure"

        rec = {
            "question_id": qid,
            "company": r["company"],
            "sector": r["sector"],
            "question_type": r["question_type"],
            "doc_name": r["doc_name"],
            "gold_page": r["gold_page"],
            "target_statement_or_section": sec,
            "target_concept": concept,
            "evidence_category": category,
            "bm25_rank_in_100": bm25_rank,
            "dense_rank_in_100": dense_rank,
            "root_cause_type": root_cause_type,
            "question": r["question"]
        }
        refined_records.append(rec)

    print("Refined 6-Category Taxonomy across 70 Failures:")
    tax_counts = Counter(r["evidence_category"] for r in refined_records)
    for cat, cnt in tax_counts.most_common():
        print(f"  {cat:<55}: {cnt:>2} ({cnt/70*100:5.1f}%)")

    print("\nRoot Cause Breakdown:")
    cause_counts = Counter(r["root_cause_type"] for r in refined_records)
    for cause, cnt in cause_counts.most_common():
        print(f"  {cause:<80}: {cnt:>2} ({cnt/70*100:5.1f}%)")

    out_refined_csv = os.path.join(OUT_DIR, "failure_70_questions_refined_taxonomy.csv")
    with open(out_refined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(refined_records[0].keys()))
        writer.writeheader()
        writer.writerows(refined_records)
    print(f"\nSaved refined taxonomy to {out_refined_csv}")

if __name__ == "__main__":
    main()
