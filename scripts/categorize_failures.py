import csv
import json
import re

def main():
    comparison_csv = "outputs/experiments/experiment_F_period_aware/question_level_comparison.csv"
    questions_file = "data/financebench/questions.jsonl"
    exp_d_csv = "outputs/experiments/experiment_D_failure_analysis/question_failure_analysis.csv"

    with open(comparison_csv, "r", encoding="utf-8") as f:
        comp_rows = {r["question_id"]: r for r in csv.DictReader(f)}

    with open(exp_d_csv, "r", encoding="utf-8") as f:
        exp_d_rows = {r["question_id"]: r for r in csv.DictReader(f)}

    with open(questions_file, "r", encoding="utf-8") as f:
        raw_questions = {q.get("financebench_id", f"FB_{i}"): q for i, q in enumerate([json.loads(l) for l in f], 1)}

    fails = [r for r in comp_rows.values() if int(r["exp_candidate_pool_hit"]) == 0]
    fails.sort(key=lambda x: x["question_id"])
    print(f"Total failures: {len(fails)}")

    # Failure category taxonomy
    # Categories:
    # 1. Company or filing mismatch
    # 2. Fiscal-period mismatch
    # 3. Vocabulary or query-wording mismatch
    # 4. Multi-page or multi-chunk evidence requirement
    # 5. RRF or candidate-pool truncation
    # 6. BM25 retrieval miss (path-level trace unknown in saved artifacts)
    # 7. Dense retrieval miss (path-level trace unknown in saved artifacts)
    # 8. Metadata retrieval miss (path-level trace unknown in saved artifacts)
    # 9. Other / Missing corpus filing

    classified_records = []

    category_counts = {
        "Company or filing mismatch": 0,
        "Fiscal-period mismatch": 0,
        "Vocabulary or query-wording mismatch": 0,
        "Multi-page or multi-chunk evidence requirement": 0,
        "RRF or candidate-pool truncation": 0,
        "BM25 retrieval miss (unverified in saved traces)": 0,
        "Dense retrieval miss (unverified in saved traces)": 0,
        "Metadata retrieval miss (unverified in saved traces)": 0,
        "Other (Corpus gap / Unindexed filing period)": 0,
        "Unknown due to insufficient trace data": 0
    }

    ratio_keywords = ["turnover", "ratio", "cycle", "ccc", "dpo", "cagr", "average capex", "working capital ratio", "roa"]

    for r in fails:
        qid = r["question_id"]
        raw_q = raw_questions.get(qid, {})
        d_row = exp_d_rows.get(qid, {})

        q_text = raw_q.get("question", r["question_text"])
        gold_doc = raw_q.get("doc_name", r["gold_doc"])
        company = raw_q.get("company", r["company"])
        evidence = raw_q.get("evidence", [])
        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        assigned_cats = []
        evidence_notes = []

        # 1. Multi-page / multi-chunk requirement
        if len(target_pages) > 1 or any(k in q_text.lower() for k in ratio_keywords) and len(target_pages) > 1:
            assigned_cats.append("Multi-page or multi-chunk evidence requirement")
            evidence_notes.append(f"Requires cross-statement or multi-page evidence across pages {target_pages}")
            category_counts["Multi-page or multi-chunk evidence requirement"] += 1

        # 2. Fiscal-period mismatch / filing mismatch
        # Check if question asked about a filing/period where corpus had gaps or ambiguity
        if qid == "financebench_id_00702":
            assigned_cats.append("Other (Corpus gap / Unindexed filing period)")
            evidence_notes.append("Query asks for 2019 for Pfizer, but corpus only contains PFIZER_2021_10K and Pfizer_2023Q2_10Q (no 2019 filing in corpus)")
            category_counts["Other (Corpus gap / Unindexed filing period)"] += 1
        elif "EARNINGS" in gold_doc and any(term in q_text.lower() for term in ["fy", "202"]):
            assigned_cats.append("Company or filing mismatch")
            evidence_notes.append(f"Target document is an Earnings transcript ({gold_doc}) competing against annual 10-K filings of the same company")
            category_counts["Company or filing mismatch"] += 1
        elif any(term in d_row.get("evidence_for_classification", "") for term in ["competing filings", "prior-year", "wrong filing"]):
            assigned_cats.append("Company or filing mismatch")
            evidence_notes.append("Multiple filings of target company compete in candidate space")
            category_counts["Company or filing mismatch"] += 1

        # 3. Vocabulary or query-wording mismatch
        # Qualitative questions, narrative phrasing, lack of exact accounting keywords
        narrative_signals = [
            "cyclicality", "dragged down", "bankrupted", "spinning off", "nature & purpose",
            "primary customers", "high growth company", "operating margin change", "improving",
            "drop in cash", "restructuring liability", "geographies that", "materially important"
        ]
        if any(sig in q_text.lower() for sig in narrative_signals) or len(target_pages) == 1:
            assigned_cats.append("Vocabulary or query-wording mismatch")
            evidence_notes.append("Narrative or qualitative phrasing lacks direct lexical keyword overlap with structured financial tables")
            category_counts["Vocabulary or query-wording mismatch"] += 1

        # 4. Path-level trace status:
        # Checkpoint did NOT log individual path ranks
        assigned_cats.append("Unknown due to insufficient trace data (path-level ranks not persisted)")
        category_counts["Unknown due to insufficient trace data"] += 1

        classified_records.append({
            "qid": qid,
            "company": company,
            "gold_doc": gold_doc,
            "target_pages": target_pages,
            "q_text": q_text,
            "categories": "; ".join(assigned_cats),
            "evidence": " | ".join(evidence_notes)
        })

    print("\n=== FAILURE CATEGORY SUMMARY COUNTS (Categories Overlap) ===")
    for cat, count in category_counts.items():
        print(f"  {cat:60s}: {count:2d} ({count/38*100:.1f}%)")

if __name__ == "__main__":
    main()
