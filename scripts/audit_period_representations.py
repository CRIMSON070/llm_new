import json
import re
from collections import defaultdict, Counter

def main():
    questions_file = "data/financebench/questions.jsonl"
    chunks_file = "data/financebench/processed/phase2_enriched_chunks.jsonl"

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    with open(chunks_file, "r", encoding="utf-8") as f:
        first_chunk = json.loads(f.readline())
        print("Chunk fields:", list(first_chunk.keys()))

    doc_pattern = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")

    # Audit all 84 documents
    doc_meta = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            d = c.get("doc_name")
            if d and d not in doc_meta:
                m = doc_pattern.match(d)
                if m:
                    comp, yr, qtr, ftype = m.groups()
                    doc_meta[d] = {
                        "company": comp,
                        "year": int(yr),
                        "quarter": int(qtr) if qtr else None,
                        "ftype": ftype
                    }
                else:
                    doc_meta[d] = {"error": "unparsed"}

    print(f"Total documents audited: {len(doc_meta)}")

    # Audit Questions for Period Mentions
    # Patterns for fiscal year, year, quarter, date ranges
    fy_pattern = re.compile(r"\b(?:FY\s*(\d{4})|FY(\d{2})\b|(\d{4})\s*fiscal\s*year|fiscal\s*year\s*(\d{4}))", re.IGNORECASE)
    year_pattern = re.compile(r"\b(20\d{2})\b")
    quarter_pattern = re.compile(r"\bQ([1-4])\b(?:\s*(?:of\s*)?(?:FY\s*)?(\d{4}))?", re.IGNORECASE)
    range_pattern = re.compile(r"\b(?:FY\s*)?(\d{4})\s*(?:-|to|and)\s*(?:FY\s*)?(\d{4})\b", re.IGNORECASE)

    period_stats = {
        "single_fy": 0,
        "single_year_no_fy": 0,
        "quarter_specified": 0,
        "multi_year_range": 0,
        "no_period_detected": 0,
        "doc_year_matches_query_period": 0,
        "doc_year_mismatch": 0
    }

    audit_records = []

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("financebench_id", f"FB_{idx}")
        q_text = q["question"]
        gold_doc = q["doc_name"]
        d_info = doc_meta.get(gold_doc, {})

        # Detect periods in question
        # 1. Check for ranges: e.g. FY2018 - FY2020, between 2022 and 2021, FY2015 to FY2016
        ranges = range_pattern.findall(q_text)
        
        # 2. Check for quarters: Q1, Q2, Q3, Q4
        quarters = quarter_pattern.findall(q_text)

        # 3. Check for FY explicitly: FY2018, FY 2021
        fys = []
        for m in fy_pattern.finditer(q_text):
            # groups: FY\s*(\d{4}), FY(\d{2}), (\d{4})\s*fiscal, fiscal\s*year\s*(\d{4})
            g = [x for x in m.groups() if x]
            if g:
                val = g[0]
                if len(val) == 2:
                    val = "20" + val
                fys.append(int(val))

        # 4. All 4-digit years in query (2010 - 2029)
        all_years = [int(y) for y in year_pattern.findall(q_text)]

        # Classify query period intent
        q_type = "unknown"
        if ranges:
            q_type = "multi_year_range"
            period_stats["multi_year_range"] += 1
        elif quarters:
            q_type = "quarter"
            period_stats["quarter_specified"] += 1
        elif fys:
            q_type = "single_fy"
            period_stats["single_fy"] += 1
        elif all_years:
            q_type = "single_year_plain"
            period_stats["single_year_no_fy"] += 1
        else:
            q_type = "no_period"
            period_stats["no_period_detected"] += 1

        # Check alignment with gold document
        gold_year = d_info.get("year")
        gold_quarter = d_info.get("quarter")
        gold_ftype = d_info.get("ftype")

        # Does query year match gold year?
        match_status = "no_query_period"
        if all_years:
            if gold_year in all_years:
                match_status = "exact_year_match"
                period_stats["doc_year_matches_query_period"] += 1
            else:
                match_status = "year_mismatch"
                period_stats["doc_year_mismatch"] += 1

        audit_records.append({
            "id": q_id,
            "question": q_text,
            "gold_doc": gold_doc,
            "gold_year": gold_year,
            "gold_quarter": gold_quarter,
            "gold_ftype": gold_ftype,
            "detected_fys": fys,
            "detected_years": all_years,
            "detected_quarters": quarters,
            "detected_ranges": ranges,
            "q_type": q_type,
            "match_status": match_status
        })

    print("\n=== PERIOD AUDIT STATS (N=150) ===")
    for k, v in period_stats.items():
        print(f"  {k:30s}: {v}")

    # Check for mismatches between query year and gold doc year
    mismatches = [r for r in audit_records if r["match_status"] == "year_mismatch"]
    print(f"\nTotal questions where gold doc year is NOT in query detected years: {len(mismatches)}")
    for m in mismatches:
        print(f"  QID: {m['id']} | Gold: {m['gold_doc']} (Yr={m['gold_year']}) | Query Yrs: {m['detected_years']} | Text: {m['question'][:90]}")

    # Check quarter questions
    quarter_qs = [r for r in audit_records if r["q_type"] == "quarter"]
    print(f"\nTotal quarter questions: {len(quarter_qs)}")
    for q in quarter_qs:
        print(f"  QID: {q['id']} | Gold: {q['gold_doc']} (Qtr={q['gold_quarter']}) | Detected Qtrs: {q['detected_quarters']} | Text: {q['question'][:90]}")

    # Check multi-year range questions
    multi_qs = [r for r in audit_records if r["q_type"] == "multi_year_range"]
    print(f"\nTotal multi-year range questions: {len(multi_qs)}")
    for q in multi_qs:
        print(f"  QID: {q['id']} | Gold: {q['gold_doc']} (Yr={q['gold_year']}) | Ranges: {q['detected_ranges']} | Text: {q['question'][:90]}")

if __name__ == "__main__":
    main()
