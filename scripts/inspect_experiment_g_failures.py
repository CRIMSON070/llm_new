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
    print(f"Total candidate pool failures in Experiment F: {len(fails)}")

    # Sort by question_id
    fails.sort(key=lambda x: x["question_id"])

    # Detailed inspection of each failure
    for i, r in enumerate(fails, 1):
        qid = r["question_id"]
        raw_q = raw_questions.get(qid, {})
        d_row = exp_d_rows.get(qid, {})

        q_text = raw_q.get("question", r["question_text"])
        gold_doc = raw_q.get("doc_name", r["gold_doc"])
        company = raw_q.get("company", r["company"])
        evidence = raw_q.get("evidence", [])
        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        d_cat = d_row.get("failure_category", "Unknown")

        print(f"[{i:2d}/38] {qid} | Company: {company} | Doc: {gold_doc} | Pages: {target_pages}")
        print(f"      Q: {q_text}")
        print(f"      Exp D Category: {d_cat}")
        print("-" * 80)

if __name__ == "__main__":
    main()
