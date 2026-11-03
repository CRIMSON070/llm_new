import csv
import json

def main():
    csv_path = "outputs/experiments/experiment_D_failure_analysis/question_failure_analysis.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        period_fails = [row for row in reader if "Filing/fiscal period confusion" in row["failure_category"]]

    print(f"Total period confusion failures in Cat C: {len(period_fails)}")
    for r in period_fails:
        qid = r["question_id"]
        gold_doc = r["gold_document"]
        q_text = r["question_text"]
        evidence = r["evidence_for_classification"]
        print(f"ID: {qid} | Gold: {gold_doc}")
        print(f"  Q: {q_text[:90]}")
        print(f"  Evidence: {evidence}")
        print("-" * 70)

if __name__ == "__main__":
    main()
