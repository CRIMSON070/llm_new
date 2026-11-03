"""
scripts/generate_70_failure_table.py
====================================
Generates comprehensive failure analysis table for the 70 candidate pool failures.
"""

import os
import json
import csv
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation")
CSV_PATH = os.path.join(OUT_DIR, "failure_70_questions_refined_taxonomy.csv")

def main():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Loaded {len(records)} failure records.")
    
    # Stratify by category
    by_cat = Counter(r["evidence_category"] for r in records)
    print("\nBreakdown by Category:")
    for cat, count in by_cat.most_common():
        print(f"  {cat}: {count} ({count/len(records)*100:.1f}%)")

    # Stratify by company
    by_comp = Counter(r["company"] for r in records)
    print("\nBreakdown by Company:")
    for comp, count in by_comp.most_common():
        print(f"  {comp}: {count} / 5 questions ({count/5*100:.0f}% failure rate)")

    # Stratify by root cause
    by_cause = Counter(r["root_cause_type"] for r in records)
    print("\nBreakdown by Root Cause:")
    for cause, count in by_cause.most_common():
        print(f"  {cause}: {count} ({count/len(records)*100:.1f}%)")

if __name__ == "__main__":
    main()
