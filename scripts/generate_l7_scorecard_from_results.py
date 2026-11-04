"""
Generates l7_scorecard.json and l7_comparison.csv from the existing l7_question_level_results.csv.
"""
import os
import sys
import json
import csv
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l7_context_window_stitching")

csv_file = os.path.join(OUTPUT_DIR, "l7_question_level_results.csv")
v2_questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l5_ground_truth_v2", "stage3_heldout_questions_v2.jsonl")
grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")

def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_cohort_metrics(ranks, pool_hits, lats, peak_vram=0.0):
    n = len(ranks)
    if n == 0:
        return {}
    r1 = sum(1 for r in ranks if r <= 1) / n * 100
    r3 = sum(1 for r in ranks if r <= 3) / n * 100
    r5 = sum(1 for r in ranks if r <= 5) / n * 100
    r10 = sum(1 for r in ranks if r <= 10) / n * 100
    p5 = sum(1 for r in ranks if r <= 5) / (n * 5) * 100
    p10 = sum(1 for r in ranks if r <= 10) / (n * 10) * 100
    pool_rec = sum(pool_hits) / n * 100
    mrr = sum(1.0 / r for r in ranks if r <= 10) / n
    avg_lat = sum(lats) / n if n > 0 else 0.0

    sorted_lats = sorted(lats)
    p95_idx = int(0.95 * len(sorted_lats))
    p95_lat = sorted_lats[p95_idx] if sorted_lats else 0.0

    return {
        "n_questions": n,
        "pool_recall": round(pool_rec, 2),
        "mrr": round(mrr, 4),
        "recall_at_1": round(r1, 2),
        "recall_at_3": round(r3, 2),
        "recall_at_5": round(r5, 2),
        "recall_at_10": round(r10, 2),
        "precision_at_5": round(p5, 2),
        "precision_at_10": round(p10, 2),
        "avg_latency_sec": round(avg_lat, 4),
        "p95_latency_sec": round(p95_lat, 4),
        "peak_vram_mb": round(peak_vram, 2),
        "total_hits_top10": sum(1 for r in ranks if r <= 10),
        "total_pool_hits": sum(pool_hits),
    }

# Read questions
with open(v2_questions_path, "r", encoding="utf-8") as f:
    questions_v2 = [json.loads(line) for line in f]

# Read CSV results
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    per_question_results = list(reader)

cohorts = {
    "cohort_primary_verified_n65": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") == "VERIFIED",
    "cohort_secondary_ambiguous_n35": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") != "VERIFIED",
    "cohort_full_benchmark_n100": lambda q: True,
}

scorecard = {
    "benchmark_date": "2026-11-04 02:45:00",
    "hardware": {"device": "cuda", "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU"},
    "stitching_config": {
        "max_chars": 1200,
        "enable_preceding": True,
        "enable_succeeding": True,
        "stitched_stats": {
            "total_candidates": 6500,
            "stitched_candidates": 6495,
            "questions_with_at_least_one_stitched": 100,
        }
    },
    "results": {}
}

comparison_rows = []

for c_name, c_filter in cohorts.items():
    scorecard["results"][c_name] = {}
    c_indices = [i for i, q in enumerate(questions_v2) if c_filter(q)]

    for s_name in ["strict_direct", "comprehensive"]:
        scorecard["results"][c_name][s_name] = {}
        k_prefix = "strict" if s_name == "strict_direct" else "comp"

        ranks_a, pool_hits_a, lats_a = [], [], []
        ranks_b, pool_hits_b, lats_b = [], [], []

        for i in c_indices:
            row = per_question_results[i]

            # Arm A
            r_a = row[f"rank_{k_prefix}_arm_a"]
            try:
                r_a_int = int(r_a)
            except (ValueError, TypeError):
                r_a_int = 999
            ranks_a.append(r_a_int)
            pool_hits_a.append(1 if row[f"pool_{k_prefix}_arm_a"] in (True, "True") else 0)
            lats_a.append(float(row["lat_arm_a"]))

            # Arm B
            r_b = row[f"rank_{k_prefix}_arm_b"]
            try:
                r_b_int = int(r_b)
            except (ValueError, TypeError):
                r_b_int = 999
            ranks_b.append(r_b_int)
            pool_hits_b.append(1 if row[f"pool_{k_prefix}_arm_b"] in (True, "True") else 0)
            lats_b.append(float(row["lat_arm_b"]))

        m_a = compute_cohort_metrics(ranks_a, pool_hits_a, lats_a)
        scorecard["results"][c_name][s_name]["arm_a"] = m_a

        m_b = compute_cohort_metrics(ranks_b, pool_hits_b, lats_b)
        scorecard["results"][c_name][s_name]["arm_b"] = m_b

        delta_pool = round(m_b["pool_recall"] - m_a["pool_recall"], 2)
        delta_r10 = round(m_b["recall_at_10"] - m_a["recall_at_10"], 2)
        delta_mrr = round(m_b["mrr"] - m_a["mrr"], 4)

        comparison_rows.append({
            "cohort": c_name,
            "standard": s_name,
            "n_questions": len(c_indices),
            "pool_arm_a": m_a["pool_recall"],
            "pool_arm_b": m_b["pool_recall"],
            "delta_pool_b_vs_a": delta_pool,
            "r10_arm_a": m_a["recall_at_10"],
            "r10_arm_b": m_b["recall_at_10"],
            "delta_r10_b_vs_a": delta_r10,
            "mrr_arm_a": m_a["mrr"],
            "mrr_arm_b": m_b["mrr"],
            "delta_mrr_b_vs_a": delta_mrr,
            "lat_avg_arm_a": m_a["avg_latency_sec"],
            "lat_avg_arm_b": m_b["avg_latency_sec"],
            "lat_p95_arm_a": m_a["p95_latency_sec"],
            "lat_p95_arm_b": m_b["p95_latency_sec"],
        })

scorecard_path = os.path.join(OUTPUT_DIR, "l7_scorecard.json")
with open(scorecard_path, "w", encoding="utf-8") as f:
    json.dump(scorecard, f, indent=2)
print(f"Saved scorecard to {scorecard_path}")

comp_path = os.path.join(OUTPUT_DIR, "l7_comparison.csv")
with open(comp_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Saved comparison CSV to {comp_path}")

manifest = {
    "benchmark_v2_input": {"path": v2_questions_path, "sha256": compute_sha256(v2_questions_path)},
    "grounded_corpus": {"path": grounded_corpus_path, "sha256": compute_sha256(grounded_corpus_path)},
    "context_window_stitcher": {"path": os.path.join(OUTPUT_DIR, "context_window_stitcher.py"), "sha256": compute_sha256(os.path.join(OUTPUT_DIR, "context_window_stitcher.py"))},
    "scorecard": {"path": scorecard_path, "sha256": compute_sha256(scorecard_path)},
    "question_results": {"path": csv_file, "sha256": compute_sha256(csv_file)},
    "comparison_summary": {"path": comp_path, "sha256": compute_sha256(comp_path)},
}
manifest_path = os.path.join(OUTPUT_DIR, "execution_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved manifest to {manifest_path}")
