import csv
import json

def main():
    csv_path = "outputs/experiments/experiment_F_period_aware/question_level_comparison.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Check the 18 period confusion questions from Exp D
    with open("outputs/experiments/experiment_D_failure_analysis/question_failure_analysis.csv", "r", encoding="utf-8") as f:
        exp_d_rows = {r["question_id"]: r for r in csv.DictReader(f)}

    period_conf_qids = [qid for qid, r in exp_d_rows.items() if "Filing/fiscal period confusion" in r["failure_category"]]
    print(f"=== 18 PERIOD CONFUSION CASES FROM EXP D (auditing {len(period_conf_qids)}) ===")
    gained_pool = []
    gained_top10 = []
    for qid in period_conf_qids:
        r = next((row for row in rows if row["question_id"] == qid), None)
        if r:
            cp_ref, cp_exp = int(r["ref_candidate_pool_hit"]), int(r["exp_candidate_pool_hit"])
            t10_ref, t10_exp = int(r["ref_top10_hit"]), int(r["exp_top10_hit"])
            if cp_exp > cp_ref:
                gained_pool.append(qid)
            if t10_exp > t10_ref:
                gained_top10.append(qid)
            print(f"{qid} | Pool: {cp_ref}->{cp_exp} | Top10: {t10_ref}->{t10_exp} | Final Rank: {r['ref_final_rank']}->{r['exp_final_rank']} | Cand Rank: {r['ref_candidate_rank']}->{r['exp_candidate_rank']} | Doc: {r['gold_doc']}")

    print(f"\nGained in Candidate Pool: {len(gained_pool)} / {len(period_conf_qids)}")
    print(f"Gained in Top 10:         {len(gained_top10)} / {len(period_conf_qids)}")

    # Check multi-period questions
    multi_period_qids = [r["question_id"] for r in rows if any(s in r["question_text"].lower() for s in ["between", "from", "average", "cagr", "growth rate", "change in"]) and ("fy" in r["question_text"].lower() or "20" in r["question_text"])]
    print(f"\n=== MULTI-PERIOD / COMPARISON QUESTIONS (total: {len(multi_period_qids)}) ===")
    multi_gained_t10 = 0
    multi_lost_t10 = 0
    for qid in multi_period_qids:
        r = next((row for row in rows if row["question_id"] == qid), None)
        if r:
            t10_ref, t10_exp = int(r["ref_top10_hit"]), int(r["exp_top10_hit"])
            if t10_exp > t10_ref: multi_gained_t10 += 1
            elif t10_exp < t10_ref: multi_lost_t10 += 1

    print(f"Multi-period questions Top-10: Gained={multi_gained_t10}, Lost={multi_lost_t10}, Net={multi_gained_t10 - multi_lost_t10:+d}")

    # Inspect all questions where Top-10 changed
    print("\n=== ALL QUESTIONS GAINED IN TOP 10 ===")
    for r in rows:
        t10_ref, t10_exp = int(r["ref_top10_hit"]), int(r["exp_top10_hit"])
        if t10_exp > t10_ref:
            print(f"  + {r['question_id']} | Rank: {r['ref_final_rank']} -> {r['exp_final_rank']} | Cand: {r['ref_candidate_rank']} -> {r['exp_candidate_rank']} | Doc: {r['gold_doc']} | Q: {r['question_text'][:70]}")

    print("\n=== ALL QUESTIONS LOST IN TOP 10 ===")
    for r in rows:
        t10_ref, t10_exp = int(r["ref_top10_hit"]), int(r["exp_top10_hit"])
        if t10_exp < t10_ref:
            print(f"  - {r['question_id']} | Rank: {r['ref_final_rank']} -> {r['exp_final_rank']} | Cand: {r['ref_candidate_rank']} -> {r['exp_candidate_rank']} | Doc: {r['gold_doc']} | Q: {r['question_text'][:70]}")

    # Inspect why MRR dropped slightly
    print("\n=== QUESTIONS WHERE MRR CHANGED ===")
    mrr_drops = [r for r in rows if float(r["mrr_delta"]) < 0]
    mrr_gains = [r for r in rows if float(r["mrr_delta"]) > 0]
    print(f"Total MRR Drops: {len(mrr_drops)} (sum delta: {sum(float(r['mrr_delta']) for r in mrr_drops):.4f})")
    print(f"Total MRR Gains: {len(mrr_gains)} (sum delta: {sum(float(r['mrr_delta']) for r in mrr_gains):.4f})")
    for r in mrr_drops[:10]:
        print(f"  Drop: {r['question_id']} | Ref Rank {r['ref_final_rank']} -> Exp Rank {r['exp_final_rank']} (MRR {r['ref_mrr']} -> {r['exp_mrr']}) | Doc: {r['gold_doc']}")

if __name__ == "__main__":
    main()
