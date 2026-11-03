import json
import os
import csv
from collections import defaultdict

c_ckpt = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
e1_ckpt = 'outputs/experiments/experiment_E1_context_window/retrieval/phase2_retrieval_checkpoint.jsonl'
q_file = 'data/financebench/questions.jsonl'

# 1. Load questions map
questions_map = {}
with open(q_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            questions_map[d['financebench_id']] = d

# 2. Load checkpoints
def load_checkpoint(path):
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                qid = d.get('financebench_id', f"q_{d.get('index')}")
                records[qid] = d
    return records

data_c = load_checkpoint(c_ckpt)
data_e1 = load_checkpoint(e1_ckpt)

assert len(data_c) == 150, f"Exp C records count: {len(data_c)}"
assert len(data_e1) == 150, f"Exp E1 records count: {len(data_e1)}"

# 3. Verify Rule 7: Upstream candidate pool identity
pool_hit_diffs = []
pool_rank_diffs = []
for qid in data_c:
    c_rec = data_c[qid]
    e1_rec = data_e1[qid]
    if c_rec['candidate_pool_hit'] != e1_rec['candidate_pool_hit']:
        pool_hit_diffs.append((qid, c_rec['candidate_pool_hit'], e1_rec['candidate_pool_hit']))
    if c_rec['best_rank_in_pool'] != e1_rec['best_rank_in_pool']:
        pool_rank_diffs.append((qid, c_rec['best_rank_in_pool'], e1_rec['best_rank_in_pool']))

print(f"Rule 7 Verification: Candidate pool hit mismatches: {len(pool_hit_diffs)}")
print(f"Rule 7 Verification: Candidate pool rank mismatches: {len(pool_rank_diffs)}")

pool_hits_c = sum(1 for r in data_c.values() if r.get('candidate_pool_hit') == 1 or r.get('best_rank_in_pool', 999) <= 50)
pool_hits_e1 = sum(1 for r in data_e1.values() if r.get('candidate_pool_hit') == 1 or r.get('best_rank_in_pool', 999) <= 50)

# 4. Macro Metrics Calculation
def calc_metrics(data):
    n = len(data)
    mrr = sum(d['rr'] for d in data.values()) / n
    r1 = sum(1 for d in data.values() if d['best_rank'] == 1) / n * 100
    r3 = sum(1 for d in data.values() if d['best_rank'] <= 3) / n * 100
    r5 = sum(1 for d in data.values() if d['best_rank'] <= 5) / n * 100
    r10 = sum(1 for d in data.values() if d['best_rank'] <= 10) / n * 100
    pool_rec = sum(1 for d in data.values() if d.get('candidate_pool_hit', 0) == 1 or d.get('best_rank_in_pool', 999) <= 50) / n * 100
    return {
        'mrr': round(mrr, 4),
        'r1': round(r1, 2),
        'r3': round(r3, 2),
        'r5': round(r5, 2),
        'r10': round(r10, 2),
        'pool_rec': round(pool_rec, 2),
        'hits_r1': sum(1 for d in data.values() if d['best_rank'] == 1),
        'hits_r3': sum(1 for d in data.values() if d['best_rank'] <= 3),
        'hits_r5': sum(1 for d in data.values() if d['best_rank'] <= 5),
        'hits_r10': sum(1 for d in data.values() if d['best_rank'] <= 10),
        'pool_hits': sum(1 for d in data.values() if d.get('candidate_pool_hit', 0) == 1 or d.get('best_rank_in_pool', 999) <= 50)
    }

m_c = calc_metrics(data_c)
m_e1 = calc_metrics(data_e1)

print("\n--- Macro Metrics Comparison ---")
print(f"Exp C  (512 chars) : MRR={m_c['mrr']:.4f} | R@1={m_c['r1']:5.2f}% ({m_c['hits_r1']}/150) | R@3={m_c['r3']:5.2f}% ({m_c['hits_r3']}/150) | R@5={m_c['r5']:5.2f}% ({m_c['hits_r5']}/150) | R@10={m_c['r10']:5.2f}% ({m_c['hits_r10']}/150) | PoolRec={m_c['pool_rec']:5.2f}% ({m_c['pool_hits']}/150)")
print(f"Exp E1 (1024 chars): MRR={m_e1['mrr']:.4f} | R@1={m_e1['r1']:5.2f}% ({m_e1['hits_r1']}/150) | R@3={m_e1['r3']:5.2f}% ({m_e1['hits_r3']}/150) | R@5={m_e1['r5']:5.2f}% ({m_e1['hits_r5']}/150) | R@10={m_e1['r10']:5.2f}% ({m_e1['hits_r10']}/150) | PoolRec={m_e1['pool_rec']:5.2f}% ({m_e1['pool_hits']}/150)")
print(f"Delta (E1 - C)     : MRR={m_e1['mrr'] - m_c['mrr']:+.4f} | R@1={m_e1['r1'] - m_c['r1']:+5.2f} pp | R@3={m_e1['r3'] - m_c['r3']:+5.2f} pp | R@5={m_e1['r5'] - m_c['r5']:+5.2f} pp | R@10={m_e1['r10'] - m_c['r10']:+5.2f} pp")

# 5. Question Movement Analysis
improved = []
regressed = []
unchanged = []

per_question_rows = []

for qid in data_c:
    c_rec = data_c[qid]
    e1_rec = data_e1[qid]
    rank_c = c_rec['best_rank']
    rank_e1 = e1_rec['best_rank']
    q_data = questions_map.get(qid, {})

    delta_rank = rank_c - rank_e1 # positive means improved (e.g. 5 - 1 = +4)

    status = "unchanged"
    if rank_e1 < rank_c:
        status = "improved"
        improved.append({
            'qid': qid,
            'company': c_rec['company'],
            'doc_name': c_rec['doc_name'],
            'rank_c': rank_c,
            'rank_e1': rank_e1,
            'pool_rank': c_rec['best_rank_in_pool'],
            'question': q_data.get('question', c_rec['question']),
            'top3_c': c_rec.get('top3_retrieved', []),
            'top3_e1': e1_rec.get('top3_retrieved', [])
        })
    elif rank_e1 > rank_c:
        status = "regressed"
        regressed.append({
            'qid': qid,
            'company': c_rec['company'],
            'doc_name': c_rec['doc_name'],
            'rank_c': rank_c,
            'rank_e1': rank_e1,
            'pool_rank': c_rec['best_rank_in_pool'],
            'question': q_data.get('question', c_rec['question']),
            'top3_c': c_rec.get('top3_retrieved', []),
            'top3_e1': e1_rec.get('top3_retrieved', [])
        })
    else:
        unchanged.append(qid)

    per_question_rows.append({
        'question_id': qid,
        'company': c_rec['company'],
        'doc_name': c_rec['doc_name'],
        'sector': c_rec['sector'],
        'target_pages': str(c_rec['target_pages']),
        'rank_exp_c': rank_c,
        'rank_exp_e1': rank_e1,
        'pool_rank': c_rec['best_rank_in_pool'],
        'candidate_pool_hit': c_rec['candidate_pool_hit'],
        'movement_status': status,
        'rr_c': c_rec['rr'],
        'rr_e1': e1_rec['rr'],
        'top3_exp_c': " | ".join(c_rec.get('top3_retrieved', [])),
        'top3_exp_e1': " | ".join(e1_rec.get('top3_retrieved', [])),
        'question_text': q_data.get('question', c_rec['question'])
    })

print(f"\nMovement Counts: Improved={len(improved)}, Regressed={len(regressed)}, Unchanged={len(unchanged)}")

print("\n--- Improved Questions ---")
for item in improved:
    print(f"  [+] {item['qid']} ({item['company']}, {item['doc_name']}): Rank #{item['rank_c']} -> #{item['rank_e1']} (Pool #{item['pool_rank']}) | Q: {item['question'][:75]}")

print("\n--- Regressed Questions ---")
for item in regressed:
    print(f"  [-] {item['qid']} ({item['company']}, {item['doc_name']}): Rank #{item['rank_c']} -> #{item['rank_e1']} (Pool #{item['pool_rank']}) | Q: {item['question'][:75]}")

# 6. Save per-question CSV
out_dir = 'outputs/experiments/experiment_E1_context_window'
csv_path = os.path.join(out_dir, 'per_question_results.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_question_rows)

print(f"\nSaved per-question results to {csv_path}")

# 7. Save JSON summary
summary = {
    'experiment': 'Experiment E1: Reranker Context Window Ablation (512 vs 1024 characters)',
    'baseline_experiment': 'Experiment C (Full-Universe Company Intent, 512 characters)',
    'upstream_retrieval_identity_verified': (len(pool_hit_diffs) == 0 and len(pool_rank_diffs) == 0),
    'metrics_comparison': {
        'experiment_c': m_c,
        'experiment_e1': m_e1,
        'delta': {
            'mrr': round(m_e1['mrr'] - m_c['mrr'], 4),
            'recall_at_1_pp': round(m_e1['r1'] - m_c['r1'], 2),
            'recall_at_3_pp': round(m_e1['r3'] - m_c['r3'], 2),
            'recall_at_5_pp': round(m_e1['r5'] - m_c['r5'], 2),
            'recall_at_10_pp': round(m_e1['r10'] - m_c['r10'], 2),
            'candidate_pool_recall_pp': round(m_e1['pool_rec'] - m_c['pool_rec'], 2)
        }
    },
    'movement_summary': {
        'improved_count': len(improved),
        'regressed_count': len(regressed),
        'unchanged_count': len(unchanged),
        'improved_questions': improved,
        'regressed_questions': regressed
    },
    'runtime_seconds': {
        'experiment_c': 32.48,
        'experiment_e1': 40.22,
        'delta_seconds': +7.74
    }
}

json_path = os.path.join(out_dir, 'experiment_E1_summary.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"Saved JSON summary to {json_path}")
