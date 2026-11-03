import json
import os

base_ckpt = 'outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl'
expA_ckpt = 'outputs/experiments/experiment_A_pool60/retrieval/phase2_retrieval_checkpoint.jsonl'
expB_ckpt = 'outputs/experiments/experiment_B_routing/retrieval/phase2_retrieval_checkpoint.jsonl'
expAB_ckpt = 'outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl'

def load_ranks(path):
    ranks = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            d = json.loads(line)
            qid = d.get('financebench_id', f"q_{d.get('index')}")
            ranks[qid] = {
                'gold_rank': d.get('best_rank', 999),
                'in_pool': d.get('candidate_pool_hit', False) if 'candidate_pool_hit' in d else (d.get('best_rank', 999) <= 40),
                'company': d.get('company', ''),
                'sector': d.get('sector', ''),
                'question': d.get('question', '')
            }
    return ranks

base = load_ranks(base_ckpt)
expA = load_ranks(expA_ckpt)
expB = load_ranks(expB_ckpt)
expAB = load_ranks(expAB_ckpt)

print(f"Total questions loaded: Base={len(base)}, A={len(expA)}, B={len(expB)}, AB={len(expAB)}")

def compare(name_a, data_a, name_b, data_b):
    improved = sum(1 for q in data_a if data_b[q]['gold_rank'] < data_a[q]['gold_rank'])
    regressed = sum(1 for q in data_a if data_b[q]['gold_rank'] > data_a[q]['gold_rank'])
    unchanged = sum(1 for q in data_a if data_b[q]['gold_rank'] == data_a[q]['gold_rank'])
    print(f"{name_b} vs {name_a}: Improved={improved}, Regressed={regressed}, Unchanged={unchanged}")

compare("Base", base, "Exp AB", expAB)
compare("Exp A", expA, "Exp AB", expAB)
compare("Exp B", expB, "Exp AB", expAB)

print("\n--- Detailed Rank Movement Analysis (Exp AB vs Base) ---")
shifts = []
for qid in base:
    b_rank = base[qid]['gold_rank']
    ab_rank = expAB[qid]['gold_rank']
    a_rank = expA[qid]['gold_rank']
    b_intent_rank = expB[qid]['gold_rank']
    if b_rank != ab_rank:
        shifts.append({
            'qid': qid,
            'company': base[qid]['company'],
            'sector': base[qid]['sector'],
            'question': base[qid]['question'],
            'base_rank': b_rank,
            'a_rank': a_rank,
            'b_rank': b_intent_rank,
            'ab_rank': ab_rank,
            'delta_rank': b_rank - ab_rank, # positive means improved (smaller rank)
            'delta_rr': (1.0/ab_rank if ab_rank <= 100 else 0) - (1.0/b_rank if b_rank <= 100 else 0)
        })

print(f"Total questions with rank changes vs Base: {len(shifts)}")
shifts.sort(key=lambda x: x['delta_rr'], reverse=True)

print("\nTop Improved Questions (by MRR gain vs Base):")
for s in shifts[:10]:
    print(f"  {s['qid']} [{s['company']} - {s['sector']}]: Base={s['base_rank']} -> ExpA={s['a_rank']} -> ExpB={s['b_rank']} -> ExpAB={s['ab_rank']} (Delta RR: +{s['delta_rr']:.4f})")
    print(f"    Q: {s['question'][:100]}")

print("\nTop Regressed Questions (by MRR loss vs Base):")
for s in shifts[-10:]:
    print(f"  {s['qid']} [{s['company']} - {s['sector']}]: Base={s['base_rank']} -> ExpA={s['a_rank']} -> ExpB={s['b_rank']} -> ExpAB={s['ab_rank']} (Delta RR: {s['delta_rr']:.4f})")
    print(f"    Q: {s['question'][:100]}")

# Top-1, Top-3, Top-5, Top-10 movements
def metric_movements(name, old_data, new_data):
    for k in [1, 3, 5, 10]:
        entered = [q for q in old_data if old_data[q]['gold_rank'] > k and new_data[q]['gold_rank'] <= k]
        exited = [q for q in old_data if old_data[q]['gold_rank'] <= k and new_data[q]['gold_rank'] > k]
        net = len(entered) - len(exited)
        print(f"  Recall@{k:2d}: Entered={len(entered):2d}, Exited={len(exited):2d}, Net={net:+2d} ({name})")

print("\nTop-K Cutoff Transitions (Exp AB vs Base):")
metric_movements("Exp AB vs Base", base, expAB)

print("\nTop-K Cutoff Transitions (Exp AB vs Exp A):")
metric_movements("Exp AB vs Exp A", expA, expAB)

print("\nTop-K Cutoff Transitions (Exp AB vs Exp B):")
metric_movements("Exp AB vs Exp B", expB, expAB)

# Look at American Express Q39 explicitly
print("\nAmerican Express Q39 Analysis:")
for qid in base:
    if 'american express' in base[qid]['company'].lower() or 'question_39' in qid or 'registered' in base[qid]['question'].lower():
        print(f"  QID: {qid} | Company: {base[qid]['company']}")
        print(f"  Question: {base[qid]['question']}")
        print(f"  Ranks: Base={base[qid]['gold_rank']}, ExpA={expA[qid]['gold_rank']}, ExpB={expB[qid]['gold_rank']}, ExpAB={expAB[qid]['gold_rank']}")


def sector_metrics(data, qids):
    rec1 = sum(1 for q in qids if data[q]['gold_rank'] == 1) / len(qids) if qids else 0
    rec3 = sum(1 for q in qids if data[q]['gold_rank'] <= 3) / len(qids) if qids else 0
    rec5 = sum(1 for q in qids if data[q]['gold_rank'] <= 5) / len(qids) if qids else 0
    rec10 = sum(1 for q in qids if data[q]['gold_rank'] <= 10) / len(qids) if qids else 0
    mrr = sum(1.0/data[q]['gold_rank'] if data[q]['gold_rank'] <= 100 else 0 for q in qids) / len(qids) if qids else 0
    pool_rec = sum(1 for q in qids if data[q]['in_pool']) / len(qids) if qids else 0
    return rec1, rec3, rec5, rec10, mrr, pool_rec

fin_qids = [q for q in base if base[q]['sector'] == 'Financials']
print(f"Financials questions count: {len(fin_qids)}")

print("\n--- Financials Sector Metrics ---")
for name, d in [('Base', base), ('Exp A', expA), ('Exp B', expB), ('Exp AB', expAB)]:
    r1, r3, r5, r10, m, pr = sector_metrics(d, fin_qids)
    print(f"{name:8s}: PoolRec={pr*100:5.2f}% | R@1={r1*100:5.2f}% | R@3={r3*100:5.2f}% | R@5={r5*100:5.2f}% | R@10={r10*100:5.2f}% | MRR={m:.4f}")

print("\n=== Questions Regressed in AB vs Exp A ===")
for q in base:
    if expAB[q]['gold_rank'] > expA[q]['gold_rank']:
        print(f"{q} [{base[q]['company']}]: ExpA rank={expA[q]['gold_rank']} -> ExpAB rank={expAB[q]['gold_rank']}")
        print(f"   Q: {base[q]['question']}")

print("\n=== Top Questions Improved in AB vs Exp A ===")
improved_A = [q for q in base if expAB[q]['gold_rank'] < expA[q]['gold_rank']]
for q in improved_A:
    print(f"{q} [{base[q]['company']}]: ExpA rank={expA[q]['gold_rank']} -> ExpAB rank={expAB[q]['gold_rank']}")
    print(f"   Q: {base[q]['question']}")

