import json
import os

ab_ckpt = 'outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl'
c_ckpt = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
base_ckpt = 'outputs/phase2/retrieval/phase2_retrieval_checkpoint.jsonl'
expA_ckpt = 'outputs/experiments/experiment_A_pool60/retrieval/phase2_retrieval_checkpoint.jsonl'
expB_ckpt = 'outputs/experiments/experiment_B_routing/retrieval/phase2_retrieval_checkpoint.jsonl'
q_path = 'data/financebench/questions.jsonl'

# Load questions map
questions_map = {}
with open(q_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        questions_map[d['financebench_id']] = d

def load_data(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            qid = d.get('financebench_id', f"q_{d.get('index')}")
            data[qid] = d
    return data

base = load_data(base_ckpt)
expA = load_data(expA_ckpt)
expB = load_data(expB_ckpt)
expAB = load_data(ab_ckpt)
expC = load_data(c_ckpt)

print(f"Total questions loaded: Base={len(base)}, A={len(expA)}, B={len(expB)}, AB={len(expAB)}, C={len(expC)}")

# 1. Macro Metrics
def calc_metrics(data):
    n = len(data)
    mrr = sum(d['rr'] for d in data.values()) / n
    r1 = sum(1 for d in data.values() if d['best_rank'] == 1) / n * 100
    r3 = sum(1 for d in data.values() if d['best_rank'] <= 3) / n * 100
    r5 = sum(1 for d in data.values() if d['best_rank'] <= 5) / n * 100
    r10 = sum(1 for d in data.values() if d['best_rank'] <= 10) / n * 100
    pool_rec = sum(1 for d in data.values() if d.get('candidate_pool_hit', d.get('best_rank_in_pool', 999) <= 50)) / n * 100
    return {'mrr': mrr, 'r1': r1, 'r3': r3, 'r5': r5, 'r10': r10, 'pool_rec': pool_rec}

m_base = calc_metrics(base)
m_a = calc_metrics(expA)
m_b = calc_metrics(expB)
m_ab = calc_metrics(expAB)
m_c = calc_metrics(expC)

print("\n--- Macro Metrics Comparison ---")
for name, m in [('Baseline', m_base), ('Exp A (Pool 60)', m_a), ('Exp B (Intent)', m_b), ('Exp AB (Combined)', m_ab), ('Exp C (Company Intent)', m_c)]:
    print(f"{name:25s}: MRR={m['mrr']:.4f} | R@1={m['r1']:5.2f}% | R@3={m['r3']:5.2f}% | R@5={m['r5']:5.2f}% | R@10={m['r10']:5.2f}% | PoolRec={m['pool_rec']:5.2f}%")

# 2. Movement vs Experiment AB
improved_ab = sum(1 for q in expAB if expC[q]['best_rank'] < expAB[q]['best_rank'])
regressed_ab = sum(1 for q in expAB if expC[q]['best_rank'] > expAB[q]['best_rank'])
unchanged_ab = sum(1 for q in expAB if expC[q]['best_rank'] == expAB[q]['best_rank'])
print(f"\nExp C vs Exp AB: Improved={improved_ab}, Regressed={regressed_ab}, Unchanged={unchanged_ab}")

# Movement vs Baseline
improved_base = sum(1 for q in base if expC[q]['best_rank'] < base[q]['best_rank'])
regressed_base = sum(1 for q in base if expC[q]['best_rank'] > base[q]['best_rank'])
unchanged_base = sum(1 for q in base if expC[q]['best_rank'] == base[q]['best_rank'])
print(f"Exp C vs Baseline: Improved={improved_base}, Regressed={regressed_base}, Unchanged={unchanged_base}")

# 3. Cross-Company Contamination
def extract_chunk_company(doc_name):
    doc_upper = doc_name.upper()
    if '3M' in doc_upper: return '3M'
    elif 'ULTA' in doc_upper: return 'ULTABEAUTY'
    elif 'ADOBE' in doc_upper: return 'ADOBE'
    elif 'AMAZON' in doc_upper: return 'AMAZON'
    elif 'MICROSOFT' in doc_upper: return 'MICROSOFT'
    elif 'JPMORGAN' in doc_upper: return 'JPMORGAN'
    elif 'JOHNSON' in doc_upper: return 'JOHNSON'
    else: return doc_upper.split('_')[0]

def measure_contamination(data):
    tot_q = len(data)
    contam_q = 0
    tot_chunks = tot_q * 3
    contam_chunks = 0
    for qid, d in data.items():
        target_comp = extract_chunk_company(d['doc_name'])
        top3 = d.get('top3_retrieved', [])
        q_contam = False
        for c_str in top3:
            parts = c_str.split('_p')[0].split('_c')[0]
            c_comp = extract_chunk_company(parts)
            if c_comp != target_comp:
                contam_chunks += 1
                q_contam = True
        if q_contam:
            contam_q += 1
    return {
        'q_count': contam_q,
        'q_pct': contam_q / tot_q * 100,
        'chunk_count': contam_chunks,
        'chunk_pct': contam_chunks / tot_chunks * 100
    }

c_ab = measure_contamination(expAB)
c_c = measure_contamination(expC)
print("\n--- Cross-Company Contamination (Top-3 Retrieved Chunks) ---")
print(f"Exp AB: Question-Level = {c_ab['q_count']}/150 ({c_ab['q_pct']:.2f}%) | Chunk-Level = {c_ab['chunk_count']}/450 ({c_ab['chunk_pct']:.2f}%)")
print(f"Exp C : Question-Level = {c_c['q_count']}/150 ({c_c['q_pct']:.2f}%) | Chunk-Level = {c_c['chunk_count']}/450 ({c_c['chunk_pct']:.2f}%)")
print(f"Delta : Question-Level = {c_c['q_pct'] - c_ab['q_pct']:+.2f} pp | Chunk-Level = {c_c['chunk_pct'] - c_ab['chunk_pct']:+.2f} pp")

# 4. Check the 19 Previously Identified Candidate-Pool Demotions (Category B from AB)
# Load question_failure_analysis.csv
import csv
cat_b_qids = []
with open('outputs/experiments/failure_analysis_AB/question_failure_analysis.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['failure_category'] == 'B':
            cat_b_qids.append(row['question_id'])

print(f"\n--- Tracking the 19 Category B Demotions from Experiment AB ---")
cat_b_recovered = []
for qid in cat_b_qids:
    ab_rank = expAB[qid]['best_rank']
    c_rank = expC[qid]['best_rank']
    ab_pool = expAB[qid]['best_rank_in_pool']
    c_pool = expC[qid].get('best_rank_in_pool', 999)
    recovered = c_rank <= 10
    if recovered:
        cat_b_recovered.append((qid, ab_rank, c_rank))
    comp = expC[qid]['company']
    q_txt = expC[qid]['question'][:60]
    print(f"  {qid} [{comp}]: AB Rank #{ab_rank} (pool #{ab_pool}) -> Exp C Rank #{c_rank} (pool #{c_pool}) | In Top 10: {recovered}")

print(f"\nTotal Category B questions moving into Top 10: {len(cat_b_recovered)} / {len(cat_b_qids)} ({len(cat_b_recovered)/len(cat_b_qids)*100:.1f}%)")

# 5. Check 3M and Registered-Debt Queries for Regression
print("\n--- Regression Check on Key Previously Solved Queries ---")
key_queries = [
    ('3M registered debt', 'financebench_id_00941'),
    ('American Express registered debt', 'financebench_id_00476'),
    ('Ulta Beauty registered debt', 'financebench_id_00746'),
    ('American Express card retention', 'financebench_id_01981'),
    ('3M Net PPNE', 'financebench_id_04672'),
    ('3M liquidity profile', 'financebench_id_00807'),
]
for label, qid in key_queries:
    b_rank = base[qid]['best_rank']
    ab_rank = expAB[qid]['best_rank']
    c_rank = expC[qid]['best_rank']
    print(f"  {label} ({qid}): Base=#{b_rank} -> Exp AB=#{ab_rank} -> Exp C=#{c_rank} (Regressed: {c_rank > ab_rank})")

# 6. Sector Breakdown
print("\n--- Sector Breakdown (Exp AB vs Exp C) ---")
sectors = set(d['sector'] for d in expC.values())
for s in sorted(list(sectors)):
    s_qids = [q for q in expC if expC[q]['sector'] == s]
    ab_mrr = sum(expAB[q]['rr'] for q in s_qids) / len(s_qids)
    c_mrr = sum(expC[q]['rr'] for q in s_qids) / len(s_qids)
    ab_r10 = sum(1 for q in s_qids if expAB[q]['best_rank'] <= 10) / len(s_qids) * 100
    c_r10 = sum(1 for q in s_qids if expC[q]['best_rank'] <= 10) / len(s_qids) * 100
print("\n--- Details of the 13 Questions Improved in Exp C vs Exp AB ---")
improved_list = []
for qid in expAB:
    if expC[qid]['best_rank'] < expAB[qid]['best_rank']:
        improved_list.append({
            'qid': qid,
            'company': expC[qid]['company'],
            'sector': expC[qid]['sector'],
            'question': expC[qid]['question'],
            'ab_rank': expAB[qid]['best_rank'],
            'c_rank': expC[qid]['best_rank'],
            'delta_rr': expC[qid]['rr'] - expAB[qid]['rr']
        })

improved_list.sort(key=lambda x: x['delta_rr'], reverse=True)
for i, item in enumerate(improved_list, 1):
    print(f"[{i:2d}/13] {item['qid']} - {item['company']} ({item['sector']})")
    print(f"       AB Rank: #{item['ab_rank']} -> Exp C Rank: #{item['c_rank']} (Delta RR: +{item['delta_rr']:.4f})")
    print(f"       Q: {item['question']}")

