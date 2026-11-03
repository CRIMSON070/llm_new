import json
import os
import sys

c_ckpt = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
q_file = 'data/financebench/questions.jsonl'

with open(q_file, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f if line.strip()]

with open(c_ckpt, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

q_map = {q['financebench_id']: q for q in questions}

cat_b_records = [r for r in records if (r.get('candidate_pool_hit') == 1 or r.get('best_rank_in_pool', 999) <= 50) and r.get('best_rank', 999) > 10]

def extract_chunk_company_and_period(doc_name):
    doc_upper = doc_name.upper()
    comp = None
    if "3M" in doc_upper: comp = "3M"
    elif "ULTA" in doc_upper: comp = "ULTABEAUTY"
    elif "ADOBE" in doc_upper: comp = "ADOBE"
    elif "AMAZON" in doc_upper: comp = "AMAZON"
    elif "MICROSOFT" in doc_upper: comp = "MICROSOFT"
    elif "JPMORGAN" in doc_upper: comp = "JPMORGAN"
    elif "JOHNSON" in doc_upper: comp = "JOHNSON"
    else: comp = doc_upper.split("_")[0]
    return comp

print(f"Total Category B questions: {len(cat_b_records)}")
for i, r in enumerate(cat_b_records, 1):
    qid = r['financebench_id']
    q_data = q_map[qid]
    q_text = q_data['question']
    doc_name = r['doc_name']
    target_comp = extract_chunk_company_and_period(doc_name)
    pool_rank = r['best_rank_in_pool']
    final_rank = r['best_rank']
    top3 = r.get('top3_retrieved', [])
    top3_docs = [t.split('_p')[0] for t in top3]
    top3_comps = [extract_chunk_company_and_period(d) for d in top3_docs]
    
    competing_wrong_comp = [c for c in top3_comps if c != target_comp]
    competing_wrong_doc = [d for d in top3_docs if d != doc_name]

    print(f"\n{i:2d}. {qid} | Company: {target_comp} | Doc: {doc_name} | Target Pages: {r.get('target_pages')}")
    print(f"    Candidate Pool Rank: #{pool_rank} -> Final Reranked Rank: #{final_rank}")
    print(f"    Question: {q_text[:90]}...")
    print(f"    Top 3 Retrieved: {top3}")
    print(f"    Competing non-target docs in Top-3: {competing_wrong_doc}")
    if competing_wrong_comp:
        print(f"    [CROSS-COMPANY CONTAMINATION IN TOP-3]: {competing_wrong_comp}")
    else:
        print(f"    [SAME-COMPANY COMPETITION]: All Top-3 chunks came from {target_comp}")
