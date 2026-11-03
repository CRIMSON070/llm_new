from scripts.diagnose_experiment_c_failures import analyzed_rows, questions_map, failures, gold_chunks_found

in_pool = [r for r in analyzed_rows if r['in_pool']]
print(f"Total In-Pool Ranking Failures: {len(in_pool)}")

for i, r in enumerate(in_pool, 1):
    qid = r['qid']
    q = questions_map[qid]
    f_orig = next(f for f in failures if f['financebench_id'] == qid)
    print(f"\n{i:2d}. {qid} | {r['target_comp']} | {r['doc_name']} | Target Pages: {r['target_pages']}")
    print(f"    Pool Rank: #{r['pool_rank']} -> Final Rank: #{r['final_rank']}")
    print(f"    Question: {r['question']}")
    print(f"    Detected Stmt: {r['detected_stmt']} | Detected Comp: {r['detected_comp']} | Detected Periods: {r['detected_periods']}")
    print(f"    Top 3 retrieved: {f_orig.get('top3_retrieved', [])}")
    for tp in r['target_pages']:
        chs = gold_chunks_found.get((r['doc_name'], tp), [])
        for ch in chs:
            t = ch.get('text', '')
            print(f"    Gold Chunk {ch.get('chunk_id')}: len={len(t)}, has_table={ch.get('tables', False)}, lead={repr(t[:120])}")
