import json
import re
from collections import Counter
from scripts.diagnose_experiment_c_failures import analyzed_rows, questions_map, gold_chunks_found, failures

print("=" * 80)
print("ANALYSIS OF 17 IN-POOL FAILURES (RANKING FAILURES)")
print("=" * 80)

in_pool = [r for r in analyzed_rows if r['in_pool']]
for i, r in enumerate(in_pool, 1):
    qid = r['qid']
    q = questions_map[qid]
    print(f"\n{i:2d}. {qid} | Company: {r['target_comp']} | Doc: {r['doc_name']} | Target Pages: {r['target_pages']}")
    print(f"    Pool Rank: #{r['pool_rank']} -> Final Rank: #{r['final_rank']}")
    print(f"    Question: {r['question']}")
    print(f"    Detected Stmt: {r['detected_stmt']} | Detected Comp: {r['detected_comp']} | Periods: {r['detected_periods']}")
    # Inspect top 3 retrieved
    f_orig = next(f for f in failures if f['financebench_id'] == qid)
    print(f"    Top 3 retrieved chunks: {f_orig.get('top3_retrieved', [])}")
    # Inspect chunks on gold page
    chs = gold_chunks_found.get((r['doc_name'], r['target_pages'][0]), [])
    for c in chs:
        print(f"    Gold Page Chunk: {c.get('chunk_id')} | Length: {len(c.get('text', ''))} chars | Has table: {bool(c.get('tables'))} | Heading: {c.get('section_heading')}")
        # Check if table row or numbers appear after char 512!
        c_text = c.get('text', '')
        if len(c_text) > 512:
            print(f"      [TRUNCATION RISK]: Chunk length is {len(c_text)} chars (>512). Truncated at 512 chars.")

print("\n" + "=" * 80)
print("ANALYSIS OF 47 ABSENT-FROM-POOL FAILURES (CANDIDATE GENERATION FAILURES)")
print("=" * 80)

absent_pool = [r for r in analyzed_rows if not r['in_pool']]
multi_page_absent = [r for r in absent_pool if r['num_pages'] > 1]
single_page_absent = [r for r in absent_pool if r['num_pages'] == 1]
print(f"Multi-page absent: {len(multi_page_absent)}")
print(f"Single-page absent: {len(single_page_absent)}")

# Group single-page absent by question reasoning / type
reasoning_counts = Counter()
for r in single_page_absent:
    q = questions_map[r['qid']]
    q_reasoning = q.get('question_reasoning') or q.get('question_type') or 'unknown'
    reasoning_counts[q_reasoning] += 1

print("\nSingle-page absent reasoning distribution:")
for k, v in reasoning_counts.most_common():
    print(f"  {k}: {v}")

print("\nSample Single-page absent cases:")
for i, r in enumerate(single_page_absent[:15], 1):
    qid = r['qid']
    q = questions_map[qid]
    f_orig = next(f for f in failures if f['financebench_id'] == qid)
    print(f"\n{i:2d}. {qid} | Company: {r['target_comp']} | Doc: {r['doc_name']} | Target Page: {r['target_pages']}")
    print(f"    Question: {r['question']}")
    print(f"    Top 3 retrieved: {f_orig.get('top3_retrieved', [])}")
    print(f"    Reasoning: {q.get('question_reasoning')} | Type: {q.get('question_type')}")
