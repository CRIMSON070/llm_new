import json
import re
from scripts.diagnose_experiment_c_failures import analyzed_rows, questions_map, gold_chunks_found

none_rows = [r for r in analyzed_rows if r['match_status'] == 'none']
print(f"Total none matches: {len(none_rows)}")
for r in none_rows:
    qid = r['qid']
    print(f"\n--- {qid} ---")
    print("Doc:", r['doc_name'], "Target pages:", r['target_pages'])
    print("Question:", r['question'])
    q = questions_map[qid]
    for ev in q['evidence']:
        print("Gold page:", ev.get('evidence_page_num'))
        print("Evidence snippet:", repr(ev.get('evidence_text', '')[:200]))
    for tp in r['target_pages']:
        chs = gold_chunks_found.get((r['doc_name'], tp), [])
        print(f"Chunks on page {tp}: {len(chs)}")
        for i, ch in enumerate(chs):
            print(f"  Chunk {i} ID: {ch.get('chunk_id')}, len={len(ch.get('text',''))}")
            print(f"  Excerpt: {repr(ch.get('text','')[:150])}")
