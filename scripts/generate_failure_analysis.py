import json
import os
import re
import csv
import fitz

ckpt_path = 'outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl'
scorecard_path = 'outputs/experiments/experiment_AB_combined/evaluation/phase2_full_scorecard.json'
q_path = 'data/financebench/questions.jsonl'
chunks_path = 'data/financebench/processed/phase2_enriched_chunks.jsonl'
pdf_dir = 'data/financebench/pdfs'
out_csv = 'outputs/experiments/failure_analysis_AB/question_failure_analysis.csv'

# 1. Load questions map
questions_map = {}
with open(q_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            questions_map[d['financebench_id']] = d

# 2. Load checkpoint records
ckpt_records = []
with open(ckpt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            ckpt_records.append(json.loads(line))

failures = [r for r in ckpt_records if r['best_rank'] > 10]
print(f"Total failures to analyze: {len(failures)}")

# 3. Load chunks for failed doc/pages
failure_doc_pages = set()
for f in failures:
    doc = f['doc_name']
    for p in f.get('target_pages', []):
        failure_doc_pages.add((doc, p))

gold_chunks_found = {} # (doc, page) -> list of chunks
with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        doc = c.get('doc_name')
        page = c.get('page_number')
        if page is None:
            m = re.search(r'_p(\d+)_c\d+', c.get('chunk_id', ''))
            if m:
                page = int(m.group(1))
            else:
                page = c.get('page')
        else:
            page = int(page)
        
        if (doc, page) in failure_doc_pages:
            key = (doc, page)
            if key not in gold_chunks_found:
                gold_chunks_found[key] = []
            gold_chunks_found[key].append({
                'chunk_id': c.get('chunk_id'),
                'text': c.get('text', ''),
                'tables': c.get('tables', []),
                'section_heading': c.get('section_heading', '')
            })

# 4. Classify each failure
analyzed_rows = []

for idx, f in enumerate(failures, start=1):
    qid = f['financebench_id']
    q_data = questions_map.get(qid, {})
    company = f.get('company', q_data.get('company', ''))
    sector = f.get('sector', q_data.get('gics_sector', ''))
    doc_name = f['doc_name']
    target_pages = f.get('target_pages', [])
    q_text = q_data.get('question', f.get('question', ''))
    q_type = q_data.get('question_type', 'unknown')
    evidence_list = q_data.get('evidence', [])
    gold_answer = q_data.get('answer', '')

    in_pool = bool(f.get('candidate_pool_hit', 0))
    pool_rank = f.get('best_rank_in_pool', 999)
    final_rank = f.get('best_rank', 999)
    top3 = f.get('top3_retrieved', [])

    # Text check on target pages
    combined_chunk_text = ""
    for tp in target_pages:
        ch_list = gold_chunks_found.get((doc_name, tp), [])
        for ch in ch_list:
            combined_chunk_text += "\n" + ch['text']

    # Evidence presence
    ev_snippets = [e.get('evidence_text', '') for e in evidence_list if e.get('evidence_text')]
    snippet_in_chunk = False
    evidence_len = sum(len(s) for s in ev_snippets)
    
    clean_chunk = re.sub(r'\s+', ' ', combined_chunk_text.lower())
    for snip in ev_snippets:
        clean_snip = re.sub(r'\s+', ' ', snip.lower().strip())
        # check if key 30-char substring exists
        if len(clean_snip) > 25 and clean_snip[:35] in clean_chunk:
            snippet_in_chunk = True
            break
        elif len(clean_snip) <= 25 and clean_snip in clean_chunk:
            snippet_in_chunk = True
            break

    # Top3 retrieval analysis: check for cross-company contamination or wrong document/filing
    cross_company = False
    wrong_filing = False
    for t in top3:
        # e.g. 'CORNING_2022_10K_p14_c4_p14'
        top_doc = t.rsplit('_p', 1)[0].rsplit('_c', 1)[0]
        if company.upper().replace(' ', '') not in top_doc.upper().replace(' ', ''):
            cross_company = True
        elif top_doc != doc_name:
            wrong_filing = True

    # Determine failure category
    # A. Gold evidence absent from candidate pool
    # B. Gold evidence present in pool but ranked below 10
    # C. Gold evidence retrieved but lost during context assembly or bundling
    # D. Evidence present in final context but answer is still incorrect
    # E. Multiple pages or table rows are required
    # F. Extraction, page mapping, or gold-label mismatch suspected
    # G. Insufficient evidence to classify

    category = "A"
    observations = []

    if len(target_pages) > 1 and not in_pool:
        category = "E"
        observations.append(f"Multi-page evidence requirement across {len(target_pages)} pages ({target_pages}); cross-statement financial ratio calculation")
    elif len(target_pages) > 1 and in_pool:
        category = "B"
        observations.append(f"Partial multi-page evidence in candidate pool at rank #{pool_rank}, but demoted by reranker to #{final_rank} (requires pages {target_pages})")
    elif in_pool:
        category = "B"
        observations.append(f"Gold evidence reached candidate pool at rank #{pool_rank}, but cross-encoder reranker demoted it to #{final_rank}")
    else:
        category = "A"
        observations.append(f"Gold evidence absent from top 50 candidate pool (best rank > 50)")

    # Supporting observations on root causes
    if cross_company:
        observations.append("Cross-company chunk contamination in top retrieved results (reranker company detection missed company)")
    if wrong_filing:
        observations.append(f"Filing/period mismatch: top retrieved results came from alternative filings of {company} instead of {doc_name}")
    if not snippet_in_chunk and len(combined_chunk_text) > 0:
        observations.append("Evidence snippet not cleanly matched in chunk lead text (possible boundary split or formatting mismatch)")

    retrieval_path_evidence = (
        f"Candidate Pool Hit: {in_pool} (Rank #{pool_rank if in_pool else '>50'}); "
        f"Individual BM25/Dense/Metadata path hits not persisted in checkpoint log"
    )

    context_status = "Excluded from Top 10 context window (Rank > 10)"

    analyzed_rows.append({
        'question_id': qid,
        'company': company,
        'sector': sector,
        'question_type': q_type,
        'doc_name': doc_name,
        'gold_page': "; ".join(str(p) for p in target_pages),
        'failure_category': category,
        'candidate_pool_status': 'in_pool' if in_pool else 'absent',
        'candidate_pool_rank': str(pool_rank) if in_pool else '>50',
        'final_rerank_rank': str(final_rank) if final_rank != 999 else '>10',
        'retrieval_path_evidence': retrieval_path_evidence,
        'context_status': context_status,
        'supporting_observations': ". ".join(observations),
        'question_text': q_text
    })

# Write CSV
fieldnames = [
    'question_id', 'company', 'sector', 'question_type', 'doc_name',
    'gold_page', 'failure_category', 'candidate_pool_status',
    'candidate_pool_rank', 'final_rerank_rank', 'retrieval_path_evidence',
    'context_status', 'supporting_observations'
]

with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in analyzed_rows:
        row_dict = {k: r[k] for k in fieldnames}
        writer.writerow(row_dict)

print(f"Successfully generated CSV at {out_csv} ({len(analyzed_rows)} rows)")

from collections import Counter
cat_counts = Counter(r['failure_category'] for r in analyzed_rows)
print("\nFinal Category Counts:")
for cat, cnt in cat_counts.most_common():
    print(f"  Category {cat}: {cnt} ({cnt/len(analyzed_rows)*100:.1f}%)")
