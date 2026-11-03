import json
import os
import re
import fitz # PyMuPDF

ckpt_path = 'outputs/experiments/experiment_AB_combined/retrieval/phase2_retrieval_checkpoint.jsonl'
scorecard_path = 'outputs/experiments/experiment_AB_combined/evaluation/phase2_full_scorecard.json'
q_path = 'data/financebench/questions.jsonl'
chunks_path = 'data/financebench/processed/phase2_enriched_chunks.jsonl'
pdf_dir = 'data/financebench/pdfs'

print("--- Step 1: Loading benchmark, checkpoint, and questions ---")
questions_map = {}
with open(q_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            questions_map[d['financebench_id']] = d

ckpt_records = []
with open(ckpt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            ckpt_records.append(json.loads(line))

print(f"Total checkpoint records: {len(ckpt_records)}")

# Identify all failures where gold was not in final top 10
failures = [r for r in ckpt_records if r['best_rank'] > 10]
print(f"Total failures (best_rank > 10): {len(failures)} out of {len(ckpt_records)}")

# Load chunks metadata for doc_name and target_pages of failures
failure_doc_pages = set()
for f in failures:
    doc = f['doc_name']
    for p in f.get('target_pages', []):
        failure_doc_pages.add((doc, p))

print(f"Unique (doc_name, target_page) pairs to check: {len(failure_doc_pages)}")

print("\n--- Step 2: Indexing extracted chunks on gold pages ---")
gold_chunks_found = {} # (doc, page) -> list of chunks
with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        doc = c.get('doc_name')
        page = c.get('page_number')
        if page is None:
            # try parsing from chunk_id e.g. '3M_2018_10K_p1_c0'
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
                'token_count': len(c.get('text', '').split()),
                'is_table': bool(c.get('tables')),
                'section_heading': c.get('section_heading', '')
            })

print(f"Indexed chunks for {len(gold_chunks_found)} / {len(failure_doc_pages)} target pages.")

# Check which target pages had zero chunks extracted
missing_chunk_pages = []
for doc, p in failure_doc_pages:
    if (doc, p) not in gold_chunks_found or len(gold_chunks_found[(doc, p)]) == 0:
        missing_chunk_pages.append((doc, p))

print(f"Target pages with ZERO chunks extracted: {len(missing_chunk_pages)}")
if missing_chunk_pages:
    print(f"  Missing chunk pages: {missing_chunk_pages[:10]}")

print("\n--- Step 3: Analyzing each failure ---")
detailed_failures = []

for f in failures:
    qid = f['financebench_id']
    q_data = questions_map.get(qid, {})
    doc_name = f['doc_name']
    target_pages = f.get('target_pages', [])
    company = f.get('company', '')
    sector = f.get('sector', '')
    question_text = q_data.get('question', f.get('question', ''))
    q_type = q_data.get('question_type', 'unknown')
    evidence_list = q_data.get('evidence', [])
    
    in_pool = bool(f.get('candidate_pool_hit', 0))
    best_rank_in_pool = f.get('best_rank_in_pool', 999)
    best_rank = f.get('best_rank', 999)
    top3 = f.get('top3_retrieved', [])

    # PDF check
    pdf_path = os.path.join(pdf_dir, f"{doc_name}.pdf")
    pdf_exists = os.path.exists(pdf_path)
    pdf_page_count = 0
    if pdf_exists:
        try:
            doc_pdf = fitz.open(pdf_path)
            pdf_page_count = len(doc_pdf)
            doc_pdf.close()
        except Exception as e:
            pdf_page_count = -1

    # Chunk extraction check
    total_chunks_on_target_pages = 0
    chunks_text_combined = ""
    for tp in target_pages:
        ch_list = gold_chunks_found.get((doc_name, tp), [])
        total_chunks_on_target_pages += len(ch_list)
        for ch in ch_list:
            chunks_text_combined += "\n" + ch['text']

    # Evidence coverage check
    # Check if evidence snippet is in the extracted chunk text
    evidence_match_status = "none"
    full_evidence_snippets = [e.get('evidence_text', '') for e in evidence_list if e.get('evidence_text')]
    
    exact_matches = 0
    partial_matches = 0
    for snip in full_evidence_snippets:
        # Take first 50 chars of snippet
        clean_snip = re.sub(r'\s+', ' ', snip.strip().lower())
        clean_chunk = re.sub(r'\s+', ' ', chunks_text_combined.strip().lower())
        if len(clean_snip) > 20 and clean_snip[:40] in clean_chunk:
            exact_matches += 1
        else:
            # check words overlap
            snip_words = set(re.findall(r'\b\w{4,}\b', clean_snip))
            chunk_words = set(re.findall(r'\b\w{4,}\b', clean_chunk))
            if snip_words and len(snip_words & chunk_words) / len(snip_words) > 0.6:
                partial_matches += 1

    # Primary and Secondary Classifications
    # A: Gold evidence absent from candidate pool
    # B: Gold evidence present in pool but ranked below 10
    # C: Gold evidence retrieved but lost during context assembly or bundling
    # D: Evidence present in final context but answer still incorrect
    # E: Multiple pages or table rows are required
    # F: Extraction, page mapping, or gold-label mismatch suspected
    # G: Insufficient evidence to classify
    
    primary_category = "G"
    evidence_notes = []

    if len(target_pages) > 1:
        # Multi-page requirement
        evidence_notes.append(f"Requires {len(target_pages)} distinct pages ({target_pages})")

    if total_chunks_on_target_pages == 0:
        primary_category = "F"
        evidence_notes.append(f"Extraction failure: 0 chunks found in index for target pages {target_pages}")
    elif in_pool:
        primary_category = "B"
        evidence_notes.append(f"Present in candidate pool at rank #{best_rank_in_pool}, but reranker demoted to #{best_rank}")
    else:
        # Gold absent from candidate pool
        if len(target_pages) > 1:
            primary_category = "E"
            evidence_notes.append(f"Candidate pool missed multi-page evidence across pages {target_pages}")
        else:
            primary_category = "A"
            evidence_notes.append(f"Candidate pool missed gold page {target_pages}; neither BM25 nor Dense retrieved into top 50 pool")

    detailed_failures.append({
        'qid': qid,
        'company': company,
        'sector': sector,
        'doc_name': doc_name,
        'question': question_text,
        'q_type': q_type,
        'target_pages': target_pages,
        'num_target_pages': len(target_pages),
        'in_pool': in_pool,
        'best_rank_in_pool': best_rank_in_pool,
        'best_rank': best_rank,
        'top3_retrieved': top3,
        'pdf_exists': pdf_exists,
        'pdf_page_count': pdf_page_count,
        'chunks_on_gold_pages': total_chunks_on_target_pages,
        'exact_snippet_matches': exact_matches,
        'partial_snippet_matches': partial_matches,
        'primary_category': primary_category,
        'evidence_notes': "; ".join(evidence_notes)
    })

# Summary of categories
from collections import Counter
cat_counts = Counter(d['primary_category'] for d in detailed_failures)
print("\n--- Failure Categories Summary ---")
for cat, cnt in cat_counts.most_common():
    print(f"Category {cat}: {cnt} questions ({cnt/len(detailed_failures)*100:.1f}%)")

sector_fail_counts = Counter(d['sector'] for d in detailed_failures)
print("\n--- Sector Failures Summary ---")
for sec, cnt in sector_fail_counts.most_common():
    print(f"Sector {sec}: {cnt} failures")

def print_cat_b():
    cat_b = [d for d in detailed_failures if d['primary_category'] == 'B']
    print(f"=== Category B Questions ({len(cat_b)}) ===")
    for i, d in enumerate(cat_b, 1):
        print(f"{i:2d}. {d['qid']} ({d['company']}): Pool Rank #{d['best_rank_in_pool']} -> Final Rank #{d['best_rank']}")
        print(f"    Target pages: {d['target_pages']} in {d['doc_name']}")
        print(f"    Q: {d['question'][:100]}")
        print(f"    Top3 retrieved: {d['top3_retrieved']}")

def print_cat_e():
    cat_e = [d for d in detailed_failures if d['primary_category'] == 'E']
    print(f"=== Category E Questions ({len(cat_e)}) ===")
    for i, d in enumerate(cat_e, 1):
        print(f"{i:2d}. {d['qid']} ({d['company']}): Target pages: {d['target_pages']} in {d['doc_name']}")
        print(f"    In pool: {d['in_pool']}, Pool rank: #{d['best_rank_in_pool']}")
        print(f"    Q: {d['question'][:100]}")
        print(f"    Top3 retrieved: {d['top3_retrieved']}")

if __name__ == '__main__':
    print_cat_b()
    print_cat_e()


