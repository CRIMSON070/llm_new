import json
import os
import re
from collections import Counter, defaultdict
import fitz # PyMuPDF

c_ckpt_path = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
q_path = 'data/financebench/questions.jsonl'
chunks_path = 'data/financebench/processed/phase2_enriched_chunks.jsonl'
pdf_dir = 'data/financebench/pdfs'

# Company mapping from Exp C
FINANCEBENCH_COMPANIES = {
    "3M": [r"\b3m\b"],
    "ACTIVISIONBLIZZARD": [r"\bactivision(\s+blizzard)?\b"],
    "ADOBE": [r"\badobe\b"],
    "AES": [r"\baes(\s+corp(oration)?)?\b"],
    "AMAZON": [r"\bamazon\b"],
    "AMCOR": [r"\bamcor\b"],
    "AMD": [r"\bamd\b", r"\badvanced\s+micro\s+devices\b"],
    "AMERICANEXPRESS": [r"\bamerican\s+express\b", r"\bamex\b"],
    "AMERICANWATERWORKS": [r"\bamerican\s+water(\s+works)?\b"],
    "BESTBUY": [r"\bbest\s+buy\b"],
    "BLOCK": [r"\bblock\s*\(\s*formerly\b", r"\bblock\s+inc\.?\b", r"\bsquare\s+inc\.?\b", r"\bblock['’]s\b"],
    "BOEING": [r"\bboeing\b"],
    "COCACOLA": [r"\bcoca[\s-]*cola\b", r"\bcoke\b"],
    "CORNING": [r"\bcorning\b"],
    "COSTCO": [r"\bcostco\b"],
    "CVSHEALTH": [r"\bcvs(\s+health)?\b"],
    "FOOTLOCKER": [r"\bfoot\s+locker\b"],
    "GENERALMILLS": [r"\bgeneral\s+mills\b"],
    "JOHNSON": [r"\bjohnson\s+(&|and)\s+johnson\b", r"\bjnj\b"],
    "JPMORGAN": [r"\bjpmorgan(\s+chase)?\b", r"\bjp\s*morgan\b", r"\bjpm\b"],
    "KRAFTHEINZ": [r"\bkraft\s+heinz\b", r"\bkraft\b"],
    "LOCKHEEDMARTIN": [r"\blockheed(\s+martin)?\b"],
    "MGMRESORTS": [r"\bmgm(\s+resorts)?\b"],
    "MICROSOFT": [r"\bmicrosoft\b", r"\bmsft\b"],
    "NETFLIX": [r"\bnetflix\b"],
    "NIKE": [r"\bnike\b"],
    "PAYPAL": [r"\bpaypal\b"],
    "PEPSICO": [r"\bpepsico\b", r"\bpepsi\b"],
    "PFIZER": [r"\bpfizer\b"],
    "ULTABEAUTY": [r"\bulta(\s*beauty)?\b"],
    "VERIZON": [r"\bverizon\b"],
    "WALMART": [r"\bwalmart\b", r"\bwal-mart\b"],
}

def detect_query_company(query):
    q_lower = query.lower()
    for canonical_name, patterns in FINANCEBENCH_COMPANIES.items():
        for p in patterns:
            if re.search(p, q_lower):
                return canonical_name
    return None

def detect_query_statement(query):
    q_lower = query.lower()
    if "cash flow" in q_lower or "cash flows" in q_lower or "capital expenditure" in q_lower or "capex" in q_lower:
        return "cash_flow_statement"
    elif "balance sheet" in q_lower or "net ppne" in q_lower or "property, plant and equipment" in q_lower or "merchandise inventories" in q_lower:
        return "balance_sheet"
    elif "income statement" in q_lower or "statement of operations" in q_lower or "net sales" in q_lower:
        return "income_statement"
    elif "cover page" in q_lower or "registered debt" in q_lower or bool(
        re.search(
            r"(?:debt.*registered|registered.*debt|securities.*registered|registered.*securities|registered\s+to\s+trade|trading\s+symbol|section\s*12\(b\)|title\s+of\s+each\s+class)",
            q_lower
        )
    ):
        return "cover_page"
    return None

def detect_query_periods(query):
    found = set(re.findall(r"\b(201\d|202\d)\b", query))
    fy_found = re.findall(r"\bfy\s*(20\d\d|\d\d)\b", query.lower())
    for fy in fy_found:
        if len(fy) == 2:
            found.add(f"20{fy}")
        else:
            found.add(fy)
    return list(found)

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

    period = None
    m = re.search(r"(201\d|202\d)", doc_upper)
    if m:
        period = m.group(1)
    return comp, period

# Load data
questions_map = {}
with open(q_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            questions_map[d['financebench_id']] = d

ckpt_records = []
with open(c_ckpt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            ckpt_records.append(json.loads(line))

failures = [r for r in ckpt_records if r['best_rank'] > 10]

# Check PDFs
pdf_cache = {}
for f in failures:
    doc = f['doc_name']
    if doc not in pdf_cache:
        p_path = os.path.join(pdf_dir, f"{doc}.pdf")
        if os.path.exists(p_path):
            doc_pdf = fitz.open(p_path)
            pdf_cache[doc] = len(doc_pdf)
            doc_pdf.close()
        else:
            pdf_cache[doc] = -1

# Load gold chunks
failure_doc_pages = set()
for f in failures:
    doc = f['doc_name']
    for p in f.get('target_pages', []):
        failure_doc_pages.add((doc, p))

gold_chunks_found = defaultdict(list)
with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        doc = c.get('doc_name')
        page = c.get('page_number')
        if page is None:
            m = re.search(r'_p(\d+)_', c.get('chunk_id', ''))
            if m:
                page = int(m.group(1))
            else:
                page = c.get('page')
        else:
            page = int(page)
        
        if (doc, page) in failure_doc_pages:
            gold_chunks_found[(doc, page)].append(c)

print("--- Checking Evidence Text in Extracted Chunks ---")
evidence_match_counts = {'exact': 0, 'partial': 0, 'none': 0}

analyzed_rows = []
for f in failures:
    qid = f['financebench_id']
    q_data = questions_map.get(qid, {})
    doc_name = f['doc_name']
    target_pages = f.get('target_pages', [])
    q_text = q_data.get('question', f.get('question', ''))
    evidence_list = q_data.get('evidence', [])
    in_pool = bool(f.get('candidate_pool_hit') == 1 or f.get('best_rank_in_pool', 999) <= 50)
    pool_rank = f.get('best_rank_in_pool', 999)
    final_rank = f.get('best_rank', 999)
    top3 = f.get('top3_retrieved', [])

    # Check evidence coverage
    target_chunks = []
    combined_chunk_text = ""
    for tp in target_pages:
        chs = gold_chunks_found.get((doc_name, tp), [])
        target_chunks.extend(chs)
        for ch in chs:
            combined_chunk_text += " " + ch.get('text', '')

    evidence_texts = [e.get('evidence_text', '') for e in evidence_list if e.get('evidence_text')]
    match_status = "none"
    clean_chunk = re.sub(r'\s+', ' ', combined_chunk_text).lower()
    
    any_exact = False
    any_partial = False
    for ev in evidence_texts:
        clean_ev = re.sub(r'\s+', ' ', ev).strip().lower()
        if len(clean_ev) > 25 and clean_ev[:35] in clean_chunk:
            any_exact = True
        else:
            # check word overlap
            ev_words = set(re.findall(r'\b\w{4,}\b', clean_ev))
            ch_words = set(re.findall(r'\b\w{4,}\b', clean_chunk))
            if ev_words and len(ev_words & ch_words) / len(ev_words) > 0.5:
                any_partial = True

    if any_exact:
        match_status = "exact"
        evidence_match_counts['exact'] += 1
    elif any_partial:
        match_status = "partial"
        evidence_match_counts['partial'] += 1
    else:
        match_status = "none"
        evidence_match_counts['none'] += 1

    # Company match check
    detected_comp = detect_query_company(q_text)
    target_comp, target_period = extract_chunk_company_and_period(doc_name)
    company_detected_correctly = (detected_comp == target_comp)

    # Statement intent check
    detected_stmt = detect_query_statement(q_text)

    # Period match check
    detected_periods = detect_query_periods(q_text)
    period_match = (target_period in detected_periods) if target_period else False

    # Check top3 contamination
    top3_companies = []
    top3_periods = []
    top3_has_target_comp = False
    top3_has_target_doc = False
    for t3_cid in top3:
        t3_doc = t3_cid.split('_p')[0]
        c_comp, c_per = extract_chunk_company_and_period(t3_doc)
        top3_companies.append(c_comp)
        top3_periods.append(c_per)
        if c_comp == target_comp:
            top3_has_target_comp = True
        if t3_doc == doc_name:
            top3_has_target_doc = True

    analyzed_rows.append({
        'qid': qid,
        'doc_name': doc_name,
        'target_pages': target_pages,
        'num_pages': len(target_pages),
        'in_pool': in_pool,
        'pool_rank': pool_rank,
        'final_rank': final_rank,
        'match_status': match_status,
        'chunks_count': len(target_chunks),
        'detected_comp': detected_comp,
        'target_comp': target_comp,
        'company_correct': company_detected_correctly,
        'detected_stmt': detected_stmt,
        'target_period': target_period,
        'detected_periods': detected_periods,
        'period_match': period_match,
        'top3_has_target_comp': top3_has_target_comp,
        'top3_has_target_doc': top3_has_target_doc,
        'top3_companies': top3_companies,
        'question': q_text
    })

print(f"Evidence match status across 64 failures: {evidence_match_counts}")

# Failure classification breakdown
print("\n--- Detailed Breakdown ---")
comp_missed = [r for r in analyzed_rows if not r['company_correct']]
print(f"Questions where company was NOT detected correctly: {len(comp_missed)}")
for r in comp_missed:
    print(f"  {r['qid']}: Target='{r['target_comp']}', Detected='{r['detected_comp']}' | Q: {r['question'][:80]}")

multi_page_absent = [r for r in analyzed_rows if r['num_pages'] > 1 and not r['in_pool']]
print(f"Multi-page absent from pool: {len(multi_page_absent)}")

in_pool_rerank_fail = [r for r in analyzed_rows if r['in_pool']]
print(f"In-pool rerank failures: {len(in_pool_rerank_fail)}")

same_comp_wrong_doc = [r for r in analyzed_rows if r['top3_has_target_comp'] and not r['top3_has_target_doc']]
print(f"Top 3 retrieved same company but WRONG filing/period: {len(same_comp_wrong_doc)}")

cross_comp_contamination = [r for r in analyzed_rows if not r['top3_has_target_comp']]
print(f"Top 3 retrieved entirely WRONG company (cross-company contamination): {len(cross_comp_contamination)}")
