import json
import os
import re
from collections import Counter, defaultdict
import fitz # PyMuPDF

c_ckpt = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
q_file = 'data/financebench/questions.jsonl'
chunks_path = 'data/financebench/processed/phase2_enriched_chunks.jsonl'
pdf_dir = 'data/financebench/pdfs'

# Company aliases dictionary from Exp C
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
with open(q_file, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f if line.strip()]

with open(c_ckpt, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

q_map = {q['financebench_id']: q for q in questions}

# Check Category C questions (47)
cat_c_records = [r for r in records if r.get('candidate_pool_hit') == 0 and r.get('best_rank_in_pool', 999) > 50]
print(f"Loaded {len(cat_c_records)} Category C records.")

# Check PDF existence and page bounds for all Category C documents
pdf_bounds = {}
for r in cat_c_records:
    doc = r['doc_name']
    if doc not in pdf_bounds:
        p_path = os.path.join(pdf_dir, f"{doc}.pdf")
        if os.path.exists(p_path):
            doc_pdf = fitz.open(p_path)
            pdf_bounds[doc] = len(doc_pdf)
            doc_pdf.close()
        else:
            pdf_bounds[doc] = -1

# Check chunks for all target pages in Category C
target_pages_set = set()
for r in cat_c_records:
    doc = r['doc_name']
    for p in r.get('target_pages', []):
        target_pages_set.add((doc, p))

chunks_by_page = defaultdict(list)
with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        doc = c.get('doc_name')
        page = c.get('page_number')
        if page is None:
            m = re.search(r'_p(\d+)_', c.get('chunk_id', ''))
            page = int(m.group(1)) if m else c.get('page')
        else:
            page = int(page)
        if (doc, page) in target_pages_set:
            chunks_by_page[(doc, page)].append(c)

print(f"Target pages in Cat C: {len(target_pages_set)}. Pages with chunks: {len(chunks_by_page)}")
assert len(target_pages_set) == len(chunks_by_page), "Missing chunks on target page!"

# Analyze each Category C question
reasons_counter = Counter()
cat_c_diagnoses = []

for r in cat_c_records:
    qid = r['financebench_id']
    q_data = q_map[qid]
    q_text = q_data['question']
    doc_name = r['doc_name']
    target_pages = r.get('target_pages', [])
    top3 = r.get('top3_retrieved', [])

    # 1. Company Detection
    detected_company = detect_query_company(q_text)
    target_company, target_period = extract_chunk_company_and_period(doc_name)
    company_detected = (detected_company is not None)
    company_match = (detected_company == target_company)

    # 2. Query Intent / Statement Routing
    detected_stmt = detect_query_statement(q_text)
    statement_detected = (detected_stmt is not None)

    # 3. Multi-page requirement
    is_multi_page = (len(target_pages) > 1)

    # 4. Period confusion / Filing confusion in top-3
    top3_docs = [t.split('_p')[0] for t in top3]
    top3_comps = [extract_chunk_company_and_period(d)[0] for d in top3_docs]
    top3_periods = [extract_chunk_company_and_period(d)[1] for d in top3_docs]
    
    same_comp_wrong_doc = any(c == target_company and d != doc_name for c, d in zip(top3_comps, top3_docs))
    cross_company = all(c != target_company for c in top3_comps) if top3_comps else False

    # 5. Question Type
    q_type = q_data.get('question_type', '')
    q_reasoning = q_data.get('question_reasoning', '')

    # Assign primary cause
    primary_causes = []
    evidence_notes = []

    if is_multi_page:
        primary_causes.append("Query requires multiple evidence passages")
        evidence_notes.append(f"Requires cross-statement evidence across {len(target_pages)} pages ({target_pages})")
    
    if same_comp_wrong_doc:
        primary_causes.append("Filing / fiscal period confusion in candidate generation")
        wrong_filings = list(set([d for d in top3_docs if d != doc_name and extract_chunk_company_and_period(d)[0] == target_company]))
        evidence_notes.append(f"Retrieved competing filings of same company: {wrong_filings}")

    if not statement_detected:
        evidence_notes.append("Statement intent unrouted (synonym not in regex)")

    if not company_detected:
        primary_causes.append("Company/entity detection failure")
        evidence_notes.append(f"Company '{target_company}' not detected in query text")

    if not primary_causes:
        primary_causes.append("Lexical-semantic vocabulary gap in sparse/dense retrieval")
        evidence_notes.append("Neither BM25 lexical terms nor BGE-M3 dense embeddings retrieved gold page into top 50")

    cat_c_diagnoses.append({
        'qid': qid,
        'doc_name': doc_name,
        'target_pages': target_pages,
        'num_pages': len(target_pages),
        'question': q_text,
        'company': target_company,
        'detected_company': detected_company,
        'company_match': company_match,
        'detected_stmt': detected_stmt,
        'top3_retrieved': top3,
        'primary_causes': primary_causes,
        'evidence_notes': "; ".join(evidence_notes)
    })
    for c in primary_causes:
        reasons_counter[c] += 1

print("\n--- Category C Primary Failure Distribution ---")
for k, v in reasons_counter.most_common():
    print(f"  {k:55s}: {v:2d} questions ({v/len(cat_c_records)*100:.1f}%)")
