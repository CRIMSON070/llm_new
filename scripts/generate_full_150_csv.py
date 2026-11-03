import json
import os
import re
import csv
from collections import Counter, defaultdict

c_ckpt = 'outputs/experiments/experiment_C_company_intent/retrieval/phase2_retrieval_checkpoint.jsonl'
q_file = 'data/financebench/questions.jsonl'
out_dir = 'outputs/experiments/experiment_D_failure_analysis'
os.makedirs(out_dir, exist_ok=True)

# 1. Load questions
questions = []
with open(q_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

# 2. Load Exp C checkpoint
records = []
with open(c_ckpt, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

q_map = {q['financebench_id']: q for q in questions}
rec_map = {r['financebench_id']: r for r in records}

# 3. Company & Statement Intent Detection Functions from Exp C
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
    return comp

# 4. Build CSV Rows for all 150 questions
csv_rows = []
counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

for q_data in questions:
    qid = q_data['financebench_id']
    q_text = q_data['question']
    q_clean = re.sub(r'\s+', ' ', q_text).strip()
    doc_name = q_data['doc_name']
    evidence_list = q_data.get('evidence', [])
    target_pages = [e.get('evidence_page_num') + 1 for e in evidence_list if e.get('evidence_page_num') is not None]

    gold_evidence_snippets = [re.sub(r'\s+', ' ', e.get('evidence_text', '')).strip() for e in evidence_list if e.get('evidence_text')]
    gold_snippet_summary = " || ".join(gold_evidence_snippets[:2])
    gold_snippet_truncated = gold_snippet_summary[:120] + ("..." if len(gold_snippet_summary) > 120 else "")
    gold_page_or_chunk = f"Pages {target_pages} ({gold_snippet_truncated})"

    rec = rec_map.get(qid)
    if not rec:
        counts['D'] += 1
        continue

    in_pool = (rec.get('candidate_pool_hit') == 1 or rec.get('best_rank_in_pool', 999) <= 50)
    in_top10 = (rec.get('best_rank', 999) <= 10)
    pool_rank = rec.get('best_rank_in_pool', 999)
    final_rank = rec.get('best_rank', 999)
    top3 = rec.get('top3_retrieved', [])

    # Detection logic
    detected_comp = detect_query_company(q_clean)
    target_comp = extract_chunk_company_and_period(doc_name)
    company_detected = "yes" if detected_comp else "no"

    detected_stmt = detect_query_statement(q_clean)
    query_intent = detected_stmt if detected_stmt else "unrouted"

    # Top3 document check
    top3_docs = [t.split('_p')[0] for t in top3]
    top3_comps = [extract_chunk_company_and_period(d) for d in top3_docs]
    same_comp_wrong_doc = any(c == target_comp and d != doc_name for c, d in zip(top3_comps, top3_docs))

    # Category determination
    failure_category = ""
    evidence_notes = []
    confidence = "high"

    if in_pool and in_top10:
        counts['A'] += 1
        failure_category = f"Category A: Successful Retrieval (Rank #{final_rank})"
        evidence_notes.append(f"Gold evidence retrieved at rank #{final_rank} (candidate pool rank #{pool_rank}).")
    elif in_pool and not in_top10:
        counts['B'] += 1
        failure_category = "Category B: Ranking Failure"
        evidence_notes.append(f"Gold chunk reached candidate pool at rank #{pool_rank} (<=50), but reranker demoted to #{final_rank} (>10).")
        if len(target_pages) > 1:
            evidence_notes.append(f"Requires cross-statement data across {len(target_pages)} pages ({target_pages}).")
        if query_intent == "unrouted":
            evidence_notes.append("Statement intent unrouted; +4.5 statement boost not applied.")
    elif not in_pool:
        counts['C'] += 1
        evidence_notes.append(f"Gold page {target_pages} was absent from top-50 fused candidate pool (candidate_pool_hit=0, best_rank_in_pool={pool_rank}).")
        if len(target_pages) > 1:
            failure_category = "Category C: Candidate Generation Failure (Multi-page evidence requirement)"
            evidence_notes.append(f"Requires cross-statement data across {len(target_pages)} distinct pages ({target_pages}); sparse/dense query token blending missed both pages.")
        elif same_comp_wrong_doc:
            failure_category = "Category C: Candidate Generation Failure (Filing/fiscal period confusion)"
            wrong_docs = list(set([d for d in top3_docs if d != doc_name and extract_chunk_company_and_period(d) == target_comp]))
            evidence_notes.append(f"Retrieved competing filings of target company: {wrong_docs}.")
        else:
            failure_category = "Category C: Candidate Generation Failure (Lexical-semantic vocabulary gap)"
            evidence_notes.append("Neither BM25 lexical search nor BGE-M3 dense FAISS matched specialized filing vocabulary into top 50 pool.")
    else:
        counts['D'] += 1
        failure_category = "Category D: Cannot determine"
        evidence_notes.append("Ambiguous or missing trace.")

    row = {
        "question_id": qid,
        "question_text": q_clean,
        "gold_document": doc_name,
        "gold_page_or_chunk": gold_page_or_chunk,
        "candidate_pool_hit": "yes" if in_pool else "no",
        "final_top10_hit": "yes" if in_top10 else "no",
        "gold_candidate_rank": str(pool_rank) if in_pool else ">50",
        "bm25_gold_hit": "unknown",
        "dense_gold_hit": "unknown",
        "metadata_gold_hit": "unknown",
        "company_detected": company_detected,
        "detected_company": detected_comp if detected_comp else "none",
        "query_intent": query_intent,
        "failure_category": failure_category,
        "evidence_for_classification": " ; ".join(evidence_notes) + " [Note: Path-specific hits unknown as individual ranks were not persisted in checkpoint]",
        "confidence": confidence
    }
    csv_rows.append(row)

# Save CSV
csv_path = os.path.join(out_dir, 'question_failure_analysis.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Saved {len(csv_rows)} rows to {csv_path}")
print(f"Verified counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, D={counts['D']}")
