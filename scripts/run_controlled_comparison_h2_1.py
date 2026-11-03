"""
scripts/run_controlled_comparison_h2_1.py
=========================================
FinSage Experiment H2.1: Correctness Fixes Only
Controlled Head-to-Head Comparison:
- Reference Baseline: Experiment F (Period-Aware Candidate Generation, K_pool=50)
- Experiment H2 Reference: Prior statement-aware run with known audit defects
- Experiment H2.1: Correctness Fixes Only:
  1. Strict Filing Identity & Period Validation:
     - Strict company identity matching before injecting statement chunks.
     - Exact fiscal year and form type matching (distinguishing 10-K from 10-Q).
     - Zero competitor injections; ambiguous cases safely skipped.
  2. Primary Consolidated Statement Selection:
     - Select chunks containing primary consolidated financial-statement headings and structure.
     - Completely eliminate preliminary MD&A / footnote table slicing bugs.
     - Resolve the single regression on financebench_id_10130 (retaining both p.70 and p.72).
  3. Quota preserved: strictly Q <= 6 statement slots, preserving >= 44 standard RRF candidates.
  4. Complete audit trails: filing_selection_audit.csv and statement_selection_audit.csv.

Outputs saved exclusively to:
outputs/experiments/experiment_H2_1_correctness_fixes/
"""

import os
import sys
import json
import time
import re
import csv
import hashlib
import importlib.util
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

# -------------------------------------------------------------------------
# Dynamic import of Experiment C's Phase 2 Reranker (Preserves baseline)
# -------------------------------------------------------------------------
EXP_C_RERANKER_PATH = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_C_company_intent", "source_backup", "phase2_reranker.py"
)
spec = importlib.util.spec_from_file_location("exp_c_reranker", EXP_C_RERANKER_PATH)
exp_c_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_c_mod)
Phase2Reranker = exp_c_mod.Phase2Reranker


# -------------------------------------------------------------------------
# Strict Company & Document Parsing
# -------------------------------------------------------------------------
DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")

COMPANY_MAP = {
    "3M": "3M",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ACTIVISIONBLIZZARD",
    "Adobe": "ADOBE",
    "Amazon": "AMAZON",
    "Amcor": "AMCOR",
    "American Express": "AMERICANEXPRESS",
    "American Water Works": "AMERICANWATERWORKS",
    "Best Buy": "BESTBUY",
    "Block": "BLOCK",
    "Boeing": "BOEING",
    "CVS Health": "CVSHEALTH",
    "Coca-Cola": "COCACOLA",
    "Corning": "CORNING",
    "Costco": "COSTCO",
    "Foot Locker": "FOOTLOCKER",
    "General Mills": "GENERALMILLS",
    "JPMorgan": "JPMORGAN",
    "Johnson & Johnson": "JOHNSON_JOHNSON",
    "Kraft Heinz": "KRAFTHEINZ",
    "Lockheed Martin": "LOCKHEEDMARTIN",
    "MGM Resorts": "MGMRESORTS",
    "Microsoft": "MICROSOFT",
    "Netflix": "NETFLIX",
    "Nike": "NIKE",
    "Paypal": "PAYPAL",
    "PepsiCo": "PEPSICO",
    "Pfizer": "PFIZER",
    "Ulta Beauty": "ULTABEAUTY",
    "Verizon": "VERIZON",
    "Walmart": "WALMART",
}

def parse_doc_meta(doc_name: str) -> Optional[Dict[str, Any]]:
    if not doc_name:
        return None
    m = DOC_PARSE_REGEX.match(doc_name)
    if m:
        comp, yr, qtr, ftype = m.groups()
        return {
            "company": comp.upper(),
            "year": int(yr),
            "quarter": int(qtr) if qtr else None,
            "ftype": ftype.upper()
        }
    return None


# -------------------------------------------------------------------------
# Temporal Intent Extraction Regexes & Document Parsing (Extended for 2-digit FY)
# -------------------------------------------------------------------------
FY_PATTERN = re.compile(r"(?:FY\s*|fiscal\s*year\s*)(\d{4})|(\d{4})\s*fiscal\s*year", re.IGNORECASE)
FY2_PATTERN = re.compile(r"\bFY\s*(\d{2})\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:FY\s*|Q[1-4]\s*(?:of\s*)?FY\s*|\b)(20\d{2})\b", re.IGNORECASE)
QUARTER_PATTERN = re.compile(r"\bQ([1-4])(?:\s*(?:of\s*)?(?:FY\s*)?(\d{2,4}))?\b", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(?:FY\s*)?(\d{2,4})\s*(?:-|to|and)\s*(?:FY\s*)?(\d{2,4})", re.IGNORECASE)

def detect_query_periods(query: str) -> Dict[str, Any]:
    q_lower = query.lower()

    ranges = RANGE_PATTERN.findall(query)
    range_years = set()
    for y1, y2 in ranges:
        y1_i = int(y1) if len(y1) == 4 else 2000 + int(y1)
        y2_i = int(y2) if len(y2) == 4 else 2000 + int(y2)
        range_years.add(y1_i)
        range_years.add(y2_i)
        if 1 <= abs(y2_i - y1_i) <= 5:
            low, high = min(y1_i, y2_i), max(y1_i, y2_i)
            for y in range(low, high + 1):
                range_years.add(y)

    quarters = set()
    q_matches = QUARTER_PATTERN.findall(query)
    for q_num, q_yr in q_matches:
        quarters.add(int(q_num))
        if q_yr:
            yr_int = int(q_yr) if len(q_yr) == 4 else 2000 + int(q_yr)
            range_years.add(yr_int)

    fy_years = set()
    for m in FY_PATTERN.finditer(query):
        for g in m.groups():
            if g:
                fy_years.add(int(g))

    for m in FY2_PATTERN.finditer(query):
        fy_years.add(2000 + int(m.group(1)))

    all_years = set(int(y) for y in YEAR_PATTERN.findall(query))
    combined_years = range_years.union(fy_years).union(all_years)

    is_forward = any(w in q_lower for w in ["guidance", "expected", "accelerate", "forecast", "outlook", "projection"])

    return {
        "years": combined_years,
        "quarters": quarters,
        "has_explicit_fy": len(fy_years) > 0,
        "is_range": len(ranges) > 0 or len(combined_years) > 1,
        "is_forward_looking": is_forward
    }


def is_chunk_in_period(doc_meta: Optional[Dict[str, Any]], period_info: Dict[str, Any]) -> bool:
    if not doc_meta or not period_info:
        return False

    req_years = period_info["years"]
    req_quarters = period_info["quarters"]

    if not req_years and not req_quarters:
        return False

    chunk_year = doc_meta["year"]
    chunk_quarter = doc_meta["quarter"]

    year_match = False
    if req_years:
        if chunk_year in req_years:
            year_match = True
        elif period_info.get("is_forward_looking") and doc_meta.get("quarter") == 4 and (chunk_year + 1) in req_years:
            year_match = True
    else:
        year_match = True

    quarter_match = True
    if req_quarters:
        if chunk_quarter is not None:
            quarter_match = (chunk_quarter in req_quarters)
        else:
            quarter_match = True

    return year_match and quarter_match


# -------------------------------------------------------------------------
# Statement Intent Detection
# -------------------------------------------------------------------------
BS_PATTERNS = re.compile(
    r"\b(balance\s*shee?ts?|statements?\s*of\s*financial\s*position|working\s*capital|quick\s*ratio|"
    r"current\s*ratio|current\s*assets|current\s*liabilities|merchandise\s*inventories|inventories\s*balance|"
    r"property[,\s]*plant\s*and\s*equipment|net\s*pp&?e|net\s*ppne|accounts\s*payable|accounts\s*receivable|"
    r"shareholders?\s*equity|stockholders?\s*equity|total\s*equity|largest\s*liability|debt\s*on\s*balance\s*sheet)\b",
    re.IGNORECASE
)

IS_PATTERNS = re.compile(
    r"\b(income\s*statements?|statements?\s*of\s*operations?|statements?\s*of\s*earnings|operating\s*margin|"
    r"gross\s*margin|net\s*margin|net\s*income|operating\s*income|net\s*sales|total\s*revenues?|ebitda?|"
    r"cogs|cost\s*of\s*goods\s*sold|operating\s*expenses|effective\s*tax\s*rate|restructuring\s*costs)\b",
    re.IGNORECASE
)

CF_PATTERNS = re.compile(
    r"\b(cash\s*flows?|statements?\s*of\s*cash\s*flows?|operating\s*cash\s*flow|free\s*cash\s*flow|"
    r"capital\s*expenditures?|capex|purchases\s*of\s*property)\b",
    re.IGNORECASE
)

def detect_statement_intents(query: str) -> List[str]:
    intents = []
    if BS_PATTERNS.search(query):
        intents.append("balance_sheet")
    if IS_PATTERNS.search(query):
        intents.append("income_statement")
    if CF_PATTERNS.search(query):
        intents.append("cash_flow_statement")
    return intents


# -------------------------------------------------------------------------
# Fix A: Strict Target Filing Resolution & Disambiguation
# -------------------------------------------------------------------------
def resolve_target_filing_strict(
    q_data: Dict[str, Any],
    p_info: Dict[str, Any],
    all_filings: List[str],
    doc_meta_cache: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], str, bool, bool]:
    """
    Returns: (selected_filing, reason, company_match, period_match)
    Strict rules:
    1. Canonical company matching: only filings matching question company prefix are considered.
    2. Period matching: fiscal year must match; 10-K required if annual, 10-Q required if quarterly.
    3. If multiple filings or unverified, returns None and preserves baseline candidate pool.
    """
    comp_name = q_data.get("company", "")
    canonical_prefix = COMPANY_MAP.get(comp_name)
    if not canonical_prefix:
        return None, f"UNKNOWN_COMPANY: {comp_name}", False, False

    comp_filings = [d for d in all_filings if doc_meta_cache.get(d, {}).get("company") == canonical_prefix]
    if not comp_filings:
        return None, f"NO_FILINGS_FOR_COMPANY: {canonical_prefix}", False, False

    years = p_info.get("years", set())
    quarters = p_info.get("quarters", set())

    if not years:
        return None, "NO_PERIOD_DETECTED", True, False

    target_year = max(years)
    req_form = "10Q" if quarters else "10K"
    req_quarter = list(quarters)[0] if quarters else None

    matches = []
    for d in comp_filings:
        meta = doc_meta_cache.get(d, {})
        m_yr = meta.get("year")
        m_qtr = meta.get("quarter")
        m_form = meta.get("ftype")

        if m_yr != target_year:
            continue

        if req_form == "10K":
            if "10K" in m_form:
                matches.append(d)
        elif req_form == "10Q":
            if "10Q" in m_form and (req_quarter is None or m_qtr == req_quarter):
                matches.append(d)

    if len(matches) == 1:
        return matches[0], "VERIFIED_EXACT_MATCH", True, True
    elif len(matches) > 1:
        return None, f"AMBIGUOUS_MULTIPLE_MATCHES: {matches}", True, False
    else:
        return None, f"NO_MATCHING_FILING (yr={target_year}, form={req_form}, qtr={req_quarter})", True, False


# -------------------------------------------------------------------------
# Fix B: Primary Statement Heading & Structure Qualification
# -------------------------------------------------------------------------
RE_BS_TITLE = re.compile(
    r"(?:Consolidated\s+Balance\s+Sheets?|"
    r"Consolidated\s+Statements?\s+of\s+Financial\s+Position|"
    r"Balance\s+Sheets?,\s+Consolidated|"
    r"\bBalance\s+Sheets?\b|"
    r"\bStatements?\s+of\s+Financial\s+Position\b)",
    re.IGNORECASE
)

RE_IS_TITLE = re.compile(
    r"(?:Consolidated\s+Statements?\s+of\s+(?:Income|Operations?|Earnings|Loss)|"
    r"Consolidated\s+Income\s+Statements?|"
    r"Statements?\s+of\s+(?:Income|Operations?|Earnings),\s+Consolidated|"
    r"\bStatements?\s+of\s+(?:Income|Operations?|Earnings|Loss)\b)",
    re.IGNORECASE
)

RE_CF_TITLE = re.compile(
    r"(?:Consolidated\s+Statements?\s+of\s+Cash\s+Flows?|"
    r"Consolidated\s+Cash\s+Flows?\s+Statements?|"
    r"Statements?\s+of\s+Cash\s+Flows?,\s+Consolidated|"
    r"\bStatements?\s+of\s+Cash\s+Flows?\b)",
    re.IGNORECASE
)

def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def evaluate_page_for_statement(page_chunks: List[Dict[str, Any]], stype: str) -> Tuple[bool, float, str]:
    if not page_chunks:
        return False, 0.0, "EMPTY"

    c0 = sorted(page_chunks, key=lambda x: x.get('chunk_id'))[0]
    txt0 = norm_ws(c0.get('text', ''))[:400]
    full_page_txt = norm_ws(" ".join(c.get('text', '') for c in page_chunks))
    lower_txt = full_page_txt.lower()

    # Reject MD&A discussion tables or preliminary footnotes
    if "management's discussion" in lower_txt[:300] or "management assessment" in lower_txt[:300]:
        return False, 0.0, "REJECT_MDA"
    if "the following table presents" in lower_txt[:300]:
        return False, 0.0, "REJECT_PRELIM_TABLE"

    score = 10.0
    matched_header = ""

    if stype == "balance_sheet":
        m = RE_BS_TITLE.search(txt0)
        if not m:
            return False, 0.0, "NO_HEADER"
        matched_header = m.group(0)
        if "assets" in lower_txt: score += 2
        if "current assets" in lower_txt or "cash and cash equivalents" in lower_txt: score += 4
        if "liabilities" in lower_txt: score += 2
        if "current liabilities" in lower_txt or "accounts payable" in lower_txt: score += 4
        if "shareholders' equity" in lower_txt or "stockholders' equity" in lower_txt or "total equity" in lower_txt: score += 3
        if "total assets" in lower_txt: score += 4
    elif stype == "income_statement":
        m = RE_IS_TITLE.search(txt0)
        if not m:
            return False, 0.0, "NO_HEADER"
        matched_header = m.group(0)
        if "revenue" in lower_txt or "net sales" in lower_txt: score += 4
        if "cost of sales" in lower_txt or "cost of goods sold" in lower_txt or "cost of products" in lower_txt: score += 4
        if "operating income" in lower_txt or "operating expenses" in lower_txt or "gross margin" in lower_txt or "gross profit" in lower_txt: score += 3
        if "net income" in lower_txt or "net earnings" in lower_txt: score += 3
        if "basic" in lower_txt and "diluted" in lower_txt: score += 3
    elif stype == "cash_flow_statement":
        m = RE_CF_TITLE.search(txt0)
        if not m:
            return False, 0.0, "NO_HEADER"
        matched_header = m.group(0)
        if "operating activities" in lower_txt: score += 4
        if "investing activities" in lower_txt: score += 4
        if "financing activities" in lower_txt: score += 4
        if "net cash" in lower_txt or "cash and cash equivalents" in lower_txt: score += 3

    is_primary = (score >= 18.0)
    return is_primary, score, matched_header


def build_primary_statement_catalog(chunk_store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds a verified catalog of PRIMARY financial statement chunks.
    Groups chunks by filing and page, evaluates page-level heading and line-item structure.
    catalog[doc_name][stype] = [chunk_ids on verified primary statement pages]
    """
    doc_page_chunks = defaultdict(lambda: defaultdict(list))
    for cid, c in chunk_store.items():
        doc_page_chunks[c.get('doc_name')][c.get('page_number')].append(c)

    catalog = defaultdict(lambda: defaultdict(list))

    for doc, pages in doc_page_chunks.items():
        for stype in ["balance_sheet", "income_statement", "cash_flow_statement"]:
            best_page = None
            best_score = 0.0
            
            for p, clist in pages.items():
                is_prim, sc, _ = evaluate_page_for_statement(clist, stype)
                if is_prim and sc > best_score:
                    best_score = sc
                    best_page = p

            if best_page is not None:
                sorted_cids = [c['chunk_id'] for c in sorted(pages[best_page], key=lambda x: x.get('chunk_id'))]
                catalog[doc][stype] = sorted_cids

    return catalog


# -------------------------------------------------------------------------
# Benchmark Execution Engine
# -------------------------------------------------------------------------
def run_benchmark_arm_h2_1(
    mode: str,
    questions: List[Dict[str, Any]],
    mpr: MultiPathRetriever,
    reranker: Phase2Reranker,
    chunk_store: Dict[str, Dict[str, Any]],
    doc_meta_cache: Dict[str, Dict[str, Any]],
    all_filings: List[str],
    primary_catalog: Dict[str, Dict[str, List[str]]],
    out_dir: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_file = os.path.join(out_dir, "phase2_retrieval_checkpoint.jsonl")
    checkpoint_out = open(checkpoint_file, "w", encoding="utf-8")

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    recall_at_10 = []
    candidate_pool_hits = []

    per_question_records = []
    filing_audit_records = []
    statement_audit_records = []

    t_start = time.time()
    print(f"\n[{mode.upper()}] Starting benchmark run across N={len(questions)} questions...")

    for idx, q_data in enumerate(questions, start=1):
        f_id = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        gold_doc = q_data["doc_name"]
        gold_comp = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence", [])

        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        # Multi-Path Retrieval (depth=50 across all paths)
        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=50)
        meta_res, _ = mpr.retrieve_metadata(q_text, k_sections=3)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res,
            "path3_metadata": meta_res
        }

        # RRF Fusion (k=60)
        p_info = detect_query_periods(q_text)
        rrf_scores = defaultdict(float)
        path_contributions = defaultdict(dict)
        for path_name, items in path_dict.items():
            for rank, (cid, raw_score) in enumerate(items, start=1):
                inc = 1.0 / (60 + rank)
                rrf_scores[cid] += inc
                path_contributions[cid][path_name] = {
                    "rank": rank,
                    "raw_score": round(raw_score, 4),
                    "rrf_contribution": round(inc, 6)
                }

        sorted_all = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        in_period_cids = []
        other_cids = []

        if p_info["years"] or p_info["quarters"]:
            for cid, score in sorted_all:
                c = chunk_store.get(cid, {})
                d = c.get("doc_name")
                d_meta = doc_meta_cache.get(d)
                if is_chunk_in_period(d_meta, p_info):
                    in_period_cids.append(cid)
                else:
                    other_cids.append(cid)
            base_fused_cids = (in_period_cids + other_cids)[:50]
        else:
            base_fused_cids = [cid for cid, score in sorted_all[:50]]

        # Candidate Selection Branching
        injected_cids = []
        displaced_cids = []
        stmt_intents = detect_statement_intents(q_text)

        if mode == "reference":
            # Exact Experiment F Reference
            fused_cids = base_fused_cids
            sel_filing, reason, comp_m, period_m = None, "REFERENCE_EXP_F", False, False
        elif mode == "experiment_h2_1":
            if not stmt_intents:
                # Narrative query: 100% bit-for-bit identical to Exp F
                fused_cids = base_fused_cids
                sel_filing, reason, comp_m, period_m = None, "NONE_NARRATIVE_QUERY", True, True
            else:
                # Statement Query: Strict Filing Resolution (Fix A)
                sel_filing, reason, comp_m, period_m = resolve_target_filing_strict(
                    q_data, p_info, all_filings, doc_meta_cache
                )

                eligible_stmt_cids = []
                if sel_filing and sel_filing in primary_catalog:
                    filing_stmts = primary_catalog[sel_filing]
                    if len(stmt_intents) == 1:
                        s_type = stmt_intents[0]
                        eligible_stmt_cids = filing_stmts.get(s_type, [])[:6]
                    else:
                        slots_per_type = max(1, 6 // len(stmt_intents))
                        for s_type in stmt_intents:
                            eligible_stmt_cids.extend(filing_stmts.get(s_type, [])[:slots_per_type])

                if eligible_stmt_cids:
                    # Pool assembly with guaranteed quota preservation:
                    # Reserve eligible statement candidates first
                    stmt_cids_unique = []
                    for cid in eligible_stmt_cids:
                        if cid not in stmt_cids_unique:
                            stmt_cids_unique.append(cid)

                    # Top candidates from base pool not already in statement set
                    remaining_slots = 50 - len(stmt_cids_unique)
                    kept_base_cids = []
                    for cid in base_fused_cids:
                        if cid not in stmt_cids_unique:
                            kept_base_cids.append(cid)
                            if len(kept_base_cids) >= remaining_slots:
                                break

                    fused_cids = kept_base_cids + stmt_cids_unique
                    injected_cids = [cid for cid in stmt_cids_unique if cid not in base_fused_cids]
                    displaced_cids = [cid for cid in base_fused_cids if cid not in fused_cids]
                else:
                    # Ambiguous / unverified filing or missing catalog: preserve base pool
                    fused_cids = base_fused_cids
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build candidate structures
        candidates = []
        for cid in fused_cids:
            c_data = chunk_store.get(cid, {})
            candidates.append({
                "chunk_id": cid,
                "text": c_data.get("text", ""),
                "doc_name": c_data.get("doc_name"),
                "page": c_data.get("page_number", 0),
                "rrf_score": round(rrf_scores[cid], 6),
                "path_contributions": path_contributions.get(cid, {}),
            })

        # Check candidate pool hit
        best_rank_in_pool = 999
        for c_idx, c in enumerate(candidates, start=1):
            if c.get("doc_name") == gold_doc and c.get("page") in target_pages:
                best_rank_in_pool = c_idx
                break
        cand_hit = 1 if best_rank_in_pool <= 50 else 0
        candidate_pool_hits.append(cand_hit)

        # Rerank with Phase 2 Reranker
        reranked = reranker.rerank(query=q_text, candidates=candidates, top_k=10, apply_boosts=True)

        best_rank = 999
        for r_idx, c in enumerate(reranked, start=1):
            if c.get("doc_name") == gold_doc and c.get("page") in target_pages:
                best_rank = r_idx
                break

        rr = 1.0 / best_rank if best_rank <= 10 else 0.0
        h1 = 1 if best_rank == 1 else 0
        h3 = 1 if best_rank <= 3 else 0
        h5 = 1 if best_rank <= 5 else 0
        h10 = 1 if best_rank <= 10 else 0

        reciprocal_ranks.append(rr)
        recall_at_1.append(h1)
        recall_at_3.append(h3)
        recall_at_5.append(h5)
        recall_at_10.append(h10)

        # Audit logs
        filing_audit_records.append({
            "question_id": f_id,
            "company": gold_comp,
            "target_filing": gold_doc,
            "selected_filing": sel_filing if sel_filing else "NONE",
            "is_statement_query": 1 if stmt_intents else 0,
            "company_match": "YES" if comp_m else "NO",
            "period_match": "YES" if period_m else "NO",
            "rejection_reason": reason,
            "injected_chunks_count": len(injected_cids)
        })

        statement_audit_records.append({
            "question_id": f_id,
            "detected_intent": "+".join(stmt_intents) if stmt_intents else "NONE",
            "selected_filing": sel_filing if sel_filing else "NONE",
            "injected_chunk_ids": ";".join(injected_cids) if injected_cids else "NONE",
            "primary_pages_selected": ";".join(str(chunk_store.get(cid, {}).get("page_number")) for cid in injected_cids) if injected_cids else "NONE",
            "gold_target_pages": ";".join(str(p) for p in target_pages)
        })

        record = {
            "index": idx,
            "financebench_id": f_id,
            "company": gold_comp,
            "doc_name": gold_doc,
            "sector": sector,
            "question": q_text[:70],
            "statement_intents": "+".join(stmt_intents) if stmt_intents else "NONE",
            "target_pages": target_pages,
            "best_rank": best_rank,
            "best_rank_in_pool": best_rank_in_pool,
            "candidate_pool_hit": cand_hit,
            "rr": round(rr, 4),
            "injected_chunks_count": len(injected_cids),
            "displaced_chunks_count": len(displaced_cids),
            "injected_chunk_ids": injected_cids,
            "displaced_chunk_ids": displaced_cids,
            "pool_chunk_ids": [c["chunk_id"] for c in candidates],
            "top3_retrieved": [c["chunk_id"] for c in reranked[:3]]
        }
        per_question_records.append(record)
        checkpoint_out.write(json.dumps(record) + "\n")

        if idx % 50 == 0 or idx == len(questions):
            print(f"  Processed {idx}/{len(questions)} queries...")

    checkpoint_out.close()
    elapsed = round(time.time() - t_start, 2)
    n = len(questions)

    summary = {
        "mode": mode,
        "pool_size": 50,
        "num_questions": n,
        "runtime_seconds": elapsed,
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "recall_at_1": round(sum(recall_at_1) / n * 100, 2),
        "recall_at_3": round(sum(recall_at_3) / n * 100, 2),
        "recall_at_5": round(sum(recall_at_5) / n * 100, 2),
        "recall_at_10": round(sum(recall_at_10) / n * 100, 2),
        "candidate_pool_recall": round(sum(candidate_pool_hits) / n * 100, 2),
        "candidate_pool_hits": sum(candidate_pool_hits),
        "top10_hits": sum(recall_at_10)
    }

    scorecard_path = os.path.join(out_dir, "scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, per_question_records, filing_audit_records, statement_audit_records


# -------------------------------------------------------------------------
# Main Execution Entrypoint
# -------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("FinSage Experiment H2.1: Correctness Fixes Only")
    print("=" * 80)

    OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_H2_1_correctness_fixes")
    REF_DIR = os.path.join(OUT_DIR, "reference_arm_exp_f")
    H2_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_H2_statement_aware")

    # Load questions
    questions_file = os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} FinanceBench questions.")

    # Load enriched chunks
    chunks_file = os.path.join(BASE_DIR, "data", "financebench", "processed", "phase2_enriched_chunks.jsonl")
    chunk_store = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store)} enriched chunks from corpus.")

    # Precompute doc metadata cache
    all_filings = sorted(list(set(c["doc_name"] for c in chunk_store.values() if c.get("doc_name"))))
    doc_meta_cache = {}
    for d in all_filings:
        doc_meta_cache[d] = parse_doc_meta(d)
    print(f"Parsed metadata for {len(all_filings)} filings.")

    # Build primary consolidated statement catalog (Fix B)
    print("\nBuilding Primary Consolidated Statement Catalog...")
    t_cat = time.time()
    primary_catalog = build_primary_statement_catalog(chunk_store)
    print(f"Primary catalog built in {round(time.time() - t_cat, 2)}s across {len(primary_catalog)} filings.")

    # Load shared retriever components
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading retrieval indexes and models on device: {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    dense_faiss = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    meta_retriever = MetadataRetriever(meta_faiss_dir, embeddings=embeddings)
    bm25 = BM25Retriever(bm25_phase2_dir, load_corpus=True)

    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25,
        chunk_store=chunk_store,
        metadata_retriever=meta_retriever,
        rrf_k=60
    )

    # Initialize Phase 2 Reranker with calibrated boosts
    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=True,
        statement_boost_weight=4.5,
        negative_disclosure_penalty=-3.0,
        company_boost_weight=1.5,
        period_boost_weight=1.5,
        table_boost_weight=0.5
    )

    # Run Arm 1: Reference Baseline (Experiment F)
    ref_summary, ref_records, _, _ = run_benchmark_arm_h2_1(
        mode="reference",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        all_filings=all_filings,
        primary_catalog=primary_catalog,
        out_dir=REF_DIR
    )

    # Run Arm 2: Experiment H2.1 (Correctness Fixes Only)
    h2_1_summary, h2_1_records, filing_audits, statement_audits = run_benchmark_arm_h2_1(
        mode="experiment_h2_1",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        all_filings=all_filings,
        primary_catalog=primary_catalog,
        out_dir=OUT_DIR
    )

    # Load prior Experiment H2 results for 3-way comparison
    h2_scorecard_file = os.path.join(H2_DIR, "scorecard.json")
    h2_checkpoint_file = os.path.join(H2_DIR, "phase2_retrieval_checkpoint.jsonl")
    h2_summary = {}
    h2_records_map = {}
    if os.path.exists(h2_scorecard_file):
        with open(h2_scorecard_file, "r", encoding="utf-8") as f:
            h2_summary = json.load(f)
    if os.path.exists(h2_checkpoint_file):
        with open(h2_checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                h2_records_map[rec["financebench_id"]] = rec

    # Generate Metrics Comparison Artifact
    comparison = {
        "benchmark": "FinanceBench 150",
        "timestamp": datetime.now().isoformat(),
        "reference_baseline_exp_f": ref_summary,
        "experiment_h2_prior": h2_summary,
        "experiment_h2_1_correctness_fixes": h2_1_summary,
        "delta_h2_1_vs_ref": {
            "mrr": round(h2_1_summary["mrr"] - ref_summary["mrr"], 4),
            "recall_at_1": round(h2_1_summary["recall_at_1"] - ref_summary["recall_at_1"], 2),
            "recall_at_3": round(h2_1_summary["recall_at_3"] - ref_summary["recall_at_3"], 2),
            "recall_at_5": round(h2_1_summary["recall_at_5"] - ref_summary["recall_at_5"], 2),
            "recall_at_10": round(h2_1_summary["recall_at_10"] - ref_summary["recall_at_10"], 2),
            "candidate_pool_recall": round(h2_1_summary["candidate_pool_recall"] - ref_summary["candidate_pool_recall"], 2),
            "pool_hits_diff": h2_1_summary["candidate_pool_hits"] - ref_summary["candidate_pool_hits"],
            "top10_hits_diff": h2_1_summary["top10_hits"] - ref_summary["top10_hits"]
        },
        "delta_h2_1_vs_h2": {
            "mrr": round(h2_1_summary["mrr"] - h2_summary.get("mrr", 0.0), 4),
            "recall_at_1": round(h2_1_summary["recall_at_1"] - h2_summary.get("recall_at_1", 0.0), 2),
            "recall_at_3": round(h2_1_summary["recall_at_3"] - h2_summary.get("recall_at_3", 0.0), 2),
            "recall_at_5": round(h2_1_summary["recall_at_5"] - h2_summary.get("recall_at_5", 0.0), 2),
            "recall_at_10": round(h2_1_summary["recall_at_10"] - h2_summary.get("recall_at_10", 0.0), 2),
            "candidate_pool_recall": round(h2_1_summary["candidate_pool_recall"] - h2_summary.get("candidate_pool_recall", 0.0), 2),
            "pool_hits_diff": h2_1_summary["candidate_pool_hits"] - h2_summary.get("candidate_pool_hits", 0),
            "top10_hits_diff": h2_1_summary["top10_hits"] - h2_summary.get("top10_hits", 0)
        }
    }

    with open(os.path.join(OUT_DIR, "metrics_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Question Level Comparison CSV
    csv_file = os.path.join(OUT_DIR, "question_level_comparison.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "financebench_id", "company", "doc_name", "sector",
            "statement_intents", "target_pages",
            "ref_pool_hit", "h2_pool_hit", "h2_1_pool_hit",
            "ref_rank", "h2_rank", "h2_1_rank",
            "ref_rr", "h2_rr", "h2_1_rr",
            "pool_status_vs_ref", "pool_status_vs_h2",
            "rank_delta_vs_ref", "injected_chunks_count", "displaced_chunks_count"
        ])

        for r_ref, r_h2_1 in zip(ref_records, h2_1_records):
            qid = r_ref["financebench_id"]
            r_h2 = h2_records_map.get(qid, {})

            ref_hit = r_ref["candidate_pool_hit"]
            h2_hit = r_h2.get("candidate_pool_hit", 0)
            h2_1_hit = r_h2_1["candidate_pool_hit"]

            # Pool status vs Ref
            if h2_1_hit > ref_hit:
                status_ref = "RECOVERY"
            elif h2_1_hit < ref_hit:
                status_ref = "REGRESSION"
            else:
                status_ref = "UNCHANGED"

            # Pool status vs H2
            if h2_1_hit > h2_hit:
                status_h2 = "H2_REPAIRED"
            elif h2_1_hit < h2_hit:
                status_h2 = "H2_REGRESSED"
            else:
                status_h2 = "UNCHANGED"

            rank_delta = (r_ref["best_rank"] - r_h2_1["best_rank"]) if (r_ref["best_rank"] <= 10 or r_h2_1["best_rank"] <= 10) else 0

            writer.writerow([
                r_ref["index"], qid, r_ref["company"], r_ref["doc_name"], r_ref["sector"],
                r_h2_1["statement_intents"], ";".join(str(p) for p in r_ref["target_pages"]),
                ref_hit, h2_hit, h2_1_hit,
                r_ref["best_rank"], r_h2.get("best_rank", 999), r_h2_1["best_rank"],
                r_ref["rr"], r_h2.get("rr", 0.0), r_h2_1["rr"],
                status_ref, status_h2,
                rank_delta, r_h2_1["injected_chunks_count"], r_h2_1["displaced_chunks_count"]
            ])

    # Filing Selection Audit CSV
    filing_audit_file = os.path.join(OUT_DIR, "filing_selection_audit.csv")
    with open(filing_audit_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "company", "target_filing", "selected_filing",
            "is_statement_query", "company_match", "period_match",
            "rejection_reason", "injected_chunks_count"
        ])
        writer.writeheader()
        for fa in filing_audits:
            writer.writerow(fa)

    # Statement Selection Audit CSV
    stmt_audit_file = os.path.join(OUT_DIR, "statement_selection_audit.csv")
    with open(stmt_audit_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "detected_intent", "selected_filing",
            "injected_chunk_ids", "primary_pages_selected", "gold_target_pages"
        ])
        writer.writeheader()
        for sa in statement_audits:
            writer.writerow(sa)

    # Sector Breakdown
    sector_stats = defaultdict(lambda: {"count": 0, "ref_pool_hits": 0, "h2_1_pool_hits": 0, "ref_rr": 0.0, "h2_1_rr": 0.0})
    for r_ref, r_h2_1 in zip(ref_records, h2_1_records):
        sec = r_ref["sector"]
        sector_stats[sec]["count"] += 1
        sector_stats[sec]["ref_pool_hits"] += r_ref["candidate_pool_hit"]
        sector_stats[sec]["h2_1_pool_hits"] += r_h2_1["candidate_pool_hit"]
        sector_stats[sec]["ref_rr"] += r_ref["rr"]
        sector_stats[sec]["h2_1_rr"] += r_h2_1["rr"]

    sector_file = os.path.join(OUT_DIR, "sector_metrics.csv")
    with open(sector_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sector", "questions", "ref_pool_recall", "h2_1_pool_recall", "delta_pool_recall", "ref_mrr", "h2_1_mrr", "delta_mrr"])
        for sec, st in sorted(sector_stats.items()):
            cnt = st["count"]
            ref_pr = round(st["ref_pool_hits"] / cnt * 100, 2)
            h2_1_pr = round(st["h2_1_pool_hits"] / cnt * 100, 2)
            ref_m = round(st["ref_rr"] / cnt, 4)
            h2_1_m = round(st["h2_1_rr"] / cnt, 4)
            writer.writerow([sec, cnt, ref_pr, h2_1_pr, round(h2_1_pr - ref_pr, 2), ref_m, h2_1_m, round(h2_1_m - ref_m, 4)])

    print("\nBenchmark Execution Completed Successfully.")
    print(f"Artifacts saved in: {OUT_DIR}")
    print(f"Reference Exp F Pool Recall: {ref_summary['candidate_pool_recall']}% | MRR: {ref_summary['mrr']}")
    print(f"Experiment H2 Prior Pool Recall: {h2_summary.get('candidate_pool_recall')}% | MRR: {h2_summary.get('mrr')}")
    print(f"Experiment H2.1 Pool Recall: {h2_1_summary['candidate_pool_recall']}% | MRR: {h2_1_summary['mrr']}")


if __name__ == "__main__":
    main()
