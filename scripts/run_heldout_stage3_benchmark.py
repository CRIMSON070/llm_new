"""
scripts/run_heldout_stage3_benchmark.py
=======================================
FinSage Stage 3: Independent Held-Out Benchmark Runner
Reproducible, Frozen Head-to-Head Evaluation:
- Arm 1: Reference Baseline (Frozen Experiment F: RRF depth=50, pool_k=50, temporal prioritization, no statement routing)
- Arm 2: Candidate System (Frozen Experiment H2.1: RRF depth=50, pool_k=50, dynamic company resolution, statement quota Q<=6, calibrated boosts)

Strict Isolation Constraints:
- Evaluates exclusively on the 100 held-out questions in outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl
- Reads chunks exclusively from data/heldout_stage3/processed/heldout_enriched_chunks.jsonl (29,335 chunks)
- Retrieves exclusively from isolated indexes in data/heldout_stage3/indexes/ (BM25 and FAISS dense)
- ZERO access to the 54,564 development chunks or development indexes in data/financebench/
- Verifies SHA-256 hashes of protected baseline modules before and after execution
- Produces paired statistical significance analysis (McNemar's test and Wilcoxon signed-rank test)
"""

import os
import sys
import json
import time
import re
import csv
import math
import hashlib
import argparse
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
# Baseline Hash Verification
# -------------------------------------------------------------------------
PROTECTED_HASHES = {
    "src/retrieval/phase2_reranker.py": "5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9",
    "scripts/run_phase2_benchmark.py": "945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1",
    "outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py": "6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED",
    "outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl": "411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C",
    "outputs/experiments/experiment_H2_statement_aware/scorecard.json": "2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2",
    "outputs/experiments/experiment_H2_statement_aware/phase2_retrieval_checkpoint.jsonl": "E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F",
    "outputs/experiments/experiment_H2_1_correctness_fixes/scorecard.json": "ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC"
}

def verify_protected_files() -> bool:
    all_ok = True
    for rel_path, expected_hash in PROTECTED_HASHES.items():
        abs_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(abs_path):
            print(f"[ERROR] Protected file missing: {rel_path}")
            all_ok = False
            continue
        with open(abs_path, "rb") as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest().upper()
        if curr_hash != expected_hash:
            print(f"[ERROR] Hash mismatch for {rel_path}!\n  Expected: {expected_hash}\n  Got:      {curr_hash}")
            all_ok = False
        else:
            print(f"[VERIFIED] {rel_path} matches protected baseline hash.")
    return all_ok

# -------------------------------------------------------------------------
# Dynamic Entity and Period Extraction
# -------------------------------------------------------------------------
DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")

# Canonical mapping for the 20 held-out corporate entities
COMPANY_MAP = {
    "Apple Inc.": "AAPL", "Apple": "AAPL", "AAPL": "AAPL",
    "Alphabet Inc.": "GOOGL", "Alphabet": "GOOGL", "Google": "GOOGL", "GOOGL": "GOOGL", "GOOG": "GOOGL",
    "NVIDIA Corporation": "NVDA", "NVIDIA": "NVDA", "NVDA": "NVDA",
    "Cisco Systems, Inc.": "CSCO", "Cisco Systems": "CSCO", "Cisco": "CSCO", "CSCO": "CSCO",
    "Broadcom Inc.": "AVGO", "Broadcom": "AVGO", "AVGO": "AVGO",
    "Meta Platforms, Inc.": "META", "Meta platforms": "META", "Meta": "META", "META": "META",
    "Bank of America Corp": "BAC", "Bank of America": "BAC", "BAC": "BAC",
    "The Goldman Sachs Group, Inc.": "GS", "Goldman Sachs": "GS", "GS": "GS",
    "Morgan Stanley": "MS", "MS": "MS",
    "Wells Fargo & Company": "WFC", "Wells Fargo": "WFC", "WFC": "WFC",
    "The Home Depot, Inc.": "HD", "Home Depot": "HD", "HD": "HD",
    "Target Corporation": "TGT", "Target": "TGT", "TGT": "TGT",
    "The Procter & Gamble Company": "PG", "Procter & Gamble": "PG", "P&G": "PG", "PG": "PG",
    "Starbucks Corporation": "SBUX", "Starbucks": "SBUX", "SBUX": "SBUX",
    "Eli Lilly and Company": "LLY", "Eli Lilly": "LLY", "LLY": "LLY",
    "UnitedHealth Group Inc.": "UNH", "UnitedHealth Group": "UNH", "UnitedHealth": "UNH", "UNH": "UNH",
    "Merck & Co., Inc.": "MRK", "Merck": "MRK", "MRK": "MRK",
    "AbbVie Inc.": "ABBV", "AbbVie": "ABBV", "ABBV": "ABBV",
    "Exxon Mobil Corporation": "XOM", "Exxon Mobil": "XOM", "ExxonMobil": "XOM", "XOM": "XOM",
    "Caterpillar Inc.": "CAT", "Caterpillar": "CAT", "CAT": "CAT"
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
# Temporal Intent Extraction Regexes & Document Matching
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
# Strict Target Filing Resolution & Disambiguation (Fix A)
# -------------------------------------------------------------------------
def resolve_target_filing_strict(
    q_data: Dict[str, Any],
    p_info: Dict[str, Any],
    all_filings: List[str],
    doc_meta_cache: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], str, bool, bool]:
    comp_name = q_data.get("company", "")
    ticker = q_data.get("ticker", "")
    
    canonical_prefix = COMPANY_MAP.get(comp_name) or COMPANY_MAP.get(ticker) or ticker
    if not canonical_prefix:
        return None, f"UNKNOWN_COMPANY: {comp_name}", False, False

    comp_filings = [d for d in all_filings if doc_meta_cache.get(d, {}).get("company") == canonical_prefix]
    if not comp_filings:
        return None, f"NO_FILINGS_FOR_COMPANY: {canonical_prefix}", False, False

    years = p_info.get("years", set())
    if not years:
        # Fall back to question's explicit fiscal_year
        if q_data.get("fiscal_year"):
            years = {int(q_data["fiscal_year"])}

    if not years:
        return None, "NO_PERIOD_DETECTED", True, False

    target_year = max(years)
    req_form = "10K" # Stage 3 benchmark is 100% Form 10-K filings

    matches = []
    for d in comp_filings:
        meta = doc_meta_cache.get(d, {})
        m_yr = meta.get("year")
        m_form = meta.get("ftype", "")

        if m_yr == target_year and "10K" in m_form:
            matches.append(d)

    if len(matches) == 1:
        return matches[0], "VERIFIED_EXACT_MATCH", True, True
    elif len(matches) > 1:
        return None, f"AMBIGUOUS_MULTIPLE_MATCHES: {matches}", True, False
    else:
        return None, f"NO_MATCHING_FILING (yr={target_year}, form={req_form})", True, False

# -------------------------------------------------------------------------
# Primary Statement Heading & Structure Qualification (Fix B)
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
        if "revenue" in lower_txt or "net sales" in lower_txt or "total revenue" in lower_txt: score += 4
        if "cost of sales" in lower_txt or "cost of goods sold" in lower_txt or "cost of products" in lower_txt: score += 4
        if "operating income" in lower_txt or "operating expenses" in lower_txt or "gross margin" in lower_txt or "gross profit" in lower_txt or "operating loss" in lower_txt: score += 3
        if "net income" in lower_txt or "net earnings" in lower_txt or "net loss" in lower_txt: score += 3
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
# Paired Statistical Significance Helpers
# -------------------------------------------------------------------------
def compute_mcnemar(b: int, c: int) -> Dict[str, Any]:
    """
    McNemar's test for paired binary outcomes:
    b: Number of discordant pairs where Exp F = 0 and H2.1 = 1 (Recoveries)
    c: Number of discordant pairs where Exp F = 1 and H2.1 = 0 (Regressions)
    """
    total_discordant = b + c
    if total_discordant == 0:
        return {
            "test": "McNemar",
            "b_recoveries": b,
            "c_regressions": c,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "odds_ratio": 1.0,
            "cohens_g": 0.0,
            "interpretation": "Identical performance across all paired evaluations."
        }
    
    # Exact two-sided binomial p-value
    p_val = 0.0
    for k in range(0, min(b, c) + 1):
        p_val += math.comb(total_discordant, k) * (0.5 ** total_discordant)
    p_val = min(1.0, 2.0 * p_val)

    # Continuity-corrected chi-squared
    chi2 = ((abs(b - c) - 1.0) ** 2) / total_discordant if abs(b - c) >= 1 else 0.0

    odds_ratio = (b / c) if c > 0 else float("inf")
    cohens_g = abs((b / total_discordant) - 0.5)

    return {
        "test": "McNemar (Exact Binomial + Continuity Corrected Chi2)",
        "b_recoveries": b,
        "c_regressions": c,
        "total_discordant": total_discordant,
        "chi2_stat": round(chi2, 4),
        "p_value": round(p_val, 5),
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else "inf",
        "cohens_g": round(cohens_g, 4),
        "significant_at_05": p_val < 0.05,
        "significant_at_01": p_val < 0.01
    }

def compute_wilcoxon_signed_rank(diffs: List[float]) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test for Reciprocal Rank differences.
    """
    non_zero_diffs = [d for d in diffs if abs(d) > 1e-6]
    n_pairs = len(non_zero_diffs)
    if n_pairs == 0:
        return {
            "test": "Wilcoxon signed-rank",
            "n_pairs": 0,
            "w_stat": 0.0,
            "p_value": 1.0,
            "interpretation": "Zero differences between paired reciprocal ranks."
        }

    # Rank absolute differences
    abs_diffs = [(abs(d), d) for d in non_zero_diffs]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign average ranks for ties
    ranks = []
    i = 0
    while i < n_pairs:
        j = i
        while j < n_pairs and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-6:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for _ in range(i, j):
            ranks.append(avg_rank)
        i = j

    w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
    w_stat = min(w_plus, w_minus)

    # Normal approximation for n >= 10
    mean_w = n_pairs * (n_pairs + 1) / 4.0
    var_w = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24.0
    sigma_w = math.sqrt(var_w)

    z = (w_stat - mean_w) / sigma_w if sigma_w > 0 else 0.0
    # Two-tailed p-value via error function approximation
    p_val = math.erfc(abs(z) / math.sqrt(2))

    return {
        "test": "Paired Wilcoxon signed-rank",
        "n_discordant_pairs": n_pairs,
        "w_plus": round(w_plus, 2),
        "w_minus": round(w_minus, 2),
        "w_stat": round(w_stat, 2),
        "z_score": round(z, 4),
        "p_value": round(p_val, 5),
        "significant_at_05": p_val < 0.05
    }

# -------------------------------------------------------------------------
# Benchmark Execution Engine
# -------------------------------------------------------------------------
def run_benchmark_arm(
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
    checkpoint_file = os.path.join(out_dir, f"phase2_retrieval_checkpoint_{mode}.jsonl")
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
    print(f"\n[{mode.upper()}] Starting benchmark run across N={len(questions)} held-out questions...")

    for idx, q_data in enumerate(questions, start=1):
        q_id = q_data.get("question_id", f"stage3_heldout_{idx:04d}")
        q_text = q_data["question"]
        gold_doc = q_data["doc_name"]
        gold_comp = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        qtype = q_data.get("question_type", "unspecified")
        evidence = q_data.get("evidence_sources", [])

        # In Stage 3, evidence_page_num is the 1-indexed physical PDF page number!
        target_pages = [e.get("evidence_page_num") for e in evidence if e.get("evidence_page_num") is not None]
        if not target_pages:
            raise ValueError(f"Malformed ground truth: {q_id} has no valid target evidence pages!")

        # 1. Multi-Path Retrieval (depth=50 for Path 1 BM25, depth=50 for Path 2 Dense)
        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=50)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res
        }

        # 2. RRF Fusion (k=60) with Temporal Prioritization
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

        # 3. Candidate Selection Branching
        injected_cids = []
        displaced_cids = []
        stmt_intents = detect_statement_intents(q_text)

        if mode == "reference_exp_f":
            # Exact Frozen Experiment F: strictly top-50 period-prioritized RRF candidates
            fused_cids = base_fused_cids
            sel_filing, reason, comp_m, period_m = None, "REFERENCE_EXP_F_NO_ROUTING", False, False

        elif mode == "candidate_h2_1":
            # Frozen Experiment H2.1: Statement-Aware Candidate Selection with strict validation
            if not stmt_intents:
                # Qualitative / narrative query: 100% bit-for-bit identical to Exp F
                fused_cids = base_fused_cids
                sel_filing, reason, comp_m, period_m = None, "NARRATIVE_QUERY_PRESERVE_BASE", True, True
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
                    # Pool assembly with guaranteed quota preservation (Q <= 6):
                    stmt_cids_unique = []
                    for cid in eligible_stmt_cids:
                        if cid not in stmt_cids_unique:
                            stmt_cids_unique.append(cid)

                    # Reserve slots: top candidates from base pool not already in statement set
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
            raise ValueError(f"Unknown benchmark mode: {mode}")

        # 4. Build candidate structures for reranking
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

        # 5. Check candidate pool recall (depth=50)
        best_rank_in_pool = 999
        for c_idx, c in enumerate(candidates, start=1):
            if c.get("doc_name") == gold_doc and c.get("page") in target_pages:
                best_rank_in_pool = c_idx
                break
        cand_hit = 1 if best_rank_in_pool <= 50 else 0
        candidate_pool_hits.append(cand_hit)

        # 6. Rerank with Phase 2 Reranker (top_k=10, apply_boosts=True)
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

        # 7. Audit records
        filing_audit_records.append({
            "question_id": q_id,
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
            "question_id": q_id,
            "detected_intent": "+".join(stmt_intents) if stmt_intents else "NONE",
            "selected_filing": sel_filing if sel_filing else "NONE",
            "injected_chunk_ids": ";".join(injected_cids) if injected_cids else "NONE",
            "primary_pages_selected": ";".join(str(chunk_store.get(cid, {}).get("page_number")) for cid in injected_cids) if injected_cids else "NONE",
            "gold_target_pages": ";".join(str(p) for p in target_pages)
        })

        record = {
            "index": idx,
            "question_id": q_id,
            "company": gold_comp,
            "doc_name": gold_doc,
            "sector": sector,
            "question_type": qtype,
            "question": q_text[:80],
            "statement_intents": "+".join(stmt_intents) if stmt_intents else "NONE",
            "target_pages": target_pages,
            "best_rank": best_rank,
            "best_rank_in_pool": best_rank_in_pool,
            "candidate_pool_hit": cand_hit,
            "recall_at_1": h1,
            "recall_at_3": h3,
            "recall_at_5": h5,
            "recall_at_10": h10,
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

        if idx % 20 == 0 or idx == len(questions):
            print(f"  Processed [{idx}/{len(questions)}] queries...")

    checkpoint_out.close()
    elapsed = round(time.time() - t_start, 2)
    n = len(questions)

    summary = {
        "mode": mode,
        "pool_size": 50,
        "num_questions": n,
        "runtime_seconds": elapsed,
        "avg_query_time_ms": round((elapsed / n) * 1000, 1),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "recall_at_1": round(sum(recall_at_1) / n * 100, 2),
        "recall_at_3": round(sum(recall_at_3) / n * 100, 2),
        "recall_at_5": round(sum(recall_at_5) / n * 100, 2),
        "recall_at_10": round(sum(recall_at_10) / n * 100, 2),
        "candidate_pool_recall": round(sum(candidate_pool_hits) / n * 100, 2),
        "candidate_pool_hits": sum(candidate_pool_hits),
        "top10_hits": sum(recall_at_10)
    }

    scorecard_path = os.path.join(out_dir, f"scorecard_{mode}.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, per_question_records, filing_audit_records, statement_audit_records

# -------------------------------------------------------------------------
# Main Execution Entrypoint
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FinSage Stage 3 Frozen Benchmark Runner")
    parser.add_argument("--questions", default="outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl")
    parser.add_argument("--chunks", default="data/heldout_stage3/processed/heldout_enriched_chunks.jsonl")
    parser.add_argument("--dense_index", default="data/heldout_stage3/indexes/faiss_bge_m3")
    parser.add_argument("--bm25_index", default="data/heldout_stage3/indexes/bm25")
    parser.add_argument("--output_dir", default="outputs/experiments/experiment_stage3_heldout_validation")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("FINSAGE STAGE 3: INDEPENDENT HELD-OUT FROZEN BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Timestamp:       {datetime.now().isoformat()}")
    print(f"Questions Path:  {args.questions}")
    print(f"Chunks Path:     {args.chunks}")
    print(f"Dense Index:     {args.dense_index}")
    print(f"BM25 Index:      {args.bm25_index}")
    print(f"Output Dir:      {args.output_dir}")

    # 1. Baseline Integrity Verification
    print("\n[Step 1] Verifying Protected Baseline Files SHA-256 Hashes...")
    if not verify_protected_files():
        print("[CRITICAL ERROR] Baseline integrity check failed. Halting runner.")
        sys.exit(1)
    print("[SUCCESS] All protected baseline hashes match perfectly.")

    # 2. Strict Isolation Assertion
    print("\n[Step 2] Auditing Strict Corpus & Index Isolation...")
    assert "heldout_stage3" in args.chunks, "Chunks must reside in heldout_stage3!"
    assert "heldout_stage3" in args.dense_index, "Dense index must reside in heldout_stage3!"
    assert "heldout_stage3" in args.bm25_index, "BM25 index must reside in heldout_stage3!"
    assert "financebench" not in args.chunks.lower() or "data/heldout_stage3" in args.chunks, "Corpus leakage detected!"
    print("[ISOLATION VERIFIED] Zero access to development corpus or indexes.")

    # 3. Load Evaluation Questions
    print("\n[Step 3] Loading Held-Out Benchmark Questions...")
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} independent evaluation questions.")
    assert len(questions) == 100, f"Expected 100 questions, got {len(questions)}"

    # 4. Load Held-Out Chunks Store
    print("\n[Step 4] Loading Held-Out Chunks Store...")
    t0 = time.time()
    chunk_store = {}
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store)} enriched chunks in {time.time() - t0:.2f}s.")
    assert len(chunk_store) == 29335, f"Expected 29,335 chunks, got {len(chunk_store)}"

    # Precompute doc metadata cache
    all_filings = sorted(list(set(c["doc_name"] for c in chunk_store.values() if c.get("doc_name"))))
    doc_meta_cache = {}
    for d in all_filings:
        doc_meta_cache[d] = parse_doc_meta(d)
    print(f"Parsed metadata for {len(all_filings)} held-out filings.")
    assert len(all_filings) == 40, f"Expected 40 filings, got {len(all_filings)}"

    # Build primary consolidated statement catalog (Fix B)
    print("\nBuilding Primary Consolidated Statement Catalog...")
    t_cat = time.time()
    primary_catalog = build_primary_statement_catalog(chunk_store)
    print(f"Primary catalog built in {round(time.time() - t_cat, 2)}s across {len(primary_catalog)} filings.")

    if args.dry_run:
        print("[DRY RUN COMPLETE] Runner preflight checks passed.")
        return

    # 5. Load Retrieval Components (Isolated to Held-Out Indexes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading retrieval indexes on device: {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    dense_faiss = FAISS.load_local(args.dense_index, embeddings, allow_dangerous_deserialization=True)
    bm25 = BM25Retriever(args.bm25_index, load_corpus=True)

    # MultiPathRetriever with zero metadata retriever (strictly 2 isolated paths: BM25 + Dense BGE-M3)
    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25,
        chunk_store=chunk_store,
        metadata_retriever=None,
        rrf_k=60
    )

    # Initialize Phase 2 Reranker with calibrated boosts
    print("\nInitializing Phase 2 Cross-Encoder Reranker...")
    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=True,
        statement_boost_weight=4.5,
        negative_disclosure_penalty=-3.0,
        company_boost_weight=1.5,
        period_boost_weight=1.5,
        table_boost_weight=0.5
    )

    # -------------------------------------------------------------------------
    # Execute Arm 1: Frozen Reference Baseline (Experiment F)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXECUTING ARM 1: REFERENCE BASELINE (FROZEN EXPERIMENT F)")
    print("=" * 80)
    exp_f_summary, exp_f_records, _, _ = run_benchmark_arm(
        mode="reference_exp_f",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        all_filings=all_filings,
        primary_catalog=primary_catalog,
        out_dir=args.output_dir
    )

    # -------------------------------------------------------------------------
    # Execute Arm 2: Frozen Candidate System (Experiment H2.1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXECUTING ARM 2: CANDIDATE SYSTEM (FROZEN EXPERIMENT H2.1)")
    print("=" * 80)
    h2_1_summary, h2_1_records, filing_audit, stmt_audit = run_benchmark_arm(
        mode="candidate_h2_1",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        all_filings=all_filings,
        primary_catalog=primary_catalog,
        out_dir=args.output_dir
    )

    # -------------------------------------------------------------------------
    # Step 6: Comparative Analysis & Paired Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING COMPARATIVE EVALUATION METRICS & PAIRED STATISTICS")
    print("=" * 80)

    # Question-level comparison
    q_comparison = []
    pool_recoveries = 0
    pool_regressions = 0
    top10_recoveries = 0
    top10_regressions = 0
    mrr_diffs = []

    for r_f, r_h in zip(exp_f_records, h2_1_records):
        assert r_f["question_id"] == r_h["question_id"]
        qid = r_f["question_id"]

        # Pool transition
        f_pool = r_f["candidate_pool_hit"]
        h_pool = r_h["candidate_pool_hit"]
        if f_pool == 0 and h_pool == 1:
            pool_recoveries += 1
            pool_transition = "RECOVERY"
        elif f_pool == 1 and h_pool == 0:
            pool_regressions += 1
            pool_transition = "REGRESSION"
        elif f_pool == 1 and h_pool == 1:
            pool_transition = "TIE_HIT"
        else:
            pool_transition = "TIE_MISS"

        # Top-10 transition
        f_top10 = r_f["recall_at_10"]
        h_top10 = r_h["recall_at_10"]
        if f_top10 == 0 and h_top10 == 1:
            top10_recoveries += 1
            top10_transition = "RECOVERY"
        elif f_top10 == 1 and h_top10 == 0:
            top10_regressions += 1
            top10_transition = "REGRESSION"
        elif f_top10 == 1 and h_top10 == 1:
            top10_transition = "TIE_HIT"
        else:
            top10_transition = "TIE_MISS"

        mrr_diff = r_h["rr"] - r_f["rr"]
        mrr_diffs.append(mrr_diff)

        q_comparison.append({
            "question_id": qid,
            "company": r_f["company"],
            "sector": r_f["sector"],
            "question_type": r_f["question_type"],
            "statement_intents": r_h["statement_intents"],
            "exp_f_pool_hit": f_pool,
            "h2_1_pool_hit": h_pool,
            "pool_transition": pool_transition,
            "exp_f_rank": r_f["best_rank"],
            "h2_1_rank": r_h["best_rank"],
            "top10_transition": top10_transition,
            "exp_f_mrr": r_f["rr"],
            "h2_1_mrr": r_h["rr"],
            "mrr_delta": round(mrr_diff, 4),
            "injected_chunks_count": r_h["injected_chunks_count"],
            "displaced_chunks_count": r_h["displaced_chunks_count"]
        })

    # Sector breakdown
    sector_stats = defaultdict(lambda: {"total": 0, "exp_f_pool": 0, "h2_1_pool": 0, "exp_f_top10": 0, "h2_1_top10": 0, "exp_f_mrr": [], "h2_1_mrr": []})
    type_stats = defaultdict(lambda: {"total": 0, "exp_f_pool": 0, "h2_1_pool": 0, "exp_f_top10": 0, "h2_1_top10": 0, "exp_f_mrr": [], "h2_1_mrr": []})

    for r_f, r_h in zip(exp_f_records, h2_1_records):
        sec = r_f["sector"]
        qt = r_f["question_type"]

        sector_stats[sec]["total"] += 1
        sector_stats[sec]["exp_f_pool"] += r_f["candidate_pool_hit"]
        sector_stats[sec]["h2_1_pool"] += r_h["candidate_pool_hit"]
        sector_stats[sec]["exp_f_top10"] += r_f["recall_at_10"]
        sector_stats[sec]["h2_1_top10"] += r_h["recall_at_10"]
        sector_stats[sec]["exp_f_mrr"].append(r_f["rr"])
        sector_stats[sec]["h2_1_mrr"].append(r_h["rr"])

        type_stats[qt]["total"] += 1
        type_stats[qt]["exp_f_pool"] += r_f["candidate_pool_hit"]
        type_stats[qt]["h2_1_pool"] += r_h["candidate_pool_hit"]
        type_stats[qt]["exp_f_top10"] += r_f["recall_at_10"]
        type_stats[qt]["h2_1_top10"] += r_h["recall_at_10"]
        type_stats[qt]["exp_f_mrr"].append(r_f["rr"])
        type_stats[qt]["h2_1_mrr"].append(r_h["rr"])

    # Statistical significance
    pool_mcnemar = compute_mcnemar(pool_recoveries, pool_regressions)
    top10_mcnemar = compute_mcnemar(top10_recoveries, top10_regressions)
    wilcoxon_res = compute_wilcoxon_signed_rank(mrr_diffs)

    # Master metrics dictionary
    master_metrics = {
        "benchmark": "FinSage Stage 3 Independent Held-Out Validation",
        "evaluation_timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "arms_comparison": {
            "arm_1_reference_exp_f": exp_f_summary,
            "arm_2_candidate_h2_1": h2_1_summary,
            "deltas": {
                "candidate_pool_recall_delta": round(h2_1_summary["candidate_pool_recall"] - exp_f_summary["candidate_pool_recall"], 2),
                "recall_at_1_delta": round(h2_1_summary["recall_at_1"] - exp_f_summary["recall_at_1"], 2),
                "recall_at_3_delta": round(h2_1_summary["recall_at_3"] - exp_f_summary["recall_at_3"], 2),
                "recall_at_5_delta": round(h2_1_summary["recall_at_5"] - exp_f_summary["recall_at_5"], 2),
                "recall_at_10_delta": round(h2_1_summary["recall_at_10"] - exp_f_summary["recall_at_10"], 2),
                "mrr_delta": round(h2_1_summary["mrr"] - exp_f_summary["mrr"], 4),
                "pool_recoveries": pool_recoveries,
                "pool_regressions": pool_regressions,
                "top10_recoveries": top10_recoveries,
                "top10_regressions": top10_regressions
            }
        },
        "statistical_tests": {
            "candidate_pool_mcnemar": pool_mcnemar,
            "recall_at_10_mcnemar": top10_mcnemar,
            "mrr_wilcoxon_signed_rank": wilcoxon_res
        },
        "breakdown_by_question_type": {
            qt: {
                "total": stats["total"],
                "exp_f_pool_recall": round(stats["exp_f_pool"] / stats["total"] * 100, 2),
                "h2_1_pool_recall": round(stats["h2_1_pool"] / stats["total"] * 100, 2),
                "exp_f_recall_at_10": round(stats["exp_f_top10"] / stats["total"] * 100, 2),
                "h2_1_recall_at_10": round(stats["h2_1_top10"] / stats["total"] * 100, 2),
                "exp_f_mrr": round(sum(stats["exp_f_mrr"]) / stats["total"], 4),
                "h2_1_mrr": round(sum(stats["h2_1_mrr"]) / stats["total"], 4)
            } for qt, stats in type_stats.items()
        },
        "breakdown_by_gics_sector": {
            sec: {
                "total": stats["total"],
                "exp_f_pool_recall": round(stats["exp_f_pool"] / stats["total"] * 100, 2),
                "h2_1_pool_recall": round(stats["h2_1_pool"] / stats["total"] * 100, 2),
                "exp_f_recall_at_10": round(stats["exp_f_top10"] / stats["total"] * 100, 2),
                "h2_1_recall_at_10": round(stats["h2_1_top10"] / stats["total"] * 100, 2),
                "exp_f_mrr": round(sum(stats["exp_f_mrr"]) / stats["total"], 4),
                "h2_1_mrr": round(sum(stats["h2_1_mrr"]) / stats["total"], 4)
            } for sec, stats in sector_stats.items()
        }
    }

    # Save artifacts
    print("\nWriting benchmark result artifacts...")
    metrics_path = os.path.join(args.output_dir, "heldout_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)
    print(f"  [SAVED] {metrics_path}")

    comparison_csv_path = os.path.join(args.output_dir, "heldout_question_level_comparison.csv")
    with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_comparison[0].keys()))
        writer.writeheader()
        writer.writerows(q_comparison)
    print(f"  [SAVED] {comparison_csv_path}")

    recovery_csv_path = os.path.join(args.output_dir, "recovery_regression_audit.csv")
    transitions = [r for r in q_comparison if r["pool_transition"] in ["RECOVERY", "REGRESSION"] or r["top10_transition"] in ["RECOVERY", "REGRESSION"]]
    with open(recovery_csv_path, "w", newline="", encoding="utf-8") as f:
        if transitions:
            writer = csv.DictWriter(f, fieldnames=list(transitions[0].keys()))
            writer.writeheader()
            writer.writerows(transitions)
        else:
            f.write("question_id,company,sector,question_type,statement_intents,exp_f_pool_hit,h2_1_pool_hit,pool_transition,exp_f_rank,h2_1_rank,top10_transition,exp_f_mrr,h2_1_mrr,mrr_delta,injected_chunks_count,displaced_chunks_count\n")
    print(f"  [SAVED] {recovery_csv_path}")

    filing_audit_path = os.path.join(args.output_dir, "filing_selection_audit.csv")
    with open(filing_audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filing_audit[0].keys()))
        writer.writeheader()
        writer.writerows(filing_audit)
    print(f"  [SAVED] {filing_audit_path}")

    stmt_audit_path = os.path.join(args.output_dir, "statement_selection_audit.csv")
    with open(stmt_audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stmt_audit[0].keys()))
        writer.writeheader()
        writer.writerows(stmt_audit)
    print(f"  [SAVED] {stmt_audit_path}")

    # Step 7: Post-Execution Baseline Hash Verification
    print("\n[Step 7] Re-Verifying Protected Baseline Hashes Post-Execution...")
    if not verify_protected_files():
        print("[CRITICAL ERROR] Baseline files modified during execution!")
        sys.exit(1)
    print("[SUCCESS] All protected baseline hashes remained 100% untouched post-execution.")

    # Print summary scorecard
    print("\n" + "=" * 80)
    print("FINSAGE STAGE 3 HELD-OUT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Exp F (Baseline)':<18} | {'H2.1 (Candidate)':<18} | {'Delta':<12}")
    print("-" * 84)
    print(f"{'Candidate Pool Recall':<30} | {exp_f_summary['candidate_pool_recall']:>17.2f}% | {h2_1_summary['candidate_pool_recall']:>17.2f}% | {master_metrics['arms_comparison']['deltas']['candidate_pool_recall_delta']:>+11.2f}%")
    print(f"{'Recall@1':<30} | {exp_f_summary['recall_at_1']:>17.2f}% | {h2_1_summary['recall_at_1']:>17.2f}% | {master_metrics['arms_comparison']['deltas']['recall_at_1_delta']:>+11.2f}%")
    print(f"{'Recall@3':<30} | {exp_f_summary['recall_at_3']:>17.2f}% | {h2_1_summary['recall_at_3']:>17.2f}% | {master_metrics['arms_comparison']['deltas']['recall_at_3_delta']:>+11.2f}%")
    print(f"{'Recall@5':<30} | {exp_f_summary['recall_at_5']:>17.2f}% | {h2_1_summary['recall_at_5']:>17.2f}% | {master_metrics['arms_comparison']['deltas']['recall_at_5_delta']:>+11.2f}%")
    print(f"{'Recall@10':<30} | {exp_f_summary['recall_at_10']:>17.2f}% | {h2_1_summary['recall_at_10']:>17.2f}% | {master_metrics['arms_comparison']['deltas']['recall_at_10_delta']:>+11.2f}%")
    print(f"{'MRR':<30} | {exp_f_summary['mrr']:>18.4f} | {h2_1_summary['mrr']:>18.4f} | {master_metrics['arms_comparison']['deltas']['mrr_delta']:>+12.4f}")
    print(f"{'Runtime (s)':<30} | {exp_f_summary['runtime_seconds']:>18.2f} | {h2_1_summary['runtime_seconds']:>18.2f} | {h2_1_summary['runtime_seconds'] - exp_f_summary['runtime_seconds']:>+12.2f}")
    print("-" * 84)
    print(f"Pool Transitions:       Recoveries={pool_recoveries}, Regressions={pool_regressions} (McNemar p={pool_mcnemar['p_value']})")
    z_score_val = wilcoxon_res.get('z_score', 0.0)
    print(f"MRR Wilcoxon Rank-Sum:  W={wilcoxon_res['w_stat']}, z={z_score_val}, p={wilcoxon_res['p_value']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
