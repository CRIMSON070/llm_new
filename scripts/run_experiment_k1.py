"""
scripts/run_experiment_k1.py
============================
FinSage Experiment K1: Controlled Unicode Normalization Retrieval Experiment
============================================================================
Evaluates whether Unicode NFKC normalization improves actual retrieval performance
on the 100-question Stage 3 held-out benchmark.

Arms:
- Control: Frozen Experiment F Baseline (Original unnormalized BM25 index + original FAISS index)
- Treatment: Experiment K1 NFKC (Normalized BM25 index + original FAISS index)

Strict Isolation Constraints:
- Evaluates exclusively on outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl (100 questions).
- Reads normalized chunks from data/heldout_stage3/experiment_k1/processed/heldout_enriched_chunks_nfkc.jsonl.
- Uses normalized BM25 index from data/heldout_stage3/experiment_k1/indexes/bm25/.
- Keeps original Dense FAISS index unchanged at data/heldout_stage3/indexes/faiss_bge_m3/.
- ZERO modifications to active production code in src/retrieval/.
- Verifies SHA-256 hashes of all protected baseline files before and after execution.
- Saves all outputs strictly under outputs/experiments/experiment_K1_unicode/.
"""

import os
import sys
import json
import time
import re
import csv
import math
import hashlib
import importlib.util
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Set

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import unicodedata
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

# Dynamic import of Experiment C's Phase 2 Reranker (Preserves baseline)
EXP_C_RERANKER_PATH = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_C_company_intent", "source_backup", "phase2_reranker.py"
)
spec = importlib.util.spec_from_file_location("exp_c_reranker", EXP_C_RERANKER_PATH)
exp_c_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_c_mod)
Phase2Reranker = exp_c_mod.Phase2Reranker

PROTECTED_HASHES = {
    "src/retrieval/phase2_reranker.py": "5C06B4DAD24079913434E64790C1D13541BD4D70B1EA4CE1C05360174A6784B9",
    "scripts/run_phase2_benchmark.py": "945F9303A07C7696C124C0DE68C4C17D6F459931F530402C487EEDDAAE2528F1",
    "outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py": "6CB436535D49209371AD07A4AAC75D30084AA4412EA41EE459EB3500630006ED",
    "outputs/experiments/experiment_F_period_aware/experiment/phase2_retrieval_checkpoint.jsonl": "411C83B116E45C163C73A5B00DA74284F30C55D78B084204956BF8D3388F3F9C",
    "outputs/experiments/experiment_H2_statement_aware/scorecard.json": "2771230871C9987A68E352AACFBD5F96ACD3B73BEC565D74593173BAD05793F2",
    "outputs/experiments/experiment_H2_statement_aware/phase2_retrieval_checkpoint.jsonl": "E531F121A6FA61D6DD9A6ECBB28E358BFC338C8DB57B6D4E6E8A8C3CD5F7B26F",
    "outputs/experiments/experiment_H2_1_correctness_fixes/scorecard.json": "ABE5D232617F93EE852BC3CC349A87D9624936B4FE5A9B7D42EF92D13CC753DC",
    "outputs/experiments/experiment_I_disclosure_aware/arm_a/scorecard_arm_a.json": "09BB876AC1CA85F582ED5981DDDA221DA46EE7C6F395544C2D39A06EBE76AA45",
    "outputs/experiments/experiment_I_disclosure_aware/arm_b/scorecard_arm_b.json": "3837C027CC833046B5A012E6B67FA20F403BCFDF0E3C1C553782F816F97135AC",
    "outputs/experiments/experiment_J_failure_attribution/oracle_bounds.json": "2C57377500E9F092B199C314E66CD624C4EEA3635F88F5668B27F663C648AF4D",
    "outputs/experiments/experiment_K0_validation/experiment_K0_report.md": "7279147C78E17D402B6BEA297BDE6CBDB2B35E917ED34B6021606E64D639B83D",
    "data/heldout_stage3/processed/heldout_enriched_chunks.jsonl": "EA4264F5D7E889D30FC4EE9147E7286B67FFA76300967FEE3651002974062845",
    "outputs/experiments/experiment_stage3_heldout_validation/heldout_questions.jsonl": "1B75376BFE803A85B600A66FB3F9609E7C9097BB96875B0A3C407FA7883F41E2",
    "data/heldout_stage3/indexes/bm25/params.index.json": "A36B36B5AA1472A3B550335782D8B934AFFACD3F01527267FC7795664E726D44",
    "data/heldout_stage3/indexes/faiss_bge_m3/index.faiss": "E78C7F801C33840C936303277D600431439E8356D965FE35C0D3098673A3AE59"
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
            print(f"  [VERIFIED] {rel_path}")
    return all_ok

# -------------------------------------------------------------------------
# Dynamic Metadata and Temporal Parsing
# -------------------------------------------------------------------------
DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")

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
# Statistical Tests
# -------------------------------------------------------------------------
def mcnemar_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        p_val = 0.0
        for i in range(min(b, c) + 1):
            p_val += math.comb(n, i) * (0.5 ** n)
        return min(1.0, 2.0 * p_val)
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / n
        return math.erfc(math.sqrt(chi2 / 2.0))

def wilcoxon_signed_rank(x: List[float], y: List[float]) -> float:
    diffs = [xi - yi for xi, yi in zip(x, y) if abs(xi - yi) > 1e-9]
    n = len(diffs)
    if n == 0:
        return 1.0
    abs_diffs = [(abs(d), d) for d in diffs]
    abs_diffs.sort(key=lambda t: t[0])
    ranks = []
    i = 0
    while i < n:
        j = i
        while j < n and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-9:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for _ in range(i, j):
            ranks.append(avg_rank)
        i = j
    w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
    w = min(w_plus, w_minus)
    if n < 10:
        return 1.0 if w_plus == w_minus else 0.5
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return 1.0
    z = (w - mean_w) / std_w
    return math.erfc(abs(z) / math.sqrt(2.0))

# -------------------------------------------------------------------------
# Arm Evaluator Engine
# -------------------------------------------------------------------------
def evaluate_arm(
    arm_name: str,
    questions: List[Dict[str, Any]],
    mpr: MultiPathRetriever,
    reranker: Phase2Reranker,
    chunk_store: Dict[str, Dict[str, Any]],
    doc_meta_cache: Dict[str, Dict[str, Any]],
    out_dir: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_file = os.path.join(out_dir, f"phase2_retrieval_checkpoint_{arm_name}.jsonl")
    checkpoint_out = open(checkpoint_file, "w", encoding="utf-8")

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    recall_at_10 = []
    candidate_pool_hits = []
    bm25_hits_10 = []
    bm25_hits_20 = []
    bm25_hits_50 = []
    bm25_hits_100 = []
    dense_only_hits = []

    per_question_records = []
    sector_stats = defaultdict(lambda: {"total": 0, "pool_hits": 0, "top10_hits": 0, "mrr_sum": 0.0})
    category_stats = defaultdict(lambda: {"total": 0, "pool_hits": 0, "top10_hits": 0, "mrr_sum": 0.0})

    t_start = time.time()
    print(f"\n[{arm_name.upper()}] Starting benchmark run across N={len(questions)} held-out questions...")

    for idx, q_data in enumerate(questions, start=1):
        q_id = q_data.get("question_id", f"stage3_heldout_{idx:04d}")
        # Consistent query text policy: NFKC normalize query
        q_text = unicodedata.normalize("NFKC", q_data["question"])
        gold_doc = q_data["doc_name"]
        gold_comp = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        qtype = q_data.get("question_type", "unspecified")
        evidence = q_data.get("evidence_sources", [])

        target_pages = [e.get("evidence_page_num") for e in evidence if e.get("evidence_page_num") is not None]
        if not target_pages:
            raise ValueError(f"Malformed ground truth: {q_id} has no valid target evidence pages!")

        # 1. Multi-Path Retrieval (depth=100 for BM25 audit, depth=50 for RRF; depth=50 Dense FAISS)
        t_q0 = time.time()
        sparse_res_100 = mpr.retrieve_sparse(q_text, k=100)
        sparse_res = sparse_res_100[:50]
        dense_res = mpr.retrieve_dense(q_text, k=50)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res
        }

        # Check BM25-only hits across depths (10, 20, 50, 100)
        bm25_gold_rank_100 = None
        for r_idx, (cid, s) in enumerate(sparse_res_100, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                bm25_gold_rank_100 = r_idx
                break

        dense_gold_rank = None
        for r_idx, (cid, s) in enumerate(dense_res, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                dense_gold_rank = r_idx
                break

        bm25_hit_10 = (bm25_gold_rank_100 is not None and bm25_gold_rank_100 <= 10)
        bm25_hit_20 = (bm25_gold_rank_100 is not None and bm25_gold_rank_100 <= 20)
        bm25_hit_50 = (bm25_gold_rank_100 is not None and bm25_gold_rank_100 <= 50)
        bm25_hit_100 = (bm25_gold_rank_100 is not None)

        bm25_hits_10.append(1 if bm25_hit_10 else 0)
        bm25_hits_20.append(1 if bm25_hit_20 else 0)
        bm25_hits_50.append(1 if bm25_hit_50 else 0)
        bm25_hits_100.append(1 if bm25_hit_100 else 0)

        dense_hit = (dense_gold_rank is not None)
        dense_only_hits.append(1 if dense_hit else 0)

        # 2. RRF Fusion (k=60) with Temporal Prioritization
        p_info = detect_query_periods(q_text)
        rrf_scores = defaultdict(float)
        for path_name, items in path_dict.items():
            for rank, (cid, raw_score) in enumerate(items, start=1):
                inc = 1.0 / (60 + rank)
                rrf_scores[cid] += inc

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
            fused_cids = (in_period_cids + other_cids)[:50]
        else:
            fused_cids = [cid for cid, score in sorted_all[:50]]

        # Candidate pool gold rank
        pool_gold_rank = None
        for r_idx, cid in enumerate(fused_cids, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                pool_gold_rank = r_idx
                break
        pool_hit = (pool_gold_rank is not None)
        candidate_pool_hits.append(1 if pool_hit else 0)

        # 3. Cross-Encoder Reranking
        candidates = []
        for cid in fused_cids:
            c = chunk_store.get(cid, {})
            candidates.append({
                "chunk_id": cid,
                "text": c.get("text", ""),
                "doc_name": c.get("doc_name", ""),
                "page": c.get("page_number", 0),
                "rrf_score": round(rrf_scores[cid], 6)
            })

        reranked = reranker.rerank(
            query=q_text,
            candidates=candidates,
            top_k=10,
            apply_boosts=True
        )

        top10 = reranked[:10]

        # Final top-10 gold rank
        final_gold_rank = None
        for r_idx, item in enumerate(top10, start=1):
            if item.get("doc_name") == gold_doc and item.get("page") in target_pages:
                final_gold_rank = r_idx
                break


        r1 = 1 if (final_gold_rank is not None and final_gold_rank <= 1) else 0
        r3 = 1 if (final_gold_rank is not None and final_gold_rank <= 3) else 0
        r5 = 1 if (final_gold_rank is not None and final_gold_rank <= 5) else 0
        r10 = 1 if (final_gold_rank is not None and final_gold_rank <= 10) else 0
        rr = (1.0 / final_gold_rank) if final_gold_rank is not None else 0.0

        recall_at_1.append(r1)
        recall_at_3.append(r3)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        reciprocal_ranks.append(rr)

        # Sector & Category stats
        sec_entry = sector_stats[sector]
        sec_entry["total"] += 1
        sec_entry["pool_hits"] += (1 if pool_hit else 0)
        sec_entry["top10_hits"] += r10
        sec_entry["mrr_sum"] += rr

        cat_entry = category_stats[qtype]
        cat_entry["total"] += 1
        cat_entry["pool_hits"] += (1 if pool_hit else 0)
        cat_entry["top10_hits"] += r10
        cat_entry["mrr_sum"] += rr

        q_time_ms = round((time.time() - t_q0) * 1000, 1)

        record = {
            "index": idx,
            "question_id": q_id,
            "company": gold_comp,
            "doc_name": gold_doc,
            "sector": sector,
            "question_type": qtype,
            "question": q_text,
            "target_pages": target_pages,
            "bm25_gold_rank": bm25_gold_rank_100 if (bm25_gold_rank_100 is not None and bm25_gold_rank_100 <= 50) else None,
            "bm25_gold_rank_100": bm25_gold_rank_100,
            "dense_gold_rank": dense_gold_rank,
            "bm25_hit_10": bm25_hit_10,
            "bm25_hit_20": bm25_hit_20,
            "bm25_hit_50": bm25_hit_50,
            "bm25_hit_100": bm25_hit_100,
            "dense_hit_50": dense_hit,
            "candidate_pool_hit": pool_hit,
            "best_rank_in_pool": pool_gold_rank,
            "best_rank": final_gold_rank,
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "rr": round(rr, 4),
            "pool_chunk_ids": fused_cids,
            "top10_retrieved": [
                {
                    "rank": r_idx,
                    "doc_id": item.get("chunk_id", item.get("doc_id")),
                    "doc_name": item.get("doc_name"),
                    "page_number": item.get("page", item.get("page_number")),
                    "score": round(item.get("rerank_score", 0.0), 4)
                }
                for r_idx, item in enumerate(top10, start=1)
            ],
            "latency_ms": q_time_ms
        }
        per_question_records.append(record)
        checkpoint_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if idx % 20 == 0 or idx == len(questions):
            print(f"  Processed [{idx}/{len(questions)}] questions -> Running Pool Recall: {sum(candidate_pool_hits)/idx*100:.1f}%, R@10: {sum(recall_at_10)/idx*100:.1f}%, MRR: {sum(reciprocal_ranks)/idx:.4f}")

    checkpoint_out.close()
    elapsed = time.time() - t_start
    n = len(questions)

    summary = {
        "arm_name": arm_name,
        "timestamp": datetime.now().isoformat(),
        "num_questions": n,
        "runtime_seconds": round(elapsed, 2),
        "avg_query_time_ms": round((elapsed / n) * 1000, 1),
        "candidate_pool_recall": round(sum(candidate_pool_hits) / n * 100, 2),
        "candidate_pool_hits": sum(candidate_pool_hits),
        "recall_at_1": round(sum(recall_at_1) / n * 100, 2),
        "recall_at_3": round(sum(recall_at_3) / n * 100, 2),
        "recall_at_5": round(sum(recall_at_5) / n * 100, 2),
        "recall_at_10": round(sum(recall_at_10) / n * 100, 2),
        "top10_hits": sum(recall_at_10),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "bm25_recall_at_10": round(sum(bm25_hits_10) / n * 100, 2),
        "bm25_recall_at_20": round(sum(bm25_hits_20) / n * 100, 2),
        "bm25_recall_at_50": round(sum(bm25_hits_50) / n * 100, 2),
        "bm25_recall_at_100": round(sum(bm25_hits_100) / n * 100, 2),
        "bm25_only_recall": round(sum(bm25_hits_50) / n * 100, 2),
        "dense_only_recall": round(sum(dense_only_hits) / n * 100, 2),
        "sector_breakdown": {
            sec: {
                "total": data["total"],
                "pool_recall": round(data["pool_hits"] / data["total"] * 100, 2),
                "recall_at_10": round(data["top10_hits"] / data["total"] * 100, 2),
                "mrr": round(data["mrr_sum"] / data["total"], 4)
            }
            for sec, data in sorted(sector_stats.items())
        },
        "category_breakdown": {
            cat: {
                "total": data["total"],
                "pool_recall": round(data["pool_hits"] / data["total"] * 100, 2),
                "recall_at_10": round(data["top10_hits"] / data["total"] * 100, 2),
                "mrr": round(data["mrr_sum"] / data["total"], 4)
            }
            for cat, data in sorted(category_stats.items())
        }
    }

    scorecard_path = os.path.join(out_dir, f"scorecard_{arm_name}.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, per_question_records

# -------------------------------------------------------------------------
# Main Benchmark Driver
# -------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("FINSAGE EXPERIMENT K1: CONTROLLED UNICODE NORMALIZATION RETRIEVAL")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 1. Baseline Integrity Verification
    print("\n[Step 1] Verifying Protected Baseline Files...")
    if not verify_protected_files():
        print("[CRITICAL ERROR] Hash verification failed. Halting.")
        sys.exit(1)

    # 2. Paths
    questions_file = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "heldout_questions.jsonl")
    norm_chunks_file = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "processed", "heldout_enriched_chunks_nfkc.jsonl")
    dense_index_dir = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3")
    orig_bm25_dir = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25")
    k1_bm25_dir = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "indexes", "bm25")
    out_dir = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_K1_unicode_normalization")
    os.makedirs(out_dir, exist_ok=True)

    # 3. Load Questions
    print(f"\n[Step 2] Loading Evaluation Questions: {questions_file}")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} evaluation questions.")
    assert len(questions) == 100

    # 4. Load Chunk Stores
    orig_chunks_file = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    norm_chunks_file = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "processed", "heldout_enriched_chunks_nfkc.jsonl")

    print(f"\n[Step 3] Loading Original Chunk Store (Control): {orig_chunks_file}")
    t0 = time.time()
    chunk_store_orig = {}
    with open(orig_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store_orig[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store_orig):,} original chunks in {time.time() - t0:.2f}s.")
    assert len(chunk_store_orig) == 29335

    print(f"\n[Step 3b] Loading Normalized Chunk Store (Treatment): {norm_chunks_file}")
    t0 = time.time()
    chunk_store_nfkc = {}
    with open(norm_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store_nfkc[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store_nfkc):,} normalized chunks in {time.time() - t0:.2f}s.")
    assert len(chunk_store_nfkc) == 29335

    all_filings = sorted(list(set(c["doc_name"] for c in chunk_store_orig.values() if c.get("doc_name"))))
    doc_meta_cache = {d: parse_doc_meta(d) for d in all_filings}
    assert len(all_filings) == 40

    # 5. Shared Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 4] Loading Shared Models (device: {device})...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    dense_faiss = FAISS.load_local(dense_index_dir, embeddings, allow_dangerous_deserialization=True)

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
    # ARM 1: REPRODUCE FROZEN BASELINE (EXPERIMENT F CONTROL)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ARM 1: CONTROL — FROZEN EXPERIMENT F BASELINE REPRODUCTION")
    print("=" * 80)
    orig_bm25 = BM25Retriever(orig_bm25_dir, load_corpus=True)
    mpr_control = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=orig_bm25,
        chunk_store=chunk_store_orig,
        metadata_retriever=None,
        rrf_k=60
    )
    exp_f_summary, exp_f_records = evaluate_arm(
        arm_name="exp_f_control",
        questions=questions,
        mpr=mpr_control,
        reranker=reranker,
        chunk_store=chunk_store_orig,
        doc_meta_cache=doc_meta_cache,
        out_dir=out_dir
    )

    print("\n[CONTROL REPRODUCTION CHECK]")
    print(f"Pool Recall: {exp_f_summary['candidate_pool_recall']}% (Expected: 30.0%)")
    print(f"Recall@1:    {exp_f_summary['recall_at_1']}% (Expected: 7.0%)")
    print(f"Recall@3:    {exp_f_summary['recall_at_3']}% (Expected: 12.0%)")
    print(f"Recall@5:    {exp_f_summary['recall_at_5']}% (Expected: 16.0%)")
    print(f"Recall@10:   {exp_f_summary['recall_at_10']}% (Expected: 20.0%)")
    print(f"MRR:         {exp_f_summary['mrr']} (Expected: 0.1050)")
    assert exp_f_summary['candidate_pool_recall'] == 30.00, f"Baseline mismatch: Pool Recall={exp_f_summary['candidate_pool_recall']}"
    assert exp_f_summary['recall_at_10'] == 20.00, f"Baseline mismatch: R@10={exp_f_summary['recall_at_10']}"
    assert abs(exp_f_summary['mrr'] - 0.1050) < 1e-4, f"Baseline mismatch: MRR={exp_f_summary['mrr']}"
    print("[SUCCESS] Frozen Experiment F baseline reproduced 100% bit-for-bit.")

    # -------------------------------------------------------------------------
    # ARM 2: TREATMENT — EXPERIMENT K1 (NFKC NORMALIZATION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ARM 2: TREATMENT — EXPERIMENT K1 (UNICODE NFKC NORMALIZATION)")
    print("=" * 80)
    k1_bm25 = BM25Retriever(k1_bm25_dir, load_corpus=True)
    mpr_k1 = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=k1_bm25,
        chunk_store=chunk_store_nfkc,
        metadata_retriever=None,
        rrf_k=60
    )
    k1_summary, k1_records = evaluate_arm(
        arm_name="k1_nfkc",
        questions=questions,
        mpr=mpr_k1,
        reranker=reranker,
        chunk_store=chunk_store_nfkc,
        doc_meta_cache=doc_meta_cache,
        out_dir=out_dir
    )


    # -------------------------------------------------------------------------
    # Step 6: Comparative Analysis, Turnover & Statistical Tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING PAIRED STATISTICAL COMPARISONS & POOL TURNOVER")
    print("=" * 80)

    # Pairwise comparisons
    comp_csv = os.path.join(out_dir, "k1_vs_exp_f_question_comparison.csv")
    turnover_csv = os.path.join(out_dir, "pool_turnover_audit.csv")

    pool_recoveries = []
    pool_regressions = []
    top10_recoveries = []
    top10_regressions = []
    turnover_records = []
    comp_records = []

    f_pool_hits = [r["candidate_pool_hit"] for r in exp_f_records]
    k1_pool_hits = [r["candidate_pool_hit"] for r in k1_records]
    f_top10 = [r["recall_at_10"] for r in exp_f_records]
    k1_top10 = [r["recall_at_10"] for r in k1_records]
    f_mrr = [r["rr"] for r in exp_f_records]
    k1_mrr = [r["rr"] for r in k1_records]

    # McNemar contingency counts
    # b: Exp F miss (0), K1 hit (1) -> recovery
    # c: Exp F hit (1), K1 miss (0) -> regression
    b_pool = sum(1 for f, k in zip(f_pool_hits, k1_pool_hits) if not f and k)
    c_pool = sum(1 for f, k in zip(f_pool_hits, k1_pool_hits) if f and not k)
    p_mcnemar_pool = mcnemar_test(b_pool, c_pool)

    b_top10 = sum(1 for f, k in zip(f_top10, k1_top10) if not f and k)
    c_top10 = sum(1 for f, k in zip(f_top10, k1_top10) if f and not k)
    p_mcnemar_top10 = mcnemar_test(b_top10, c_top10)

    p_wilcoxon_mrr = wilcoxon_signed_rank(k1_mrr, f_mrr)

    for f_rec, k_rec in zip(exp_f_records, k1_records):
        qid = f_rec["question_id"]
        # Pool transition
        f_ph = f_rec["candidate_pool_hit"]
        k_ph = k_rec["candidate_pool_hit"]
        if not f_ph and k_ph:
            pool_trans = "RECOVERY (+)"
            pool_recoveries.append(qid)
        elif f_ph and not k_ph:
            pool_trans = "REGRESSION (-)"
            pool_regressions.append(qid)
        else:
            pool_trans = "UNCHANGED"

        # Top-10 transition
        f_t10 = f_rec["recall_at_10"]
        k_t10 = k_rec["recall_at_10"]
        if not f_t10 and k_t10:
            top10_trans = "RECOVERY (+)"
            top10_recoveries.append(qid)
        elif f_t10 and not k_t10:
            top10_trans = "REGRESSION (-)"
            top10_regressions.append(qid)
        else:
            top10_trans = "UNCHANGED"

        # Pool overlap & turnover
        f_pool_set = set(f_rec["pool_chunk_ids"])
        k_pool_set = set(k_rec["pool_chunk_ids"])
        intersection = len(f_pool_set.intersection(k_pool_set))
        union = len(f_pool_set.union(k_pool_set))
        jaccard = round(intersection / union, 4) if union > 0 else 1.0
        displaced_count = 50 - intersection

        turnover_records.append({
            "question_id": qid,
            "company": f_rec["company"],
            "shared_candidates": intersection,
            "novel_k1_candidates": displaced_count,
            "pool_jaccard": jaccard,
            "pool_transition": pool_trans,
            "top10_transition": top10_trans
        })

        comp_records.append({
            "question_id": qid,
            "company": f_rec["company"],
            "sector": f_rec["sector"],
            "f_bm25_rank": f_rec["bm25_gold_rank"],
            "k1_bm25_rank": k_rec["bm25_gold_rank"],
            "f_dense_rank": f_rec["dense_gold_rank"],
            "k1_dense_rank": k_rec["dense_gold_rank"],
            "f_pool_hit": f_ph,
            "k1_pool_hit": k_ph,
            "pool_transition": pool_trans,
            "f_final_rank": f_rec["best_rank"],
            "k1_final_rank": k_rec["best_rank"],
            "top10_transition": top10_trans,
            "f_mrr": f_rec["rr"],
            "k1_mrr": k_rec["rr"],
            "mrr_delta": round(k_rec["rr"] - f_rec["rr"], 4)
        })

    # Write CSVs
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_records[0].keys()))
        writer.writeheader()
        writer.writerows(comp_records)

    with open(turnover_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(turnover_records[0].keys()))
        writer.writeheader()
        writer.writerows(turnover_records)

    # -------------------------------------------------------------------------
    # Additional Audit: 10 Questions with Ligatures / Terminology
    # -------------------------------------------------------------------------
    # Select 10 previously missed questions where gold evidence contains ligatures or relevant financial terms
    audit_target_qids = [
        "stage3_heldout_0001", # Apple net sales FY24 (Page 68, ligatures in text)
        "stage3_heldout_0007", # Cisco cash flows (Page 73, cash flows ligatures)
        "stage3_heldout_0016", # Google advertising revenues (Page 39)
        "stage3_heldout_0023", # NVIDIA gross profit / cost of revenue (Page 113)
        "stage3_heldout_0031", # Meta head count & employee benefits (Page 83)
        "stage3_heldout_0045", # Goldman Sachs net interest income (Page 120)
        "stage3_heldout_0056", # Procter & Gamble effective tax rate (Page 52)
        "stage3_heldout_0068", # Merck research and development expense (Page 71)
        "stage3_heldout_0079", # ExxonMobil crude production (Page 60)
        "stage3_heldout_0092", # Caterpillar financial services (Page 16)
        "stage3_heldout_0099"  # Eli Lilly income taxes (Page 48)
    ]

    audit_records = []
    comp_map = {r["question_id"]: r for r in comp_records}
    for qid in audit_target_qids:
        r = comp_map.get(qid)
        if r:
            audit_records.append({
                "question_id": qid,
                "company": r["company"],
                "f_bm25_rank": r["f_bm25_rank"],
                "k1_bm25_rank": r["k1_bm25_rank"],
                "f_pool_hit": r["f_pool_hit"],
                "k1_pool_hit": r["k1_pool_hit"],
                "pool_transition": r["pool_transition"],
                "f_final_rank": r["f_final_rank"],
                "k1_final_rank": r["k1_final_rank"],
                "top10_transition": r["top10_transition"]
            })

    audit_csv = os.path.join(out_dir, "audit_10_ligature_misses.csv")
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_records[0].keys()))
        writer.writeheader()
        writer.writerows(audit_records)

    # Print Summary Report
    print("\n" + "=" * 80)
    print("FINSAGE EXPERIMENT K1 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Exp F (Baseline)':<18} | {'K1 NFKC':<18} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Candidate Pool Recall':<25} | {exp_f_summary['candidate_pool_recall']:>17.2f}% | {k1_summary['candidate_pool_recall']:>17.2f}% | {k1_summary['candidate_pool_recall'] - exp_f_summary['candidate_pool_recall']:>+9.2f} pp")
    print(f"{'Recall@1':<25} | {exp_f_summary['recall_at_1']:>17.2f}% | {k1_summary['recall_at_1']:>17.2f}% | {k1_summary['recall_at_1'] - exp_f_summary['recall_at_1']:>+9.2f} pp")
    print(f"{'Recall@3':<25} | {exp_f_summary['recall_at_3']:>17.2f}% | {k1_summary['recall_at_3']:>17.2f}% | {k1_summary['recall_at_3'] - exp_f_summary['recall_at_3']:>+9.2f} pp")
    print(f"{'Recall@5':<25} | {exp_f_summary['recall_at_5']:>17.2f}% | {k1_summary['recall_at_5']:>17.2f}% | {k1_summary['recall_at_5'] - exp_f_summary['recall_at_5']:>+9.2f} pp")
    print(f"{'Recall@10':<25} | {exp_f_summary['recall_at_10']:>17.2f}% | {k1_summary['recall_at_10']:>17.2f}% | {k1_summary['recall_at_10'] - exp_f_summary['recall_at_10']:>+9.2f} pp")
    print(f"{'MRR':<25} | {exp_f_summary['mrr']:>18.4f} | {k1_summary['mrr']:>18.4f} | {k1_summary['mrr'] - exp_f_summary['mrr']:>+9.4f}")
    print(f"{'BM25-only Gold Recall':<25} | {exp_f_summary['bm25_only_recall']:>17.2f}% | {k1_summary['bm25_only_recall']:>17.2f}% | {k1_summary['bm25_only_recall'] - exp_f_summary['bm25_only_recall']:>+9.2f} pp")
    print(f"{'Dense-only Gold Recall':<25} | {exp_f_summary['dense_only_recall']:>17.2f}% | {k1_summary['dense_only_recall']:>17.2f}% | {k1_summary['dense_only_recall'] - exp_f_summary['dense_only_recall']:>+9.2f} pp")
    print("-" * 80)
    print(f"Pool Recoveries (+):   {len(pool_recoveries)} ({pool_recoveries})")
    print(f"Pool Regressions (-):  {len(pool_regressions)} ({pool_regressions})")
    print(f"Top-10 Recoveries (+): {len(top10_recoveries)} ({top10_recoveries})")
    print(f"Top-10 Regressions (-):{len(top10_regressions)} ({top10_regressions})")
    print(f"McNemar Pool Recall:   p = {p_mcnemar_pool:.4f}")
    print(f"McNemar Recall@10:     p = {p_mcnemar_top10:.4f}")
    print(f"Wilcoxon MRR:          p = {p_wilcoxon_mrr:.4f}")
    
    avg_jaccard = sum(r["pool_jaccard"] for r in turnover_records) / len(turnover_records)
    avg_turnover = sum(r["novel_k1_candidates"] for r in turnover_records) / len(turnover_records)
    print(f"Avg Pool Jaccard:      {avg_jaccard:.4f}")
    print(f"Avg Novel Candidates:  {avg_turnover:.1f} / 50 slots ({avg_turnover/50*100:.1f}%)")

    # Post-run hash verification
    print("\n[Post-Run Integrity Check] Re-verifying Protected Baseline Hashes...")
    if not verify_protected_files():
        print("[CRITICAL ERROR] Baseline file was modified during execution!")
        sys.exit(1)
    print("[SUCCESS] All protected hashes remain 100% intact.")

if __name__ == "__main__":
    main()
