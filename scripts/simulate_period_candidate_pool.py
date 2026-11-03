import os
import sys
import json
import re
import torch
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

# Period detection regexes
FY_PATTERN = re.compile(r"(?:FY\s*|fiscal\s*year\s*)(\d{4})|(\d{4})\s*fiscal\s*year", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:FY\s*|Q[1-4]\s*(?:of\s*)?FY\s*|\b)(20\d{2})\b", re.IGNORECASE)
QUARTER_PATTERN = re.compile(r"\bQ([1-4])\b(?:\s*(?:of\s*)?(?:FY\s*)?(\d{4}))?", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(?:FY\s*)?(\d{4})\s*(?:-|to|and)\s*(?:FY\s*)?(\d{4})", re.IGNORECASE)
DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")

def parse_doc_meta(doc_name: str):
    if not doc_name:
        return None
    m = DOC_PARSE_REGEX.match(doc_name)
    if m:
        comp, yr, qtr, ftype = m.groups()
        return {
            "company": comp.upper(),
            "year": int(yr),
            "quarter": int(qtr) if qtr else None,
            "ftype": ftype
        }
    return None

def detect_query_periods(query: str):
    """
    Detects fiscal periods from query text without any ground truth labels.
    Returns:
        dict with:
            years: set of int years
            quarters: set of int quarters (or empty)
            has_explicit_fy: bool
            is_range: bool
    """
    q_lower = query.lower()
    
    # 1. Check for ranges
    ranges = RANGE_PATTERN.findall(query)
    range_years = set()
    for y1, y2 in ranges:
        y1_i, y2_i = int(y1), int(y2)
        range_years.add(y1_i)
        range_years.add(y2_i)
        # If range like 2018-2020, add intermediate years too
        if 1 <= abs(y2_i - y1_i) <= 5:
            low, high = min(y1_i, y2_i), max(y1_i, y2_i)
            for y in range(low, high + 1):
                range_years.add(y)

    # 2. Check for explicit quarters
    quarters = set()
    q_matches = QUARTER_PATTERN.findall(query)
    for q_num, q_yr in q_matches:
        quarters.add(int(q_num))
        if q_yr:
            range_years.add(int(q_yr))

    # 3. Check for explicit FYs
    fy_years = set()
    for m in FY_PATTERN.finditer(query):
        for g in m.groups():
            if g:
                fy_years.add(int(g))

    # 4. All 4-digit years
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

def is_chunk_in_period(doc_meta: dict, period_info: dict) -> bool:
    if not doc_meta or not period_info:
        return False
    
    req_years = period_info["years"]
    req_quarters = period_info["quarters"]

    if not req_years and not req_quarters:
        return False

    chunk_year = doc_meta["year"]
    chunk_quarter = doc_meta["quarter"]

    # Year check:
    year_match = False
    if req_years:
        if chunk_year in req_years:
            year_match = True
        elif period_info.get("is_forward_looking") and doc_meta.get("quarter") == 4 and (chunk_year + 1) in req_years:
            # Q4 earnings call of prior year provides guidance for next year
            year_match = True
    else:
        year_match = True

    # Quarter check:
    quarter_match = True
    if req_quarters:
        if chunk_quarter is not None:
            quarter_match = (chunk_quarter in req_quarters)
        else:
            quarter_match = True  # 10-K is acceptable as fallback for quarterly inquiries

    return year_match and quarter_match

def main():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")
    enriched_chunks_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")

    chunk_store = {}
    doc_meta_cache = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
            d = c.get("doc_name")
            if d and d not in doc_meta_cache:
                doc_meta_cache[d] = parse_doc_meta(d)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    with open(os.path.join(data_dir, "questions.jsonl"), "r", encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f]

    print(f"Evaluating candidate pool recall across N={len(all_questions)} questions...")

    hits_baseline_pool = 0
    hits_method_a_pool = 0
    hits_method_b_pool = 0

    improvements = []
    regressions = []

    for idx, q_data in enumerate(all_questions, start=1):
        f_id = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        gold_doc = q_data["doc_name"]
        target_pages = [e.get("evidence_page_num") + 1 for e in q_data.get("evidence", []) if e.get("evidence_page_num") is not None]

        # Multi-Path Retrieval (depth=50)
        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=50)
        meta_res, _ = mpr.retrieve_metadata(q_text, k_sections=3)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res,
            "path3_metadata": meta_res
        }

        # 1. Baseline RRF (Experiment C: [:50])
        baseline_candidates = mpr.reciprocal_rank_fusion(path_dict, k_pool=50)
        hit_base = any(c.get("doc_name") == gold_doc and c.get("page") in target_pages for c in baseline_candidates)
        if hit_base: hits_baseline_pool += 1

        # 2. Method A: Period-Aware Partitioning with Fallback
        p_info = detect_query_periods(q_text)
        
        # Calculate full RRF scores without [:50] cutoff
        rrf_scores = defaultdict(float)
        for p_name, items in path_dict.items():
            for rank, (cid, score) in enumerate(items, start=1):
                rrf_scores[cid] += 1.0 / (60 + rank)
        
        sorted_all_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        in_period_cids = []
        other_cids = []

        if p_info["years"] or p_info["quarters"]:
            for cid, score in sorted_all_cids:
                c = chunk_store.get(cid, {})
                d = c.get("doc_name")
                d_meta = doc_meta_cache.get(d)
                if is_chunk_in_period(d_meta, p_info):
                    in_period_cids.append(cid)
                else:
                    other_cids.append(cid)
            
            # Combine: in-period first, then fallback from other_cids to reach 50
            fused_method_a = (in_period_cids + other_cids)[:50]
        else:
            fused_method_a = [cid for cid, score in sorted_all_cids[:50]]

        hit_method_a = any(chunk_store.get(cid, {}).get("doc_name") == gold_doc and chunk_store.get(cid, {}).get("page_number") in target_pages for cid in fused_method_a)
        if hit_method_a: hits_method_a_pool += 1

        if hit_method_a and not hit_base:
            improvements.append((f_id, gold_doc, q_text[:70]))
        elif hit_base and not hit_method_a:
            regressions.append((f_id, gold_doc, q_text[:70]))

    print("\n" + "=" * 70)
    print(f"CANDIDATE POOL EVALUATION (N={len(all_questions)})")
    print(f"  Baseline Exp C Pool Hits: {hits_baseline_pool} / {len(all_questions)} ({hits_baseline_pool/len(all_questions)*100:.2f}%)")
    print(f"  Method A (Period-Aware) : {hits_method_a_pool} / {len(all_questions)} ({hits_method_a_pool/len(all_questions)*100:.2f}%)")
    print(f"  Net Change              : {hits_method_a_pool - hits_baseline_pool:+d} questions ({(hits_method_a_pool - hits_baseline_pool)/len(all_questions)*100:+.2f} pp)")
    print(f"  Improvements            : {len(improvements)}")
    for qid, doc, txt in improvements:
        print(f"    + [GAINED] {qid} | Doc: {doc} | {txt}")
    print(f"  Regressions             : {len(regressions)}")
    for qid, doc, txt in regressions:
        print(f"    - [LOST]   {qid} | Doc: {doc} | {txt}")
    print("=" * 70)

if __name__ == "__main__":
    main()
