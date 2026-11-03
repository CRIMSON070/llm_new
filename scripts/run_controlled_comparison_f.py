"""
FinSage Experiment F: Fiscal-Period-Aware Candidate Generation
Controlled Head-to-Head Comparison: Reference (Exp C) vs Experiment F
"""

import os
import sys
import json
import time
import re
import csv
import importlib.util
from collections import defaultdict
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
# Dynamic import of Experiment C's Phase 2 Reranker (preserves active workspace)
# -------------------------------------------------------------------------
EXP_C_RERANKER_PATH = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_C_company_intent", "source_backup", "phase2_reranker.py"
)
spec = importlib.util.spec_from_file_location("exp_c_reranker", EXP_C_RERANKER_PATH)
exp_c_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_c_mod)
Phase2Reranker = exp_c_mod.Phase2Reranker


# -------------------------------------------------------------------------
# Temporal Intent Extraction Regexes & Document Parsing
# -------------------------------------------------------------------------
FY_PATTERN = re.compile(r"(?:FY\s*|fiscal\s*year\s*)(\d{4})|(\d{4})\s*fiscal\s*year", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:FY\s*|Q[1-4]\s*(?:of\s*)?FY\s*|\b)(20\d{2})\b", re.IGNORECASE)
QUARTER_PATTERN = re.compile(r"\bQ([1-4])\b(?:\s*(?:of\s*)?(?:FY\s*)?(\d{4}))?", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(?:FY\s*)?(\d{4})\s*(?:-|to|and)\s*(?:FY\s*)?(\d{4})", re.IGNORECASE)
DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")


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
            "ftype": ftype
        }
    return None


def detect_query_periods(query: str) -> Dict[str, Any]:
    """
    Extracts query temporal intent using ONLY user query string.
    Zero evaluation ground truth is accessible.
    """
    q_lower = query.lower()

    # 1. Range extraction
    ranges = RANGE_PATTERN.findall(query)
    range_years = set()
    for y1, y2 in ranges:
        y1_i, y2_i = int(y1), int(y2)
        range_years.add(y1_i)
        range_years.add(y2_i)
        if 1 <= abs(y2_i - y1_i) <= 5:
            low, high = min(y1_i, y2_i), max(y1_i, y2_i)
            for y in range(low, high + 1):
                range_years.add(y)

    # 2. Quarter extraction
    quarters = set()
    q_matches = QUARTER_PATTERN.findall(query)
    for q_num, q_yr in q_matches:
        quarters.add(int(q_num))
        if q_yr:
            range_years.add(int(q_yr))

    # 3. Explicit FY
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


def is_chunk_in_period(doc_meta: Optional[Dict[str, Any]], period_info: Dict[str, Any]) -> bool:
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
            quarter_match = True  # 10-K covers all quarters

    return year_match and quarter_match


def run_benchmark(
    mode: str,
    questions: List[Dict[str, Any]],
    mpr: MultiPathRetriever,
    reranker: Phase2Reranker,
    chunk_store: Dict[str, Dict[str, Any]],
    doc_meta_cache: Dict[str, Dict[str, Any]],
    out_dir: str
) -> Dict[str, Any]:
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

    t_start = time.time()
    print(f"\n[{mode.upper()}] Starting benchmark run across N={len(questions)} questions...")

    for idx, q_data in enumerate(questions, start=1):
        f_id = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence", [])

        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        # Multi-Path Retrieval (depth=50)
        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=50)
        meta_res, _ = mpr.retrieve_metadata(q_text, k_sections=3)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res,
            "path3_metadata": meta_res
        }

        if mode == "reference":
            # Exact Experiment C configuration
            candidates = mpr.reciprocal_rank_fusion(path_dict, k_pool=50)
        elif mode == "experiment":
            # Experiment F: Fiscal-Period-Aware Candidate Generation
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
                # In-period first, then safe backfill
                fused_cids = (in_period_cids + other_cids)[:50]
            else:
                fused_cids = [cid for cid, score in sorted_all[:50]]

            candidates = []
            for cid in fused_cids:
                c_data = chunk_store.get(cid, {})
                candidates.append({
                    "chunk_id": cid,
                    "text": c_data.get("text", ""),
                    "doc_name": c_data.get("doc_name"),
                    "page": c_data.get("page_number", 0),
                    "rrf_score": round(rrf_scores[cid], 6),
                    "path_contributions": path_contributions[cid],
                })
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Check candidate pool hit
        best_rank_in_pool = 999
        for c_idx, c in enumerate(candidates, start=1):
            if c.get("doc_name") == doc_name and c.get("page") in target_pages:
                best_rank_in_pool = c_idx
                break
        cand_hit = 1 if best_rank_in_pool <= 50 else 0
        candidate_pool_hits.append(cand_hit)

        # Rerank with Phase 2 Reranker (BAAI/bge-reranker-base, 512 context, calibrated boosts)
        reranked = reranker.rerank(query=q_text, candidates=candidates, top_k=10, apply_boosts=True)

        best_rank = 999
        for r_idx, c in enumerate(reranked, start=1):
            if c.get("doc_name") == doc_name and c.get("page") in target_pages:
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

        rec_record = {
            "index": idx,
            "financebench_id": f_id,
            "company": q_data.get("company"),
            "doc_name": doc_name,
            "sector": sector,
            "question": q_text[:70],
            "target_pages": target_pages,
            "best_rank": best_rank,
            "best_rank_in_pool": best_rank_in_pool,
            "candidate_pool_hit": cand_hit,
            "rr": round(rr, 4),
            "top3_retrieved": [f"{c['chunk_id']}_p{c['page']}" for c in reranked[:3]]
        }
        checkpoint_out.write(json.dumps(rec_record) + "\n")
        per_question_records.append(rec_record)

        if idx % 25 == 0 or idx == len(questions):
            interim_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            interim_r10 = sum(recall_at_10) / len(recall_at_10) * 100
            interim_cpool = sum(candidate_pool_hits) / len(candidate_pool_hits) * 100
            print(f"  [{mode:10s}] Processed [{idx:3d}/{len(questions)}] | Pool: {interim_cpool:.1f}% | R@10: {interim_r10:.1f}% | MRR: {interim_mrr:.4f}")

    checkpoint_out.close()
    runtime_sec = round(time.time() - t_start, 2)

    macro_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    macro_r1 = sum(recall_at_1) / len(recall_at_1) * 100
    macro_r3 = sum(recall_at_3) / len(recall_at_3) * 100
    macro_r5 = sum(recall_at_5) / len(recall_at_5) * 100
    macro_r10 = sum(recall_at_10) / len(recall_at_10) * 100
    cand_recall = sum(candidate_pool_hits) / len(candidate_pool_hits) * 100

    scorecard = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "total_questions": len(questions),
        "runtime_seconds": runtime_sec,
        "metrics": {
            "mrr": round(macro_mrr, 4),
            "recall_at_1": round(macro_r1, 2),
            "recall_at_3": round(macro_r3, 2),
            "recall_at_5": round(macro_r5, 2),
            "recall_at_10": round(macro_r10, 2),
            "candidate_pool_recall": round(cand_recall, 2),
            "candidate_pool_hits": sum(candidate_pool_hits),
            "top10_hits": sum(recall_at_10)
        }
    }

    with open(os.path.join(out_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    return {
        "scorecard": scorecard,
        "per_question": per_question_records
    }


def main():
    print("=" * 80)
    print("FINTEL / FINSAGE EXPERIMENT F: CONTROLLED COMPARISON")
    print("Reference (Experiment C) vs Experiment F (Fiscal-Period-Aware)")
    print("=" * 80)

    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")
    enriched_chunks_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")
    questions_file = os.path.join(data_dir, "questions.jsonl")

    # Load chunk store & build metadata cache
    print("\n1. Loading chunk store & building metadata cache...")
    chunk_store = {}
    doc_meta_cache = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
            d = c.get("doc_name")
            if d and d not in doc_meta_cache:
                doc_meta_cache[d] = parse_doc_meta(d)
    print(f"Loaded {len(chunk_store)} enriched chunks across {len(doc_meta_cache)} filings.")

    # Load questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} FinanceBench questions.")

    # Load indexes
    print("\n2. Loading FAISS Dense Index & Metadata Index (CUDA)...")
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

    # Initialize Reranker from Exp C
    print("\n3. Initializing Phase 2 Production Reranker from Experiment C...")
    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=True,
        statement_boost_weight=4.5,
        negative_disclosure_penalty=-3.0,
        company_boost_weight=1.5,
        period_boost_weight=1.5,
        table_boost_weight=0.5
    )

    out_base = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_F_period_aware")
    ref_dir = os.path.join(out_base, "reference")
    exp_dir = os.path.join(out_base, "experiment")

    # Run A: Reference Run (Exp C)
    res_ref = run_benchmark(
        mode="reference",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        out_dir=ref_dir
    )

    # Run B: Experiment F (Fiscal-Period-Aware)
    res_exp = run_benchmark(
        mode="experiment",
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        out_dir=exp_dir
    )

    # Generate Metrics Comparison JSON
    m_ref = res_ref["scorecard"]["metrics"]
    m_exp = res_exp["scorecard"]["metrics"]

    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scale": "N=150 FinanceBench questions across 84 SEC filings",
        "reference_experiment_c": {
            "runtime_seconds": res_ref["scorecard"]["runtime_seconds"],
            "metrics": m_ref
        },
        "experiment_f_period_aware": {
            "runtime_seconds": res_exp["scorecard"]["runtime_seconds"],
            "metrics": m_exp
        },
        "delta": {
            "candidate_pool_recall": round(m_exp["candidate_pool_recall"] - m_ref["candidate_pool_recall"], 2),
            "candidate_pool_hits": m_exp["candidate_pool_hits"] - m_ref["candidate_pool_hits"],
            "recall_at_1": round(m_exp["recall_at_1"] - m_ref["recall_at_1"], 2),
            "recall_at_3": round(m_exp["recall_at_3"] - m_ref["recall_at_3"], 2),
            "recall_at_5": round(m_exp["recall_at_5"] - m_ref["recall_at_5"], 2),
            "recall_at_10": round(m_exp["recall_at_10"] - m_ref["recall_at_10"], 2),
            "top10_hits": m_exp["top10_hits"] - m_ref["top10_hits"],
            "mrr": round(m_exp["mrr"] - m_ref["mrr"], 4),
            "runtime_overhead_seconds": round(res_exp["scorecard"]["runtime_seconds"] - res_ref["scorecard"]["runtime_seconds"], 2)
        }
    }

    metrics_comp_path = os.path.join(out_base, "metrics_comparison.json")
    with open(metrics_comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Generate Question-Level Comparison CSV
    csv_path = os.path.join(out_base, "question_level_comparison.csv")
    ref_by_qid = {r["financebench_id"]: r for r in res_ref["per_question"]}
    exp_by_qid = {r["financebench_id"]: r for r in res_exp["per_question"]}

    improved_count = 0
    regressed_count = 0
    unchanged_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question_id", "company", "gold_doc", "question_text",
            "ref_candidate_pool_hit", "exp_candidate_pool_hit", "candidate_pool_delta",
            "ref_candidate_rank", "exp_candidate_rank",
            "ref_final_rank", "exp_final_rank",
            "ref_top10_hit", "exp_top10_hit", "top10_delta",
            "ref_mrr", "exp_mrr", "mrr_delta",
            "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            qid = q.get("financebench_id")
            r_ref = ref_by_qid[qid]
            r_exp = exp_by_qid[qid]

            cp_ref = r_ref["candidate_pool_hit"]
            cp_exp = r_exp["candidate_pool_hit"]
            t10_ref = 1 if r_ref["best_rank"] <= 10 else 0
            t10_exp = 1 if r_exp["best_rank"] <= 10 else 0

            mrr_ref = r_ref["rr"]
            mrr_exp = r_exp["rr"]

            if t10_exp > t10_ref or (t10_exp == t10_ref and cp_exp > cp_ref) or mrr_exp > mrr_ref:
                status = "IMPROVED"
                improved_count += 1
            elif t10_exp < t10_ref or (t10_exp == t10_ref and cp_exp < cp_ref) or mrr_exp < mrr_ref:
                status = "REGRESSED"
                regressed_count += 1
            else:
                status = "UNCHANGED"
                unchanged_count += 1

            writer.writerow({
                "question_id": qid,
                "company": q.get("company"),
                "gold_doc": q.get("doc_name"),
                "question_text": q.get("question"),
                "ref_candidate_pool_hit": cp_ref,
                "exp_candidate_pool_hit": cp_exp,
                "candidate_pool_delta": cp_exp - cp_ref,
                "ref_candidate_rank": r_ref["best_rank_in_pool"],
                "exp_candidate_rank": r_exp["best_rank_in_pool"],
                "ref_final_rank": r_ref["best_rank"],
                "exp_final_rank": r_exp["best_rank"],
                "ref_top10_hit": t10_ref,
                "exp_top10_hit": t10_exp,
                "top10_delta": t10_exp - t10_ref,
                "ref_mrr": mrr_ref,
                "exp_mrr": mrr_exp,
                "mrr_delta": round(mrr_exp - mrr_ref, 4),
                "status": status
            })

    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Reference (Exp C)':<18} | {'Experiment F':<18} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Candidate Pool Recall':<25} | {m_ref['candidate_pool_recall']:>16.2f}% | {m_exp['candidate_pool_recall']:>16.2f}% | {comparison['delta']['candidate_pool_recall']:>+8.2f}%")
    print(f"{'Recall@1':<25} | {m_ref['recall_at_1']:>16.2f}% | {m_exp['recall_at_1']:>16.2f}% | {comparison['delta']['recall_at_1']:>+8.2f}%")
    print(f"{'Recall@3':<25} | {m_ref['recall_at_3']:>16.2f}% | {m_exp['recall_at_3']:>16.2f}% | {comparison['delta']['recall_at_3']:>+8.2f}%")
    print(f"{'Recall@5':<25} | {m_ref['recall_at_5']:>16.2f}% | {m_exp['recall_at_5']:>16.2f}% | {comparison['delta']['recall_at_5']:>+8.2f}%")
    print(f"{'Recall@10':<25} | {m_ref['recall_at_10']:>16.2f}% | {m_exp['recall_at_10']:>16.2f}% | {comparison['delta']['recall_at_10']:>+8.2f}%")
    print(f"{'Mean Reciprocal Rank':<25} | {m_ref['mrr']:>16.4f}  | {m_exp['mrr']:>16.4f}  | {comparison['delta']['mrr']:>+8.4f}")
    print(f"{'Runtime (seconds)':<25} | {res_ref['scorecard']['runtime_seconds']:>16.2f}s | {res_exp['scorecard']['runtime_seconds']:>16.2f}s | {comparison['delta']['runtime_overhead_seconds']:>+8.2f}s")
    print("=" * 80)
    print(f"Per-Question Status: {improved_count} IMPROVED, {regressed_count} REGRESSED, {unchanged_count} UNCHANGED")
    print("=" * 80)


if __name__ == "__main__":
    main()
