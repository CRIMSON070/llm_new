"""
scripts/run_controlled_comparison_h1.py
=======================================
FinSage Experiment H1: Dense Retrieval Depth Ablation
Controlled Head-to-Head Comparison:
- Reference Baseline: Experiment F (Dense Depth k=50, Pool K=50)
- Experiment H1a: Dense Depth k=100 (Pool K=50)
- Experiment H1b: Dense Depth k=150 (Pool K=50)

Strict constraints:
1. Evaluates identical 150 FinanceBench benchmark questions.
2. Changes ONLY dense retrieval depth (k=50 vs k=100 vs k=150).
3. Candidate pool kept strictly at top 50 across all runs.
4. Preserves all existing experiment results, checkpoints, and backups.
5. Writes outputs exclusively to:
   outputs/experiments/experiment_H1_dense_depth/
6. Uses pre-existing indexes without rebuilding.
"""

import os
import sys
import json
import time
import re
import csv
import importlib.util
from datetime import datetime
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
# Temporal Intent Extraction Regexes & Document Parsing (Identical to Exp F)
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
    q_lower = query.lower()

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

    quarters = set()
    q_matches = QUARTER_PATTERN.findall(query)
    for q_num, q_yr in q_matches:
        quarters.add(int(q_num))
        if q_yr:
            range_years.add(int(q_yr))

    fy_years = set()
    for m in FY_PATTERN.finditer(query):
        for g in m.groups():
            if g:
                fy_years.add(int(g))

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
# Single Run Execution Engine
# -------------------------------------------------------------------------
def run_evaluation(
    run_label: str,
    dense_depth: int,
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

    print(f"\n[{run_label}] Running benchmark across N={len(questions)} questions (dense_k={dense_depth}, pool_k=50)...")

    for idx, q_data in enumerate(questions, start=1):
        f_id = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence", [])

        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        # Multi-Path Retrieval: sparse=50, dense=dense_depth, meta=3
        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=dense_depth)
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

        # Check candidate pool hit
        best_rank_in_pool = 999
        for c_idx, c in enumerate(candidates, start=1):
            if c.get("doc_name") == doc_name and c.get("page") in target_pages:
                best_rank_in_pool = c_idx
                break
        cand_hit = 1 if best_rank_in_pool <= 50 else 0
        candidate_pool_hits.append(cand_hit)

        # Rerank with Phase 2 Reranker (BAAI/bge-reranker-base, context=512, calibrated boosts)
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

        record = {
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
        "run_label": run_label,
        "dense_depth": dense_depth,
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

    return {
        "summary": summary,
        "records": per_question_records
    }


# -------------------------------------------------------------------------
# Main Controller Routine
# -------------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_H1_dense_depth")
    os.makedirs(out_root, exist_ok=True)

    h1a_dir = os.path.join(out_root, "h1a_depth100")
    h1b_dir = os.path.join(out_root, "h1b_depth150")
    ref_dir = os.path.join(out_root, "reference_depth50")

    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")
    enriched_chunks_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")

    # 1. Load questions (all 150)
    questions = []
    with open(os.path.join(data_dir, "questions.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"[Init] Loaded {len(questions)} FinanceBench benchmark questions.")

    # 2. Identify the 38 Experiment G failure questions
    exp_f_checkpoint = os.path.join(
        BASE_DIR, "outputs", "experiments", "experiment_F_period_aware", "experiment", "phase2_retrieval_checkpoint.jsonl"
    )
    failed_38_ids = set()
    with open(exp_f_checkpoint, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("candidate_pool_hit") == 0:
                failed_38_ids.add(r["financebench_id"])
    print(f"[Init] Identified {len(failed_38_ids)} reference candidate pool misses from Experiment F.")

    # 3. Load chunk store & doc metadata
    print("[Init] Loading chunk store and metadata cache...")
    chunk_store = {}
    doc_meta_cache = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
            dname = c.get("doc_name")
            if dname and dname not in doc_meta_cache:
                doc_meta_cache[dname] = parse_doc_meta(dname)

    # 4. Initialize Multi-Path Retriever & Reranker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading retrieval indexes and models on device: {device}...")
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

    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=True,
        statement_boost_weight=4.5,
        negative_disclosure_penalty=-3.0,
        company_boost_weight=1.5,
        period_boost_weight=1.5,
        table_boost_weight=0.5
    )

    # ---------------------------------------------------------------------
    # Execute Evaluation Runs
    # ---------------------------------------------------------------------
    # Run 0: Reference Baseline (dense_k=50)
    res_ref = run_evaluation(
        run_label="Reference_Depth50",
        dense_depth=50,
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        out_dir=ref_dir
    )

    # Run 1: Experiment H1a (dense_k=100)
    res_h1a = run_evaluation(
        run_label="Experiment_H1a_Depth100",
        dense_depth=100,
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        out_dir=h1a_dir
    )

    # Run 2: Experiment H1b (dense_k=150)
    res_h1b = run_evaluation(
        run_label="Experiment_H1b_Depth150",
        dense_depth=150,
        questions=questions,
        mpr=mpr,
        reranker=reranker,
        chunk_store=chunk_store,
        doc_meta_cache=doc_meta_cache,
        out_dir=h1b_dir
    )

    # ---------------------------------------------------------------------
    # Pool Turnover Analysis & Question-Level Comparison
    # ---------------------------------------------------------------------
    print("\n[Analysis] Computing pool turnover, regressions, and question deltas...")
    q_map_ref = {r["financebench_id"]: r for r in res_ref["records"]}
    q_map_h1a = {r["financebench_id"]: r for r in res_h1a["records"]}
    q_map_h1b = {r["financebench_id"]: r for r in res_h1b["records"]}

    comparison_rows = []
    turnover_rows = []

    # 38-failure tracking
    recovered_in_h1a = []
    recovered_in_h1b = []

    for q in questions:
        qid = q["financebench_id"]
        r_ref = q_map_ref[qid]
        r_h1a = q_map_h1a[qid]
        r_h1b = q_map_h1b[qid]

        pool_ref = set(r_ref["pool_chunk_ids"])
        pool_h1a = set(r_h1a["pool_chunk_ids"])
        pool_h1b = set(r_h1b["pool_chunk_ids"])

        # Turnover vs Reference
        jaccard_h1a = len(pool_ref.intersection(pool_h1a)) / len(pool_ref.union(pool_h1a))
        jaccard_h1b = len(pool_ref.intersection(pool_h1b)) / len(pool_ref.union(pool_h1b))
        new_chunks_h1a = pool_h1a - pool_ref
        new_chunks_h1b = pool_h1b - pool_ref
        displaced_chunks_h1a = pool_ref - pool_h1a
        displaced_chunks_h1b = pool_ref - pool_h1b

        turnover_rows.append({
            "question_id": qid,
            "company": q.get("company"),
            "ref_pool_hit": r_ref["candidate_pool_hit"],
            "h1a_pool_hit": r_h1a["candidate_pool_hit"],
            "h1b_pool_hit": r_h1b["candidate_pool_hit"],
            "h1a_jaccard_sim": round(jaccard_h1a, 4),
            "h1b_jaccard_sim": round(jaccard_h1b, 4),
            "h1a_new_chunk_count": len(new_chunks_h1a),
            "h1b_new_chunk_count": len(new_chunks_h1b),
            "h1a_displaced_chunk_count": len(displaced_chunks_h1a),
            "h1b_displaced_chunk_count": len(displaced_chunks_h1b)
        })

        # Failure Slice check
        if qid in failed_38_ids:
            if r_h1a["candidate_pool_hit"] == 1:
                recovered_in_h1a.append(qid)
            if r_h1b["candidate_pool_hit"] == 1:
                recovered_in_h1b.append(qid)

        # Status for H1b vs Ref
        if r_ref["candidate_pool_hit"] == 0 and r_h1b["candidate_pool_hit"] == 1:
            status = "RECOVERED"
        elif r_ref["candidate_pool_hit"] == 1 and r_h1b["candidate_pool_hit"] == 0:
            status = "REGRESSED"
        elif r_ref["candidate_pool_hit"] == 1 and r_h1b["candidate_pool_hit"] == 1:
            status = "STABLE_HIT"
        else:
            status = "STABLE_MISS"

        comparison_rows.append({
            "question_id": qid,
            "company": q.get("company"),
            "sector": q.get("gics_sector"),
            "is_exp_g_38_miss": 1 if qid in failed_38_ids else 0,
            "ref_cand_hit": r_ref["candidate_pool_hit"],
            "h1a_cand_hit": r_h1a["candidate_pool_hit"],
            "h1b_cand_hit": r_h1b["candidate_pool_hit"],
            "ref_cand_rank": r_ref["best_rank_in_pool"],
            "h1a_cand_rank": r_h1a["best_rank_in_pool"],
            "h1b_cand_rank": r_h1b["best_rank_in_pool"],
            "ref_final_rank": r_ref["best_rank"],
            "h1a_final_rank": r_h1a["best_rank"],
            "h1b_final_rank": r_h1b["best_rank"],
            "ref_mrr": r_ref["rr"],
            "h1a_mrr": r_h1a["rr"],
            "h1b_mrr": r_h1b["rr"],
            "h1b_vs_ref_status": status
        })

    # Save Turnover CSV
    turnover_csv = os.path.join(out_root, "pool_turnover_analysis.csv")
    with open(turnover_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=turnover_rows[0].keys())
        writer.writeheader()
        writer.writerows(turnover_rows)

    # Save Comparison CSV
    comp_csv = os.path.join(out_root, "question_level_comparison.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    # ---------------------------------------------------------------------
    # Sector Breakdown
    # ---------------------------------------------------------------------
    sectors = sorted(list(set(q.get("gics_sector", "General") for q in questions)))
    sector_summary = []
    for sec in sectors:
        sec_qids = [q["financebench_id"] for q in questions if q.get("gics_sector", "General") == sec]
        n_sec = len(sec_qids)
        if n_sec == 0: continue

        ref_cp = sum(q_map_ref[qid]["candidate_pool_hit"] for qid in sec_qids) / n_sec * 100
        h1a_cp = sum(q_map_h1a[qid]["candidate_pool_hit"] for qid in sec_qids) / n_sec * 100
        h1b_cp = sum(q_map_h1b[qid]["candidate_pool_hit"] for qid in sec_qids) / n_sec * 100

        ref_r10 = sum(1 for qid in sec_qids if q_map_ref[qid]["best_rank"] <= 10) / n_sec * 100
        h1a_r10 = sum(1 for qid in sec_qids if q_map_h1a[qid]["best_rank"] <= 10) / n_sec * 100
        h1b_r10 = sum(1 for qid in sec_qids if q_map_h1b[qid]["best_rank"] <= 10) / n_sec * 100

        ref_mrr = sum(q_map_ref[qid]["rr"] for qid in sec_qids) / n_sec
        h1b_mrr = sum(q_map_h1b[qid]["rr"] for qid in sec_qids) / n_sec

        sector_summary.append({
            "sector": sec,
            "count": n_sec,
            "ref_pool_recall": round(ref_cp, 2),
            "h1a_pool_recall": round(h1a_cp, 2),
            "h1b_pool_recall": round(h1b_cp, 2),
            "h1b_pool_delta": round(h1b_cp - ref_cp, 2),
            "ref_recall_at_10": round(ref_r10, 2),
            "h1b_recall_at_10": round(h1b_r10, 2),
            "ref_mrr": round(ref_mrr, 4),
            "h1b_mrr": round(h1b_mrr, 4),
        })

    sector_csv = os.path.join(out_root, "sector_metrics.csv")
    with open(sector_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sector_summary[0].keys())
        writer.writeheader()
        writer.writerows(sector_summary)

    # ---------------------------------------------------------------------
    # Save Metrics Comparison JSON & Report Markdown
    # ---------------------------------------------------------------------
    m_ref = res_ref["summary"]
    m_h1a = res_h1a["summary"]
    m_h1b = res_h1b["summary"]

    comparison_json = {
        "timestamp": timestamp,
        "reference_experiment_f": m_ref,
        "experiment_h1a_depth100": m_h1a,
        "experiment_h1b_depth150": m_h1b,
        "h1a_delta_vs_ref": {
            "candidate_pool_recall": round(m_h1a["candidate_pool_recall"] - m_ref["candidate_pool_recall"], 2),
            "candidate_pool_hits": m_h1a["candidate_pool_hits"] - m_ref["candidate_pool_hits"],
            "recall_at_1": round(m_h1a["recall_at_1"] - m_ref["recall_at_1"], 2),
            "recall_at_3": round(m_h1a["recall_at_3"] - m_ref["recall_at_3"], 2),
            "recall_at_5": round(m_h1a["recall_at_5"] - m_ref["recall_at_5"], 2),
            "recall_at_10": round(m_h1a["recall_at_10"] - m_ref["recall_at_10"], 2),
            "mrr": round(m_h1a["mrr"] - m_ref["mrr"], 4),
            "recovered_exp_g_misses": len(recovered_in_h1a)
        },
        "h1b_delta_vs_ref": {
            "candidate_pool_recall": round(m_h1b["candidate_pool_recall"] - m_ref["candidate_pool_recall"], 2),
            "candidate_pool_hits": m_h1b["candidate_pool_hits"] - m_ref["candidate_pool_hits"],
            "recall_at_1": round(m_h1b["recall_at_1"] - m_ref["recall_at_1"], 2),
            "recall_at_3": round(m_h1b["recall_at_3"] - m_ref["recall_at_3"], 2),
            "recall_at_5": round(m_h1b["recall_at_5"] - m_ref["recall_at_5"], 2),
            "recall_at_10": round(m_h1b["recall_at_10"] - m_ref["recall_at_10"], 2),
            "mrr": round(m_h1b["mrr"] - m_ref["mrr"], 4),
            "recovered_exp_g_misses": len(recovered_in_h1b)
        }
    }

    metrics_json_path = os.path.join(out_root, "metrics_comparison.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2)

    # Generate Markdown Report
    report_path = os.path.join(out_root, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Experiment H1: Dense Retrieval Depth Ablation Report\n\n")
        f.write(f"**Execution Timestamp:** `{timestamp}`  \n")
        f.write(f"**Benchmark Scale:** N=150 FinanceBench Questions across 84 SEC Filings  \n")
        f.write(f"**Isolated Directory:** `outputs/experiments/experiment_H1_dense_depth/`  \n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary & Benchmark Matrix\n\n")
        f.write("| Configuration | Dense Depth | Candidate Pool Size | Candidate Pool Recall | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Runtime |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Reference (Exp F)** | 50 | 50 | **{m_ref['candidate_pool_recall']:.2f}%** ({m_ref['candidate_pool_hits']}/150) | {m_ref['recall_at_1']:.2f}% | {m_ref['recall_at_3']:.2f}% | {m_ref['recall_at_5']:.2f}% | {m_ref['recall_at_10']:.2f}% | {m_ref['mrr']:.4f} | {m_ref['runtime_seconds']}s |\n")
        f.write(f"| **Experiment H1a** | 100 | 50 | **{m_h1a['candidate_pool_recall']:.2f}%** ({m_h1a['candidate_pool_hits']}/150) | {m_h1a['recall_at_1']:.2f}% | {m_h1a['recall_at_3']:.2f}% | {m_h1a['recall_at_5']:.2f}% | {m_h1a['recall_at_10']:.2f}% | {m_h1a['mrr']:.4f} | {m_h1a['runtime_seconds']}s |\n")
        f.write(f"| **Experiment H1b** | 150 | 50 | **{m_h1b['candidate_pool_recall']:.2f}%** ({m_h1b['candidate_pool_hits']}/150) | {m_h1b['recall_at_1']:.2f}% | {m_h1b['recall_at_3']:.2f}% | {m_h1b['recall_at_5']:.2f}% | {m_h1b['recall_at_10']:.2f}% | {m_h1b['mrr']:.4f} | {m_h1b['runtime_seconds']}s |\n\n")

        f.write("## 2. Recovery on the 38 Experiment G Failed Questions\n\n")
        f.write(f"- **Recovered in H1a (depth=100):** {len(recovered_in_h1a)} / 38 ({len(recovered_in_h1a)/38*100:.1f}%)  \n")
        f.write(f"- **Recovered in H1b (depth=150):** {len(recovered_in_h1b)} / 38 ({len(recovered_in_h1b)/38*100:.1f}%)  \n\n")

        if recovered_in_h1b:
            f.write("| Question ID | Company | Exp F Pool Rank | H1a Pool Rank | H1b Pool Rank | Final Reranked Rank |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
            for qid in recovered_in_h1b:
                f.write(f"| `{qid}` | {q_map_ref[qid]['company']} | >50 | {q_map_h1a[qid]['best_rank_in_pool']} | {q_map_h1b[qid]['best_rank_in_pool']} | {q_map_h1b[qid]['best_rank']} |\n")
            f.write("\n")

        f.write("## 3. Candidate Pool Turnover & Stability Analysis\n\n")
        avg_turnover_h1a = sum(t["h1a_new_chunk_count"] for t in turnover_rows) / len(turnover_rows)
        avg_turnover_h1b = sum(t["h1b_new_chunk_count"] for t in turnover_rows) / len(turnover_rows)
        avg_jaccard_h1a = sum(t["h1a_jaccard_sim"] for t in turnover_rows) / len(turnover_rows)
        avg_jaccard_h1b = sum(t["h1b_jaccard_sim"] for t in turnover_rows) / len(turnover_rows)
        f.write(f"- **Average Top-50 Jaccard Similarity (H1a vs Ref):** {avg_jaccard_h1a*100:.2f}% (Average {avg_turnover_h1a:.1f} new chunks per question)\n")
        f.write(f"- **Average Top-50 Jaccard Similarity (H1b vs Ref):** {avg_jaccard_h1b*100:.2f}% (Average {avg_turnover_h1b:.1f} new chunks per question)\n\n")

        f.write("## 4. Sector-Level Breakdown\n\n")
        f.write("| GICS Sector | Count | Exp F Pool Recall | H1b Pool Recall | Delta | Exp F Recall@10 | H1b Recall@10 | Exp F MRR | H1b MRR |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in sector_summary:
            f.write(f"| **{s['sector']}** | {s['count']} | {s['ref_pool_recall']:.1f}% | {s['h1b_pool_recall']:.1f}% | {s['h1b_pool_delta']:+.1f}% | {s['ref_recall_at_10']:.1f}% | {s['h1b_recall_at_10']:.1f}% | {s['ref_mrr']:.4f} | {s['h1b_mrr']:.4f} |\n")
        f.write("\n")

        f.write("## 5. Summary & Conclusions\n\n")
        delta_cp = m_h1b['candidate_pool_recall'] - m_ref['candidate_pool_recall']
        delta_r10 = m_h1b['recall_at_10'] - m_ref['recall_at_10']
        delta_mrr = m_h1b['mrr'] - m_ref['mrr']
        f.write(f"- Candidate Pool Recall changed by **{delta_cp:+.2f}%** ({m_h1b['candidate_pool_hits'] - m_ref['candidate_pool_hits']:+d} net questions).\n")
        f.write(f"- Recall@10 changed by **{delta_r10:+.2f}%** ({m_h1b['top10_hits'] - m_ref['top10_hits']:+d} net questions).\n")
        f.write(f"- MRR changed by **{delta_mrr:+.4f}**.\n")

    print(f"\n[Done] Experiment H1 execution completed. Results saved in {out_root}.")


if __name__ == "__main__":
    main()
