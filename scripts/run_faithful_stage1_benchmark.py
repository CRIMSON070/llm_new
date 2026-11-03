"""
FinSage Stage 1: Faithful-Methodology Reproduction with Local Model Adaptations
================================================================================
Evaluation Pipeline for all 150 FinanceBench Questions.

Paper Reference:
    FinSage: A Multi-aspect RAG System for Financial Filings Question Answering
    (arXiv:2504.14493, CIKM 2025) Section 3.2, 4.2.1, and Appendix B.

Reproduction Label:
    "Stage 1 faithful-methodology reproduction with local model adaptations"
    (Local adaptations: Qwen2.5-7B for HyDE rather than Qwen2.5-72B; locally-trained
     domain DPO adapter on BAAI/bge-reranker-v2-gemma).

Pipeline Architecture & Parameters:
    1. Multi-Path Retrieval (MPR):
       - Path 1: BM25 Sparse Lexical (depth k=40)
       - Path 2: BGE-M3 Dense FAISS (depth k=40)
       - Path 3: Heading-Summary Metadata FAISS (k_sections=4)
       - Path 4: HyDE-1 via Qwen2.5-7B-Instruct (3 hypothetical passages, k_per_hypo=20)
    2. Reciprocal Rank Fusion (RRF):
       - k_rrf = 60, Candidate Pool Size K_pool = 40
    3. Three-Way Evaluation (Fixed Candidate Pools):
       - Stage 1 Baseline: Initial Multi-Path RRF ranking (prior to cross-encoder)
       - Arm 1: Base Cross-Encoder (BAAI/bge-reranker-v2-gemma, 4-bit NF4)
       - Arm 2: Domain Re-Ranker (BAAI/bge-reranker-v2-gemma + outputs/phase1_drr_adapter/)
       - Zero heuristic boosts applied (+0.0 boost across all candidates)
    4. Metrics Separation:
       - Set 1 (Raw Candidate Retrieval): MRR, nDCG@10, Recall@1, 3, 5, 10, Pool Recall@40
       - Set 2 (Bundled Evidence Coverage): Post-bundling coverage (tau_exp=0.70, max_size=5)
    5. Dataset & Leakage Stratification:
       - Full 150 questions evaluated (paper reports 149; unknown excluded item)
       - Metrics stratified across DPO train (125), DPO val (17), and strictly held-out (8)
"""

import os
import sys
import gc
import json
import time
import math
import hashlib
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.hyde_generator import GenerativeHyDE
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.chunk_bundler import ChunkBundler


# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
REPRODUCTION_LABEL = "Stage 1 faithful-methodology reproduction with local model adaptations"
QUESTIONS_FILE = os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "financebench", "processed", "parsed_chunks.jsonl")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "faiss_index")
BM25_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "bm25_index")
META_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "metadata_faiss_index")
SECTIONS_FILE = os.path.join(BASE_DIR, "data", "financebench", "processed", "parsed_sections.jsonl")

DRR_ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter")
BASE_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
DPO_TRAIN_FILE = os.path.join(BASE_DIR, "data", "financebench", "drr", "train_drr.jsonl")
DPO_VAL_FILE = os.path.join(BASE_DIR, "data", "financebench", "drr", "val_drr.jsonl")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reproduction_stage1_faithful")
HYDE_CACHE_FILE = os.path.join(OUTPUT_DIR, "hyde_generated_passages_cache.json")

# Core RRF and Bundling parameters
RRF_K = 60
K_POOL = 40
TAU_EXP = 0.70
MAX_BUNDLE_SIZE = 5

CROSS_ENCODER_INSTRUCTION = (
    "Given a query A and a passage B, determine whether the passage contains an "
    "answer to the query by providing a prediction of either 'Yes' or 'No'."
)


# -----------------------------------------------------------------------------
# Integrity & Hash Auditing
# -----------------------------------------------------------------------------
PROTECTED_TARGETS = [
    os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter", "adapter_model.safetensors"),
    os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter", "adapter_config.json"),
    os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl"),
    os.path.join(BASE_DIR, "data", "financebench", "drr", "train_drr.jsonl"),
]

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.isfile(filepath):
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def record_protected_hashes() -> Dict[str, str]:
    """Capture hashes of protected artifacts."""
    return {p: compute_file_sha256(p) for p in PROTECTED_TARGETS}

def verify_protected_hashes(baseline_hashes: Dict[str, str]):
    """Verify no protected file was altered during the run."""
    current_hashes = record_protected_hashes()
    tampered = []
    for path, expected in baseline_hashes.items():
        curr = current_hashes.get(path)
        if curr != expected:
            tampered.append((path, expected, curr))
    if tampered:
        raise RuntimeError(f"CRITICAL: Protected files modified during run! {tampered}")
    print("[Integrity] Verified all protected SHA-256 hashes remain identical.")


# -----------------------------------------------------------------------------
# Data Loading & DPO Stratification
# -----------------------------------------------------------------------------
def load_questions_with_splits(questions_path: str) -> List[Dict[str, Any]]:
    """
    Load all 150 questions and tag their DPO leakage split:
    - 'train_triplet': question was used in DPO training (125 items)
    - 'val_triplet': question was used in DPO validation (17 items)
    - 'strictly_held_out': question had zero exposure to DPO training (8 items)
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Load DPO splits
    train_ids = set()
    if os.path.exists(DPO_TRAIN_FILE):
        with open(DPO_TRAIN_FILE, "r", encoding="utf-8") as f:
            train_ids = {json.loads(line).get("financebench_id") for line in f if line.strip()}

    val_ids = set()
    if os.path.exists(DPO_VAL_FILE):
        with open(DPO_VAL_FILE, "r", encoding="utf-8") as f:
            val_ids = {json.loads(line).get("financebench_id") for line in f if line.strip()}

    tagged_questions = []
    for q in questions:
        qid = q.get("financebench_id")
        if qid in train_ids:
            split = "train_triplet"
        elif qid in val_ids:
            split = "val_triplet"
        else:
            split = "strictly_held_out"
        
        q_copy = dict(q)
        q_copy["dpo_leakage_split"] = split
        
        # Parse ground-truth target pages: 0-indexed in questions.jsonl -> 1-indexed PDF
        target_pages = []
        target_doc = q.get("doc_name")
        for ev in q.get("evidence", []):
            ev_p = ev.get("evidence_page_num")
            if ev_p is not None:
                target_pages.append(ev_p + 1)
        q_copy["gold_target_doc"] = target_doc
        q_copy["gold_target_pages"] = sorted(list(set(target_pages)))
        tagged_questions.append(q_copy)

    return tagged_questions


# -----------------------------------------------------------------------------
# HyDE Generation Phase (Sequential & Decoupled)
# -----------------------------------------------------------------------------
def generate_or_load_hyde_cache(
    questions: List[Dict[str, Any]],
    cache_path: str,
    device: str = "cuda"
) -> Dict[str, List[str]]:
    """
    Generate or load cached HyDE hypothetical passages for all questions.
    Uses local Ollama API if running; otherwise initializes Qwen2.5-7B in 4-bit,
    generates all passages, and cleanly unloads to free VRAM.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"[HyDE] Loading existing HyDE passage cache from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        missing = [q["financebench_id"] for q in questions if q["financebench_id"] not in cache]
        if not missing:
            print(f"[HyDE] Cache complete with {len(cache)} entries. Zero GPU memory consumed.")
            return cache
        print(f"[HyDE] Cache contains {len(cache)} entries; {len(missing)} missing entries to generate.")
    else:
        cache = {}

    print(f"[HyDE] Initializing GenerativeHyDE generator...")
    hyde_gen = GenerativeHyDE(model_name="qwen2.5:7b")
    
    total_q = len(questions)
    for idx, q in enumerate(questions, start=1):
        qid = q["financebench_id"]
        if qid in cache:
            continue
        query_text = q["question"]
        t0 = time.time()
        passages = hyde_gen.generate_hypothetical_passages(
            query=query_text,
            num_hypo=3,
            max_new_tokens=256,
            temperature=0.3
        )
        cache[qid] = passages
        elapsed = time.time() - t0
        if idx % 10 == 0 or idx == total_q:
            print(f"[HyDE Generation] [{idx}/{total_q}] Generated passages for {qid} in {elapsed:.2f}s")

    # Save cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"[HyDE] Saved full passage cache to {cache_path}.")

    # Unload generator model if HF fallback was used
    if hasattr(hyde_gen, "hf_model") and hyde_gen.hf_model is not None:
        print("[HyDE] Unloading Hugging Face Qwen2.5-7B model from GPU memory...")
        del hyde_gen.hf_model
        del hyde_gen.hf_tokenizer
        del hyde_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[HyDE] Qwen model unloaded. Active VRAM: {round(torch.cuda.memory_allocated() / 1e6, 2)} MB")

    return cache


# -----------------------------------------------------------------------------
# Fixed Candidate Pool Generation (Multi-Path Retrieval + RRF)
# -----------------------------------------------------------------------------
def build_fixed_candidate_pools(
    questions: List[Dict[str, Any]],
    hyde_cache: Dict[str, List[str]],
    bge_embeddings: HuggingFaceEmbeddings,
    k_pool: int = 40
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Execute MPR (BM25, Dense FAISS, Metadata FAISS, HyDE) and RRF fusion (k=60)
    to produce FIXED 40-candidate pools for each question.
    """
    print("\n" + "=" * 70)
    print(f"[MPR] Initializing Multi-Path Retrieval with RRF (k_rrf={RRF_K}, k_pool={k_pool})...")
    print("=" * 70)

    # 1. Load BM25 Retriever
    bm25 = BM25Retriever(BM25_INDEX_DIR)

    # 2. Load FAISS Dense Index
    faiss_index = FAISS.load_local(
        FAISS_INDEX_DIR,
        bge_embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. Load Chunk Store
    chunk_store = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
    print(f"[MPR] Chunk store loaded with {len(chunk_store)} chunks.")

    # 4. Load Metadata Retriever
    metadata_retriever = MetadataRetriever(
        metadata_index_dir=META_INDEX_DIR,
        embeddings=bge_embeddings
    )

    # 5. Load HyDE Retriever (using pre-generated cache)
    class CachedHyDEGenerator:
        def __init__(self, cache: Dict[str, List[str]]):
            self.cache = cache
            self.current_qid = None
        def generate_hypothetical_passages(self, query: str, **kwargs) -> List[str]:
            if self.current_qid and self.current_qid in self.cache:
                return self.cache[self.current_qid]
            return [query]

    cached_generator = CachedHyDEGenerator(hyde_cache)
    hyde_retriever = HyDERetriever(
        faiss_index=faiss_index,
        generator=cached_generator
    )

    # 6. Initialize MPR Coordinator
    mpr = MultiPathRetriever(
        faiss_index=faiss_index,
        bm25_retriever=bm25,
        chunk_store=chunk_store,
        metadata_retriever=metadata_retriever,
        hyde_retriever=hyde_retriever,
        rrf_k=RRF_K
    )

    candidate_pools = []
    t_start = time.time()
    for idx, q in enumerate(questions, start=1):
        qid = q["financebench_id"]
        cached_generator.current_qid = qid
        query_text = q["question"]

        mpr_res = mpr.retrieve_multi_path(
            query=query_text,
            k_pool=k_pool,
            enable_bm25=True,
            enable_dense=True,
            enable_metadata=True,
            enable_hyde=True
        )

        candidates = mpr_res["candidates"]
        
        # Check pool evidence hit
        gold_doc = q["gold_target_doc"]
        gold_pages = set(q["gold_target_pages"])
        pool_has_hit = False
        for c in candidates:
            if c.get("doc_name") == gold_doc and c.get("page") in gold_pages:
                pool_has_hit = True
                break

        pool_entry = {
            "financebench_id": qid,
            "doc_name": gold_doc,
            "question": query_text,
            "dpo_leakage_split": q["dpo_leakage_split"],
            "gold_target_pages": q["gold_target_pages"],
            "pool_size": len(candidates),
            "pool_has_hit": pool_has_hit,
            "candidates": candidates,  # Fixed pool of 40 candidates ordered by RRF
            "telemetry": mpr_res.get("telemetry", {})
        }
        candidate_pools.append(pool_entry)

        if idx % 25 == 0 or idx == len(questions):
            print(f"[MPR] Processed candidate pool [{idx}/{len(questions)}] ({time.time() - t_start:.1f}s)")

    pool_hits = sum(1 for p in candidate_pools if p["pool_has_hit"])
    print(f"\n[MPR Complete] Candidate Pool Recall@{k_pool}: {pool_hits}/{len(candidate_pools)} ({pool_hits/len(candidate_pools)*100:.2f}%)")

    return candidate_pools, chunk_store


# -----------------------------------------------------------------------------
# Baseline Evaluation: Initial Multi-Path RRF Ranking (Prior to Cross-Encoder)
# -----------------------------------------------------------------------------
def evaluate_initial_rrf_ranking(
    candidate_pools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluates the original RRF-ranked candidate list before any cross-encoder reranking.
    Uses the exact same evidence-matching logic, relevance definition, and metrics.
    """
    print("\n[Baseline RRF] Evaluating Initial Multi-Path RRF ranking (pre-reranker)...")
    results = []

    for pool_data in candidate_pools:
        qid = pool_data["financebench_id"]
        gold_doc = pool_data["doc_name"]
        gold_pages = set(pool_data["gold_target_pages"])
        cands = pool_data["candidates"]

        scored_candidates = []
        for c in cands:
            cid = c["chunk_id"]
            is_hit = (c.get("doc_name") == gold_doc and c.get("page") in gold_pages)
            scored_candidates.append({
                "chunk_id": cid,
                "doc_name": c.get("doc_name"),
                "page": c.get("page"),
                "rrf_score": c.get("rrf_score"),
                "reranker_score": float(c.get("rrf_score", 0.0)),
                "is_hit": is_hit
            })

        # Candidates are already sorted by rrf_score descending from RRF
        hit_rank = None
        for r, sc in enumerate(scored_candidates, start=1):
            if sc["is_hit"]:
                hit_rank = r
                break

        results.append({
            "financebench_id": qid,
            "dpo_leakage_split": pool_data["dpo_leakage_split"],
            "gold_doc": gold_doc,
            "gold_target_pages": pool_data["gold_target_pages"],
            "pool_has_hit": pool_data["pool_has_hit"],
            "hit_rank": hit_rank,
            "ranked_candidates": scored_candidates
        })

    return results


# -----------------------------------------------------------------------------
# Cross-Encoder Reranking (Base vs DPO with Fixed Pools & Zero Boosts)
# -----------------------------------------------------------------------------
def format_cross_encoder_text(query: str, passage: str) -> str:
    """Format prompt for BAAI/bge-reranker-v2-gemma."""
    return f"A: {query}\nB: {passage}\n{CROSS_ENCODER_INSTRUCTION}"

def score_passage(
    model,
    tokenizer,
    query: str,
    passage: str,
    yes_token_id: int,
    max_length: int = 512
) -> float:
    """Compute pure cross-encoder logit without any heuristic boost (+0.0)."""
    text = format_cross_encoder_text(query, passage)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False
    ).to(model.device)

    with torch.no_grad():
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            base_backbone = model.base_model.model.model
            lm_head = model.base_model.model.lm_head
        elif hasattr(model, "model") and hasattr(model, "lm_head"):
            base_backbone = model.model
            lm_head = model.lm_head
        else:
            outputs = model(**inputs)
            return float(outputs.logits[:, -1, yes_token_id].item())

        outputs = base_backbone(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score = lm_head(last_hidden)[:, yes_token_id].item()
    return float(score)

def run_reranking_arm(
    candidate_pools: List[Dict[str, Any]],
    model,
    tokenizer,
    yes_token_id: int,
    arm_name: str
) -> List[Dict[str, Any]]:
    """
    Reranks fixed candidate pools with zero heuristic boosts (+0.0).
    Returns question results with ranked candidates.
    """
    print(f"\n[Reranker] Running {arm_name} on {len(candidate_pools)} questions (Zero Heuristic Boosts)...")
    t0 = time.time()
    results = []
    total_q = len(candidate_pools)

    for idx, pool_data in enumerate(candidate_pools, start=1):
        qid = pool_data["financebench_id"]
        query = pool_data["question"]
        gold_doc = pool_data["doc_name"]
        gold_pages = set(pool_data["gold_target_pages"])
        cands = pool_data["candidates"]

        scored_candidates = []
        for c in cands:
            cid = c["chunk_id"]
            text = c["text"]
            score = score_passage(model, tokenizer, query, text, yes_token_id)
            is_hit = (c.get("doc_name") == gold_doc and c.get("page") in gold_pages)
            scored_candidates.append({
                "chunk_id": cid,
                "doc_name": c.get("doc_name"),
                "page": c.get("page"),
                "rrf_score": c.get("rrf_score"),
                "reranker_score": round(score, 6),
                "is_hit": is_hit
            })

        # Sort descending by reranker_score strictly
        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Determine rank of first hit
        hit_rank = None
        for r, sc in enumerate(scored_candidates, start=1):
            if sc["is_hit"]:
                hit_rank = r
                break

        results.append({
            "financebench_id": qid,
            "dpo_leakage_split": pool_data["dpo_leakage_split"],
            "gold_doc": gold_doc,
            "gold_target_pages": pool_data["gold_target_pages"],
            "pool_has_hit": pool_data["pool_has_hit"],
            "hit_rank": hit_rank,
            "ranked_candidates": scored_candidates
        })

        if idx % 25 == 0 or idx == total_q:
            print(f"[{arm_name}] Processed [{idx}/{total_q}] questions ({time.time() - t0:.1f}s)")

    return results


# -----------------------------------------------------------------------------
# Chunk Bundling Execution (Separate Metric Calculation)
# -----------------------------------------------------------------------------
def apply_chunk_bundling(
    reranked_results: List[Dict[str, Any]],
    chunk_bundler: ChunkBundler,
    tau_exp: float = 0.70,
    max_bundle_size: int = 5,
    lead_k_cutoff: int = 3
) -> List[Dict[str, Any]]:
    """
    Applies ChunkBundler to top lead candidates.
    Assembles contiguous context bundles and checks post-bundling evidence coverage.
    """
    print(f"\n[ChunkBundler] Applying expansion (tau_exp={tau_exp}, max_size={max_bundle_size}) on top-{lead_k_cutoff} leads...")
    bundled_results = []

    for item in reranked_results:
        qid = item["financebench_id"]
        gold_doc = item["gold_doc"]
        gold_pages = set(item["gold_target_pages"])
        ranked_cands = item["ranked_candidates"]

        lead_cands = ranked_cands[:lead_k_cutoff]
        bundled_context_cids = []
        covered_pages = set()

        for lead in lead_cands:
            lead_cid = lead["chunk_id"]
            bundle_cids, trace = chunk_bundler.bundle_single_chunk(
                lead_cid=lead_cid,
                tau_exp=tau_exp,
                max_bundle_size=max_bundle_size
            )
            for bcid in bundle_cids:
                if bcid not in bundled_context_cids:
                    bundled_context_cids.append(bcid)
                    c_info = chunk_bundler.chunk_store.get(bcid, {})
                    if c_info.get("doc_name") == gold_doc:
                        covered_pages.add(c_info.get("page_number"))

        bundle_has_hit = bool(covered_pages.intersection(gold_pages))

        bundled_results.append({
            "financebench_id": qid,
            "dpo_leakage_split": item["dpo_leakage_split"],
            "gold_doc": gold_doc,
            "gold_target_pages": item["gold_target_pages"],
            "raw_hit_rank": item["hit_rank"],
            "lead_count": len(lead_cands),
            "total_bundled_chunks": len(bundled_context_cids),
            "covered_pages": sorted(list(covered_pages)),
            "bundle_has_hit": bundle_has_hit,
            "bundled_context_cids": bundled_context_cids
        })

    return bundled_results


# -----------------------------------------------------------------------------
# Metric Aggregation & Reporting
# -----------------------------------------------------------------------------
def compute_metrics_slice(
    raw_results: List[Dict[str, Any]],
    bundled_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes separate raw retrieval metrics (including MRR, nDCG@10, Recall@1, 3, 5, 10,
    Pool Recall@40) and post-bundling evidence coverage.
    """
    N = len(raw_results)
    if N == 0:
        return {}

    # 1. Raw Retrieval Metrics (BEFORE Chunk Bundling)
    ranks = [r["hit_rank"] for r in raw_results]
    mrr = sum(1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks) / N
    rec1 = sum(1.0 if (r is not None and r <= 1) else 0.0 for r in ranks) / N
    rec3 = sum(1.0 if (r is not None and r <= 3) else 0.0 for r in ranks) / N
    rec5 = sum(1.0 if (r is not None and r <= 5) else 0.0 for r in ranks) / N
    rec10 = sum(1.0 if (r is not None and r <= 10) else 0.0 for r in ranks) / N
    pool_rec40 = sum(1.0 if r.get("pool_has_hit") else 0.0 for r in raw_results) / N

    # Binary nDCG@10
    ndcg10_list = []
    for r in ranks:
        if r is not None and 1 <= r <= 10:
            ndcg10_list.append(1.0 / math.log2(r + 1))
        else:
            ndcg10_list.append(0.0)
    ndcg10 = sum(ndcg10_list) / N

    # 2. Bundled Evidence Coverage (AFTER Chunk Bundling)
    bundle_hits = sum(1.0 if b["bundle_has_hit"] else 0.0 for b in bundled_results) / N
    avg_bundle_chunks = sum(b["total_bundled_chunks"] for b in bundled_results) / N

    return {
        "question_count": N,
        "raw_retrieval_metrics": {
            "mrr": round(mrr, 4),
            "ndcg_at_10": round(ndcg10, 4),
            "recall_at_1": round(rec1 * 100, 2),
            "recall_at_3": round(rec3 * 100, 2),
            "recall_at_5": round(rec5 * 100, 2),
            "recall_at_10": round(rec10 * 100, 2),
            "candidate_pool_recall_at_40": round(pool_rec40 * 100, 2)
        },
        "bundled_evidence_coverage": {
            "evidence_coverage_rate": round(bundle_hits * 100, 2),
            "average_bundled_chunks_per_query": round(avg_bundle_chunks, 2),
            "bundle_threshold": TAU_EXP,
            "max_bundle_size": MAX_BUNDLE_SIZE
        }
    }

def aggregate_and_stratify(
    raw_results: List[Dict[str, Any]],
    bundled_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes metrics across all 150 questions and stratifies by DPO leakage split.
    """
    bundled_map = {b["financebench_id"]: b for b in bundled_results}

    # Splits
    splits = {
        "overall_150": raw_results,
        "strictly_held_out_8": [r for r in raw_results if r["dpo_leakage_split"] == "strictly_held_out"],
        "dpo_val_17": [r for r in raw_results if r["dpo_leakage_split"] == "val_triplet"],
        "dpo_train_125": [r for r in raw_results if r["dpo_leakage_split"] == "train_triplet"]
    }

    report = {}
    for split_name, items in splits.items():
        bundled_items = [bundled_map[r["financebench_id"]] for r in items]
        report[split_name] = compute_metrics_slice(items, bundled_items)

    return report


# -----------------------------------------------------------------------------
# Main Runner Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 faithful-methodology reproduction with local model adaptations."
    )
    parser.add_argument("--dry_run", action="store_true", help="Run on first 5 questions for verification.")
    parser.add_argument("--question_limit", type=int, default=None, help="Limit number of evaluation questions.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    args = parser.parse_args()

    print("=" * 80)
    print("FINTEL / FINSAGE REPRODUCTION SUITE")
    print(f"Task: {REPRODUCTION_LABEL}")
    print(f"Publication Note: Published paper reports N=149 questions (exact excluded item unknown).")
    print(f"Evaluation Target: All 150 available questions from {QUESTIONS_FILE}")
    print("=" * 80)

    # 1. Audit baseline file integrity
    baseline_hashes = record_protected_hashes()
    print("[Integrity] Recorded protected hashes for DPO adapter, questions, and datasets.")

    try:
        # 2. Load and tag questions
        questions = load_questions_with_splits(QUESTIONS_FILE)
        if args.dry_run:
            print("[Dry Run] Limiting execution to first 5 questions.")
            questions = questions[:5]
        elif args.question_limit:
            print(f"[Limit] Restricting execution to first {args.question_limit} questions.")
            questions = questions[:args.question_limit]

        print(f"\n[Questions Loaded] Total to evaluate: {len(questions)}")
        split_counts = defaultdict(int)
        for q in questions:
            split_counts[q["dpo_leakage_split"]] += 1
        print("DPO Stratification Breakdown:")
        for split, count in split_counts.items():
            print(f"  - {split}: {count} questions")

        # 3. Phase A: HyDE Generation / Cache Loading
        hyde_cache = generate_or_load_hyde_cache(
            questions=questions,
            cache_path=HYDE_CACHE_FILE,
            device=args.device
        )

        # 4. Phase B: MPR and Fixed Candidate Pools (BGE-M3 + BM25 + FAISS)
        print("\n[Embeddings] Loading BAAI/bge-m3 dense embedding model...")
        bge_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": args.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        candidate_pools, chunk_store = build_fixed_candidate_pools(
            questions=questions,
            hyde_cache=hyde_cache,
            bge_embeddings=bge_embeddings,
            k_pool=K_POOL
        )

        # 5. Initialize ChunkBundler (using same BGE-M3 embeddings)
        chunk_bundler = ChunkBundler(
            chunks_file=CHUNKS_FILE,
            embeddings=bge_embeddings,
            tau_exp=TAU_EXP,
            max_bundle_size=MAX_BUNDLE_SIZE,
            device=args.device
        )

        # 6. Baseline Evaluation: Initial Multi-Path RRF Ranking (Pre-Reranker)
        raw_rrf_results = evaluate_initial_rrf_ranking(candidate_pools)
        bundled_rrf_results = apply_chunk_bundling(
            reranked_results=raw_rrf_results,
            chunk_bundler=chunk_bundler,
            tau_exp=TAU_EXP,
            max_bundle_size=MAX_BUNDLE_SIZE
        )
        rrf_scorecard = aggregate_and_stratify(raw_rrf_results, bundled_rrf_results)

        # 7. Phase C: Reranker Loading (Base Gemma + Local DPO Adapter)
        print("\n" + "=" * 70)
        print(f"[Reranker Loading] Loading {BASE_RERANKER_MODEL} (4-bit NF4) + DPO Adapter...")
        print("=" * 70)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_RERANKER_MODEL)
        yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_RERANKER_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        drr_model = PeftModel.from_pretrained(base_model, DRR_ADAPTER_DIR)
        drr_model.eval()
        print(f"[Reranker Loaded] Total VRAM footprint: {round(torch.cuda.memory_allocated() / 1e6, 2)} MB")

        # 8. Arm 1 Evaluation: Base Gemma Reranker (Adapter Disabled, Fixed Pools)
        with drr_model.disable_adapter():
            raw_base_results = run_reranking_arm(
                candidate_pools=candidate_pools,
                model=drr_model,
                tokenizer=tokenizer,
                yes_token_id=yes_token_id,
                arm_name="Arm 1: Base Cross-Encoder (BAAI/bge-reranker-v2-gemma)"
            )
        bundled_base_results = apply_chunk_bundling(
            reranked_results=raw_base_results,
            chunk_bundler=chunk_bundler,
            tau_exp=TAU_EXP,
            max_bundle_size=MAX_BUNDLE_SIZE
        )
        base_scorecard = aggregate_and_stratify(raw_base_results, bundled_base_results)

        # 9. Arm 2 Evaluation: DPO Adapted Reranker (Adapter Enabled, Same Fixed Pools)
        raw_dpo_results = run_reranking_arm(
            candidate_pools=candidate_pools,
            model=drr_model,
            tokenizer=tokenizer,
            yes_token_id=yes_token_id,
            arm_name="Arm 2: Stage 1 DRR Adapter (Zero Heuristic Boosts)"
        )
        bundled_dpo_results = apply_chunk_bundling(
            reranked_results=raw_dpo_results,
            chunk_bundler=chunk_bundler,
            tau_exp=TAU_EXP,
            max_bundle_size=MAX_BUNDLE_SIZE
        )
        dpo_scorecard = aggregate_and_stratify(raw_dpo_results, bundled_dpo_results)

        # 10. Clean up GPU Memory
        del drr_model
        del base_model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 11. Compile and Save Outputs
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_scorecard = {
            "metadata": {
                "reproduction_label": REPRODUCTION_LABEL,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_questions_evaluated": len(questions),
                "publication_reference_count": 149,
                "publication_note": "Paper reports 149 questions; exact excluded item not identified in open-source release. All 150 available questions evaluated.",
                "parameters": {
                    "rrf_k": RRF_K,
                    "candidate_pool_size": K_POOL,
                    "chunk_bundling_tau_exp": TAU_EXP,
                    "max_bundle_size": MAX_BUNDLE_SIZE,
                    "reranker_heuristic_boosts": 0.0
                }
            },
            "initial_rrf_baseline": rrf_scorecard,
            "arm1_base_cross_encoder": base_scorecard,
            "arm2_dpo_adapted_reranker": dpo_scorecard
        }

        scorecard_path = os.path.join(OUTPUT_DIR, "scorecard_stage1_full.json")
        with open(scorecard_path, "w", encoding="utf-8") as f:
            json.dump(full_scorecard, f, indent=2)
        print(f"\n[Saved] Full reproduction scorecard -> {scorecard_path}")

        # Save detailed question-level retrieval checkpoints
        checkpoints_path = os.path.join(OUTPUT_DIR, "stage1_full_retrieval_checkpoint.jsonl")
        with open(checkpoints_path, "w", encoding="utf-8") as f:
            for r_raw, b_raw, d_raw, r_bdl, b_bdl, d_bdl in zip(
                raw_rrf_results, raw_base_results, raw_dpo_results,
                bundled_rrf_results, bundled_base_results, bundled_dpo_results
            ):
                entry = {
                    "financebench_id": b_raw["financebench_id"],
                    "dpo_leakage_split": b_raw["dpo_leakage_split"],
                    "gold_doc": b_raw["gold_doc"],
                    "gold_target_pages": b_raw["gold_target_pages"],
                    "initial_rrf": {
                        "raw_hit_rank": r_raw["hit_rank"],
                        "bundle_has_hit": r_bdl["bundle_has_hit"],
                        "total_bundled_chunks": r_bdl["total_bundled_chunks"],
                        "top5_cids": [c["chunk_id"] for c in r_raw["ranked_candidates"][:5]]
                    },
                    "arm1_base": {
                        "raw_hit_rank": b_raw["hit_rank"],
                        "bundle_has_hit": b_bdl["bundle_has_hit"],
                        "total_bundled_chunks": b_bdl["total_bundled_chunks"],
                        "top5_cids": [c["chunk_id"] for c in b_raw["ranked_candidates"][:5]]
                    },
                    "arm2_dpo": {
                        "raw_hit_rank": d_raw["hit_rank"],
                        "bundle_has_hit": d_bdl["bundle_has_hit"],
                        "total_bundled_chunks": d_bdl["total_bundled_chunks"],
                        "top5_cids": [c["chunk_id"] for c in d_raw["ranked_candidates"][:5]]
                    }
                }
                f.write(json.dumps(entry) + "\n")
        print(f"[Saved] Detailed question-level checkpoints -> {checkpoints_path}")

        # 12. Generate Markdown Summary Report
        report_path = os.path.join(OUTPUT_DIR, "stage1_reproduction_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# {REPRODUCTION_LABEL}\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")
            f.write(f"**Dataset:** 150 questions from `data/financebench/questions.jsonl` (Published paper reports 149 questions; exact excluded question unknown).\n\n")
            f.write(f"**Integrity Safeguards:** Zero heuristic boosts applied (+0.0). Fixed identical candidate pools across Base vs DPO reranker.\n\n")
            
            f.write("## Three-Way Benchmark Comparison (All 150 Questions)\n\n")
            f.write("| Evaluation Stage | Metric | Initial Multi-Path RRF | Arm 1: Base Gemma Reranker | Arm 2: DPO Adapted Gemma | RRF -> DPO Delta | Base -> DPO Delta |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
            
            r_raw_m = rrf_scorecard["overall_150"]["raw_retrieval_metrics"]
            b_raw_m = base_scorecard["overall_150"]["raw_retrieval_metrics"]
            d_raw_m = dpo_scorecard["overall_150"]["raw_retrieval_metrics"]

            r_cov_m = rrf_scorecard["overall_150"]["bundled_evidence_coverage"]
            b_cov_m = base_scorecard["overall_150"]["bundled_evidence_coverage"]
            d_cov_m = dpo_scorecard["overall_150"]["bundled_evidence_coverage"]

            f.write(f"| **Raw Retrieval** | Candidate Pool Recall@40 | {r_raw_m['candidate_pool_recall_at_40']}% | {b_raw_m['candidate_pool_recall_at_40']}% | {d_raw_m['candidate_pool_recall_at_40']}% | 0.00% (Identical Pool) | 0.00% (Identical Pool) |\n")
            f.write(f"| **Raw Retrieval** | MRR | {r_raw_m['mrr']:.4f} | {b_raw_m['mrr']:.4f} | {d_raw_m['mrr']:.4f} | {d_raw_m['mrr'] - r_raw_m['mrr']:+.4f} | {d_raw_m['mrr'] - b_raw_m['mrr']:+.4f} |\n")
            f.write(f"| **Raw Retrieval** | nDCG@10 | {r_raw_m['ndcg_at_10']:.4f} | {b_raw_m['ndcg_at_10']:.4f} | {d_raw_m['ndcg_at_10']:.4f} | {d_raw_m['ndcg_at_10'] - r_raw_m['ndcg_at_10']:+.4f} | {d_raw_m['ndcg_at_10'] - b_raw_m['ndcg_at_10']:+.4f} |\n")
            f.write(f"| **Raw Retrieval** | Recall@1 | {r_raw_m['recall_at_1']}% | {b_raw_m['recall_at_1']}% | {d_raw_m['recall_at_1']}% | {d_raw_m['recall_at_1'] - r_raw_m['recall_at_1']:+.2f}% | {d_raw_m['recall_at_1'] - b_raw_m['recall_at_1']:+.2f}% |\n")
            f.write(f"| **Raw Retrieval** | Recall@3 | {r_raw_m['recall_at_3']}% | {b_raw_m['recall_at_3']}% | {d_raw_m['recall_at_3']}% | {d_raw_m['recall_at_3'] - r_raw_m['recall_at_3']:+.2f}% | {d_raw_m['recall_at_3'] - b_raw_m['recall_at_3']:+.2f}% |\n")
            f.write(f"| **Raw Retrieval** | Recall@5 | {r_raw_m['recall_at_5']}% | {b_raw_m['recall_at_5']}% | {d_raw_m['recall_at_5']}% | {d_raw_m['recall_at_5'] - r_raw_m['recall_at_5']:+.2f}% | {d_raw_m['recall_at_5'] - b_raw_m['recall_at_5']:+.2f}% |\n")
            f.write(f"| **Raw Retrieval** | Recall@10 | {r_raw_m['recall_at_10']}% | {b_raw_m['recall_at_10']}% | {d_raw_m['recall_at_10']}% | {d_raw_m['recall_at_10'] - r_raw_m['recall_at_10']:+.2f}% | {d_raw_m['recall_at_10'] - b_raw_m['recall_at_10']:+.2f}% |\n")
            f.write(f"| **Bundled Coverage** | Evidence Coverage Rate | {r_cov_m['evidence_coverage_rate']}% | {b_cov_m['evidence_coverage_rate']}% | {d_cov_m['evidence_coverage_rate']}% | {d_cov_m['evidence_coverage_rate'] - r_cov_m['evidence_coverage_rate']:+.2f}% | {d_cov_m['evidence_coverage_rate'] - b_cov_m['evidence_coverage_rate']:+.2f}% |\n")
            f.write(f"| **Bundled Coverage** | Avg Chunks per Query | {r_cov_m['average_bundled_chunks_per_query']} | {b_cov_m['average_bundled_chunks_per_query']} | {d_cov_m['average_bundled_chunks_per_query']} | {d_cov_m['average_bundled_chunks_per_query'] - r_cov_m['average_bundled_chunks_per_query']:+.2f} | {d_cov_m['average_bundled_chunks_per_query'] - b_cov_m['average_bundled_chunks_per_query']:+.2f} |\n\n")

            f.write("## DPO Leakage Stratification (Initial RRF vs Base Gemma vs DPO Gemma)\n\n")
            f.write("| Sub-Split | Questions | RRF MRR | Base MRR | DPO MRR | RRF R@5 | Base R@5 | DPO R@5 | RRF Cov | Base Cov | DPO Cov |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for split_key, split_label in [
                ("strictly_held_out_8", "Strictly Held-Out (Zero Exposure)"),
                ("dpo_val_17", "DPO Validation Split"),
                ("dpo_train_125", "DPO Training Split")
            ]:
                rs = rrf_scorecard[split_key]
                bs = base_scorecard[split_key]
                ds = dpo_scorecard[split_key]
                f.write(f"| {split_label} | {bs['question_count']} | {rs['raw_retrieval_metrics']['mrr']:.4f} | {bs['raw_retrieval_metrics']['mrr']:.4f} | {ds['raw_retrieval_metrics']['mrr']:.4f} | {rs['raw_retrieval_metrics']['recall_at_5']}% | {bs['raw_retrieval_metrics']['recall_at_5']}% | {ds['raw_retrieval_metrics']['recall_at_5']}% | {rs['bundled_evidence_coverage']['evidence_coverage_rate']}% | {bs['bundled_evidence_coverage']['evidence_coverage_rate']}% | {ds['bundled_evidence_coverage']['evidence_coverage_rate']}% |\n")

            f.write("\n## Methodological Interpretation of DPO Splits & Generalization\n\n")
            f.write("- **125 Questions (DPO Training Split):** These questions were mined as positive/negative triplets during the training of the domain reranker adapter (`outputs/phase1_drr_adapter/`). Performance on this subset reflects in-distribution contrastive fitting and memorization capacity rather than generalization.\n")
            f.write("- **17 Questions (DPO Validation Split):** Used for checkpoint selection and hyperparameter validation during DPO training.\n")
            f.write("- **8 Questions (Strictly Held-Out):** Questions `financebench_id_03029`, `financebench_id_04672`, `financebench_id_00746`, `financebench_id_00521`, `financebench_id_00601`, `financebench_id_00603`, `financebench_id_00605`, and `financebench_id_00606` had zero exposure during DPO training or validation. They represent the only strictly unseen subset for assessing DPO generalization in this evaluation set.\n")
            f.write("- **Critical Methodological Note:** The 142 training/validation questions must not be treated as an independent DPO generalization evaluation. Furthermore, results on the eight held-out questions must be interpreted cautiously due to the small sample size ($N=8$) and high statistical variance. They should not be interpreted as definitive proof of broad out-of-distribution generalization.\n")

        print(f"[Saved] Markdown summary report -> {report_path}")

    finally:
        # Guaranteed post-execution integrity verification
        print("\n[Integrity] Executing post-execution SHA-256 verification...")
        verify_protected_hashes(baseline_hashes)

    print("\n" + "=" * 80)
    print("STAGE 1 REPRODUCTION RUN COMPLETE SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
