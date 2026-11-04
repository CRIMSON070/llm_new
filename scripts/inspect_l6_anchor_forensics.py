import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import json
import pickle
from pathlib import Path

# Load Benchmark V2
v2_path = Path("outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl")
target_ids = [
    "stage3_heldout_0075",
    "stage3_heldout_0074",
    "stage3_heldout_0099",
    "stage3_heldout_0100",
    "stage3_heldout_0014",
    "stage3_heldout_0025",
]

v2_questions = {}
with open(v2_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["question_id"] in target_ids:
            v2_questions[item["question_id"]] = item

# Load V1 evaluation results for comparison if available
# V1 comparison CSV
v1_csv_path = Path("outputs/experiments/experiment_stage3_heldout_validation/heldout_question_level_comparison.csv")
v1_results = {}
if v1_csv_path.exists():
    import csv
    with open(v1_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question_id") in target_ids:
                v1_results[row["question_id"]] = row

# We run a lightweight script to get exact RRF and Reranker ranks for the 6 questions
# by running mpr and reranker for only these 6 questions across Arm A, B, and C
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.financial_expander import FinancialQueryExpander
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.phase2_reranker import Phase2Reranker
from outputs.experiments.experiment_l2_section_aware.section_aware_retriever import SectionAwareMultiPathRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})

base_corpus_path = "data/heldout_stage3/processed/heldout_enriched_chunks.jsonl"
grounded_corpus_path = "data/heldout_stage3/processed/heldout_grounded_chunks.jsonl"

arm_a_chunks = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(base_corpus_path, "r", encoding="utf-8")}
arm_b_chunks = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(grounded_corpus_path, "r", encoding="utf-8")}

arm_a_faiss = FAISS.load_local("data/heldout_stage3/indexes/faiss_bge_m3", embeddings, allow_dangerous_deserialization=True)
arm_a_bm25 = BM25Retriever("data/heldout_stage3/indexes/bm25", load_corpus=True)

arm_b_faiss = FAISS.load_local("data/heldout_stage3/indexes_grounded_exp_l/faiss_bge_m3", embeddings, allow_dangerous_deserialization=True)
arm_b_bm25 = BM25Retriever("data/heldout_stage3/indexes_grounded_exp_l/bm25", load_corpus=True)

expander = FinancialQueryExpander(enable_coreference=True)
reranker = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True)

mpr_arm_a = MultiPathRetriever(faiss_index=arm_a_faiss, bm25_retriever=arm_a_bm25, chunk_store=arm_a_chunks, rrf_k=60, expander=expander)
mpr_arm_b_c = SectionAwareMultiPathRetriever(faiss_index=arm_b_faiss, bm25_retriever=arm_b_bm25, chunk_store=arm_b_chunks, rrf_k=60, expander=expander, narrative_table_cap=4, item1_boost=0.05)

arms = [
    ("Arm A (Baseline K=50)", mpr_arm_a, 50, "baseline"),
    ("Arm B (Exp L.1 Grounded K=65)", mpr_arm_b_c, 65, "control"),
    ("Arm C (Exp L.2 Section-Aware K=65)", mpr_arm_b_c, 65, "section_aware"),
]

forensic_records = []

for qid in target_ids:
    q_data = v2_questions[qid]
    q_text = q_data["question"]
    qtype = q_data.get("question_type")
    doc_name = q_data["doc_name"]
    ev_sources = q_data["evidence_sources"]
    
    direct_ev = [e for e in ev_sources if e.get("evidence_type") == "direct_evidence"]
    context_ev = [e for e in ev_sources if e.get("evidence_type") == "contextual_table_header"]
    hist_ev = [e for e in ev_sources if e.get("evidence_type") == "historical_annotated_reference"]
    
    print(f"\n=======================================================")
    print(f"QUESTION ID: {qid}")
    print(f"Question: {q_text}")
    print(f"Document: {doc_name} | Type: {qtype}")
    print(f"V2 Status: {q_data.get('verification_metadata', {}).get('benchmark_v2_status')}")
    print(f"Original Classification: {q_data.get('verification_metadata', {}).get('original_annotation_classification')}")
    print("Corrected Evidence Sources:")
    for ev in ev_sources:
        print(f"  - Page {ev.get('evidence_page_num')} [{ev.get('evidence_type')}]: chunks={ev.get('matching_chunk_ids')} | '{ev.get('relevant_passage', '')[:60]}...'")
    
    target_cids = [cid for e in direct_ev for cid in e.get("matching_chunk_ids", [])]
    target_pages = [e.get("evidence_page_num") for e in direct_ev]
    comp_pages = [e.get("evidence_page_num") for e in ev_sources]
    
    for arm_label, mpr, k_pool, mode in arms:
        if mode == "baseline":
            ret = mpr.retrieve_multi_path(query=q_text, k_pool=k_pool, enable_bm25=True, enable_dense=True)
        elif mode == "control":
            ret = mpr.retrieve_multi_path(query=q_text, k_pool=k_pool, enable_bm25=True, enable_dense=True, arm_mode="control")
        else:
            ret = mpr.retrieve_multi_path(query=q_text, k_pool=k_pool, enable_bm25=True, enable_dense=True, question_type=qtype, arm_mode="section_aware")
        
        cands = ret["candidates"]
        reranked = reranker.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        
        cand_ids = [c["chunk_id"] for c in cands]
        rerank_ids = [r["chunk_id"] for r in reranked]
        
        print(f"  [{arm_label}]")
        # Check matching for each direct evidence chunk
        for ev in direct_ev:
            pnum = ev.get("evidence_page_num")
            cids = ev.get("matching_chunk_ids", [])
            for cid in cids:
                in_pool = cid in cand_ids
                rrf_rank = (cand_ids.index(cid) + 1) if in_pool else "OUT"
                final_rank = (rerank_ids.index(cid) + 1) if cid in rerank_ids else "OUT"
                print(f"    Direct Page {pnum} Chunk {cid}: InPool={in_pool}, RRF_Rank={rrf_rank}, Reranker_Rank={final_rank}")
        
        # Check contextual header
        for ev in context_ev:
            pnum = ev.get("evidence_page_num")
            cids = ev.get("matching_chunk_ids", [])
            for cid in cids:
                in_pool = cid in cand_ids
                rrf_rank = (cand_ids.index(cid) + 1) if in_pool else "OUT"
                final_rank = (rerank_ids.index(cid) + 1) if cid in rerank_ids else "OUT"
                print(f"    Header Page {pnum} Chunk {cid}: InPool={in_pool}, RRF_Rank={rrf_rank}, Reranker_Rank={final_rank}")
