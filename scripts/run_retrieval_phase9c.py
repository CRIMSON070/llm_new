"""
Phase 9C: Step 1 - Retrieval Ablations
Evaluates 4 retrieval configurations on Q1-Q6:
1. Dense Only (BGE-M3 + FAISS)
2. Dense + BM25 (RRF fusion without query expansion)
3. Hybrid Expanded (Dense + BM25 + Financial Query Expansion via RRF)
4. Hybrid Expanded + Statement-Aware Reranked (BGE-Reranker-base + GAAP Metadata Adjustments)

Saves candidate pools and metadata to outputs/phase9c/retrieval_ablation_data.json.
"""

import os
import json
import time
import torch
import warnings
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.financial_expander import FinancialQueryExpander
from src.retrieval.statement_reranker import StatementAwareReranker
from src.retrieval.hybrid_retriever import HybridRetriever

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    out_dir = os.path.join(base_dir, "outputs", "phase9c")
    os.makedirs(out_dir, exist_ok=True)
    
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_dir = os.path.join(data_dir, "bm25_index")
    
    test_cases = [
        {
            "id": "Q1",
            "query": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
            "gold_answer": "$1577.00",
            "gold_chunks": ["3M_2018_10K_p46_c4", "3M_2018_10K_p60_c1", "3M_2018_10K_p49_c4"],
            "financial_metric": "Capital Expenditures (PP&E Purchases)",
            "reporting_period": "FY2018"
        },
        {
            "id": "Q2",
            "query": "Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.",
            "gold_answer": "$8.70",
            "gold_chunks": ["3M_2018_10K_p58_c1", "3M_2018_10K_p58_c0", "3M_2018_10K_mm_58_1"],
            "financial_metric": "Property, plant and equipment, net",
            "reporting_period": "FY2018"
        },
        {
            "id": "Q3",
            "query": "Which debt securities are registered to trade on a national securities exchange under Ulta Beauty's name as of FY2023?",
            "gold_answer": "There are none",
            "gold_chunks": ["ULTABEAUTY_2023_10K_p1_c1", "ULTABEAUTY_2023_10K_p1_c0"],
            "financial_metric": "Registered Debt Securities",
            "reporting_period": "FY2023"
        },
        {
            "id": "Q4",
            "query": "What are major acquisitions that Ulta Beauty has done in FY2023 and FY2022?",
            "gold_answer": "Ulta Beauty did not make any acquisitions in FY2023 and FY2022.",
            "gold_chunks": ["ULTABEAUTY_2023_10K_p57_c0", "ULTABEAUTY_2023Q4_EARNINGS_p1_c2", "ULTABEAUTY_2023Q4_EARNINGS_mm_1_1"],
            "financial_metric": "Major Acquisitions",
            "reporting_period": "FY2022 / FY2023"
        },
        {
            "id": "Q5",
            "query": "Based on the statements of income, what is the absolute difference in net sales for 3M from 2017 to 2018? Also calculate the percentage change.",
            "gold_answer": "Net sales in 2018 - Net sales in 2017. Approx 3%.",
            "gold_chunks": ["3M_2018_10K_p56_c0", "3M_2018_10K_p127_c1"],
            "financial_metric": "Net Sales Difference",
            "reporting_period": "2017 to 2018"
        },
        {
            "id": "Q6",
            "query": "What is the difference between Ulta Beauty's merchandise inventories at the end of FY2023 and the end of FY2022?",
            "gold_answer": "$1,603,451 - $1,499,218 = $104,233",
            "gold_chunks": ["ULTABEAUTY_2023_10K_p41_c3", "ULTABEAUTY_2023_10K_p57_c0"],
            "financial_metric": "Merchandise Inventories Difference",
            "reporting_period": "FY2023 vs FY2022"
        }
    ]

    print("Step 1: Loading Chunks, FAISS & BM25 Indexes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    faiss_index = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever(bm25_dir, load_corpus=True)
    
    chunk_store = {}
    with open(orig_chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            chunk_store[record["chunk_id"]] = record
            chunk_store[i] = record
            
    if os.path.exists(new_chunks_file):
        with open(new_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                chunk_store[record["chunk_id"]] = record

    expander = FinancialQueryExpander()
    reranker = StatementAwareReranker(use_fp16=(device=='cuda'))
    hybrid_retriever = HybridRetriever(
        faiss_index=faiss_index,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        expander=expander,
        reranker=reranker
    )

    print("\nStep 2: Executing Retrieval Ablations on Q1–Q6...")
    ablation_results = {
        "dense_only": [],
        "dense_plus_bm25": [],
        "hybrid_expanded": [],
        "hybrid_expanded_statement_reranked": []
    }

    for tc in test_cases:
        qid = tc["id"]
        q = tc["query"]
        gold_cids = set(tc["gold_chunks"])
        print(f"\nProcessing {qid}...")
        
        # Path 1: Dense Only
        t0 = time.time()
        dense_hits = hybrid_retriever.retrieve_dense(q, k=40)
        dt_dense = time.time() - t0
        dense_candidates = [cid for cid, s in dense_hits]
        dense_final_3 = dense_candidates[:3]
        dense_final_5 = dense_candidates[:5]
        
        ablation_results["dense_only"].append({
            "qid": qid,
            "query": q,
            "candidates_count": len(dense_candidates),
            "candidates": dense_candidates,
            "final_top3": dense_final_3,
            "final_top5": dense_final_5,
            "recall_at_10": any(cid in gold_cids for cid in dense_candidates[:10]),
            "recall_at_40": any(cid in gold_cids for cid in dense_candidates[:40]),
            "gold_in_top3": any(cid in gold_cids for cid in dense_final_3),
            "gold_in_top5": any(cid in gold_cids for cid in dense_final_5),
            "latency": round(dt_dense, 3)
        })

        # Path 2: Dense + BM25 (RRF without expansion)
        t0 = time.time()
        p2_paths = {
            "dense_orig": hybrid_retriever.retrieve_dense(q, k=20),
            "sparse_orig": hybrid_retriever.retrieve_sparse(q, k=20)
        }
        p2_fused = hybrid_retriever.reciprocal_rank_fusion(p2_paths, k_pool=40)
        dt_p2 = time.time() - t0
        p2_candidates = [c["chunk_id"] for c in p2_fused]
        p2_final_3 = p2_candidates[:3]
        p2_final_5 = p2_candidates[:5]
        
        ablation_results["dense_plus_bm25"].append({
            "qid": qid,
            "query": q,
            "candidates_count": len(p2_candidates),
            "candidates": p2_candidates,
            "final_top3": p2_final_3,
            "final_top5": p2_final_5,
            "recall_at_10": any(cid in gold_cids for cid in p2_candidates[:10]),
            "recall_at_40": any(cid in gold_cids for cid in p2_candidates[:40]),
            "gold_in_top3": any(cid in gold_cids for cid in p2_final_3),
            "gold_in_top5": any(cid in gold_cids for cid in p2_final_5),
            "latency": round(dt_p2, 3)
        })

        # Path 3: Hybrid Expanded (Dense + BM25 + Financial Query Expansion via RRF)
        t0 = time.time()
        p3_candidates, exp_res = hybrid_retriever.get_candidate_pool(q, k_pool=40, use_expansion=True)
        dt_p3 = time.time() - t0
        p3_cids = [c["chunk_id"] for c in p3_candidates]
        p3_final_3 = p3_cids[:3]
        p3_final_5 = p3_cids[:5]
        
        ablation_results["hybrid_expanded"].append({
            "qid": qid,
            "query": q,
            "expansion": exp_res.trace_log,
            "candidates_count": len(p3_cids),
            "candidates": p3_cids,
            "final_top3": p3_final_3,
            "final_top5": p3_final_5,
            "recall_at_10": any(cid in gold_cids for cid in p3_cids[:10]),
            "recall_at_40": any(cid in gold_cids for cid in p3_cids[:40]),
            "gold_in_top3": any(cid in gold_cids for cid in p3_final_3),
            "gold_in_top5": any(cid in gold_cids for cid in p3_final_5),
            "latency": round(dt_p3, 3)
        })

        # Path 4: Hybrid Expanded + Statement-Aware Reranked
        t0 = time.time()
        p4_ret = hybrid_retriever.retrieve(q, k_pool=40, k_final=5, use_expansion=True, use_statement_reranking=True)
        dt_p4 = time.time() - t0
        p4_candidates = [c["chunk_id"] for c in p4_ret["candidates"]]
        p4_final_3 = [c["chunk_id"] for c in p4_ret["final_chunks"][:3]]
        p4_final_5 = [c["chunk_id"] for c in p4_ret["final_chunks"]]
        
        ablation_results["hybrid_expanded_statement_reranked"].append({
            "qid": qid,
            "query": q,
            "detected_statement": p4_ret["expansion"].detected_statement,
            "statement_confidence": p4_ret["expansion"].statement_confidence,
            "candidates_count": len(p4_candidates),
            "candidates": p4_candidates,
            "final_top3": p4_final_3,
            "final_top5": p4_final_5,
            "reranked_details": [
                {
                    "chunk_id": c["chunk_id"],
                    "page": c.get("page"),
                    "raw_score": c.get("raw_rerank_score"),
                    "statement_boost": c.get("statement_boost"),
                    "table_boost": c.get("table_boost"),
                    "final_score": c.get("final_score"),
                    "statement_type": c.get("chunk_statement_type")
                }
                for c in p4_ret["final_chunks"]
            ],
            "recall_at_10": any(cid in gold_cids for cid in p4_candidates[:10]),
            "recall_at_40": any(cid in gold_cids for cid in p4_candidates[:40]),
            "gold_in_top3": any(cid in gold_cids for cid in p4_final_3),
            "gold_in_top5": any(cid in gold_cids for cid in p4_final_5),
            "latency": round(dt_p4, 3)
        })

    # Save retrieval ablation intermediate data
    retrieval_data_file = os.path.join(out_dir, "retrieval_ablation_data.json")
    with open(retrieval_data_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_cases": test_cases,
            "ablation_results": ablation_results
        }, f, indent=2)
    print(f"\nStep 1 Complete! Retrieval ablation data saved to {retrieval_data_file}")

if __name__ == "__main__":
    main()
