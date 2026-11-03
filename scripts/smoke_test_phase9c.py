import os
import json
import torch
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.financial_expander import FinancialQueryExpander
from src.retrieval.statement_reranker import StatementAwareReranker
from src.retrieval.hybrid_retriever import HybridRetriever

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_dir = os.path.join(data_dir, "bm25_index")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embeddings on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Loading FAISS index...")
    faiss_index = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    
    print("Loading BM25 index...")
    bm25_retriever = BM25Retriever(bm25_dir, load_corpus=True)
    
    print("Loading chunk store...")
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
                
    print(f"Loaded {len(chunk_store)} chunk records.")
    
    expander = FinancialQueryExpander()
    reranker = StatementAwareReranker(use_fp16=(device=='cuda'))
    
    retriever = HybridRetriever(
        faiss_index=faiss_index,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        expander=expander,
        reranker=reranker
    )
    
    # Smoke test on Q1
    q1 = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
    print(f"\nRunning Smoke Test on Q1:\n{q1}")
    
    # 1. Base Dense
    print("\n--- 1. Base Dense Only (Top 5) ---")
    dense_res = retriever.retrieve_dense(q1, k=5)
    for r, (cid, s) in enumerate(dense_res, 1):
        c = chunk_store.get(cid, {})
        print(f"Rank {r}: {cid} (Page {c.get('page_number')}) - Score {s:.4f}")
        
    # 2. Hybrid with Expansion & Statement-Aware Reranking
    print("\n--- 2. Hybrid + Expansion + Statement Reranker (Top 5) ---")
    full_res = retriever.retrieve(q1, k_pool=40, k_final=5, use_expansion=True, use_statement_reranking=True)
    print(f"Detected Statement: {full_res['expansion'].detected_statement} (conf {full_res['expansion'].statement_confidence})")
    print(f"Candidate pool count: {full_res['candidate_count']}")
    for r, c in enumerate(full_res['final_chunks'], 1):
        print(f"Rank {r}: {c['chunk_id']} (Page {c.get('page')}) - Raw: {c.get('raw_rerank_score')} | StmtBoost: {c.get('statement_boost')} | Final: {c.get('final_score')}")
        print(f"   Snippet: {c.get('text', '')[:120]}...\n")
        
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    main()
