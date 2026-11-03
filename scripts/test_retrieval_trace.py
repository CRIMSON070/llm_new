import os
import sys
import json
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

def main():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")
    enriched_chunks_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")

    chunk_store = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c

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

    test_qids = ["financebench_id_01226", "financebench_id_00438", "financebench_id_00005", "financebench_id_00299"]
    questions = []
    with open(os.path.join(data_dir, "questions.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("financebench_id") in test_qids:
                questions.append(q)

    for q in questions:
        qid = q["financebench_id"]
        q_text = q["question"]
        gold_doc = q["doc_name"]
        target_pages = [e.get("evidence_page_num") + 1 for e in q.get("evidence", []) if e.get("evidence_page_num") is not None]

        print(f"\n=======================================================")
        print(f"QID: {qid} | Gold Doc: {gold_doc} | Target Pages: {target_pages}")
        print(f"Q: {q_text}")

        # Retrieve at depth 50
        sparse_50 = mpr.retrieve_sparse(q_text, k=50)
        dense_50 = mpr.retrieve_dense(q_text, k=50)
        sparse_100 = mpr.retrieve_sparse(q_text, k=100)
        dense_100 = mpr.retrieve_dense(q_text, k=100)

        # Check where gold doc and page appear in sparse and dense
        print(f"Sparse (k=50) matches on gold doc {gold_doc}:")
        sparse_gold_pages = []
        for rank, (cid, score) in enumerate(sparse_50, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc:
                p = c.get("page_number")
                hit = " [GOLD HIT!]" if p in target_pages else ""
                sparse_gold_pages.append(f"rank {rank}: p{p}{hit}")
        print("  " + (", ".join(sparse_gold_pages[:5]) if sparse_gold_pages else "NONE"))

        print(f"Dense (k=50) matches on gold doc {gold_doc}:")
        dense_gold_pages = []
        for rank, (cid, score) in enumerate(dense_50, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc:
                p = c.get("page_number")
                hit = " [GOLD HIT!]" if p in target_pages else ""
                dense_gold_pages.append(f"rank {rank}: p{p}{hit}")
        print("  " + (", ".join(dense_gold_pages[:5]) if dense_gold_pages else "NONE"))

        # In k=100:
        print(f"Sparse (k=100) matches on gold doc {gold_doc}:")
        s100_hits = [f"rank {rank}: p{chunk_store.get(cid,{}).get('page_number')}" for rank, (cid, score) in enumerate(sparse_100, start=1) if chunk_store.get(cid,{}).get("doc_name") == gold_doc and chunk_store.get(cid,{}).get("page_number") in target_pages]
        print("  " + (", ".join(s100_hits) if s100_hits else "NONE"))

        print(f"Dense (k=100) matches on gold doc {gold_doc}:")
        d100_hits = [f"rank {rank}: p{chunk_store.get(cid,{}).get('page_number')}" for rank, (cid, score) in enumerate(dense_100, start=1) if chunk_store.get(cid,{}).get("doc_name") == gold_doc and chunk_store.get(cid,{}).get("page_number") in target_pages]
        print("  " + (", ".join(d100_hits) if d100_hits else "NONE"))

if __name__ == "__main__":
    main()
