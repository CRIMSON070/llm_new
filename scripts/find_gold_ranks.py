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

def main():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
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
    bm25 = BM25Retriever(bm25_phase2_dir, load_corpus=True)

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

        # Search top 500 in BM25
        ids, scores = bm25.invoke(q_text, k=500)
        bm25_rank = None
        for r, cid in enumerate(ids, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                bm25_rank = r
                break

        # Search top 500 in Dense
        results = dense_faiss.similarity_search_with_score(q_text, k=500)
        dense_rank = None
        for r, (doc, score) in enumerate(results, start=1):
            cid = doc.metadata.get("chunk_id")
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                dense_rank = r
                break

        print(f"  BM25 Rank (k=500): {bm25_rank}")
        print(f"  Dense Rank (k=500): {dense_rank}")

if __name__ == "__main__":
    main()
