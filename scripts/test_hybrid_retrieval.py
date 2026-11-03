import os
import json
import torch
import requests
import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure src can be imported for BM25
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.bm25Retriever import BM25Retriever

def generate_hyde(query: str, model: str = "llama3.2:latest") -> str:
    """Use local Ollama to generate a Hypothetical Document Embedding (HyDE) response."""
    prompt = f"Please write a factual and detailed financial paragraph that answers the following question. Write it as if it were extracted directly from an official 10-K financial report. Question: {query}\n\nAnswer:"
    
    print(f"\n[HyDE] Generating hypothetical document using {model}...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }
        )
        if response.status_code == 200:
            hyde_doc = response.json().get("response", "").strip()
            print(f"[HyDE] Successfully generated hypothetical document (length: {len(hyde_doc)} chars)")
            return hyde_doc
        else:
            print(f"[HyDE] Ollama API error: {response.text}")
            return query
    except Exception as e:
        print(f"[HyDE] Failed to connect to Ollama: {e}")
        return query

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_dir = os.path.join(data_dir, "bm25_index")
    
    print("1. Loading Embedding Model (BGE-M3) on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("2. Loading FAISS Dense Index...")
    faiss_index = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    
    print("3. Loading BM25 Sparse Index...")
    bm25_retriever = BM25Retriever(bm25_dir, load_corpus=True)
    
    # Test Query from FinanceBench
    test_query = "What is the FY2018 operating margin for 3M?"
    print(f"\n==================================================")
    print(f"TEST QUERY: {test_query}")
    print(f"==================================================")
    
    # --- PHASE 7: HyDE ---
    hypothetical_doc = generate_hyde(test_query)
    
    print("\n4. Executing Hybrid Retrieval...")
    
    # Dense Retrieval (using both Original Query and HyDE)
    # We retrieve top 5 for original query and top 5 for HyDE
    print(" -> Querying FAISS (Dense)...")
    dense_results_orig = faiss_index.similarity_search_with_score(test_query, k=5)
    dense_results_hyde = faiss_index.similarity_search_with_score(hypothetical_doc, k=5)
    
    # Extract unique chunks from dense retrieval
    dense_chunks = {}
    for doc, score in dense_results_orig + dense_results_hyde:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in dense_chunks:
            dense_chunks[chunk_id] = {"doc": doc, "score": score, "source": "FAISS"}
            
    # Sparse Retrieval (using Original Query)
    print(" -> Querying BM25 (Sparse)...")
    bm25_ids, bm25_scores = bm25_retriever.invoke(test_query, k=5)
    
    # We need to map bm25_ids back to text, but the current bm25Retriever only returns chunk_ids.
    # In a full pipeline we would fetch the text from a docstore, but for this test we can extract
    # it from the FAISS docstore since it has all the metadata and page_content mapped!
    
    hybrid_results = list(dense_chunks.values())
    
    for chunk_id, score in zip(bm25_ids, bm25_scores):
        if chunk_id not in dense_chunks:
            # Look up the document in FAISS docstore to get the text
            # FAISS docstore is a dict mapping UUID to Document
            found = False
            for doc_uuid, doc in faiss_index.docstore._dict.items():
                if doc.metadata.get("chunk_id") == chunk_id:
                    hybrid_results.append({"doc": doc, "score": score, "source": "BM25"})
                    found = True
                    break
            if not found:
                print(f"Warning: Chunk {chunk_id} not found in FAISS docstore.")
                
    print(f"\n[Hybrid Retrieval Complete] Found {len(hybrid_results)} unique candidate chunks.")
    
    print("\nTop 3 Candidate Chunks:")
    for i, res in enumerate(hybrid_results[:3]):
        doc = res["doc"]
        source = res["source"]
        score = res["score"]
        print(f"\n--- Rank {i+1} ({source}) ---")
        print(f"Document: {doc.metadata.get('doc_name')} | Page: {doc.metadata.get('page_number')}")
        print(f"Text Snippet: {doc.page_content[:300]}...")

if __name__ == "__main__":
    main()
