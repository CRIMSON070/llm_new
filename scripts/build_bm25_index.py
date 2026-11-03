import os
import json
import sys
from langchain_core.documents import Document

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.bm25Retriever import load_from_chroma_and_save

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    index_dir = os.path.join(data_dir, "bm25_index")
    
    input_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print("Loading chunks from disk...")
    docs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            metadata = {
                "doc_name": record["doc_name"],
                "page_number": record["page_number"],
                "doc_id": record["chunk_id"] # The existing code expects 'doc_id' in metadata
            }
            doc = Document(page_content=record["text"], metadata=metadata)
            docs.append(doc)
            
    print(f"Loaded {len(docs)} chunks. Building BM25 index (this is entirely CPU-bound)...")
    
    os.makedirs(index_dir, exist_ok=True)
    load_from_chroma_and_save(docs, index_dir)
    
    print("BM25 index built successfully!")

if __name__ == "__main__":
    main()
