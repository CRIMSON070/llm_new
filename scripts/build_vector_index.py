import os
import json
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    index_dir = os.path.join(data_dir, "faiss_index")
    
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
                "chunk_id": record["chunk_id"]
            }
            doc = Document(page_content=record["text"], metadata=metadata)
            docs.append(doc)
            
    print(f"Loaded {len(docs)} chunks.")
    
    print("Initializing BGE-M3 model on GPU (if available)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    # HuggingFaceEmbeddings automatically pulls the model from Hugging Face if not locally cached
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("Generating dense embeddings and building FAISS index...")
    print("This step can take several minutes for 54k+ chunks. Please wait.")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(index_dir, exist_ok=True)
    print(f"Saving FAISS index to {index_dir}...")
    vectorstore.save_local(index_dir)
    print("Vector index built successfully!")

if __name__ == "__main__":
    main()
