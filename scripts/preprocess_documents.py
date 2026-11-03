import os
import json
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(pdf_path, doc_name):
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            if not text.strip():
                continue
                
            page_chunks = text_splitter.split_text(text)
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_record = {
                    "doc_name": doc_name,
                    "page_number": page_num + 1,
                    "chunk_id": f"{doc_name}_p{page_num+1}_c{chunk_idx}",
                    "text": chunk,
                    "tables": [], # Placeholder: A full multimodal parser (like MinerU) would populate this.
                    "section_heading": None # Placeholder for structure-aware parsing
                }
                chunks.append(chunk_record)
                
    except Exception as e:
        print(f"Error parsing {doc_name}: {e}")
        
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    pdfs_dir = os.path.join(data_dir, "pdfs")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    output_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]
    
    print(f"Starting preprocessing of {len(pdf_files)} PDFs...")
    
    total_chunks = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for i, pdf_file in enumerate(pdf_files):
            doc_name = pdf_file.replace(".pdf", "")
            pdf_path = os.path.join(pdfs_dir, pdf_file)
            print(f"[{i+1}/{len(pdf_files)}] Processing {doc_name}...")
            
            chunks = process_pdf(pdf_path, doc_name)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
            total_chunks += len(chunks)
            print(f"  -> Extracted {len(chunks)} chunks.")
            
    print(f"\nPreprocessing complete. Total chunks generated: {total_chunks}")
    print("NOTE: We are using a standard text extractor (PyMuPDF). This does not fully reproduce")
    print("the paper's complex multimodal MinerU preprocessing (which flawlessly extracts tables and headings).")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
