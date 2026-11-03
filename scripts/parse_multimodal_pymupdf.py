import os
import json
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter

def process_pdf(pdf_path, doc_name, text_splitter):
    print(f"Processing {doc_name} with pymupdf4llm...")
    chunks = []
    
    try:
        md_pages = pymupdf4llm.to_markdown(
            doc=pdf_path,
            page_chunks=True,
            write_images=False, # Wait, user said do not integrate vision model yet.
        )
        
        for page_data in md_pages:
            page_number = page_data.get('metadata', {}).get('page_number', 0)
            text = page_data.get('text', '')
            
            if not text.strip():
                continue
                
            page_chunks = text_splitter.split_text(text)
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_record = {
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_id": f"{doc_name}_mm_{page_number}_{chunk_idx}",
                    "text": chunk,
                    "type": "markdown_text_and_table",
                    "metadata": {
                        "extraction_method": "pymupdf4llm"
                    }
                }
                chunks.append(chunk_record)
                
    except Exception as e:
        print(f"Error processing {doc_name}: {e}")
        
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    pdfs_dir = os.path.join(data_dir, "pdfs")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    output_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    
    test_pdfs = ["3M_2018_10K.pdf", "ULTABEAUTY_2023Q4_EARNINGS.pdf"]
    
    text_splitter = MarkdownTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    
    all_chunks = []
    for pdf_name in test_pdfs:
        pdf_path = os.path.join(pdfs_dir, pdf_name)
        if os.path.exists(pdf_path):
            doc_name = pdf_name.replace('.pdf', '')
            chunks = process_pdf(pdf_path, doc_name, text_splitter)
            all_chunks.extend(chunks)
        else:
            print(f"File not found: {pdf_path}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
            
    print(f"\nExtracted {len(all_chunks)} chunks using pymupdf4llm.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
