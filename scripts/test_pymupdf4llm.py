import os
import json
import pymupdf4llm

def process_pdf(pdf_path, doc_name, out_dir):
    print(f"Processing {pdf_path} with pymupdf4llm...")
    
    # Extract markdown with page-level chunking
    # pymupdf4llm can extract tables and output markdown
    # page_chunks=True returns a list of dictionaries per page
    try:
        md_chunks = pymupdf4llm.to_markdown(
            doc=pdf_path,
            page_chunks=True,
            write_images=True,
            image_path=os.path.join(out_dir, "images"),
            image_format="png"
        )
        
        # Save output to a debug file for manual inspection
        out_file = os.path.join(out_dir, f"{doc_name}_extracted.md")
        with open(out_file, 'w', encoding='utf-8') as f:
            for chunk in md_chunks:
                f.write(f"<!-- Page {chunk.get('metadata', {}).get('page', 'Unknown')} -->\n")
                f.write(chunk.get("text", ""))
                f.write("\n\n")
                
        print(f"Saved markdown to {out_file}")
        
    except Exception as e:
        print(f"Error processing {doc_name}: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "financebench", "pdfs")
    out_dir = os.path.join(base_dir, "data", "financebench", "pymupdf_test")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    
    test_pdfs = ["3M_2018_10K.pdf", "ULTABEAUTY_2023Q4_EARNINGS.pdf"]
    
    for pdf_name in test_pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if os.path.exists(pdf_path):
            doc_name = pdf_name.replace('.pdf', '')
            process_pdf(pdf_path, doc_name, out_dir)
        else:
            print(f"File not found: {pdf_path}")

if __name__ == "__main__":
    main()
