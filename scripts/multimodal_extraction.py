import os
import json
import uuid
import subprocess
import requests

def summarize_image(image_path: str, model: str = "llama3.2-vision") -> str:
    """Use Ollama's vision model to summarize a chart/figure."""
    prompt = "This is a chart or figure extracted from a financial report. Describe the key data points, trends, and takeaways in detail."
    # Wait, Ollama API supports passing images as base64 in the 'images' array.
    import base64
    try:
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [b64_img],
                "stream": False,
                "options": {"temperature": 0.2}
            }
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Vision Error] {e}")
    return "A chart or figure containing financial data."

def extract_multimodal_chunks(pdf_path: str, out_dir: str):
    """Run magic-pdf (MinerU) to extract markdown tables and figure crops."""
    print(f"\n[MinerU] Processing {os.path.basename(pdf_path)}...")
    pdf_out = os.path.join(out_dir, os.path.basename(pdf_path).replace('.pdf', ''))
    
    # Run magic-pdf CLI
    cmd = f"magic-pdf -p {pdf_path} -o {pdf_out} -m txt"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running magic-pdf on {pdf_path}: {result.stderr}")
        return []
    
    chunks = []
    
    # magic-pdf outputs a folder containing a .md file and an /images folder
    # We will search the output directory for the markdown content.
    # In a full implementation, we would parse layout.json to extract exact bounding boxes
    # and link images to their textual context, but for Phase 4 incremental integration,
    # we just need to ingest the tables and pass the images to the Vision LLM.
    
    md_file = None
    for root, dirs, files in os.walk(pdf_out):
        for file in files:
            if file.endswith('.md'):
                md_file = os.path.join(root, file)
                break
                
    if md_file:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Simple chunking for the markdown (split by headers)
            # This ensures tables (which are markdown formatted) stay intact
            md_chunks = content.split('\n## ')
            for i, chunk in enumerate(md_chunks):
                chunk_id = f"{os.path.basename(pdf_path).replace('.pdf', '')}_mm_{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk.strip(),
                    "metadata": {
                        "doc_name": os.path.basename(pdf_path).replace('.pdf', ''),
                        "type": "markdown_text_and_table"
                    }
                })
                
    # Process Images
    img_dir = os.path.join(pdf_out, "images")
    if os.path.exists(img_dir):
        print(f"Found images in {img_dir}. Querying llama3.2-vision...")
        for img in os.listdir(img_dir):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, img)
                summary = summarize_image(img_path)
                
                chunk_id = f"{os.path.basename(pdf_path).replace('.pdf', '')}_fig_{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": f"[Figure Summary] {summary}",
                    "metadata": {
                        "doc_name": os.path.basename(pdf_path).replace('.pdf', ''),
                        "type": "figure_summary",
                        "image_path": img_path
                    }
                })
                
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "financebench", "pdfs")
    mm_out_dir = os.path.join(base_dir, "data", "financebench", "multimodal_out")
    os.makedirs(mm_out_dir, exist_ok=True)
    
    output_file = os.path.join(base_dir, "data", "financebench", "processed", "multimodal_chunks.jsonl")
    
    # Process just the first PDF as a test to avoid OOM
    test_pdf = os.path.join(pdf_dir, "3M_2018_10K.pdf")
    if not os.path.exists(test_pdf):
        print(f"Test PDF not found: {test_pdf}")
        return
        
    print("Starting Multimodal Extraction...")
    chunks = extract_multimodal_chunks(test_pdf, mm_out_dir)
    
    print(f"Extracted {len(chunks)} multimodal chunks (Markdown Tables + Figure Summaries).")
    
    # Append to JSONL
    with open(output_file, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
            
    print(f"Appended chunks to {output_file}.")
    print("\nNext Step: Rerun `build_vector_index.py` and `build_bm25_index.py` over this new file to append them to the RAG database!")

if __name__ == "__main__":
    main()
