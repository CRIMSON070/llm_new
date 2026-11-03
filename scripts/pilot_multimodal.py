import os
import json
import uuid
import subprocess
import requests
import base64
import time

def summarize_image(image_path: str, model: str = "llama3.2-vision") -> str:
    """Use Ollama's vision model to summarize a chart/figure."""
    prompt = "State the chart title, metric, units, time period, visible trend, and relevant values only when supported by the image. Mark unreadable values as uncertain instead of guessing."
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
                "options": {"temperature": 0.1}
            }
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"[Vision Error] API returned status {response.status_code}")
    except Exception as e:
        print(f"[Vision Error] {e}")
    return "uncertain"

def run_magic_pdf(pdf_path: str, out_dir: str, env_path: str):
    """Run magic-pdf (MinerU) using the isolated venv."""
    print(f"\n[MinerU] Processing {os.path.basename(pdf_path)}...")
    pdf_out = os.path.join(out_dir, os.path.basename(pdf_path).replace('.pdf', ''))
    
    magic_pdf_exe = os.path.join(env_path, "Scripts", "magic-pdf.exe")
    if not os.path.exists(magic_pdf_exe):
        magic_pdf_exe = "magic-pdf" # Fallback if in PATH
        
    cmd = f"{magic_pdf_exe} -p {pdf_path} -o {out_dir} -m txt"
    print(f"Running: {cmd}")
    
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running magic-pdf on {pdf_path}:\n{result.stderr}")
        return None
        
    print(f"MinerU extraction completed in {time.time()-t0:.2f}s")
    
    # magic-pdf creates a folder with the same name as the pdf (without extension) inside out_dir
    return pdf_out

def parse_magic_pdf_output(pdf_out_dir: str, doc_name: str):
    """Parse the output markdown and images to create multimodal chunks."""
    chunks = []
    
    md_file = None
    for root, dirs, files in os.walk(pdf_out_dir):
        for file in files:
            if file.endswith('.md'):
                md_file = os.path.join(root, file)
                break
                
    if not md_file:
        print(f"Warning: No markdown file found in {pdf_out_dir}")
        return chunks
        
    print(f"Parsing markdown from {md_file}")
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Basic markdown parsing by headers.
    md_parts = content.split('\n#')
    for i, part in enumerate(md_parts):
        text = part.strip()
        if not text:
            continue
            
        chunk_type = "table" if "|" in text and "-|-" in text else "text"
        
        chunk_id = f"{doc_name}_mm_{i}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "page_number": None, # Unavailable in plain markdown without layout json
            "section_heading": None,
            "type": chunk_type,
            "text": text,
            "metadata": {
                "extraction_method": "MinerU_Markdown",
            }
        })
        
    # Process Images
    img_dir = os.path.join(pdf_out_dir, "images")
    if os.path.exists(img_dir):
        images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} images in {img_dir}. Summarizing...")
        for img in images:
            img_path = os.path.join(img_dir, img)
            print(f"  -> Summarizing {img}...")
            summary = summarize_image(img_path)
            
            chunk_id = f"{doc_name}_fig_{uuid.uuid4().hex[:6]}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "page_number": None,
                "section_heading": None,
                "type": "figure_summary",
                "text": f"[Figure Summary] {summary}",
                "image_path": img_path,
                "metadata": {
                    "extraction_method": "MinerU_Vision_Llama3.2",
                }
            })
            
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "financebench", "pdfs")
    mm_out_dir = os.path.join(base_dir, "data", "financebench", "multimodal_out")
    os.makedirs(mm_out_dir, exist_ok=True)
    
    output_file = os.path.join(base_dir, "data", "financebench", "processed", "multimodal_chunks.jsonl")
    env_path = os.path.join(base_dir, "mineru_env")
    
    # Select 2 pilot PDFs
    test_pdfs = ["3M_2018_10K.pdf", "ULTABEAUTY_2023Q4_EARNINGS.pdf"]
    
    all_chunks = []
    
    for pdf_name in test_pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
            
        doc_name = pdf_name.replace('.pdf', '')
        
        # Step 2: Run MinerU
        pdf_out_dir = run_magic_pdf(pdf_path, mm_out_dir, env_path)
        
        if pdf_out_dir and os.path.exists(pdf_out_dir):
            # Step 3 & 4: Parse outputs and summarize charts
            chunks = parse_magic_pdf_output(pdf_out_dir, doc_name)
            all_chunks.extend(chunks)
            print(f"Generated {len(chunks)} multimodal chunks for {doc_name}")
            
    # Save output chunks
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
            
    print(f"\nPilot Multimodal extraction complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
