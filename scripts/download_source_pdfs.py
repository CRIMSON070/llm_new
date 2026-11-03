import os
import json
import urllib.request
import time

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    questions_file = os.path.join(data_dir, "questions.jsonl")
    pdfs_dir = os.path.join(data_dir, "pdfs")
    manifest_file = os.path.join(data_dir, "manifest.json")
    
    os.makedirs(pdfs_dir, exist_ok=True)
    
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found.")
        return
        
    unique_docs = set()
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if 'doc_name' in record:
                unique_docs.add(record['doc_name'])
                
    print(f"Found {len(unique_docs)} unique documents to download.")
    
    base_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
    manifest = {}
    
    for i, doc_name in enumerate(list(unique_docs)):
        pdf_filename = f"{doc_name}.pdf"
        local_path = os.path.join(pdfs_dir, pdf_filename)
        url = f"{base_url}{pdf_filename}"
        
        print(f"[{i+1}/{len(unique_docs)}] Processing {pdf_filename}...")
        
        if os.path.exists(local_path):
            print(f"  Already exists: {pdf_filename}")
            manifest[doc_name] = {"local_path": os.path.relpath(local_path, data_dir), "status": "downloaded"}
            continue
            
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as out_file:
                        out_file.write(response.read())
                    manifest[doc_name] = {"local_path": os.path.relpath(local_path, data_dir).replace('\\', '/'), "status": "downloaded"}
                else:
                    print(f"  Failed to download {pdf_filename}: HTTP {response.status}")
                    manifest[doc_name] = {"status": "failed", "error": f"HTTP {response.status}"}
        except urllib.error.HTTPError as e:
            print(f"  Failed to download {pdf_filename}: HTTP Error {e.code}")
            manifest[doc_name] = {"status": "failed", "error": f"HTTP Error {e.code}"}
        except Exception as e:
            print(f"  Error downloading {pdf_filename}: {e}")
            manifest[doc_name] = {"status": "failed", "error": str(e)}
        
        # Sleep slightly to avoid overwhelming the server
        time.sleep(0.5)
        
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest saved to {manifest_file}.")
    print("Download process completed.")

if __name__ == "__main__":
    main()
