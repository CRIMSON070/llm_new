import os
import json

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    questions_file = os.path.join(data_dir, "questions.jsonl")
    manifest_file = os.path.join(data_dir, "manifest.json")
    validation_file = os.path.join(data_dir, "validation_report.json")
    pdfs_dir = os.path.join(data_dir, "pdfs")
    
    if not os.path.exists(questions_file) or not os.path.exists(manifest_file):
        print("Required files missing.")
        return
        
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    records = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
            
    num_questions = len(records)
    unique_doc_names = set(r.get('doc_name') for r in records if r.get('doc_name'))
    
    downloaded_pdfs = [d for d in os.listdir(pdfs_dir) if d.endswith('.pdf')]
    
    missing_pdfs_ref = []
    missing_qa = []
    duplicate_q_ids = set()
    seen_q_ids = set()
    
    unmapped_questions = []
    
    for r in records:
        q_id = r.get('financebench_id')
        if q_id in seen_q_ids:
            duplicate_q_ids.add(q_id)
        seen_q_ids.add(q_id)
        
        if not r.get('question') or not r.get('answer'):
            missing_qa.append(q_id)
            
        doc_name = r.get('doc_name')
        if doc_name not in manifest or manifest[doc_name].get('status') != 'downloaded':
            missing_pdfs_ref.append(doc_name)
            unmapped_questions.append(q_id)
            
    report = {
        "num_questions_downloaded": num_questions,
        "num_unique_document_identifiers": len(unique_doc_names),
        "num_pdfs_downloaded": len(downloaded_pdfs),
        "missing_pdf_references": list(set(missing_pdfs_ref)),
        "missing_questions_or_answers_ids": missing_qa,
        "duplicate_question_ids": list(duplicate_q_ids),
        "unmapped_questions_ids": unmapped_questions,
        "all_questions_mapped_to_source": len(unmapped_questions) == 0
    }
    
    print(json.dumps(report, indent=4))
    
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation report saved to {validation_file}.")

if __name__ == "__main__":
    main()
