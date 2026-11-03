import os
import json
from datasets import load_dataset

def main():
    print("Loading PatronusAI/financebench dataset...")
    try:
        dataset = load_dataset("PatronusAI/financebench", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Number of records: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"First record: {dataset[0]}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "financebench")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "questions.jsonl")
    
    print(f"Saving dataset to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in dataset:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print("Download and save completed successfully.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    main()
