import json
import os

def load_results(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_file = os.path.join(base_dir, "data", "financebench", "baseline_results.jsonl")
    upgraded_file = os.path.join(base_dir, "data", "financebench", "upgraded_results.jsonl")
    
    baseline = load_results(baseline_file)
    upgraded = load_results(upgraded_file)
    
    print("| Q# | Question | Baseline Correct Doc | Upgraded Correct Doc | Baseline Answer | Upgraded Answer | Expected |")
    print("|---|---|---|---|---|---|---|")
    
    for i in range(len(baseline)):
        b = baseline[i]
        u = upgraded[i]
        
        q_short = b['query'][:40] + "..."
        b_doc = "Yes" if b['correct_doc_retrieved'] else "No"
        u_doc = "Yes" if u['correct_doc_retrieved'] else "No"
        
        b_ans = b['generated_answer'].replace('\n', ' ')[:50] + "..."
        u_ans = u['generated_answer'].replace('\n', ' ')[:50] + "..."
        exp = b['expected_answer'].replace('\n', ' ')[:50] + "..."
        
        print(f"| {i+1} | {q_short} | {b_doc} | {u_doc} | {b_ans} | {u_ans} | {exp} |")

if __name__ == '__main__':
    main()
