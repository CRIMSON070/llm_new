"""
Phase 9C Master Evaluation Script:
Orchestrates:
1. Step 1: Retrieval Ablations (Dense, Dense+BM25, Hybrid Expanded, Hybrid Expanded + Statement-Aware Reranked)
2. Step 2: Downstream Extraction Evaluation (Top-3 vs Top-5 retrieved contexts with 3-epoch DPO model)
Runs in separate clean sub-processes to avoid CUDA/mmap memory pressure on Windows.
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_exec = sys.executable
    
    print("=" * 70)
    print("FINSAGE PHASE 9C: HYBRID RETRIEVAL & STATEMENT-AWARE RERANKING")
    print("=" * 70)
    
    step1_script = os.path.join(base_dir, "scripts", "run_retrieval_phase9c.py")
    step2_script = os.path.join(base_dir, "scripts", "run_generation_phase9c.py")
    
    print("\n>>> Launching Step 1: Retrieval Ablations...")
    res1 = subprocess.run([py_exec, step1_script], check=True)
    if res1.returncode != 0:
        print("Step 1 failed!")
        sys.exit(res1.returncode)
        
    print("\n>>> Launching Step 2: Downstream Generation Evaluation...")
    res2 = subprocess.run([py_exec, step2_script], check=True)
    if res2.returncode != 0:
        print("Step 2 failed!")
        sys.exit(res2.returncode)

    print("\n" + "=" * 70)
    print("Phase 9C Evaluation Completed Successfully!")
    print("Deliverables available at: outputs/phase9c/")
    print("=" * 70)

if __name__ == "__main__":
    main()
