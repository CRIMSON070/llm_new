import os
import shutil
import difflib

# 1. Prepare phase2_reranker.py for Experiment E1
src_c_reranker = 'outputs/experiments/experiment_C_company_intent/source_backup/phase2_reranker.py'
with open(src_c_reranker, 'r', encoding='utf-8') as f:
    reranker_code = f.read()

target = 'c["text"][:512]'
replacement = 'c["text"][:1024]'

if target not in reranker_code:
    raise ValueError(f"Target '{target}' not found in {src_c_reranker}")

e1_reranker_code = reranker_code.replace(target, replacement, 1)

with open('src/retrieval/phase2_reranker.py', 'w', encoding='utf-8') as f:
    f.write(e1_reranker_code)

# 2. Prepare run_phase2_benchmark.py for Experiment E1
src_c_runner = 'outputs/experiments/experiment_C_company_intent/source_backup/run_phase2_benchmark.py'
with open(src_c_runner, 'r', encoding='utf-8') as f:
    runner_code = f.read()

target_out = 'outputs", "experiments", "experiment_C_company_intent"'
replacement_out = 'outputs", "experiments", "experiment_E1_context_window"'

if target_out not in runner_code:
    raise ValueError(f"Target '{target_out}' not found in {src_c_runner}")

e1_runner_code = runner_code.replace(target_out, replacement_out, 1)

with open('scripts/run_phase2_benchmark.py', 'w', encoding='utf-8') as f:
    f.write(e1_runner_code)

# 3. Save copy to experiment_E1 source_backup
shutil.copy2('src/retrieval/phase2_reranker.py', 'outputs/experiments/experiment_E1_context_window/source_backup/phase2_reranker.py')
shutil.copy2('scripts/run_phase2_benchmark.py', 'outputs/experiments/experiment_E1_context_window/source_backup/run_phase2_benchmark.py')

print("Successfully written E1 phase2_reranker.py and run_phase2_benchmark.py.")

# Verify diffs
with open(src_c_reranker, 'r', encoding='utf-8') as f:
    c_lines = f.readlines()
with open('src/retrieval/phase2_reranker.py', 'r', encoding='utf-8') as f:
    e1_lines = f.readlines()

diff = list(difflib.unified_diff(c_lines, e1_lines, fromfile='Exp_C_reranker.py', tofile='Exp_E1_reranker.py'))
print("\n--- Diff between Exp C and Exp E1 phase2_reranker.py ---")
for line in diff:
    print(line, end='')

with open(src_c_runner, 'r', encoding='utf-8') as f:
    c_runner_lines = f.readlines()
with open('scripts/run_phase2_benchmark.py', 'r', encoding='utf-8') as f:
    e1_runner_lines = f.readlines()

diff_runner = list(difflib.unified_diff(c_runner_lines, e1_runner_lines, fromfile='Exp_C_runner.py', tofile='Exp_E1_runner.py'))
print("\n--- Diff between Exp C and Exp E1 run_phase2_benchmark.py ---")
for line in diff_runner:
    print(line, end='')
