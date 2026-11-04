import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

out_dir = r"outputs/Group_NUMBER/figures"
os.makedirs(out_dir, exist_ok=True)

def draw_rounded_rect(ax, x, y, w, h, title, subtitle="", bg_color="#ffffff", border_color="#30363d", text_color="#1f2328", sub_color="#57606a", title_size=11, sub_size=8.5, lw=1.5, ls="-"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        linestyle=ls,
        zorder=2
    )
    ax.add_patch(box)
    
    # Text positioning
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", color=text_color, fontsize=title_size, weight="bold", zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", color=sub_color, fontsize=sub_size, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", color=text_color, fontsize=title_size, weight="bold", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color="#57606a", lw=1.5, zorder=1):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14, zorder=zorder)
    )

# -----------------------------------------------------------------------------
# 1. Original Research Method Diagram
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Canvas background
fig.patch.set_facecolor("#f8f9fa")

# Main Title Header
ax.text(7.5, 8.1, "Original FinSage Research Architecture (arXiv:2504.14493, ACM CIKM 2025)", ha="center", va="center", fontsize=15, weight="bold", color="#1f2328")
ax.text(7.5, 7.75, "Four Sequential Pillars: Multi-Modal Ingestion, Multi-Path Fusion, DPO Cross-Encoder, and Large Generator", ha="center", va="center", fontsize=10, color="#57606a")

# 4 Major Subgraph Containers
subgraph_colors = ["#eef2f6", "#f0f4f8", "#f3f0f7", "#fdf6ec"]
subgraph_borders = ["#cbd5e1", "#cbd5e1", "#d8b4fe", "#fed7aa"]
titles = [
    "1. Financial Filing Preprocessing (FFP)",
    "2. Multi-Path Retrieval & Bundling",
    "3. Domain-Specialized Re-ranker (DRR)",
    "4. 72B Generation & Evaluation"
]

col_w = 3.3
xs = [0.4, 3.95, 7.5, 11.05]

for i in range(4):
    cont = patches.FancyBboxPatch((xs[i], 0.4), col_w, 7.0, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=subgraph_colors[i], edgecolor=subgraph_borders[i], lw=1.5, zorder=1)
    ax.add_patch(cont)
    ax.text(xs[i] + col_w/2, 7.15, titles[i], ha="center", va="center", fontsize=10.5, weight="bold", color="#1e293b", zorder=2)

# Column 1: FFP
draw_rounded_rect(ax, 0.6, 6.1, 2.9, 0.75, "SEC Filing PDFs", "10-K, 10-Q, 8-K Reports", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 4.9, 2.9, 0.75, "MinerU Layout Engine", "magic-pdf: text, tables, visual boxes", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 3.7, 2.9, 0.75, "VLM Table-to-Text", "Mineru2Base: synthetic prose narratives", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 2.5, 2.9, 0.75, "LLM Coreference & Summaries", "Pronoun scanning (4 preceding chunks)", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 0.7, 2.9, 1.25, "Structured Chunk Store", "256 tokens per chunk\nDeduplication (TF-IDF tau=0.70)\nBGE-M3 Section Heading Index", "#ffffff", "#64748b", lw=2)

draw_arrow(ax, 2.05, 6.1, 2.05, 5.65)
draw_arrow(ax, 2.05, 4.9, 2.05, 4.45)
draw_arrow(ax, 2.05, 3.7, 2.05, 3.25)
draw_arrow(ax, 2.05, 2.5, 2.05, 1.95)

# Inter-column arrow 1 -> 2
draw_arrow(ax, 3.5, 1.3, 3.95, 1.3, color="#0284c7", lw=2)

# Column 2: MPR
draw_rounded_rect(ax, 4.15, 6.1, 2.9, 0.75, "User Financial Query", "Natural language query tokens", "#e0f2fe", "#0284c7")
draw_rounded_rect(ax, 4.15, 5.0, 2.9, 0.65, "Path 1: BM25 Sparse", "Lexical keyword scoring (k1=1.5, b=0.75)", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 4.15, 4.1, 2.9, 0.65, "Path 2: BGE-M3 Dense", "1024-dim dense FAISS search", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 4.15, 3.2, 2.9, 0.65, "Path 3: Metadata Heading", "Heading summaries fetch full sections", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 4.15, 2.3, 2.9, 0.65, "Path 4: Generative HyDE", "InPars Qwen-7B-SFT (3 domain passages)", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 4.15, 0.7, 2.9, 1.15, "Chunk Bundling Fusion", "Candidate Union (Pool >= 40)\nAdjacent Cosine Expansion (tau=0.85)\nCapacity Cap K_bundle = 5", "#ffffff", "#0284c7", lw=2)

draw_arrow(ax, 4.5, 6.1, 4.5, 5.65)
draw_arrow(ax, 5.0, 6.1, 5.0, 4.75)
draw_arrow(ax, 5.5, 6.1, 5.5, 3.85)
draw_arrow(ax, 6.0, 6.1, 6.0, 2.95)

draw_arrow(ax, 5.6, 5.0, 5.6, 1.85)
draw_arrow(ax, 5.6, 4.1, 5.6, 1.85)
draw_arrow(ax, 5.6, 3.2, 5.6, 1.85)
draw_arrow(ax, 5.6, 2.3, 5.6, 1.85)

# Inter-column arrow 2 -> 3
draw_arrow(ax, 7.05, 1.3, 7.5, 1.3, color="#9333ea", lw=2)

# Column 3: DRR
draw_rounded_rect(ax, 7.7, 5.5, 2.9, 1.25, "Cross-Encoder Backbone", "BAAI/bge-reranker-v2-gemma\n2.5B parameter causal backbone\nScored via Yes/No token logits", "#faf5ff", "#9333ea")
draw_rounded_rect(ax, 7.7, 3.7, 2.9, 1.25, "DPO Pairwise Training", "Trained on Lotus (s+, s-) pairs\nLoRA r=32, a=64, 10 epochs\nDirect Preference Optimization", "#ffffff", "#a855f7")
draw_rounded_rect(ax, 7.7, 2.2, 2.9, 0.95, "Date Time Bonus", "K(s,q) = sigma(w^T CE + b + beta*Time)\nStrictly zero statement/table boosts", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 7.7, 0.7, 2.9, 0.95, "Top-5 Context Selection", "Delivers top 5 ranked chunks\nPrecision: 78.8%, Norm Recall: 79.1%", "#ffffff", "#9333ea", lw=2)

draw_arrow(ax, 9.15, 1.85, 9.15, 2.2)
draw_arrow(ax, 9.15, 3.7, 9.15, 5.5)
draw_arrow(ax, 9.15, 2.2, 9.15, 1.65)

# Inter-column arrow 3 -> 4
draw_arrow(ax, 10.6, 1.15, 11.05, 1.15, color="#ea580c", lw=2)

# Column 4: Generation
draw_rounded_rect(ax, 11.25, 5.2, 2.9, 1.5, "Qwen2.5-72B-Instruct", "72 Billion Parameter LLM (AWQ 4-bit)\nDeployed on NVIDIA A100 (80GB VRAM)\nGenerates natural conversational text", "#fff7ed", "#ea580c")
draw_rounded_rect(ax, 11.25, 3.1, 2.9, 1.4, "Dual Evaluation Protocol", "1. LlamaIndex GPT-4o CorrectnessJudge\n2. Expert Domain Human Analysts\nEvaluated on full 149 FinanceBench Qs", "#ffffff", "#ea580c")
draw_rounded_rect(ax, 11.25, 0.7, 2.9, 1.7, "Published Benchmark Metrics", "FinanceBench: 49.7% LLM / 57.1% Manual\nCompany Dataset: 85.3% LLM / 88.0% Manual\nOutperformed prior RAG (19.0%-36.9%)", "#ffffff", "#16a34a", lw=2)

draw_arrow(ax, 12.7, 5.2, 12.7, 4.5)
draw_arrow(ax, 12.7, 3.1, 12.7, 2.4)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "original_method_diagram.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved original_method_diagram.png")

# -----------------------------------------------------------------------------
# 2. Modified Method Diagram (Our Implementation - Stages 1 & 2)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")

ax.text(7.5, 8.1, "Our Implemented & Modified FinSage Architecture (Stage 1 Reproduction + Stage 2 Enhancements)", ha="center", va="center", fontsize=14, weight="bold", color="#1f2328")
ax.text(7.5, 7.75, "Modifications Highlighted: Ingestion Table Propagation, Coreference Resolver, Stitching, Statement Calibrations, and Sandboxed Math", ha="center", va="center", fontsize=9.5, color="#57606a")

# Containers
for i in range(4):
    cont = patches.FancyBboxPatch((xs[i], 0.4), col_w, 7.0, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=subgraph_colors[i], edgecolor=subgraph_borders[i], lw=1.5, zorder=1)
    ax.add_patch(cont)
    ax.text(xs[i] + col_w/2, 7.15, titles[i], ha="center", va="center", fontsize=10.5, weight="bold", color="#1e293b", zorder=2)

# Column 1: Ingestion
draw_rounded_rect(ax, 0.6, 6.1, 2.9, 0.75, "84 SEC Filing PDFs", "data/financebench/pdfs/ (10-K, 10-Q)", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 4.8, 2.9, 0.95, "PyMuPDF fitz Extractor", "Raw text & exact Markdown tables\n(Omitted synthetic VLM rewriting)", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 0.6, 3.4, 2.9, 0.95, "Recursive Chunking", "1,000 chars, 200 overlap\n54,564 Chunks in parsed_chunks.jsonl", "#ffffff", "#94a3b8")
# MODIFICATION 1: Table Header Propagation
draw_rounded_rect(ax, 0.6, 0.7, 2.9, 2.2, "MODIFICATION 1 (Stage 2):\nTable Header Propagation", "Enriched 2,902 continuation chunks\nPrepended doc title & column headers\nEliminates orphan table bisection\nPool Recall: 75.3% -> 86.0% (+10.7 pp)\nQ1 CapEx jumped #21 -> #1 natively", "#dcfce7", "#16a34a", text_color="#14532d", lw=2)

draw_arrow(ax, 2.05, 6.1, 2.05, 5.75)
draw_arrow(ax, 2.05, 4.8, 2.05, 4.35)
draw_arrow(ax, 2.05, 3.4, 2.05, 2.9)
draw_arrow(ax, 3.5, 1.8, 3.95, 1.8, color="#16a34a", lw=2)

# Column 2: MPR & Coref
# MODIFICATION 2: Coreference Resolver
draw_rounded_rect(ax, 4.15, 5.65, 2.9, 1.25, "MODIFICATION 2 (Stage 2):\nCoreference & Brand Resolver", "COREFERENCE_TAXONOMY mapping\nMounjaro -> Eli Lilly, YouTube -> Alphabet\nPronoun anaphora (it, they, its, their)\n+6.0 pp R@5 (FB), +4.0 pp R@10 (Stage 3)", "#dbeafe", "#2563eb", text_color="#1e3a8a", lw=2)

draw_rounded_rect(ax, 4.15, 4.6, 2.9, 0.8, "4-Path Hybrid Retrieval", "BM25S Lexical + BGE-M3 FAISS (1024-d)\n+ Metadata Heading + Prompted HyDE", "#ffffff", "#94a3b8")
draw_rounded_rect(ax, 4.15, 3.45, 2.9, 0.85, "Reciprocal Rank Fusion (RRF)", "RRF standard k=60, Pool K=40/50/65\nMerges sparse, dense, and metadata", "#ffffff", "#94a3b8")
# MODIFICATION 3: Chunk Bundling Threshold
draw_rounded_rect(ax, 4.15, 0.7, 2.9, 2.4, "MODIFICATION 3 (Stage 1 Adapt):\nSemantic Chunk Bundling", "Adjacency cosine test (K_bundle=5)\nPaper claimed tau=0.85 (halted on CapEx)\nEmpirically adapted tau=0.70\nCaptures 0.7172 table bisection in Q1\nStitches operating to investing cash flows", "#fef3c7", "#d97706", text_color="#78350f", lw=2)

draw_arrow(ax, 5.6, 5.65, 5.6, 5.4)
draw_arrow(ax, 5.6, 4.6, 5.6, 4.3)
draw_arrow(ax, 5.6, 3.45, 5.6, 3.1)
draw_arrow(ax, 7.05, 1.8, 7.5, 1.8, color="#9333ea", lw=2)

# Column 3: DRR & Stitcher
draw_rounded_rect(ax, 7.7, 5.65, 2.9, 1.25, "Stage 1: Faithfully Trained DRR", "BAAI/bge-reranker-v2-gemma (2.5B)\n4-bit NF4 QLoRA (r=32, a=64, 10 ep)\nPure logits (ZERO HEURISTIC BOOSTS)\nMRR: 0.3503 -> 0.5379 (+0.1876)", "#faf5ff", "#9333ea", text_color="#581c87", lw=1.8)

# MODIFICATION 4: Statement Routing & Distractor Penalty
draw_rounded_rect(ax, 7.7, 3.5, 2.9, 1.85, "MODIFICATION 4 (Stage 2):\nCalibrated Statement Routing", "Dynamic query statement routing:\n+3.5 to +4.5 target statement boost\n-3.0 negative narrative penalty\nSuppresses environmental/MD&A noise\nEliminates ranking collisions", "#f3e8ff", "#9333ea", text_color="#581c87", lw=2)

# MODIFICATION 5: Forward Stitcher
draw_rounded_rect(ax, 7.7, 0.7, 2.9, 2.4, "MODIFICATION 5 (Stage 2):\nL.9 Forward Context Stitching", "ContextWindowStitcher (mode='forward')\nAsymmetric next-chunk concatenation\nMax 1,200 chars window preservation\nProtects candidate chunk ID in pool\nRecovers cross-page boundary sentences", "#e0e7ff", "#4338ca", text_color="#312e81", lw=2)

draw_arrow(ax, 9.15, 5.65, 9.15, 5.35)
draw_arrow(ax, 9.15, 3.5, 9.15, 3.1)
draw_arrow(ax, 10.6, 1.8, 11.05, 1.8, color="#ea580c", lw=2)

# Column 4: 7B Generator & Math Sandbox
draw_rounded_rect(ax, 11.25, 5.6, 2.9, 1.3, "Stage 1 Generator Adaptation", "Qwen2.5-7B-Instruct (4-bit NF4)\nFits in 8GB RTX 5060 Laptop GPU\nGreedy decoding, max_tokens=256\nOutputs strict operational JSON schema", "#fff7ed", "#ea580c", text_color="#7c2d12")

# MODIFICATION 6: Arithmetic Sandbox Engine
draw_rounded_rect(ax, 11.25, 2.8, 2.9, 2.4, "MODIFICATION 6 (Stage 2):\nPython Arithmetic Sandbox", "arithmetic_evaluator.py (WP4 Engine)\nExtracts formula & raw line operands\nEvaluates math deterministically in Python:\n+, -, %, scale conversions (M to B)\n100% precision on extracted operands\nEliminates LLM arithmetic hallucinations", "#ffedd5", "#c2410c", text_color="#7c2d12", lw=2)

draw_rounded_rect(ax, 11.25, 0.7, 2.9, 1.7, "Verification & Interactive UI", "Harness V2: Decoupled offline evaluator\n(Semantic, numeric, and schema checks)\nInteractive Streamlit Web Dashboard\n(app.py on http://localhost:8501)", "#ffffff", "#0284c7", lw=1.8)

draw_arrow(ax, 12.7, 5.6, 12.7, 5.2)
draw_arrow(ax, 12.7, 2.8, 12.7, 2.4)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "modified_method_diagram.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved modified_method_diagram.png")
