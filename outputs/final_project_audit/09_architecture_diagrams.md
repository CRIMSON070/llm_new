# FinSage Architecture Block Diagrams

**Document:** `outputs/final_project_audit/09_architecture_diagrams.md`  
**Reference Paper:** *FinSage: A Multi-aspect RAG System for Financial Filings Question Answering* (Wang et al., arXiv:2504.14493, ACM CIKM 2025)  
**Governing Standard:** Precise Architectural Block Diagrams Comparing Published vs. Implemented Systems  

---

## 1. Diagram 1: Original FinSage Research Architecture

The diagram below illustrates the exact architecture described in the published paper (*arXiv:2504.14493*):

```mermaid
flowchart TD
    subgraph S1["1. Financial Filing Preprocessing (FFP)"]
        A1["SEC PDF Filings (10-K, 10-Q, 8-K)"] --> A2["MinerU Layout Engine (magic-pdf)"]
        A2 --> A3["Text Blocks"]
        A2 --> A4["Tables"]
        A2 --> A5["Infographics & Charts"]
        A4 --> A6["VLM Table-to-Text (Mineru2Base / GPT-4o)"]
        A3 --> A7["LLM Co-reference Resolution (4 Preceding Chunks)"]
        A6 --> A8["Section Heading-Summary Metadata (BGE-M3)"]
        A7 --> A8
        A8 --> A9["Deduplication (TF-IDF Cosine tau=0.70)"]
        A9 --> A10[("Structured Chunk Store (256 Tokens)")]
    end

    subgraph S2["2. Multi-Path Retrieval (MPR)"]
        B1["User Financial Query"] --> B2["Path 1: BM25 Lexical Retrieval"]
        B1 --> B3["Path 2: Dense Semantic Search (BGE-M3 FAISS)"]
        B1 --> B4["Path 3: Metadata Heading-Summary Search (FAISS)"]
        B1 --> B5["Path 4: Generative HyDE (InPars Qwen-7B-SFT)"]
        
        A10 --> B2
        A10 --> B3
        A10 --> B4
        A10 --> B5
        
        B2 --> B6["Candidate Pool Union"]
        B3 --> B6
        B4 --> B6
        B5 --> B6
        
        B6 --> B7["Chunk Bundling (Adjacency Cosine tau=0.85, K=5)"]
    end

    subgraph S3["3. Domain-Specialized Re-ranking (DRR)"]
        B7 --> C1["Cross-Encoder Reranker (BAAI/bge-reranker-v2-gemma)"]
        C2["DPO Fine-Tuned Weights (LoRA r=32, a=64, 10 Epochs)"] --> C1
        C3["Publication Date Time Bonus f(m_i)"] --> C1
        C1 --> C4["Top-5 Ranked Context Chunks (K=5)"]
    end

    subgraph S4["4. Downstream Generation & Evaluation"]
        C4 --> D1["Generator LLM: Qwen2.5-72B-Instruct-AWQ (NVIDIA A100 80GB)"]
        B1 --> D1
        D1 --> D2["Natural Conversational Text Answer"]
        D2 --> D3["Evaluation: LlamaIndex GPT-4o + Human Expert Review"]
    end
```

---

## 2. Diagram 2: Implemented FinSage System Architecture

The diagram below captures our **actual workspace implementation**, explicitly highlighting faithfully reproduced components, hardware/API adaptations, and production enhancements:

```mermaid
flowchart TD
    subgraph IMP1["1. Implemented Ingestion & Preprocessing"]
        I1["84 SEC PDF Filings (data/financebench/pdfs/)"] --> I2["PyMuPDF fitz Text & Table Extraction"]
        I2 --> I3["Raw Markdown Tables Preserved (NO Synthetic Narratives)"]
        I2 --> I4["Page-Preserved Raw Text (NO Pronoun Resolution)"]
        I3 --> I5["RecursiveCharacterTextSplitter (1000 Chars, 200 Overlap)"]
        I4 --> I5
        I5 --> I6[("parsed_chunks.jsonl: 54,564 Chunks (57.1 MB)")]
        
        I6 -.->|"Stage 2 Enhancement"| I7["Table Header Propagation (2,902 Split Chunks Enriched)"]
    end

    subgraph IMP2["2. Implemented Multi-Path Retrieval (MPR)"]
        I6 --> J1[("Sparse Index: bm25s")]
        I6 --> J2[("Dense FAISS: BGE-M3 (54,564 Vectors, 223 MB)")]
        I6 --> J3[("Metadata FAISS: 2,678 Section Summaries (10.9 MB)")]
        I7 -.-> J1
        
        Q["User Financial Query"] --> P1["Path 1: BM25 Lexical Search (bm25s)"]
        Q --> P2["Path 2: Dense Semantic Search (BGE-M3 FAISS)"]
        Q --> P3["Path 3: Metadata Section Search (Fetches Segment Blocks)"]
        Q --> P4["Path 4: Generative HyDE (Prompted Qwen2.5-7B NF4)"]
        
        J1 --> P1
        J2 --> P2
        J3 --> P3
        J2 --> P4
        
        P1 --> F1["Reciprocal Rank Fusion RRF (k=60, K_pool=40)"]
        P2 --> F1
        P3 --> F1
        P4 --> F1
        
        F1 --> F2["Semantic Chunk Bundling (tau_exp=0.70 Calibrated, K=5)"]
    end

    subgraph IMP3["3. Implemented Reranker (Dual Mode Support)"]
        F2 --> K1{"Reranker Execution Mode"}
        
        K1 -->|"Stage 1: Paper Reproduction"| K2["Trained Gemma DRR Adapter (ZERO HEURISTIC BOOSTS +0.0)"]
        K1 -->|"Stage 2: Production FinSage"| K3["Phase 2 Calibrated Reranker (+3.5 Statement / -3.0 Penalty)"]
        
        K4[("outputs/phase1_drr_adapter/adapter_model.safetensors (29.5 MB)")] --> K2
        K4 --> K3
        
        K2 --> K5["Assembled Top-Context Window (K=5 Chunks)"]
        K3 --> K5
    end

    subgraph IMP4["4. Implemented Generation & Evaluation"]
        K5 --> G1["Generator LLM: Qwen2.5-7B-Instruct (4-bit NF4 on RTX 5060 8GB)"]
        Q --> G1
        G1 --> G2["Operational JSON (operation, operands) + Explanation"]
        
        G2 -.->|"Stage 2 Enhancement"| G3["Deterministic Python Sandbox Math Engine"]
        
        G2 --> E1["Stage 1 Eval: Harness V2 Scorecard (Semantic, Numeric, Schema on Q1-Q6)"]
        G3 --> E2["Stage 2 Eval: Full FinanceBench Scorecard (N=150 across 9 Sectors)"]
        
        G2 --> UI["Interactive Streamlit Dashboard (app.py on http://localhost:8501)"]
    end
```

---

## 3. Key Differences Between Architecture 1 and Architecture 2

| Architectural Dimension | Original Research Paper (Diagram 1) | Implemented System (Diagram 2) | Design Rationale & Impact |
|---|---|---|---|
| **Document Ingestion** | `MinerU` layout engine + VLM Table-to-Text synthetic prose rewriting | `PyMuPDF` text + exact Markdown tables | Prevents synthetic numerical hallucinations; preserves exact accounting rows/columns. |
| **Table Continuity** | MinerU multi-page table stitching | Stage 1: Chunk Bundling ($\tau=0.70$). Stage 2: Table Header Propagation (2,902 chunks). | Natively stitches bisected statements (e.g. 3M FY18 CapEx) across chunk boundaries. |
| **Generative HyDE** | `Qwen2.5-7B` fine-tuned via InPars on 72B synthetic pairs | `Qwen2.5-7B` instruction-prompted with financial system prompts | Eliminates compute overhead of synthetic InPars pre-training while generating 3 domain passages. |
| **Reranker Training** | Full `bfloat16` fine-tuning on multi-GPU enterprise cluster | 4-bit NormalFloat4 (NF4) QLoRA on 8.55 GB RTX 5060 GPU | Enables complete DPO training from scratch on local hardware within 3.64 GB peak VRAM. |
| **Reranker Scoring** | Pure cross-encoder logits + Time bonus | Dual Mode: Stage 1 = Zero boosts (+0.0). Stage 2 = Calibrated routing (+3.5 / -3.0). | Allows instant toggling between academic paper reproduction and production optimization. |
| **Generator Model** | `Qwen2.5-72B-Instruct-AWQ` on A100 (80 GB) | `Qwen2.5-7B-Instruct` (4-bit NF4) on RTX 5060 (8.55 GB) | Accessible consumer hardware deployment with deterministic greedy decoding. |
| **Downstream Verification** | Online GPT-4o CorrectnessEvaluator API | Offline deterministic Harness V2 + Python math sandbox | Eliminates external API dependencies and guarantees 100% reproducible numerical auditing. |
| **User Interface** | Internal API server | Interactive Streamlit Dashboard (`app.py` on `:8501`) | Provides executive web UI with live mode switching and telemetry. |
