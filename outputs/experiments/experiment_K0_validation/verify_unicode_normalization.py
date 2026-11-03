"""
outputs/experiments/experiment_K0_validation/verify_unicode_normalization.py
=============================================================================
Audit Unicode ligatures in Stage 3 held-out corpus:
1. Scan all 29,335 chunks in data/heldout_stage3/processed/heldout_enriched_chunks.jsonl.
2. Quantify exact frequencies of all ligature glyphs (U+FB00 to U+FB06, etc.).
3. Test NFKC normalization on affected text, checking numbers, symbols, punctuation.
4. Test BM25 (bm25s) and Dense (BGE-M3) and Reranker (bge-reranker-base) tokenizers
   on representative financial terms and real chunk excerpts before and after NFKC.
5. Quantify exact lexical mismatch (unnormalized vs normalized) across financial vocabulary.
"""

import os
import sys
import json
import unicodedata
import collections
from typing import Dict, List, Tuple
import bm25s
import Stemmer
from transformers import AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_K0_validation")

# Common Latin ligatures in Unicode
LIGATURE_MAP = {
    '\ufb00': ('ff', 'LATIN SMALL LIGATURE FF'),
    '\ufb01': ('fi', 'LATIN SMALL LIGATURE FI'),
    '\ufb02': ('fl', 'LATIN SMALL LIGATURE FL'),
    '\ufb03': ('ffi', 'LATIN SMALL LIGATURE FFI'),
    '\ufb04': ('ffl', 'LATIN SMALL LIGATURE FFL'),
    '\ufb05': ('ft', 'LATIN SMALL LIGATURE LONG S T / FT'),
    '\ufb06': ('st', 'LATIN SMALL LIGATURE ST'),
    '\u00a0': (' ', 'NO-BREAK SPACE'),
    '\u200b': ('', 'ZERO WIDTH SPACE'),
    '\u2013': ('-', 'EN DASH'),
    '\u2014': ('-', 'EM DASH'),
    '\u2018': ("'", 'LEFT SINGLE QUOTATION MARK'),
    '\u2019': ("'", 'RIGHT SINGLE QUOTATION MARK'),
    '\u201c': ('"', 'LEFT DOUBLE QUOTATION MARK'),
    '\u201d': ('"', 'RIGHT DOUBLE QUOTATION MARK'),
}

def audit_ligatures():
    print(f"Loading chunks from: {CHUNKS_FILE}")
    ligature_counts = collections.Counter()
    chunks_with_ligatures = set()
    total_chunks = 0
    doc_ligature_chunks = collections.defaultdict(int)
    doc_total_chunks = collections.defaultdict(int)
    
    # Track financial words corrupted by ligatures
    corrupted_word_counts = collections.Counter()
    sample_affected_chunks = []
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            total_chunks += 1
            text = rec["text"]
            doc_name = rec["doc_name"]
            doc_total_chunks[doc_name] += 1
            
            chunk_has_lig = False
            for char in text:
                if char in LIGATURE_MAP:
                    ligature_counts[char] += 1
                    chunk_has_lig = True
                    
            if chunk_has_lig:
                chunks_with_ligatures.add(rec["chunk_id"])
                doc_ligature_chunks[doc_name] += 1
                
                # Check for corrupted financial keywords
                for raw_word in text.split():
                    # check if any ligature in word
                    if any(c in raw_word for c in ['\ufb00', '\ufb01', '\ufb02', '\ufb03', '\ufb04']):
                        norm_word = unicodedata.normalize('NFKC', raw_word).lower().strip('.,;:()[]"\'')
                        if norm_word in [
                            'financial', 'financials', 'fiscal', 'flow', 'flows', 'effective',
                            'officers', 'officer', 'defined', 'benefit', 'benefits', 'deferred',
                            'first', 'operating', 'profit', 'profits', 'liabilities', 'affiliates',
                            'significant', 'difficult', 'inflation', 'influence', 'sufficient'
                        ]:
                            corrupted_word_counts[norm_word] += 1
                            
                if len(sample_affected_chunks) < 5 and any(c in text for c in ['\ufb01', '\ufb00', '\ufb02']):
                    sample_affected_chunks.append({
                        "chunk_id": rec["chunk_id"],
                        "doc_name": rec["doc_name"],
                        "page_number": rec["page_number"],
                        "text_sample": text[:300]
                    })
                    
    print(f"Total chunks scanned: {total_chunks}")
    print(f"Chunks containing ligatures: {len(chunks_with_ligatures)} ({len(chunks_with_ligatures)/total_chunks*100:.2f}%)")
    
    print("\n--- Ligature Glyph Frequencies ---")
    for char, count in ligature_counts.most_common():
        ascii_equiv, name = LIGATURE_MAP.get(char, ('?', 'UNKNOWN'))
        print(f"  U+{ord(char):04X} ({name}) -> '{ascii_equiv}': {count:,} occurrences")
        
    print("\n--- Top Corrupted Financial Keywords ---")
    for word, cnt in corrupted_word_counts.most_common(15):
        print(f"  {word}: {cnt:,} occurrences")
        
    return {
        "total_chunks": total_chunks,
        "chunks_with_ligatures": len(chunks_with_ligatures),
        "ligature_counts": {f"U+{ord(k):04X} ({LIGATURE_MAP[k][0]})": v for k, v in ligature_counts.items()},
        "corrupted_word_counts": dict(corrupted_word_counts.most_common(20)),
        "doc_stats": {
            doc: {
                "ligature_chunks": doc_ligature_chunks[doc],
                "total_chunks": doc_total_chunks[doc],
                "pct": round(doc_ligature_chunks[doc] / doc_total_chunks[doc] * 100, 1)
            }
            for doc in sorted(doc_total_chunks.keys())
        },
        "samples": sample_affected_chunks
    }

def test_normalization_integrity():
    print("\n" + "="*80)
    print("TESTING UNICODE NFKC NORMALIZATION INTEGRITY")
    print("="*80)
    
    test_cases = [
        # Financial phrases with ligatures
        "Consolidated Statements of Cash ﬂows for the ﬁscal year ended December 31, 2024",
        "The eﬀective income tax rate was 18.2% compared to 16.5% in 2023.",
        "Senior executive oﬃcers received beneﬁts under the deﬁned contribution plan.",
        # Table row with numbers, currency symbols, and delimiters
        "Total net revenues | $383,285 | $394,328 | (2.8)% | €10,500 | ¥1,200,000",
        "Operating income (loss): $(1,452.5) million; Diluted EPS: $0.85 per share.",
        "Note 14 — Subsequent Events & Commitments: 10.5% Senior Notes due 2031 [Unaudited]"
    ]
    
    results = []
    for tc in test_cases:
        norm = unicodedata.normalize('NFKC', tc)
        results.append({
            "original": tc,
            "normalized": norm,
            "changed": tc != norm
        })
        print(f"Orig: {tc}")
        print(f"Norm: {norm}")
        print(f"Changed: {tc != norm}\n")
        
    # Check that numbers, punctuation, currency, and pipes are 100% identical
    for tc in test_cases:
        norm = unicodedata.normalize('NFKC', tc)
        # Verify currency symbols preserved
        for sym in ['$', '€', '¥', '%', '|', '(', ')', ':', ',', '-', '.']:
            assert tc.count(sym) == norm.count(sym), f"Symbol '{sym}' count mismatch!"
    print("[VERIFIED] All currency symbols, percentages, numbers, and punctuation are 100% preserved.")
    return results

def test_tokenizer_behavior():
    print("\n" + "="*80)
    print("TESTING TOKENIZER BEHAVIOR: BM25 (bm25s) vs BGE-M3 vs BGE-Reranker")
    print("="*80)
    
    stemmer = Stemmer.Stemmer('english')
    
    # Representative queries and document strings
    test_pairs = [
        {
            "query": "Apple fiscal 2024 financial statements",
            "doc_raw": "Apple Inc. Consolidated ﬁnancial Statements for the ﬁscal year ended September 28, 2024.",
            "doc_norm": "Apple Inc. Consolidated financial Statements for the fiscal year ended September 28, 2024."
        },
        {
            "query": "Alphabet cash flows from operating activities",
            "doc_raw": "Alphabet Inc. Consolidated Statements of Cash ﬂows Operating activities cash ﬂow.",
            "doc_norm": "Alphabet Inc. Consolidated Statements of Cash flows Operating activities cash flow."
        },
        {
            "query": "effective income tax rate benefits",
            "doc_raw": "The eﬀective tax rate reﬂected tax beneﬁts from research and development credits.",
            "doc_norm": "The effective tax rate reflected tax benefits from research and development credits."
        }
    ]
    
    # 1. BM25s tokenizer
    print("--- BM25s Tokenization Test ---")
    bm25_results = []
    for pair in test_pairs:
        q = pair["query"]
        raw = pair["doc_raw"]
        norm = pair["doc_norm"]
        
        # Token strings for inspection
        q_tokens = bm25s.tokenize([q], stopwords="english", stemmer=stemmer, return_ids=False)[0]
        raw_tokens = bm25s.tokenize([raw], stopwords="english", stemmer=stemmer, return_ids=False)[0]
        norm_tokens = bm25s.tokenize([norm], stopwords="english", stemmer=stemmer, return_ids=False)[0]
        
        # Token objects for scoring
        q_tok_obj = bm25s.tokenize([q], stopwords="english", stemmer=stemmer)
        raw_tok_obj = bm25s.tokenize([raw], stopwords="english", stemmer=stemmer)
        norm_tok_obj = bm25s.tokenize([norm], stopwords="english", stemmer=stemmer)
        
        # Exact BM25 scoring with a single document index
        # 1) raw index
        bm_raw = bm25s.BM25()
        bm_raw.index(raw_tok_obj)
        _, raw_score = bm_raw.retrieve(q_tok_obj, k=1)
        
        # 2) norm index
        bm_norm = bm25s.BM25()
        bm_norm.index(norm_tok_obj)
        _, norm_score = bm_norm.retrieve(q_tok_obj, k=1)
        
        # Overlap
        raw_overlap = set(q_tokens).intersection(set(raw_tokens))
        norm_overlap = set(q_tokens).intersection(set(norm_tokens))

        
        res = {
            "query": q,
            "q_tokens": q_tokens,
            "raw_tokens": raw_tokens,
            "norm_tokens": norm_tokens,
            "raw_overlap": list(raw_overlap),
            "norm_overlap": list(norm_overlap),
            "raw_bm25_score": float(raw_score[0][0]),
            "norm_bm25_score": float(norm_score[0][0])
        }
        bm25_results.append(res)
        print(f"\nQuery: '{q}' -> tokens: {q_tokens}")
        print(f"  Raw doc tokens:  {raw_tokens}")
        print(f"  Norm doc tokens: {norm_tokens}")
        print(f"  Raw match:  {raw_overlap} | BM25 Score: {res['raw_bm25_score']:.4f}")
        print(f"  Norm match: {norm_overlap} | BM25 Score: {res['norm_bm25_score']:.4f}")
        print(f"  Score delta: +{res['norm_bm25_score'] - res['raw_bm25_score']:.4f} (+{(res['norm_bm25_score'] - res['raw_bm25_score'])/max(res['raw_bm25_score'], 1e-6)*100:.1f}%)")

    # 2. BGE-M3 Tokenizer test
    print("\n--- BGE-M3 Tokenizer Test (BAAI/bge-m3) ---")
    bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_results = []
    for pair in test_pairs:
        raw = pair["doc_raw"]
        norm = pair["doc_norm"]
        
        raw_subwords = bge_tok.tokenize(raw)
        norm_subwords = bge_tok.tokenize(norm)
        
        bge_results.append({
            "raw": raw,
            "norm": norm,
            "raw_subwords": raw_subwords,
            "norm_subwords": norm_subwords,
            "raw_token_count": len(raw_subwords),
            "norm_token_count": len(norm_subwords)
        })
        print(f"\nRaw text:  {raw}")
        print(f"  Raw BGE-M3 subwords ({len(raw_subwords)}):  {raw_subwords}")
        print(f"Norm text: {norm}")
        print(f"  Norm BGE-M3 subwords ({len(norm_subwords)}): {norm_subwords}")

    # 3. BGE-Reranker Tokenizer test
    print("\n--- BGE-Reranker Tokenizer Test (BAAI/bge-reranker-base) ---")
    rerank_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    rerank_results = []
    for pair in test_pairs:
        raw = pair["doc_raw"]
        norm = pair["doc_norm"]
        
        raw_subwords = rerank_tok.tokenize(raw)
        norm_subwords = rerank_tok.tokenize(norm)
        
        rerank_results.append({
            "raw": raw,
            "norm": norm,
            "raw_subwords": raw_subwords,
            "norm_subwords": norm_subwords,
            "raw_token_count": len(raw_subwords),
            "norm_token_count": len(norm_subwords)
        })
        print(f"\nRaw text:  {raw}")
        print(f"  Raw Reranker subwords ({len(raw_subwords)}):  {raw_subwords}")
        print(f"Norm text: {norm}")
        print(f"  Norm Reranker subwords ({len(norm_subwords)}): {norm_subwords}")

    return {
        "bm25_results": bm25_results,
        "bge_results": bge_results,
        "rerank_results": rerank_results
    }

if __name__ == "__main__":
    audit_data = audit_ligatures()
    norm_data = test_normalization_integrity()
    tok_data = test_tokenizer_behavior()
    
    summary_path = os.path.join(OUT_DIR, "unicode_normalization_audit.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit_data": audit_data,
            "tokenizer_tests": tok_data
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n[DONE] Saved audit results to: {summary_path}")
