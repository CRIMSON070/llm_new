"""
Unit Test Suite for Phase 2 Query Intent Routing Fix (Experiment B)
Governing Requirements:
1. Exact American Express query: "Which debt securities are registered to trade on a national securities exchange under American Express?" -> cover_page
2. Reordered version of same question: "Under American Express, what securities that are debt have been registered to trade on national exchanges?" -> cover_page
3. Genuine registered-debt cover-page query: "Does Ulta Beauty have any registered debt securities as of FY 2023? Answer with details from the 10-K cover page." -> cover_page
4. Unrelated debt maturity query: "What is the total principal amount of senior notes and long-term debt maturing in 2025 according to Note 8 debt footnote?" -> None (MUST NOT trigger cover_page)
5. Unrelated debt footnote query: "What are the terms, credit facility commitments, and interest rates on commercial paper issuances?" -> None (MUST NOT trigger cover_page)
6. Existing statement-intent query: "What is the FY2018 capital expenditure amount for 3M in the cash flow statement?" -> cash_flow_statement
7. Existing balance sheet query: "What is the net property, plant, and equipment on the balance sheet?" -> balance_sheet
8. Existing income statement query: "What was the net sales difference in the consolidated statement of income?" -> income_statement
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.phase2_reranker import Phase2Reranker

def run_tests():
    reranker = Phase2Reranker.__new__(Phase2Reranker)
    reranker.STATEMENT_PATTERNS = Phase2Reranker.STATEMENT_PATTERNS
    
    test_cases = [
        # 1. Exact AMEX query
        (
            "Which debt securities are registered to trade on a national securities exchange under American Express?",
            "cover_page",
            "Exact American Express query"
        ),
        # 2. Reordered version
        (
            "Under American Express, what securities that are debt have been registered to trade on national exchanges?",
            "cover_page",
            "Reordered American Express query"
        ),
        # 3. Genuine registered-debt cover-page query
        (
            "Does Ulta Beauty have any registered debt securities as of FY 2023? Answer with details from the 10-K cover page.",
            "cover_page",
            "Ulta Beauty registered debt query"
        ),
        # 4. Another reordered phrasing
        (
            "Are any classes of debt securities registered pursuant to section 12(b) for 3M Company?",
            "cover_page",
            "Section 12(b) registration query"
        ),
        # 5. Unrelated debt maturity query (NEGATIVE TEST)
        (
            "What is the total principal amount of senior notes and long-term debt maturing in 2025 according to Note 8 debt footnote?",
            None,
            "Unrelated debt maturity query (must NOT trigger cover_page)"
        ),
        # 6. Unrelated credit facility / commercial paper (NEGATIVE TEST)
        (
            "What are the terms, credit facility commitments, and interest rates on commercial paper issuances?",
            None,
            "Credit facility and commercial paper query (must NOT trigger cover_page)"
        ),
        # 7. Unrelated total debt query (NEGATIVE TEST)
        (
            "How much total debt does the company carry on its consolidated balance sheet?",
            "balance_sheet",  # balance sheet intent detected because of 'balance sheet'
            "Debt on balance sheet query"
        ),
        # 8. Unrelated debt note query without statement keyword (NEGATIVE TEST)
        (
            "What covenants apply to the outstanding senior notes under the indenture agreement?",
            None,
            "Debt covenant query (must NOT trigger cover_page)"
        ),
        # 9. Existing statement query: Cash flow
        (
            "What is the FY2018 capital expenditure amount for 3M in the cash flow statement?",
            "cash_flow_statement",
            "Cash flow statement query"
        ),
        # 10. Existing statement query: Balance sheet
        (
            "What is the net property, plant, and equipment on the balance sheet?",
            "balance_sheet",
            "Balance sheet query"
        ),
        # 11. Existing statement query: Income statement
        (
            "What was the net sales difference in the consolidated statement of income?",
            "income_statement",
            "Income statement query"
        )
    ]
    
    passed = 0
    failed = 0
    print("=" * 80)
    print("RUNNING INTENT ROUTING UNIT TESTS (EXPERIMENT B)")
    print("=" * 80)
    
    for query, expected, desc in test_cases:
        actual = reranker.detect_query_statement_intent(query)
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed += 1
            print(f"[{status}] {desc}")
            print(f"       Query:    \"{query[:70]}...\"")
            print(f"       Result:   {actual} (Expected: {expected})\n")
        else:
            failed += 1
            print(f"[{status}] {desc}")
            print(f"       Query:    \"{query}\"")
            print(f"       Result:   {actual} (Expected: {expected})\n")
            
    print("=" * 80)
    print(f"UNIT TEST SUMMARY: {passed}/{len(test_cases)} PASSED (Failed: {failed})")
    print("=" * 80)
    
    assert failed == 0, f"{failed} unit test(s) failed!"

if __name__ == "__main__":
    run_tests()
