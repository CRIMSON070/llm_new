import unittest
import re
from typing import Optional

# Canonical patterns for all 32 FinanceBench companies
FINANCEBENCH_COMPANIES = {
    "3M": [r"\b3m\b"],
    "ACTIVISIONBLIZZARD": [r"\bactivision(\s+blizzard)?\b"],
    "ADOBE": [r"\badobe\b"],
    "AES": [r"\baes(\s+corp(oration)?)?\b"],
    "AMAZON": [r"\bamazon\b"],
    "AMCOR": [r"\bamcor\b"],
    "AMD": [r"\bamd\b", r"\badvanced\s+micro\s+devices\b"],
    "AMERICANEXPRESS": [r"\bamerican\s+express\b", r"\bamex\b"],
    "AMERICANWATERWORKS": [r"\bamerican\s+water(\s+works)?\b"],
    "BESTBUY": [r"\bbest\s+buy\b"],
    "BLOCK": [r"\bblock\s*\(\s*formerly\b", r"\bblock\s+inc\.?\b", r"\bsquare\s+inc\.?\b", r"\bblock['’]s\b"],
    "BOEING": [r"\bboeing\b"],
    "COCACOLA": [r"\bcoca[\s-]*cola\b", r"\bcoke\b"],
    "CORNING": [r"\bcorning\b"],
    "COSTCO": [r"\bcostco\b"],
    "CVSHEALTH": [r"\bcvs(\s+health)?\b"],
    "FOOTLOCKER": [r"\bfoot\s+locker\b"],
    "GENERALMILLS": [r"\bgeneral\s+mills\b"],
    "JOHNSON": [r"\bjohnson\s+(&|and)\s+johnson\b", r"\bjnj\b"],
    "JPMORGAN": [r"\bjpmorgan(\s+chase)?\b", r"\bjp\s*morgan\b", r"\bjpm\b"],
    "KRAFTHEINZ": [r"\bkraft\s+heinz\b", r"\bkraft\b"],
    "LOCKHEEDMARTIN": [r"\blockheed(\s+martin)?\b"],
    "MGMRESORTS": [r"\bmgm(\s+resorts)?\b"],
    "MICROSOFT": [r"\bmicrosoft\b", r"\bmsft\b"],
    "NETFLIX": [r"\bnetflix\b"],
    "NIKE": [r"\bnike\b"],
    "PAYPAL": [r"\bpaypal\b"],
    "PEPSICO": [r"\bpepsico\b", r"\bpepsi\b"],
    "PFIZER": [r"\bpfizer\b"],
    "ULTABEAUTY": [r"\bulta(\s*beauty)?\b"],
    "VERIZON": [r"\bverizon\b"],
    "WALMART": [r"\bwalmart\b", r"\bwal-mart\b"],
}

def detect_query_company(query: str) -> Optional[str]:
    q_lower = query.lower()
    for canonical_name, patterns in FINANCEBENCH_COMPANIES.items():
        for p in patterns:
            if re.search(p, q_lower):
                return canonical_name
    return None


class TestCompanyDetection(unittest.TestCase):
    """Unit test suite for company intent detection across all 32 FinanceBench companies."""

    def test_positive_cases_all_32_companies(self):
        """Verify positive detection for every one of the 32 companies."""
        test_cases = [
            ("What is the FY2018 capex for 3M?", "3M"),
            ("What was Activision Blizzard's 3 year average capex?", "ACTIVISIONBLIZZARD"),
            ("Does Adobe have an improving Free cashflow conversion?", "ADOBE"),
            ("What is the quantity of restructuring costs for AES Corporation?", "AES"),
            ("What is Amazon's FY2017 days payable outstanding?", "AMAZON"),
            ("Has AMCOR's quick ratio improved or declined?", "AMCOR"),
            ("From FY21 to FY22, in which AMD reporting segment did sales increase?", "AMD"),
            ("Was American Express able to retain card members during 2022?", "AMERICANEXPRESS"),
            ("How much did American Water Works pay out in cash dividends?", "AMERICANWATERWORKS"),
            ("What was Best Buy's merchandise inventory in FY2023?", "BESTBUY"),
            ("What is Block's (formerly known as Square) working capital ratio?", "BLOCK"),
            ("What production rate changes is Boeing forecasting for FY2023?", "BOEING"),
            ("What is Coca-Cola's FY2021 COGS % margin?", "COCACOLA"),
            ("What is the FY2020 days payable outstanding for Corning?", "CORNING"),
            ("What was Costco's total revenue in FY2021?", "COSTCO"),
            ("What is the FY2018 fixed asset turnover ratio for CVS Health?", "CVSHEALTH"),
            ("What was Foot Locker's store count in FY2022?", "FOOTLOCKER"),
            ("What is the FY2019 cash conversion cycle for General Mills?", "GENERALMILLS"),
            ("Which business segment of JnJ will be treated as discontinued?", "JOHNSON"),
            ("Which of JPM's business segments had the lowest net revenue?", "JPMORGAN"),
            ("What is Kraft Heinz's FY2019 inventory turnover ratio?", "KRAFTHEINZ"),
            ("What is Lockheed Martin's FY2021 net working capital?", "LOCKHEEDMARTIN"),
            ("Which region had the worst topline performance for MGM Resorts?", "MGMRESORTS"),
            ("What is Microsoft's FY2023 cloud revenue?", "MICROSOFT"),
            ("What is Netflix's year end FY2017 total current liabilities?", "NETFLIX"),
            ("What is Nike's gross margin for FY2021?", "NIKE"),
            ("What was PayPal's transaction take rate in FY2022?", "PAYPAL"),
            ("What is the FY2022 unadjusted EBITDA % margin for PepsiCo?", "PEPSICO"),
            ("As of Q2'2023, is Pfizer spinning off any large business segments?", "PFIZER"),
            ("Did Ulta Beauty make any major acquisitions in FY 2022?", "ULTABEAUTY"),
            ("Is Verizon a capital intensive business based on FY 2022 data?", "VERIZON"),
            ("What is Walmart's FY2018 days payable outstanding?", "WALMART"),
        ]
        for query, expected_comp in test_cases:
            with self.subTest(query=query, expected=expected_comp):
                result = detect_query_company(query)
                self.assertEqual(result, expected_comp, f"Failed on query: '{query}'")

    def test_aliases_and_abbreviations(self):
        """Verify common financial aliases."""
        alias_cases = [
            ("Does AMEX have an improving operating margin profile?", "AMERICANEXPRESS"),
            ("What is JnJ's inventory turnover ratio?", "JOHNSON"),
            ("Johnson & Johnson reported earnings yesterday", "JOHNSON"),
            ("Johnson and Johnson consolidated balance sheet", "JOHNSON"),
            ("What did JPM report for net income in Q2?", "JPMORGAN"),
            ("JP Morgan Chase financial results", "JPMORGAN"),
            ("Pepsi revenue breakdown by geographic region", "PEPSICO"),
            ("Coke dividend distribution trend", "COCACOLA"),
            ("What did Ulta report for net sales?", "ULTABEAUTY"),
            ("MSFT intelligent cloud revenue growth", "MICROSOFT"),
            ("Lockheed revenue CAGR over 2 years", "LOCKHEEDMARTIN"),
            ("MGM interest coverage ratio using Adjusted EBIT", "MGMRESORTS"),
            ("CVS reported pharmacy revenue", "CVSHEALTH"),
            ("Kraft inventory turnover ratio", "KRAFTHEINZ"),
            ("Wal-Mart store operations", "WALMART"),
            ("Advanced Micro Devices segment revenue", "AMD"),
        ]
        for query, expected_comp in alias_cases:
            with self.subTest(query=query, expected=expected_comp):
                result = detect_query_company(query)
                self.assertEqual(result, expected_comp, f"Failed alias query: '{query}'")

    def test_avoid_ambiguous_aliases_negative_cases(self):
        """Verify that ambiguous acronyms like CAT, GM, or WM do not trigger false positives."""
        negative_cases = [
            # Word 'cat' should not match Caterpillar
            ("The cat jumped over the fence and ran into the house", None),
            ("We categorize our operating expenses into three buckets", None),
            # Word 'gm' should not match General Mills
            ("The plant manager said gm to the shifts in the morning", None),
            ("In general, mills in the rural area produce flour", None),
            # Word 'wm' or generic waste management
            ("Total waste management was mentioned as an ESG goal", None),
            # Generic word 'block'
            ("He put a block on the credit card account", None),
            ("These are the building blocks of financial modeling", None),
            # Generic financial queries with no company
            ("What is the definition of operating margin?", None),
            ("How is working capital ratio calculated on a balance sheet?", None),
            ("What are the covenants on the revolving credit facility?", None),
            ("What are the common stock voting rights?", None),
            ("Who are the board member nominees?", None),
        ]
        for query, expected_comp in negative_cases:
            with self.subTest(query=query, expected=expected_comp):
                result = detect_query_company(query)
                self.assertIsNone(result, f"False positive match '{result}' for negative query: '{query}'")

    def test_no_cross_company_collisions(self):
        """Verify overlapping names do not cross-collide."""
        overlap_cases = [
            ("American Express annual report", "AMERICANEXPRESS"),
            ("American Water Works 10-K filing", "AMERICANWATERWORKS"),
            ("Amcor packaging revenue", "AMCOR"),
            ("AMD processor market share", "AMD"),
        ]
        for query, expected_comp in overlap_cases:
            with self.subTest(query=query, expected=expected_comp):
                result = detect_query_company(query)
                self.assertEqual(result, expected_comp)


if __name__ == "__main__":
    unittest.main()
