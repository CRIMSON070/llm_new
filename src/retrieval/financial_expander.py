"""
Financial Query Expander & Statement Intent Analyzer for FinSage Phase 4.

Provides:
- Improvement 1: Comprehensive 32 FinanceBench + 20 Stage 3 company detection, ticker resolution, and filing matching.
- Improvement 2: Explicit fiscal year, quarter, and filing type (10-K, 10-Q, 8-K, EARNINGS) detection.
- Improvement 3: Unicode NFKC normalization and curated GAAP terminology expansion.
- Improvement 4: Multi-statement evidence retrieval intent and component decomposition.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from src.retrieval.coreference_resolver import CoreferenceResolver, ResolvedQuery


def normalize_unicode(text: str) -> str:
    """
    Applies Unicode NFKC normalization to resolve ligatures (e.g. \ufb01 -> fi, \ufb02 -> fl)
    and standardizes quotation marks and hyphens without altering numerical values, units, or signs.
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    # Standardize typographic quotes and hyphens while preserving symbols and math
    norm = norm.replace("\u2018", "'").replace("\u2019", "'")
    norm = norm.replace("\u201c", '"').replace("\u201d", '"')
    norm = norm.replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    # Standardize non-breaking and thin spaces
    norm = norm.replace("\u00a0", " ").replace("\u2009", " ")
    return norm


DOC_PARSE_REGEX = re.compile(r"^([A-Za-z0-9_]+?)_(\d{4})(?:Q([1-4]))?_(.+)$")


def parse_doc_metadata(doc_name: str) -> Optional[Dict[str, Any]]:
    """Extracts company identifier, year, quarter, and filing type from doc_name."""
    if not doc_name:
        return None
    m = DOC_PARSE_REGEX.match(doc_name)
    if m:
        comp, yr, qtr, ftype = m.groups()
        return {
            "company": comp.upper(),
            "year": int(yr),
            "quarter": int(qtr) if qtr else None,
            "filing_type": ftype.upper(),
            "doc_name": doc_name
        }
    return None


@dataclass
class FinancialExpansionResult:
    original_query: str
    detected_statement: Optional[str]
    statement_confidence: float
    detected_intent: str  # balance, flow, change, ratio, explanation
    target_concepts: List[str]
    expansion_terms: List[str]
    dense_query: str
    bm25_query: str
    trace_log: Dict[str, Any] = field(default_factory=dict)
    # Extracted Intelligence
    detected_company: Optional[str] = None
    detected_ticker: Optional[str] = None
    detected_metric: Optional[str] = None
    detected_filing_type: Optional[str] = None
    detected_periods: List[str] = field(default_factory=list)
    detected_quarters: List[int] = field(default_factory=list)
    detected_operation: str = "extract_single_value"
    targeted_period_queries: List[str] = field(default_factory=list)
    # Phase 4 Multi-Statement Intelligence
    is_multi_statement: bool = False
    multi_statement_type: Optional[str] = None
    multi_statement_components: List[Dict[str, str]] = field(default_factory=list)
    # Phase 4 Coreference Resolution Intelligence (Section 3.1 & Appendix A.6)
    is_coreference_resolved: bool = False
    coreference_term: Optional[str] = None
    coreference_type: str = "none"
    resolved_company_name: Optional[str] = None


class FinancialQueryExpander:
    """
    Expands financial user queries with official GAAP reporting line items
    and detects statement targeting, company, period, and multi-statement intent.
    """

    STATEMENT_PATTERNS = {
        "cash_flow_statement": [
            r"cash\s*flow\s*statements?",
            r"statements?\s*of\s*cash\s*flows?",
            r"cash\s*flows?\s*from\s*(operating|investing|financing)",
            r"investing\s*activities",
            r"operating\s*activities",
            r"financing\s*activities",
        ],
        "balance_sheet": [
            r"balance\s*sheets?",
            r"statements?\s*of\s*financial\s*position",
            r"consolidated\s*balance\s*sheets?",
        ],
        "income_statement": [
            r"statements?\s*of\s*income",
            r"income\s*statements?",
            r"statements?\s*of\s*operations?",
            r"statements?\s*of\s*earnings",
            r"\bp&l\b",
            r"profit\s*and\s*loss",
        ],
        "cover_page": [
            r"registered\s*to\s*trade\s*on\s*a\s*national\s*securities\s*exchange",
            r"title\s*of\s*each\s*class",
            r"trading\s*symbol",
            r"section\s*12\(b\)",
            r"section\s*12\(g\)",
            r"exchange\s*on\s*which\s*registered",
            r"registered\s*debt\s*securities",
        ],
    }

    INTENT_PATTERNS = {
        "change": [
            r"\bdifference\b",
            r"\bpercentage\s*change\b",
            r"\bchange\b",
            r"\bgrowth\b",
            r"\bincreased?\b",
            r"\bdecreased?\b",
            r"\byear-over-year\b",
            r"\byoy\b",
        ],
        "ratio": [
            r"\bratio\b",
            r"\bmargin\b",
            r"\b%\s*margin\b",
            r"\bturnover\b",
            r"\bconversion\b",
            r"\breturn\s*on\b",
            r"\bquick\s*ratio\b",
        ],
        "explanation": [
            r"\bwhat\s*drove\b",
            r"\bwhy\b",
            r"\bexplain\b",
            r"\bmajor\s*acquisitions\b",
            r"\bproducts?\s*and\s*services\b",
            r"\bwhich\s*debt\s*securities\b",
        ],
        "balance": [
            r"\bas\s*of\b",
            r"\bat\s*the\s*end\s*of\b",
            r"\byear\s*end\b",
            r"\bbalance\b",
            r"\boutstanding\b",
        ],
        "flow": [
            r"\bfor\s*the\s*(fiscal\s*year|quarter|period|year)\b",
            r"\bexpenditure\b",
            r"\bduring\b",
            r"\bspent\b",
        ],
    }

    # Canonical Company Taxonomy: All 32 FinanceBench + 20 Stage 3 Companies
    COMPANY_TAXONOMY = {
        # 32 FinanceBench Companies
        "3M": {
            "ticker": "MMM",
            "doc_prefixes": ["3M"],
            "patterns": [r"\b3m\b", r"\bmmm\b", r"\bminnesota\s+mining\b"],
        },
        "ACTIVISIONBLIZZARD": {
            "ticker": "ATVI",
            "doc_prefixes": ["ACTIVISIONBLIZZARD"],
            "patterns": [r"\bactivision(\s+blizzard)?\b", r"\batvi\b"],
        },
        "ADOBE": {
            "ticker": "ADBE",
            "doc_prefixes": ["ADOBE"],
            "patterns": [r"\badobe(\s+inc\.?|\s+systems)?\b", r"\badbe\b"],
        },
        "AES": {
            "ticker": "AES",
            "doc_prefixes": ["AES"],
            "patterns": [r"\baes(\s+corp(oration)?)?\b"],
        },
        "AMAZON": {
            "ticker": "AMZN",
            "doc_prefixes": ["AMAZON"],
            "patterns": [r"\bamazon(\.com)?\b", r"\bamzn\b"],
        },
        "AMCOR": {
            "ticker": "AMCR",
            "doc_prefixes": ["AMCOR"],
            "patterns": [r"\bamcor(\s+plc)?\b", r"\bamcr\b"],
        },
        "AMD": {
            "ticker": "AMD",
            "doc_prefixes": ["AMD"],
            "patterns": [r"\bamd\b", r"\badvanced\s+micro\s+devices\b"],
        },
        "AMERICANEXPRESS": {
            "ticker": "AXP",
            "doc_prefixes": ["AMERICANEXPRESS"],
            "patterns": [r"\bamerican\s+express\b", r"\bamex\b", r"\baxp\b"],
        },
        "AMERICANWATERWORKS": {
            "ticker": "AWK",
            "doc_prefixes": ["AMERICANWATERWORKS"],
            "patterns": [r"\bamerican\s+water(\s+works)?\b", r"\bawk\b"],
        },
        "BESTBUY": {
            "ticker": "BBY",
            "doc_prefixes": ["BESTBUY"],
            "patterns": [r"\bbest\s+buy\b", r"\bbby\b"],
        },
        "BLOCK": {
            "ticker": "SQ",
            "doc_prefixes": ["BLOCK"],
            "patterns": [r"\bblock\b", r"\bsquare\b", r"\bsq\b"],
        },
        "BOEING": {
            "ticker": "BA",
            "doc_prefixes": ["BOEING"],
            "patterns": [r"\bboeing\b", r"\bba\b"],
        },
        "COCACOLA": {
            "ticker": "KO",
            "doc_prefixes": ["COCACOLA"],
            "patterns": [r"\bcoca[\s-]*cola\b", r"\bcoke\b", r"\bko\b"],
        },
        "CORNING": {
            "ticker": "GLW",
            "doc_prefixes": ["CORNING"],
            "patterns": [r"\bcorning\b", r"\bglw\b"],
        },
        "COSTCO": {
            "ticker": "COST",
            "doc_prefixes": ["COSTCO"],
            "patterns": [r"\bcostco(\s+wholesale)?\b"],
        },
        "CVSHEALTH": {
            "ticker": "CVS",
            "doc_prefixes": ["CVSHEALTH"],
            "patterns": [r"\bcvs(\s+health|\s+caremark)?\b"],
        },
        "FOOTLOCKER": {
            "ticker": "FL",
            "doc_prefixes": ["FOOTLOCKER"],
            "patterns": [r"\bfoot\s+locker\b", r"\bfl\b"],
        },
        "GENERALMILLS": {
            "ticker": "GIS",
            "doc_prefixes": ["GENERALMILLS"],
            "patterns": [r"\bgeneral\s+mills\b", r"\bgis\b"],
        },
        "JOHNSON": {
            "ticker": "JNJ",
            "doc_prefixes": ["JOHNSON"],
            "patterns": [r"\bjohnson\s+(&|and)\s+johnson\b", r"\bj[&n]j\b"],
        },
        "JPMORGAN": {
            "ticker": "JPM",
            "doc_prefixes": ["JPMORGAN"],
            "patterns": [r"\bjpmorgan(\s+chase)?\b", r"\bjp\s*morgan\b", r"\bjpm\b"],
        },
        "KRAFTHEINZ": {
            "ticker": "KHC",
            "doc_prefixes": ["KRAFTHEINZ"],
            "patterns": [r"\bkraft\s+heinz\b", r"\bkraft\b", r"\bkhc\b"],
        },
        "LOCKHEEDMARTIN": {
            "ticker": "LMT",
            "doc_prefixes": ["LOCKHEEDMARTIN"],
            "patterns": [r"\blockheed(\s+martin)?\b", r"\blmt\b"],
        },
        "MGMRESORTS": {
            "ticker": "MGM",
            "doc_prefixes": ["MGMRESORTS"],
            "patterns": [r"\bmgm(\s+resorts)?(\s+international)?\b"],
        },
        "MICROSOFT": {
            "ticker": "MSFT",
            "doc_prefixes": ["MICROSOFT"],
            "patterns": [r"\bmicrosoft\b", r"\bmsft\b"],
        },
        "NETFLIX": {
            "ticker": "NFLX",
            "doc_prefixes": ["NETFLIX"],
            "patterns": [r"\bnetflix\b", r"\bnflx\b"],
        },
        "NIKE": {
            "ticker": "NKE",
            "doc_prefixes": ["NIKE"],
            "patterns": [r"\bnike\b", r"\bnke\b"],
        },
        "PAYPAL": {
            "ticker": "PYPL",
            "doc_prefixes": ["PAYPAL"],
            "patterns": [r"\bpaypal(\s+holdings)?\b", r"\bpypl\b"],
        },
        "PEPSICO": {
            "ticker": "PEP",
            "doc_prefixes": ["PEPSICO"],
            "patterns": [r"\bpepsico\b", r"\bpepsi\b", r"\bpep\b"],
        },
        "PFIZER": {
            "ticker": "PFE",
            "doc_prefixes": ["PFIZER", "Pfizer"],
            "patterns": [r"\bpfizer\b", r"\bpfe\b"],
        },
        "ULTABEAUTY": {
            "ticker": "ULTA",
            "doc_prefixes": ["ULTABEAUTY"],
            "patterns": [r"\bulta(\s*beauty)?\b"],
        },
        "VERIZON": {
            "ticker": "VZ",
            "doc_prefixes": ["VERIZON"],
            "patterns": [r"\bverizon(\s+communications)?\b", r"\bvz\b"],
        },
        "WALMART": {
            "ticker": "WMT",
            "doc_prefixes": ["WALMART"],
            "patterns": [r"\bwalmart\b", r"\bwal-mart\b", r"\bwmt\b"],
        },
        # 20 Stage 3 Held-Out Companies
        "APPLE": {
            "ticker": "AAPL",
            "doc_prefixes": ["AAPL", "APPLE"],
            "patterns": [r"\bapple\b", r"\baapl\b"],
        },
        "ALPHABET": {
            "ticker": "GOOGL",
            "doc_prefixes": ["GOOGL", "GOOG", "ALPHABET", "GOOGLE"],
            "patterns": [r"\bgoogle\b", r"\balphabet\b", r"\bgoogl?\b"],
        },
        "NVIDIA": {
            "ticker": "NVDA",
            "doc_prefixes": ["NVDA", "NVIDIA"],
            "patterns": [r"\bnvidia\b", r"\bnvda\b"],
        },
        "CISCO": {
            "ticker": "CSCO",
            "doc_prefixes": ["CSCO", "CISCO"],
            "patterns": [r"\bcisco(\s+systems)?\b", r"\bcsco\b"],
        },
        "BROADCOM": {
            "ticker": "AVGO",
            "doc_prefixes": ["AVGO", "BROADCOM"],
            "patterns": [r"\bbroadcom\b", r"\bavgo\b"],
        },
        "META": {
            "ticker": "META",
            "doc_prefixes": ["META"],
            "patterns": [r"\bmeta(\s+platforms)?\b", r"\bfacebook\b"],
        },
        "BANKOFAMERICA": {
            "ticker": "BAC",
            "doc_prefixes": ["BAC", "BANKOFAMERICA"],
            "patterns": [r"\bbank\s+of\s+america\b", r"\bbac\b", r"\bbofa\b"],
        },
        "GOLDMANSACHS": {
            "ticker": "GS",
            "doc_prefixes": ["GS", "GOLDMANSACHS"],
            "patterns": [r"\bgoldman\s+sachs\b", r"\bgs\b"],
        },
        "MORGANSTANLEY": {
            "ticker": "MS",
            "doc_prefixes": ["MS", "MORGANSTANLEY"],
            "patterns": [r"\bmorgan\s+stanley\b", r"\bms\b"],
        },
        "WELLSFARGO": {
            "ticker": "WFC",
            "doc_prefixes": ["WFC", "WELLSFARGO"],
            "patterns": [r"\bwells\s+fargo\b", r"\bwfc\b"],
        },
        "HOMEDEPOT": {
            "ticker": "HD",
            "doc_prefixes": ["HD", "HOMEDEPOT"],
            "patterns": [r"\bhome\s+depot\b", r"\bhd\b"],
        },
        "TARGET": {
            "ticker": "TGT",
            "doc_prefixes": ["TGT", "TARGET"],
            "patterns": [r"\btarget(\s+corp(oration)?)?\b", r"\btgt\b"],
        },
        "PROCTERGAMBLE": {
            "ticker": "PG",
            "doc_prefixes": ["PG", "PROCTERGAMBLE"],
            "patterns": [r"\bprocter\s+(&|and)\s+gamble\b", r"\bp&g\b", r"\bpg\b"],
        },
        "STARBUCKS": {
            "ticker": "SBUX",
            "doc_prefixes": ["SBUX", "STARBUCKS"],
            "patterns": [r"\bstarbucks\b", r"\bsbux\b"],
        },
        "ELILILLY": {
            "ticker": "LLY",
            "doc_prefixes": ["LLY", "ELILILLY"],
            "patterns": [r"\beli\s+lilly\b", r"\blilly\b", r"\blly\b"],
        },
        "UNITEDHEALTH": {
            "ticker": "UNH",
            "doc_prefixes": ["UNH", "UNITEDHEALTH"],
            "patterns": [r"\bunited\s*health(care)?\b", r"\bunh\b"],
        },
        "MERCK": {
            "ticker": "MRK",
            "doc_prefixes": ["MRK", "MERCK"],
            "patterns": [r"\bmerck\b", r"\bmrk\b"],
        },
        "ABBVIE": {
            "ticker": "ABBV",
            "doc_prefixes": ["ABBV", "ABBVIE"],
            "patterns": [r"\babbvie\b", r"\babbv\b"],
        },
        "EXXONMOBIL": {
            "ticker": "XOM",
            "doc_prefixes": ["XOM", "EXXONMOBIL"],
            "patterns": [r"\bexxon\s*mobil\b", r"\bexxon\b", r"\bxom\b"],
        },
        "CATERPILLAR": {
            "ticker": "CAT",
            "doc_prefixes": ["CAT", "CATERPILLAR"],
            "patterns": [r"\bcaterpillar\b", r"\bcat\b"],
        },
    }

    PREFIX_TO_CANONICAL: Dict[str, str] = {}
    for canon, meta in COMPANY_TAXONOMY.items():
        for pfx in meta["doc_prefixes"]:
            PREFIX_TO_CANONICAL[pfx.upper()] = canon
        if "ticker" in meta and meta["ticker"]:
            PREFIX_TO_CANONICAL[meta["ticker"].upper()] = canon

    # Multi-Statement Ratio Rules
    MULTI_STATEMENT_RULES = [
        {
            "id": "fixed_asset_turnover",
            "patterns": [r"\bfixed\s*asset\s*turnover\b"],
            "components": [
                {"statement": "income_statement", "cue": "revenue net sales turnover"},
                {"statement": "balance_sheet", "cue": "property plant and equipment net PP&E"}
            ]
        },
        {
            "id": "operating_cash_flow_ratio",
            "patterns": [r"\boperating\s*cash\s*flow\s*ratio\b"],
            "components": [
                {"statement": "cash_flow_statement", "cue": "cash flows from operating activities cash from operations"},
                {"statement": "balance_sheet", "cue": "current liabilities total current liabilities"}
            ]
        },
        {
            "id": "roa",
            "patterns": [r"\breturn\s*on\s*assets\b", r"\broa\b"],
            "components": [
                {"statement": "income_statement", "cue": "net income net earnings"},
                {"statement": "balance_sheet", "cue": "total assets"}
            ]
        },
        {
            "id": "roe",
            "patterns": [r"\breturn\s*on\s*(equity|stockholders['’]?\s*equity)\b", r"\broe\b"],
            "components": [
                {"statement": "income_statement", "cue": "net income net earnings"},
                {"statement": "balance_sheet", "cue": "stockholders equity shareholders equity"}
            ]
        },
        {
            "id": "dpo",
            "patterns": [r"\bdays\s*payable\s*outstanding\b", r"\bdpo\b"],
            "components": [
                {"statement": "balance_sheet", "cue": "accounts payable merchandise inventories"},
                {"statement": "income_statement", "cue": "cost of goods sold cost of sales cogs"}
            ]
        },
        {
            "id": "inventory_turnover",
            "patterns": [r"\binventory\s*turnover\b"],
            "components": [
                {"statement": "income_statement", "cue": "cost of goods sold cost of sales"},
                {"statement": "balance_sheet", "cue": "inventories merchandise inventories"}
            ]
        },
        {
            "id": "cash_conversion_cycle",
            "patterns": [r"\bcash\s*conversion\s*cycle\b", r"\bccc\b"],
            "components": [
                {"statement": "balance_sheet", "cue": "accounts receivable inventories accounts payable"},
                {"statement": "income_statement", "cue": "revenue sales cost of sales"}
            ]
        },
        {
            "id": "capex_to_sales",
            "patterns": [r"\bcapex\s*(to|\/)\s*sales\b", r"\bcapital\s*expenditures?\s*(to|\/)\s*(sales|revenue)\b"],
            "components": [
                {"statement": "cash_flow_statement", "cue": "capital expenditures purchases of property plant"},
                {"statement": "income_statement", "cue": "revenue net sales"}
            ]
        }
    ]

    # Concept Mappings: Official GAAP Terminology Expansions
    CONCEPT_TAXONOMY = [
        {
            "id": "capex_ppe_purchases",
            "triggers": [
                r"\bcapital\s*expenditures?\b",
                r"\bcapex\b",
                r"\bcapital\s*spending\b",
                r"\bcapital\s*additions\b",
            ],
            "expansions": [
                "Purchases of property, plant and equipment",
                "Purchases of property, plant and equipment (PP&E)",
                "Capital expenditures",
                "Additions to property, plant and equipment",
                "Investments in property, plant and equipment",
            ],
            "favored_statement": "cash_flow_statement",
        },
        {
            "id": "net_ppe",
            "triggers": [
                r"\bnet\s*ppne\b",
                r"\bnet\s*pp&e\b",
                r"\bnet\s*property[,\s]*plant\s*and\s*equipment\b",
                r"\bproperty[,\s]*plant\s*and\s*equipment[,\s]*net\b",
            ],
            "expansions": [
                "Property, plant and equipment, net",
                "Property, plant and equipment - net",
                "Property, plant and equipment less accumulated depreciation",
                "Land, buildings and equipment, net",
            ],
            "favored_statement": "balance_sheet",
        },
        {
            "id": "debt_securities_issued",
            "triggers": [
                r"\bdebt\s*securities\b",
                r"\bregistered\s*debt\b",
                r"\bsecurities\s*registered\b",
                r"\btrading\s*symbol\b",
            ],
            "expansions": [
                "Securities registered pursuant to Section 12(b) of the Act",
                "Title of each class",
                "Trading symbol(s)",
                "Name of each exchange on which registered",
            ],
            "favored_statement": "cover_page",
        },
        {
            "id": "revenue_topline",
            "triggers": [
                r"\bnet\s*sales\b",
                r"\btotal\s*revenue\b",
                r"\btotal\s*sales\b",
                r"\btopline\b",
                r"\brevenues?\b",
            ],
            "expansions": [
                "Total net sales",
                "Net revenues",
                "Total revenues",
                "Revenue from contracts with customers",
            ],
            "favored_statement": "income_statement",
        },
        {
            "id": "cogs_gross_profit",
            "triggers": [
                r"\bcogs\b",
                r"\bcost\s*of\s*goods\s*sold\b",
                r"\bcost\s*of\s*sales\b",
                r"\bgross\s*profit\b",
                r"\bgross\s*margin\b",
            ],
            "expansions": [
                "Cost of sales",
                "Cost of products sold",
                "Cost of goods sold",
                "Gross profit",
            ],
            "favored_statement": "income_statement",
        },
        {
            "id": "operating_income_ebit",
            "triggers": [
                r"\boperating\s*income\b",
                r"\boperating\s*profit\b",
                r"\bebit\b",
                r"\boperating\s*earnings\b",
            ],
            "expansions": [
                "Operating income",
                "Operating profit",
                "Operating earnings",
                "Income from operations",
            ],
            "favored_statement": "income_statement",
        },
        {
            "id": "net_income_earnings",
            "triggers": [
                r"\bnet\s*income\b",
                r"\bnet\s*earnings\b",
                r"\bbottom\s*line\b",
            ],
            "expansions": [
                "Net income",
                "Net earnings",
                "Net income attributable to common stockholders",
                "Net income attributable to the company",
            ],
            "favored_statement": "income_statement",
        },
        {
            "id": "merchandise_inventory",
            "triggers": [
                r"\bmerchandise\s*inventories\b",
                r"\btotal\s*inventories\b",
                r"\binventory\b",
                r"\binventories\b",
            ],
            "expansions": [
                "Merchandise inventories",
                "Merchandise inventories, net",
                "Inventories: Finished goods, Work in process, Raw materials",
                "Change in merchandise inventories",
            ],
            "favored_statement": "balance_sheet",
        },
        {
            "id": "total_debt_obligations",
            "triggers": [
                r"\btotal\s*debt\b",
                r"\blong-term\s*debt\b",
                r"\bshort-term\s*debt\b",
                r"\bcurrent\s*portion\s*of\s*long-term\s*debt\b",
            ],
            "expansions": [
                "Long-term debt",
                "Short-term borrowings",
                "Current portion of long-term debt",
                "Total debt",
            ],
            "favored_statement": "balance_sheet",
        },
        {
            "id": "working_capital_current_assets",
            "triggers": [
                r"\bworking\s*capital\b",
                r"\bcurrent\s*assets\b",
                r"\bcurrent\s*liabilities\b",
                r"\bworking\s*capital\s*ratio\b",
                r"\bcurrent\s*ratio\b",
            ],
            "expansions": [
                "Total current assets",
                "Total current liabilities",
                "Cash and cash equivalents",
                "Accounts receivable, net",
            ],
            "favored_statement": "balance_sheet",
        },
        {
            "id": "operating_cash_flow",
            "triggers": [
                r"\bcash\s*from\s*operations\b",
                r"\boperating\s*cash\s*flow\b",
                r"\bcash\s*flows?\s*from\s*operating\s*activities\b",
            ],
            "expansions": [
                "Net cash provided by operating activities",
                "Net cash provided by (used in) operating activities",
                "Cash flows from operating activities",
            ],
            "favored_statement": "cash_flow_statement",
        },
        {
            "id": "sga_expenses",
            "triggers": [
                r"\bsg&a\b",
                r"\bsga\b",
                r"\bselling,\s*general\s*and\s*administrative\b",
                r"\bselling\s*general\s*and\s*administrative\b",
            ],
            "expansions": [
                "Selling, general and administrative expenses",
                "Selling, general and administrative",
            ],
            "favored_statement": "income_statement",
        },
    ]

    METRIC_PATTERNS = [
        ("capital expenditures", [r"\bcapital\s*expenditures?\b", r"\bcapex\b", r"\bcapital\s*spending\b", r"\bpurchases\s*of\s*property[,\s]*plant\b"]),
        ("property, plant and equipment, net", [r"\bnet\s*pp&?e\b", r"\bnet\s*ppne\b", r"\bproperty[,\s]*plant\s*and\s*equipment[,\s]*net\b", r"\bnet\s*property[,\s]*plant[,\s]*and\s*equipment\b", r"\bproperty[,\s]*plant\s*and\s*equipment\b"]),
        ("debt securities", [r"\bdebt\s*securities\b", r"\bsecurities\s*registered\b"]),
        ("total debt", [r"\btotal\s*debt\b", r"\blong-term\s*debt\b", r"\bshort-term\s*debt\b", r"\bnotes\s*payable\b"]),
        ("revenue", [r"\brevenues?\b", r"\bnet\s*sales\b", r"\btotal\s*sales\b", r"\btotal\s*revenue\b"]),
        ("net income", [r"\bnet\s*income\b", r"\bnet\s*earnings\b", r"\bprofit\b"]),
        ("operating income", [r"\boperating\s*income\b", r"\boperating\s*profit\b", r"\bebit\b", r"\bebitda\b"]),
        ("free cash flow", [r"\bfree\s*cash\s*flow\b", r"\bfcf\b", r"\bcash\s*flow\s*conversion\b"]),
        ("gross profit", [r"\bgross\s*profit\b", r"\bgross\s*margin\b", r"\bcogs\b"]),
        ("inventories", [r"\bmerchandise\s*inventor(?:y|ies)\b", r"\binventor(?:y|ies)\b"]),
        ("days payable outstanding", [r"\bdays\s*payable\s*outstanding\b", r"\bdpo\b"]),
        ("cash conversion cycle", [r"\bcash\s*conversion\s*cycle\b", r"\bccc\b"]),
        ("working capital", [r"\bworking\s*capital\b"]),
        ("dividends", [r"\bcash\s*dividends?\b", r"\bdividends?\s*paid\b"]),
        ("restructuring costs", [r"\brestructuring\s*costs?\b", r"\brestructuring\s*charges?\b"]),
        ("store count", [r"\bstore\s*count\b", r"\bnumber\s*of\s*stores\b"]),
    ]

    def __init__(
        self,
        enable_statement_detection: bool = True,
        coref_resolver: Optional[CoreferenceResolver] = None,
        enable_coreference: bool = True,
    ):
        self.enable_statement_detection = enable_statement_detection
        self.enable_coreference = enable_coreference
        self.coref_resolver = (coref_resolver or CoreferenceResolver()) if enable_coreference else None

    def detect_company(self, query: str) -> Optional[str]:
        """Detect company name deterministically from query."""
        canon, _ = self.detect_company_and_ticker(query)
        return canon

    def detect_company_and_ticker(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect company name and ticker symbol deterministically from query.
        Returns (canonical_company_key, ticker_symbol) or (None, None).
        Orders matches by earliest position in the query to prioritize subject entities.
        Includes product/brand/subsidiary coreference resolution (Section 3.1 & Appendix A.6).
        """
        q_norm = normalize_unicode(query)
        q_lower = q_norm.lower()

        # 1. Direct Company Taxonomy Match
        matches = []
        for canon, meta in self.COMPANY_TAXONOMY.items():
            for p in meta["patterns"]:
                m = re.search(p, q_lower)
                if m:
                    matches.append((m.start(), canon, meta["ticker"]))
                    break

        if matches:
            matches.sort(key=lambda x: x[0])
            return matches[0][1], matches[0][2]

        # 2. Product Brand / Subsidiary / Drug Coreference Resolution
        if self.coref_resolver:
            coref_res = self.coref_resolver.resolve_query(q_norm)
            if coref_res.is_coreference_resolved and coref_res.canonical_company:
                return coref_res.canonical_company, coref_res.ticker

        # 3. General entity heuristics for unseen companies
        possessive_match = re.search(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)?)[’']s\b", q_norm)
        if possessive_match:
            cand = possessive_match.group(1).strip()
            if cand.lower() not in {"what", "how", "company", "which", "whose", "when", "where", "why", "there", "it"}:
                c_key = cand.upper().replace(" ", "")
                return c_key, c_key

        corp_match = re.search(r"(?:for|at|of|in|by)?\s*([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)?)\s+(?:Inc\.?|Corp\.?|Corporation|Co\.?|Ltd\.?|PLC)\b", q_norm)
        if corp_match:
            cand = corp_match.group(1).strip()
            if cand.lower() not in {"what", "how", "company", "which", "whose", "when", "where", "why", "the", "a", "an"}:
                c_key = cand.upper().replace(" ", "")
                return c_key, c_key

        return None, None

    def match_company(self, query_comp: Optional[str], doc_name: str) -> bool:
        """Determines if the detected company matches the filing doc_name."""
        if not query_comp or not doc_name:
            return False
        doc_pfx = doc_name.split("_")[0].upper()
        doc_canon = self.PREFIX_TO_CANONICAL.get(doc_pfx, doc_pfx)
        q_canon = query_comp.upper()
        return q_canon == doc_canon.upper() or q_canon == doc_pfx

    def detect_periods(self, query: str) -> List[str]:
        """Detect reporting periods/years deterministically from query without inventing."""
        q_norm = normalize_unicode(query)
        periods = set()

        for yr in re.findall(r"\b(199\d|20[0-2]\d|2030)\b", q_norm):
            periods.add(yr)

        for fy in re.findall(r"\bFY\s*['’]?(\d{2,4})\b", q_norm, re.IGNORECASE):
            if len(fy) == 2:
                fy_full = f"20{fy}" if int(fy) < 50 else f"19{fy}"
                periods.add(fy_full)
            else:
                periods.add(fy)

        for fy in re.findall(r"\bfiscal\s*(?:year\s*)?(\d{2,4})\b", q_norm, re.IGNORECASE):
            if len(fy) == 2:
                periods.add(f"20{fy}" if int(fy) < 50 else f"19{fy}")
            else:
                periods.add(fy)

        return sorted(list(periods), key=lambda x: int(x) if x.isdigit() else x)

    def detect_quarters(self, query: str) -> List[int]:
        """Detect explicit quarter specifications (Q1, Q2, Q3, Q4)."""
        q_norm = normalize_unicode(query)
        quarters = set()
        for q in re.findall(r"\bQ([1-4])\b", q_norm, re.IGNORECASE):
            quarters.add(int(q))
        return sorted(list(quarters))

    def detect_filing_type(self, query: str) -> Optional[str]:
        """Detect filing type if explicitly specified in query."""
        q_lower = normalize_unicode(query).lower()
        if re.search(r"\b10-?k\b|annual\s*report", q_lower):
            return "10-K"
        if re.search(r"\b10-?q\b|quarterly\s*report", q_lower):
            return "10-Q"
        if re.search(r"\b8-?k\b|current\s*report", q_lower):
            return "8-K"
        if re.search(r"\bearnings\b", q_lower):
            return "EARNINGS"
        return None

    def detect_multi_statement(self, query: str) -> Tuple[bool, Optional[str], List[Dict[str, str]]]:
        """
        Detects if query requires evidence from multiple financial statements.
        Returns: (is_multi_statement, rule_id, components_list)
        """
        q_lower = normalize_unicode(query).lower()

        # 1. Check known multi-statement ratio definitions
        for rule in self.MULTI_STATEMENT_RULES:
            for p in rule["patterns"]:
                if re.search(p, q_lower):
                    return True, rule["id"], rule["components"]

        # 2. Check explicit mentions of multiple statements in query text
        stmts_found = []
        if any(re.search(p, q_lower) for p in [r"\bbalance\s*sheets?\b", r"\bstatements?\s*of\s*financial\s*position\b"]):
            stmts_found.append(("balance_sheet", "balance sheet financial position"))
        if any(re.search(p, q_lower) for p in [r"\bcash\s*flows?\s*statements?\b", r"\bstatements?\s*of\s*cash\s*flows?\b"]):
            stmts_found.append(("cash_flow_statement", "cash flows operating activities"))
        if any(re.search(p, q_lower) for p in [r"\bincome\s*statements?\b", r"\bstatements?\s*of\s*operations?\b", r"\bstatements?\s*of\s*earnings\b", r"\bp&l\b"]):
            stmts_found.append(("income_statement", "income statement operations revenues"))

        if len(stmts_found) >= 2:
            return True, "explicit_multi_statement", [{"statement": s[0], "cue": s[1]} for s in stmts_found]

        return False, None, []

    def detect_metric(self, query: str) -> Optional[str]:
        """Detect requested financial metric deterministically."""
        q_lower = normalize_unicode(query).lower()
        for metric_name, patterns in self.METRIC_PATTERNS:
            for p in patterns:
                if re.search(p, q_lower):
                    return metric_name
        return None

    def detect_operation(self, query: str, periods: List[str]) -> str:
        """Detect requested financial operation."""
        q_lower = normalize_unicode(query).lower()
        if any(re.search(p, q_lower) for p in [r"\bgrowth\b", r"\bpercentage\s*change\b", r"\b%\s*change\b", r"\byoy\b"]):
            return "percentage_change"
        if len(periods) >= 2 and any(w in q_lower for w in ["change", "growth", "increase", "decrease", "compare", "between", "from"]):
            return "percentage_change"
        if any(re.search(p, q_lower) for p in [r"\bdifference\b", r"\bvariance\b", r"\bhow\s*much\s*did.*?increase\b"]):
            return "difference"
        if any(re.search(p, q_lower) for p in [r"\bratio\b", r"\bmargin\b", r"\bturnover\b", r"\bconversion\b", r"\breturn\s*on\b"]):
            return "ratio"
        if any(re.search(p, q_lower) for p in [r"\bwhy\b", r"\bwhat\s*drove\b", r"\bexplain\b", r"\bdescribe\b", r"\bwhich\b"]):
            return "explanation"
        return "extract_single_value"

    def generate_targeted_period_queries(
        self,
        query: str,
        company: Optional[str],
        metric: Optional[str],
        periods: List[str]
    ) -> List[str]:
        """Generate targeted search queries for specific reporting periods."""
        targeted = []
        comp_display = company if company else ""
        metric_display = metric if metric else ""

        if len(periods) >= 2:
            for p in periods:
                tokens = [t for t in [comp_display, metric_display, p] if t]
                if tokens:
                    targeted.append(" ".join(tokens))
        elif len(periods) == 1 and (comp_display or metric_display):
            tokens = [t for t in [comp_display, metric_display, periods[0]] if t]
            if len(tokens) >= 2:
                targeted.append(" ".join(tokens))

        return targeted

    def detect_statement(self, query: str) -> Tuple[Optional[str], float]:
        """Detect the requested financial statement type with confidence."""
        q_lower = normalize_unicode(query).lower()
        for stmt, patterns in self.STATEMENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, q_lower):
                    return stmt, 1.0

        for item in self.CONCEPT_TAXONOMY:
            for trig in item["triggers"]:
                if re.search(trig, q_lower):
                    fav = item.get("favored_statement")
                    if fav:
                        return fav, 0.6

        return None, 0.0

    def detect_intent(self, query: str) -> str:
        """Detect the question's operational and information intent."""
        q_lower = normalize_unicode(query).lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, q_lower):
                    return intent
        return "flow"

    def expand(self, query: str) -> FinancialExpansionResult:
        """
        Analyze financial query and generate structured expansion result with Phase 4 intelligence.
        """
        norm_query = normalize_unicode(query)
        q_lower = norm_query.lower()
        detected_stmt, stmt_conf = self.detect_statement(norm_query)
        detected_intent = self.detect_intent(norm_query)

        # Intelligence Extraction
        detected_company, detected_ticker = self.detect_company_and_ticker(norm_query)
        detected_periods = self.detect_periods(norm_query)
        detected_quarters = self.detect_quarters(norm_query)
        detected_metric = self.detect_metric(norm_query)
        detected_filing = self.detect_filing_type(norm_query)
        detected_operation = self.detect_operation(norm_query, detected_periods)
        targeted_queries = self.generate_targeted_period_queries(
            query=norm_query,
            company=detected_company,
            metric=detected_metric,
            periods=detected_periods
        )

        # Multi-Statement Intent Detection
        is_multi, multi_type, multi_components = self.detect_multi_statement(norm_query)

        target_concepts = []
        expansion_terms = []
        trace_log = {
            "query": query,
            "norm_query": norm_query,
            "matched_concepts": [],
            "statement_cues": [],
            "detected_company": detected_company,
            "detected_ticker": detected_ticker,
            "detected_periods": detected_periods,
            "detected_quarters": detected_quarters,
            "detected_metric": detected_metric,
            "detected_filing": detected_filing,
            "detected_operation": detected_operation,
            "targeted_queries": targeted_queries,
            "is_multi_statement": is_multi,
            "multi_statement_type": multi_type,
            "multi_statement_components": multi_components
        }

        # Analyze matching concepts
        for item in self.CONCEPT_TAXONOMY:
            matched = False
            for trig in item["triggers"]:
                if re.search(trig, q_lower):
                    matched = True
                    trace_log["matched_concepts"].append(item["id"])
                    break
            if matched:
                target_concepts.append(item["id"])
                for term in item["expansions"]:
                    if term not in expansion_terms:
                        expansion_terms.append(term)

        # Coreference Resolution Check (Section 3.1 & Appendix A.6)
        coref_res = self.coref_resolver.resolve_query(norm_query) if self.coref_resolver else None
        is_coref = bool(coref_res and coref_res.is_coreference_resolved)
        coref_term = coref_res.coreference_term if is_coref else None
        coref_type = coref_res.coreference_type if is_coref else "none"
        resolved_cname = coref_res.company_name if is_coref else None

        if is_coref and not detected_company and coref_res.canonical_company:
            detected_company = coref_res.canonical_company
            detected_ticker = coref_res.ticker

        # Build augmented queries preserving original query
        dense_expansion_str = " | ".join(expansion_terms[:3]) if expansion_terms else ""
        dense_query = f"{norm_query} [Official Terms: {dense_expansion_str}]" if dense_expansion_str else norm_query

        # BM25 expansion query contains distinct keywords
        bm25_keywords = set()
        for term in expansion_terms:
            for word in re.findall(r'[A-Za-z0-9&]+', term):
                if len(word) > 2 and word.lower() not in {"and", "the", "for", "with", "from"}:
                    bm25_keywords.add(word)

        if is_coref and detected_ticker and detected_ticker.lower() not in norm_query.lower():
            bm25_keywords.add(detected_ticker)

        bm25_extra = " ".join(list(bm25_keywords)[:10])
        bm25_query = f"{norm_query} {bm25_extra}".strip() if bm25_extra else norm_query

        trace_log["detected_statement"] = detected_stmt
        trace_log["statement_confidence"] = stmt_conf
        trace_log["detected_intent"] = detected_intent
        trace_log["expansion_count"] = len(expansion_terms)
        trace_log["is_coreference_resolved"] = is_coref
        trace_log["coreference_term"] = coref_term
        trace_log["coreference_type"] = coref_type
        trace_log["resolved_company_name"] = resolved_cname

        return FinancialExpansionResult(
            original_query=query,
            detected_statement=detected_stmt,
            statement_confidence=stmt_conf,
            detected_intent=detected_intent,
            target_concepts=target_concepts,
            expansion_terms=expansion_terms,
            dense_query=dense_query,
            bm25_query=bm25_query,
            trace_log=trace_log,
            detected_company=detected_company,
            detected_ticker=detected_ticker,
            detected_metric=detected_metric,
            detected_filing_type=detected_filing,
            detected_periods=detected_periods,
            detected_quarters=detected_quarters,
            detected_operation=detected_operation,
            targeted_period_queries=targeted_queries,
            is_multi_statement=is_multi,
            multi_statement_type=multi_type,
            multi_statement_components=multi_components,
            is_coreference_resolved=is_coref,
            coreference_term=coref_term,
            coreference_type=coref_type,
            resolved_company_name=resolved_cname
        )
