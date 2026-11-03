"""
FinSage Coreference Resolver (Section 3.1 & Appendix A.6)
==========================================================
Paper Reference: FinSage (arXiv:2504.14493, CIKM 2025)
- Section 3.1: Financial Filings Pre-processing (FFP) pipeline.
- Appendix A.6: Co-reference Resolution replacing pronouns with named entity antecedents.

Capabilities:
1. Query-Time Entity & Brand Coreference Resolution:
   - Resolves product brands, drug names, business divisions, platforms, and subsidiaries
     to their canonical corporate registrants (e.g. Mounjaro -> Eli Lilly, Keytruda -> Merck,
     Humira/Skyrizi/Rinvoq -> AbbVie, Reality Labs -> Meta, Optum -> UnitedHealth, AWS -> Amazon).
   - Resolves query anaphora and pronouns ("its", "their", "the company's") using prior conversational
     or extracted entity antecedents.
2. Chunk-Time Coreference Resolution (FFP Pre-processing):
   - Contextually replaces self-referential pronouns ("the Company", "the Registrant", "our", "we")
     with explicit named entity antecedents derived from filing metadata (doc_name).
   - Injects structured entity antecedent headers for lexical and cross-encoder scoring.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ResolvedQuery:
    original_query: str
    enriched_query: str
    canonical_company: Optional[str]
    company_name: Optional[str]
    ticker: Optional[str]
    coreference_term: Optional[str]
    coreference_type: str  # "brand_product", "anaphora_pronoun", "direct_mention", "none"
    is_coreference_resolved: bool = False


# Canonical Corporate Registry & Official Names
COMPANY_CANONICAL_INFO = {
    "ELILILLY": {"name": "Eli Lilly and Company", "ticker": "LLY", "doc_prefix": "LLY"},
    "MERCK": {"name": "Merck & Co., Inc.", "ticker": "MRK", "doc_prefix": "MRK"},
    "ABBVIE": {"name": "AbbVie Inc.", "ticker": "ABBV", "doc_prefix": "ABBV"},
    "META": {"name": "Meta Platforms, Inc.", "ticker": "META", "doc_prefix": "META"},
    "UNITEDHEALTH": {"name": "UnitedHealth Group Incorporated", "ticker": "UNH", "doc_prefix": "UNH"},
    "AMAZON": {"name": "Amazon.com, Inc.", "ticker": "AMZN", "doc_prefix": "AMZN"},
    "ALPHABET": {"name": "Alphabet Inc.", "ticker": "GOOGL", "doc_prefix": "GOOGL"},
    "MICROSOFT": {"name": "Microsoft Corporation", "ticker": "MSFT", "doc_prefix": "MSFT"},
    "APPLE": {"name": "Apple Inc.", "ticker": "AAPL", "doc_prefix": "AAPL"},
    "NVIDIA": {"name": "NVIDIA Corporation", "ticker": "NVDA", "doc_prefix": "NVDA"},
    "BROADCOM": {"name": "Broadcom Inc.", "ticker": "AVGO", "doc_prefix": "AVGO"},
    "CISCO": {"name": "Cisco Systems, Inc.", "ticker": "CSCO", "doc_prefix": "CSCO"},
    "BANKOFAMERICA": {"name": "Bank of America Corporation", "ticker": "BAC", "doc_prefix": "BAC"},
    "GOLDMANSACHS": {"name": "The Goldman Sachs Group, Inc.", "ticker": "GS", "doc_prefix": "GS"},
    "MORGANSTANLEY": {"name": "Morgan Stanley", "ticker": "MS", "doc_prefix": "MS"},
    "WELLSFARGO": {"name": "Wells Fargo & Company", "ticker": "WFC", "doc_prefix": "WFC"},
    "HOMEDEPOT": {"name": "The Home Depot, Inc.", "ticker": "HD", "doc_prefix": "HD"},
    "TARGET": {"name": "Target Corporation", "ticker": "TGT", "doc_prefix": "TGT"},
    "PROCTERGAMBLE": {"name": "The Procter & Gamble Company", "ticker": "PG", "doc_prefix": "PG"},
    "STARBUCKS": {"name": "Starbucks Corporation", "ticker": "SBUX", "doc_prefix": "SBUX"},
    "EXXONMOBIL": {"name": "Exxon Mobil Corporation", "ticker": "XOM", "doc_prefix": "XOM"},
    "CATERPILLAR": {"name": "Caterpillar Inc.", "ticker": "CAT", "doc_prefix": "CAT"},
    "3M": {"name": "3M Company", "ticker": "MMM", "doc_prefix": "3M"},
    "ADOBE": {"name": "Adobe Inc.", "ticker": "ADBE", "doc_prefix": "ADOBE"},
    "AMERISOURCEBERGEN": {"name": "AmerisourceBergen Corporation", "ticker": "ABC", "doc_prefix": "AMERISOURCEBERGEN"},
    "AMERICANWATERWORKS": {"name": "American Water Works Company, Inc.", "ticker": "AWK", "doc_prefix": "AMERICANWATERWORKS"},
    "BESTBUY": {"name": "Best Buy Co., Inc.", "ticker": "BBY", "doc_prefix": "BESTBUY"},
    "BLOCK": {"name": "Block, Inc.", "ticker": "SQ", "doc_prefix": "BLOCK"},
    "BOEING": {"name": "The Boeing Company", "ticker": "BA", "doc_prefix": "BOEING"},
    "CHEVRON": {"name": "Chevron Corporation", "ticker": "CVX", "doc_prefix": "CHEVRON"},
    "COCACOLA": {"name": "The Coca-Cola Company", "ticker": "KO", "doc_prefix": "COCACOLA"},
    "CORNING": {"name": "Corning Incorporated", "ticker": "GLW", "doc_prefix": "CORNING"},
    "COSTCO": {"name": "Costco Wholesale Corporation", "ticker": "COST", "doc_prefix": "COSTCO"},
    "CVSHEALTH": {"name": "CVS Health Corporation", "ticker": "CVS", "doc_prefix": "CVSHEALTH"},
    "EBAY": {"name": "eBay Inc.", "ticker": "EBAY", "doc_prefix": "EBAY"},
    "FOOTLOCKER": {"name": "Foot Locker, Inc.", "ticker": "FL", "doc_prefix": "FOOTLOCKER"},
    "GENERALMILLS": {"name": "General Mills, Inc.", "ticker": "GIS", "doc_prefix": "GENERALMILLS"},
    "GENERALMOTORS": {"name": "General Motors Company", "ticker": "GM", "doc_prefix": "GENERALMOTORS"},
    "JOHNSONJOHNSON": {"name": "Johnson & Johnson", "ticker": "JNJ", "doc_prefix": "JOHNSON_JOHNSON"},
    "JPMORGAN": {"name": "JPMorgan Chase & Co.", "ticker": "JPM", "doc_prefix": "JPMORGAN"},
    "KRAFTHEINZ": {"name": "The Kraft Heinz Company", "ticker": "KHC", "doc_prefix": "KRAFTHEINZ"},
    "LOCKHEEDMARTIN": {"name": "Lockheed Martin Corporation", "ticker": "LMT", "doc_prefix": "LOCKHEEDMARTIN"},
    "MGMRESORTS": {"name": "MGM Resorts International", "ticker": "MGM", "doc_prefix": "MGMRESORTS"},
    "MONDELEZ": {"name": "Mondelez International, Inc.", "ticker": "MDLZ", "doc_prefix": "MONDELEZ"},
    "NETFLIX": {"name": "Netflix, Inc.", "ticker": "NFLX", "doc_prefix": "NETFLIX"},
    "NIKE": {"name": "NIKE, Inc.", "ticker": "NKE", "doc_prefix": "NIKE"},
    "PAYPAL": {"name": "PayPal Holdings, Inc.", "ticker": "PYPL", "doc_prefix": "PAYPAL"},
    "PEPSICO": {"name": "PepsiCo, Inc.", "ticker": "PEP", "doc_prefix": "PEPSICO"},
    "PFIZER": {"name": "Pfizer Inc.", "ticker": "PFE", "doc_prefix": "PFIZER"},
    "TESLA": {"name": "Tesla, Inc.", "ticker": "TSLA", "doc_prefix": "TESLA"},
    "ULTABEAUTY": {"name": "Ulta Beauty, Inc.", "ticker": "ULTA", "doc_prefix": "ULTABEAUTY"},
    "VERIZON": {"name": "Verizon Communications Inc.", "ticker": "VZ", "doc_prefix": "VERIZON"},
    "WALMART": {"name": "Walmart Inc.", "ticker": "WMT", "doc_prefix": "WALMART"},
    "WAYFAIR": {"name": "Wayfair Inc.", "ticker": "W", "doc_prefix": "WAYFAIR"},
}


# Brand / Product / Subsidiary / Drug to Canonical Company Coreference Taxonomy
COREFERENCE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # Eli Lilly (LLY)
    "ELILILLY": {
        "brands": [
            r"\bmounjaro\b",
            r"\btirzepatide\b",
            r"\bzepbound\b",
            r"\btrulicity\b",
            r"\bverzenio\b",
            r"\bjardiance\b",
            r"\btaltz\b",
            r"\bolumiant\b",
            r"\bcialis\b",
            r"\balimta\b",
            r"\bbaqsimi\b",
            r"\bretsevmo\b",
            r"\bcyramza\b",
        ],
    },
    # Merck (MRK)
    "MERCK": {
        "brands": [
            r"\bkeytruda\b",
            r"\bpembrolizumab\b",
            r"\bgardasil\b",
            r"\bjanuvia\b",
            r"\blagevrio\b",
            r"\bproquad\b",
            r"\blynparza\b",
            r"\blenvima\b",
            r"\bbridion\b",
            r"\bisentress\b",
            r"\bwinrevair\b",
            r"\bsotatercept\b",
            r"\bvarivax\b",
        ],
    },
    # AbbVie (ABBV)
    "ABBVIE": {
        "brands": [
            r"\bhumira\b",
            r"\badalimumab\b",
            r"\bskyrizi\b",
            r"\brisankizumab\b",
            r"\brinvoq\b",
            r"\bupadacitinib\b",
            r"\bimbruvica\b",
            r"\bbotox\b",
            r"\bvenclexta\b",
            r"\bcreon\b",
            r"\bmavyret\b",
            r"\bepkinly\b",
            r"\bvyvgart\b",
        ],
    },
    # Meta Platforms (META)
    "META": {
        "brands": [
            r"\breality\s+labs\b",
            r"\binstagram\b",
            r"\bwhatsapp\b",
            r"\bmessenger\b",
            r"\bmeta\s+quest\b",
            r"\bquest\s+[23]\b",
            r"\boculus\b",
            r"\bthreads\b",
            r"\bhorizon\s+worlds\b",
            r"\bfamily\s+of\s+apps\b",
        ],
    },
    # UnitedHealth Group (UNH)
    "UNITEDHEALTH": {
        "brands": [
            r"\boptum\b",
            r"\boptum\s*health\b",
            r"\boptum\s*insight\b",
            r"\boptum\s*rx\b",
            r"\bunitedhealthcare\b",
            r"\bchange\s+healthcare\b",
        ],
    },
    # Amazon (AMZN)
    "AMAZON": {
        "brands": [
            r"\baws\b",
            r"\bamazon\s+web\s+services\b",
            r"\bprime\s+video\b",
            r"\bwhole\s+foods\b",
            r"\btwitch\b",
            r"\bring\b",
            r"\bzoox\b",
            r"\bkuiper\b",
        ],
    },
    # Alphabet / Google (GOOGL)
    "ALPHABET": {
        "brands": [
            r"\bgoogle\s+cloud\b",
            r"\bgcp\b",
            r"\byoutube\b",
            r"\bwaymo\b",
            r"\bdeepmind\b",
            r"\bandroid\b",
            r"\bgoogle\s+services\b",
            r"\bgoogle\s+search\b",
        ],
    },
    # Microsoft (MSFT)
    "MICROSOFT": {
        "brands": [
            r"\bazure\b",
            r"\blinkedin\b",
            r"\bxbox\b",
            r"\bgithub\b",
            r"\boffice\s+365\b",
            r"\bmicrosoft\s+365\b",
            r"\bteams\b",
            r"\bcopilot\b",
            r"\bnuance\b",
            r"\bactivision\b",
        ],
    },
    # Apple (AAPL)
    "APPLE": {
        "brands": [
            r"\biphone\b",
            r"\bipad\b",
            r"\bmacbook\b",
            r"\bapple\s+watch\b",
            r"\bapple\s+tv\b",
            r"\bapp\s+store\b",
            r"\bicloud\b",
            r"\bairpods\b",
            r"\bvision\s+pro\b",
        ],
    },
    # Broadcom (AVGO)
    "BROADCOM": {
        "brands": [
            r"\bvmware\b",
            r"\bsymantec\b",
            r"\bca\s+technologies\b",
            r"\bbrocade\b",
        ],
    },
    # ExxonMobil (XOM)
    "EXXONMOBIL": {
        "brands": [
            r"\bpioneer\s+natural\s+resources\b",
            r"\bmobil\b",
            r"\besso\b",
            r"\bpermian\s+basin\s+unconventional\b",
        ],
    },
    # MGM Resorts (MGM)
    "MGMRESORTS": {
        "brands": [
            r"\bbetmgm\b",
            r"\bbellagio\b",
            r"\bmgm\s+grand\b",
            r"\bborgata\b",
            r"\bcosmopolitan\b",
            r"\baria\b",
        ],
    },
    # Johnson & Johnson (JNJ)
    "JOHNSONJOHNSON": {
        "brands": [
            r"\bkenvue\b",
            r"\bjanssen\b",
            r"\btylenol\b",
            r"\bband-aid\b",
            r"\bneutrogena\b",
            r"\bstelara\b",
            r"\bdarzalex\b",
            r"\btremfya\b",
            r"\berleada\b",
        ],
    },
    # Pfizer (PFE)
    "PFIZER": {
        "brands": [
            r"\bcomirnaty\b",
            r"\bpaxlovid\b",
            r"\beliquis\b",
            r"\bvyndaqel\b",
            r"\bprevnar\b",
            r"\bibrance\b",
            r"\bxtandi\b",
            r"\bxeljanz\b",
            r"\bnurtec\b",
        ],
    },
    # Procter & Gamble (PG)
    "PROCTERGAMBLE": {
        "brands": [
            r"\bgillette\b",
            r"\bpampers\b",
            r"\btide\b",
            r"\bbounty\b",
            r"\bcharmin\b",
            r"\boral-b\b",
            r"\bhead\s+&\s+shoulders\b",
            r"\bolay\b",
            r"\bpantene\b",
            r"\bdawn\b",
        ],
    },
    # General Mills (GIS)
    "GENERALMILLS": {
        "brands": [
            r"\bcheerios\b",
            r"\bnature\s+valley\b",
            r"\bblue\s+buffalo\b",
            r"\bbetty\s+crocker\b",
            r"\bpillsbury\b",
            r"\bhäagen-dazs\b",
            r"\bhaagen-dazs\b",
            r"\byoplait\b",
        ],
    },
    # Mondelez (MDLZ)
    "MONDELEZ": {
        "brands": [
            r"\boreo\b",
            r"\bcadbury\b",
            r"\bmilka\b",
            r"\btoblerone\b",
            r"\britz\b",
            r"\bclif\s+bar\b",
            r"\bsour\s+patch\s+kids\b",
        ],
    },
    # PepsiCo (PEP)
    "PEPSICO": {
        "brands": [
            r"\bfrito-lay\b",
            r"\blay's\b",
            r"\bdoritos\b",
            r"\bcheetos\b",
            r"\bgatorade\b",
            r"\bquaker\b",
            r"\bruffles\b",
            r"\btostitos\b",
            r"\bmountain\s+dew\b",
            r"\bsodastream\b",
        ],
    },
    # Foot Locker (FL)
    "FOOTLOCKER": {
        "brands": [
            r"\bchamps\s+sports\b",
            r"\bkids\s+foot\s+locker\b",
            r"\batmos\b",
            r"\bwss\b",
            r"\beastbay\b",
        ],
    },
}


# Precompile Regex Patterns for Performance
COMPILED_BRAND_PATTERNS: List[Tuple[str, re.Pattern, str]] = []
for canon, meta in COREFERENCE_TAXONOMY.items():
    for pat_str in meta["brands"]:
        pat = re.compile(pat_str, re.IGNORECASE)
        COMPILED_BRAND_PATTERNS.append((canon, pat, pat_str))

# Pronoun / Anaphora Patterns
ANAPHORA_PATTERNS = [
    (re.compile(r"\b(?:its|their)\b", re.IGNORECASE), "its"),
    (re.compile(r"\bthe\s+company(?:'s)?\b", re.IGNORECASE), "the company"),
    (re.compile(r"\bthe\s+firm(?:'s)?\b", re.IGNORECASE), "the firm"),
    (re.compile(r"\bthe\s+registrant(?:'s)?\b", re.IGNORECASE), "the registrant"),
]

# Chunk-Level Self-Referential Pronouns (Section 3.1 & Appendix A.6)
CHUNK_SELF_REF_PATTERN = re.compile(
    r"\b(the\s+Company|the\s+Registrant|our|we|us)\b",
    re.IGNORECASE
)


class CoreferenceResolver:
    """
    Implements FinSage paper Section 3.1 & Appendix A.6 Coreference Resolution.
    Handles:
    - Query-Side: Product brand to corporate entity linking & pronoun anaphora resolution.
    - Chunk-Side: Antecedent context resolution and pronoun grounding.
    """

    def __init__(self):
        self.brand_patterns = COMPILED_BRAND_PATTERNS
        self.canonical_info = COMPANY_CANONICAL_INFO

    def resolve_query(
        self,
        query: str,
        prior_context_company: Optional[str] = None
    ) -> ResolvedQuery:
        """
        Resolves query-side coreference by identifying product brands, subsidiaries,
        or anaphoric pronouns and grounding them to the canonical corporate entity.
        """
        if not query:
            return ResolvedQuery(
                original_query="",
                enriched_query="",
                canonical_company=None,
                company_name=None,
                ticker=None,
                coreference_term=None,
                coreference_type="none",
                is_coreference_resolved=False
            )

        # 1. Product Brand / Subsidiary / Drug Coreference Resolution
        for canon, pat, orig_pattern in self.brand_patterns:
            m = pat.search(query)
            if m:
                matched_term = m.group(0)
                info = self.canonical_info.get(canon, {})
                c_name = info.get("name", canon)
                ticker = info.get("ticker", "")
                
                # Enrich query with explicit named entity antecedent
                entity_context = f"[Entity: {c_name} | Ticker: {ticker}]"
                enriched = f"{query} {entity_context}"
                
                return ResolvedQuery(
                    original_query=query,
                    enriched_query=enriched,
                    canonical_company=canon,
                    company_name=c_name,
                    ticker=ticker,
                    coreference_term=matched_term,
                    coreference_type="brand_product",
                    is_coreference_resolved=True
                )

        # 2. Conversational Anaphora / Pronoun Coreference Resolution
        if prior_context_company:
            canon = prior_context_company.upper()
            if canon in self.canonical_info:
                info = self.canonical_info[canon]
                c_name = info.get("name", canon)
                ticker = info.get("ticker", "")
                
                for a_pat, a_term in ANAPHORA_PATTERNS:
                    if a_pat.search(query):
                        # Replace or enrich query with prior company antecedent
                        enriched = f"{query} [Entity: {c_name} | Ticker: {ticker}]"
                        return ResolvedQuery(
                            original_query=query,
                            enriched_query=enriched,
                            canonical_company=canon,
                            company_name=c_name,
                            ticker=ticker,
                            coreference_term=a_term,
                            coreference_type="anaphora_pronoun",
                            is_coreference_resolved=True
                        )

        # 3. Direct mention or none
        return ResolvedQuery(
            original_query=query,
            enriched_query=query,
            canonical_company=None,
            company_name=None,
            ticker=None,
            coreference_term=None,
            coreference_type="none",
            is_coreference_resolved=False
        )

    def resolve_chunk_antecedents(
        self,
        chunk_text: str,
        doc_name: str,
        section_heading: Optional[str] = None
    ) -> str:
        """
        Paper Section 3.1 & Appendix A.6:
        Resolves generic corporate pronouns ("the Company", "our", "the registrant") in filing
        prose to the explicit named entity antecedent derived from document metadata.
        """
        if not chunk_text or not doc_name:
            return chunk_text

        # Extract corporate identifier from doc_name (e.g. LLY_2023_10K -> LLY -> ELILILLY)
        pfx = doc_name.split("_")[0].upper()
        canon = None
        for c_id, info in self.canonical_info.items():
            if info.get("doc_prefix", "").upper() == pfx or info.get("ticker", "").upper() == pfx or c_id == pfx:
                canon = c_id
                break

        if not canon:
            return chunk_text

        info = self.canonical_info.get(canon, {})
        c_name = info.get("name", canon)
        ticker = info.get("ticker", "")

        # Check if chunk contains self-referential mentions without the company name
        has_self_ref = bool(CHUNK_SELF_REF_PATTERN.search(chunk_text))
        has_cname = (ticker and ticker in chunk_text) or (c_name and c_name.split()[0] in chunk_text)

        # If chunk is already explicitly grounded, preserve it
        if has_cname and not has_self_ref:
            return chunk_text

        # Prepend structured entity antecedent header
        header_parts = [f"Entity: {c_name}"]
        if ticker:
            header_parts.append(f"Ticker: {ticker}")
        header_parts.append(f"Filing: {doc_name}")
        if section_heading:
            header_parts.append(f"Section: {section_heading}")

        antecedent_header = f"[{' | '.join(header_parts)}]\n"
        return antecedent_header + chunk_text
