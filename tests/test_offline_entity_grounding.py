"""
tests/test_offline_entity_grounding.py
======================================
Unit tests for OfflineEntityGrounder (Experiment L).
Verifies:
1. Deterministic corporate identity resolution from doc_name & manifest.
2. Accurate brand/product attachment when present in chunk.
3. Absence of brand tags when brands are not present in chunk.
4. Prevention of cross-company brand pollution (e.g. Mounjaro in Apple 10-K).
5. Preservation of existing [TABLE CONTINUATION | ...] headers.
6. Strict idempotence (no duplicate [CORPORATE CONTEXT | ...] headers).
7. Preservation of original chunk content.
"""

import unittest
from src.indexing.offline_entity_grounder import OfflineEntityGrounder


class TestOfflineEntityGrounder(unittest.TestCase):

    def setUp(self):
        self.grounder = OfflineEntityGrounder()

    def test_resolve_filing_identity(self):
        # From manifest or doc_name
        id_lly = self.grounder.resolve_filing_identity("LLY_2023_10K")
        self.assertEqual(id_lly["company_name"], "Eli Lilly and Company")
        self.assertEqual(id_lly["ticker"], "LLY")
        self.assertEqual(id_lly["fiscal_year"], "2023")

        id_aapl = self.grounder.resolve_filing_identity("AAPL_2024_10K")
        self.assertEqual(id_aapl["company_name"], "Apple Inc.")
        self.assertEqual(id_aapl["ticker"], "AAPL")
        self.assertEqual(id_aapl["fiscal_year"], "2024")

        id_avgo = self.grounder.resolve_filing_identity("AVGO_2024_10K")
        self.assertEqual(id_avgo["company_name"], "Broadcom Inc.")
        self.assertEqual(id_avgo["ticker"], "AVGO")

    def test_brand_detection_in_chunk(self):
        # Chunk with brand
        lly_chunk = "During 2023, revenue from Mounjaro increased significantly following launch."
        canon = self.grounder.get_canonical_company_id("LLY", "LLY_2023_10K")
        brands = self.grounder.detect_chunk_brands(lly_chunk, canonical_company=canon)
        self.assertIn("Mounjaro", brands)

        # Chunk without brand
        generic_chunk = "Cash and cash equivalents at the end of the year was $2.5 billion."
        no_brands = self.grounder.detect_chunk_brands(generic_chunk, canonical_company=canon)
        self.assertEqual(no_brands, [])

    def test_no_cross_company_brand_pollution(self):
        # If Apple chunk somehow mentions "Keytruda" (e.g. general discussion or market index),
        # it should NOT be attributed to Apple because Keytruda is Merck's product.
        aapl_chunk = "Pharmaceutical investments such as Keytruda are tracked by healthcare funds."
        canon_apple = self.grounder.get_canonical_company_id("AAPL", "AAPL_2024_10K")
        brands = self.grounder.detect_chunk_brands(aapl_chunk, canonical_company=canon_apple)
        self.assertEqual(brands, [], "Cross-company brand should not be attached to Apple!")

    def test_ground_chunk_structure_and_preservation(self):
        doc_name = "LLY_2023_10K"
        raw_text = "The Company reported robust growth in diabetes treatments, with Mounjaro driving volume."
        grounded = self.grounder.ground_chunk(doc_name, raw_text)

        self.assertTrue(grounded.startswith("[CORPORATE CONTEXT |"))
        self.assertIn("Entity: Eli Lilly and Company", grounded)
        self.assertIn("Ticker: LLY", grounded)
        self.assertIn("Filing: LLY_2023_10K", grounded)
        self.assertIn("Products/Brands: Mounjaro", grounded)
        self.assertTrue(grounded.endswith(raw_text))

    def test_preserve_table_continuation_headers(self):
        doc_name = "BAC_2024_10K"
        table_chunk = "[TABLE CONTINUATION | Statement: Consolidated Balance Sheets | Units: (in millions)]\nAssets: $3,261,450"
        grounded = self.grounder.ground_chunk(doc_name, table_chunk)

        # Context header should be prepended before table continuation
        self.assertTrue(grounded.startswith("[CORPORATE CONTEXT |"))
        self.assertIn("Entity: Bank of America", grounded)
        self.assertIn("[TABLE CONTINUATION | Statement: Consolidated Balance Sheets | Units: (in millions)]", grounded)
        self.assertIn("Assets: $3,261,450", grounded)

    def test_idempotence(self):
        doc_name = "META_2024_10K"
        text = "Our Reality Labs segment continues research on virtual reality and Quest 3 headsets."
        grounded_once = self.grounder.ground_chunk(doc_name, text)
        grounded_twice = self.grounder.ground_chunk(doc_name, grounded_once)

        self.assertEqual(grounded_once, grounded_twice, "Grounding must be strictly idempotent!")
        self.assertEqual(grounded_twice.count("[CORPORATE CONTEXT |"), 1)


if __name__ == "__main__":
    unittest.main()
