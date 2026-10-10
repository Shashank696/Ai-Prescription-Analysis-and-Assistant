import unittest
from pathlib import Path

from src.recommendation.medicine_recommender import (
    CHECK_CONFLICT,
    TIER_POSSIBLE_EQUIVALENT,
    TIER_SAME_PRODUCT,
    TIER_WEAKER_MATCH,
    MedicineRecommendationEngine,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "medicine_recommender_sample.csv"
)


class MedicineRecommendationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MedicineRecommendationEngine(str(FIXTURE_PATH))
        cls.source = next(
            record for record in cls.engine.records
            if record["SETID"] == "source-1"
        )

    def test_matching_product_attributes_produce_possible_equivalent(self):
        result = self.engine.recommend_equivalents(self.source)
        candidate = next(
            item for item in result["candidates"]
            if item["record"]["SETID"] == "equivalent-1"
        )

        self.assertEqual(result["recommendation_status"], "ready")
        self.assertEqual(
            candidate["recommendation_tier"],
            TIER_POSSIBLE_EQUIVALENT,
        )
        self.assertFalse(candidate["conflicts"])

    def test_same_primary_rxcui_is_separate_from_equivalent_products(self):
        result = self.engine.recommend_equivalents(self.source)
        candidate = next(
            item for item in result["candidates"]
            if item["record"]["SETID"] == "same-product-1"
        )

        self.assertEqual(candidate["recommendation_tier"], TIER_SAME_PRODUCT)

    def test_known_conflicts_are_never_possible_equivalents(self):
        result = self.engine.recommend_equivalents(self.source)
        by_setid = {
            item["record"]["SETID"]: item
            for item in result["candidates"]
        }

        for setid in (
            "strength-conflict-1",
            "form-conflict-1",
            "release-conflict-1",
        ):
            self.assertEqual(
                by_setid[setid]["recommendation_tier"],
                TIER_WEAKER_MATCH,
            )

        self.assertEqual(
            by_setid["strength-conflict-1"]["attribute_checks"]["strength"],
            CHECK_CONFLICT,
        )
        self.assertEqual(
            by_setid["form-conflict-1"]["attribute_checks"]["dosage_form"],
            CHECK_CONFLICT,
        )
        self.assertEqual(
            by_setid["release-conflict-1"]["attribute_checks"]["release"],
            CHECK_CONFLICT,
        )

    def test_active_moiety_only_match_is_weaker_not_equivalent(self):
        result = self.engine.recommend_equivalents(self.source)
        candidate = next(
            item for item in result["candidates"]
            if item["record"]["SETID"] == "active-moiety-only-1"
        )

        self.assertEqual(candidate["recommendation_tier"], TIER_WEAKER_MATCH)
        self.assertTrue(candidate["conflicts"])

    def test_missing_source_attributes_prevent_recommendations(self):
        incomplete_source = next(
            record for record in self.engine.records
            if record["SETID"] == "missing-strength-1"
        )

        result = self.engine.recommend_equivalents(incomplete_source)

        self.assertEqual(
            result["recommendation_status"],
            "insufficient_source_data",
        )
        self.assertEqual(result["candidates"], [])


if __name__ == "__main__":
    unittest.main()
