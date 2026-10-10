import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.schemas.prescription_schema import Medicine, Prescription
from src.verification.medicine_verifier import (
    CHARACTERISTIC_CONFLICT,
    STRENGTH_ALIGNED,
    STRENGTH_CONFLICT,
    VERIFICATION_CANDIDATE,
    VERIFICATION_EXACT,
    VERIFICATION_UNCERTAIN,
    VERIFICATION_UNMATCHED,
    MedicineVerifier,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "medicine_verifier_sample.csv"
)


class MedicineVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = MedicineVerifier(str(FIXTURE_PATH))

    def test_generic_name_and_matching_strength_is_exact(self):
        result = self.verifier.verify_medicine("Tab. naproxen", "500 MG")

        self.assertEqual(result["verification_status"], VERIFICATION_EXACT)
        self.assertTrue(result["verified"])
        self.assertEqual(result["strength_check"], STRENGTH_ALIGNED)
        self.assertEqual(len(result["records"]), 1)

    def test_duplicate_spl_versions_resolve_to_one_rxnorm_product(self):
        data = pd.read_csv(FIXTURE_PATH)
        duplicate = data.loc[
            data["primary_rxcui"].eq(198014)
        ].iloc[0].copy()
        duplicate["SETID"] = "duplicate-spl-version"
        duplicate["SPL_VERSION"] = 2

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            dataset_path = Path(directory) / "duplicate_product.csv"
            pd.concat(
                [data, pd.DataFrame([duplicate])],
                ignore_index=True,
            ).to_csv(dataset_path, index=False)
            verifier = MedicineVerifier(str(dataset_path))
            result = verifier.verify_medicine("naproxen", "500 MG")

        self.assertEqual(result["verification_status"], VERIFICATION_EXACT)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["records"][0]["primary_rxcui"], 198014)

    def test_generic_name_without_strength_is_uncertain(self):
        result = self.verifier.verify_medicine("naproxen")

        self.assertEqual(result["verification_status"], VERIFICATION_UNCERTAIN)
        self.assertFalse(result["verified"])

    def test_brand_match_preserves_multiple_candidates(self):
        result = self.verifier.verify_medicine("Augmentin", "500 MG / 125 MG")

        self.assertEqual(result["verification_status"], VERIFICATION_CANDIDATE)
        self.assertEqual(result["match_type"], "brand_name")
        self.assertGreaterEqual(result["candidate_count"], 3)
        self.assertIsInstance(result["records"], list)
        self.assertTrue(all(isinstance(row, dict) for row in result["records"]))

    def test_combination_strength_is_compared_component_by_component(self):
        result = self.verifier.verify_medicine(
            "amoxicillin / clavulanate",
            "500 MG / 125 MG",
        )

        self.assertEqual(result["verification_status"], VERIFICATION_EXACT)
        self.assertEqual(result["strength_check"], STRENGTH_ALIGNED)
        self.assertEqual(result["candidate_count"], 1)

    def test_combination_strength_total_is_not_treated_as_equivalent(self):
        result = self.verifier.verify_medicine(
            "amoxicillin / clavulanate",
            "625 MG",
        )

        self.assertEqual(result["verification_status"], VERIFICATION_UNCERTAIN)
        self.assertEqual(result["strength_check"], STRENGTH_CONFLICT)

    def test_missing_and_unknown_names_are_unmatched(self):
        self.assertEqual(
            self.verifier.verify_medicine(None)["verification_status"],
            VERIFICATION_UNMATCHED,
        )
        self.assertEqual(
            self.verifier.verify_medicine("unknown medicine")["verification_status"],
            VERIFICATION_UNMATCHED,
        )

    def test_base_ingredient_match_is_uncertain(self):
        result = self.verifier.verify_medicine("benazepril")

        self.assertEqual(result["verification_status"], VERIFICATION_UNCERTAIN)
        self.assertEqual(result["match_type"], "base_medicine")
        self.assertGreaterEqual(result["candidate_count"], 2)

    def test_fda_proprietary_name_preserves_all_candidates(self):
        result = self.verifier.verify_medicine("SampleFda")

        self.assertEqual(result["verification_status"], VERIFICATION_CANDIDATE)
        self.assertEqual(result["match_type"], "fda_proprietary_name")
        self.assertEqual(result["candidate_count"], 2)

    def test_dosage_form_conflict_is_explicit(self):
        result = self.verifier.verify_medicine(
            "testdrug oral capsule",
            "10 MG",
        )

        self.assertEqual(result["verification_status"], VERIFICATION_UNCERTAIN)
        self.assertEqual(result["dosage_form_check"], CHARACTERISTIC_CONFLICT)

    def test_release_conflict_is_explicit(self):
        result = self.verifier.verify_medicine(
            "12 HR testdrug oral tablet",
            "10 MG",
        )

        self.assertEqual(result["verification_status"], VERIFICATION_UNCERTAIN)
        self.assertEqual(result["release_check"], CHARACTERISTIC_CONFLICT)

    def test_grouped_result_contract_matches_pipeline_categories(self):
        prescription = Prescription(
            medicines=[
                Medicine(medicine_name="naproxen", strength="500 MG"),
                Medicine(medicine_name="Augmentin", strength="500 MG / 125 MG"),
                Medicine(medicine_name="benazepril hydrochloride"),
                Medicine(medicine_name="unknown medicine"),
            ]
        )

        exact, candidate, uncertain, unmatched = (
            self.verifier.get_verified_medicines(prescription)
        )

        self.assertEqual(len(exact), 1)
        self.assertEqual(len(candidate), 1)
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(len(unmatched), 1)
        for group in (exact, candidate, uncertain, unmatched):
            for item in group:
                self.assertIn(item["verification_status"], {
                    VERIFICATION_EXACT,
                    VERIFICATION_CANDIDATE,
                    VERIFICATION_UNCERTAIN,
                    VERIFICATION_UNMATCHED,
                })
                self.assertIsInstance(item["database_records"], list)

    def test_pipeline_preserves_the_four_verification_groups(self):
        fake_prescription = Prescription()
        expected_groups = ([{"id": "exact"}], [{"id": "candidate"}], [{"id": "uncertain"}], [{"id": "unmatched"}])

        extractor_module = types.ModuleType(
            "src.extraction.prescription_extractor"
        )
        extractor_module.extraction_prescription = lambda _: fake_prescription

        preprocessing_module = types.ModuleType(
            "src.extraction.image_preprocessing"
        )
        preprocessing_module.preprocess_image = lambda path: path

        verifier_module = types.ModuleType(
            "src.verification.medicine_verifier"
        )

        class FakeVerifier:
            def __init__(self, dataset_path):
                self.dataset_path = dataset_path

            def get_verified_medicines(self, prescription):
                self.last_prescription = prescription
                return expected_groups

        verifier_module.MedicineVerifier = FakeVerifier

        sys.modules.pop("src.pipeline.prescription_pipeline", None)
        with patch.dict(
            sys.modules,
            {
                "src.extraction.prescription_extractor": extractor_module,
                "src.extraction.image_preprocessing": preprocessing_module,
                "src.verification.medicine_verifier": verifier_module,
            },
        ):
            pipeline = importlib.import_module(
                "src.pipeline.prescription_pipeline"
            )
            result = pipeline.process_prescription("image.jpg")

        sys.modules.pop("src.pipeline.prescription_pipeline", None)
        self.assertIs(result["prescription"], fake_prescription)
        self.assertIs(result["verified_medicines"], expected_groups[0])
        self.assertIs(result["candidate_medicines"], expected_groups[1])
        self.assertIs(result["uncertain_medicines"], expected_groups[2])
        self.assertIs(result["unverified_medicines"], expected_groups[3])


if __name__ == "__main__":
    unittest.main()
