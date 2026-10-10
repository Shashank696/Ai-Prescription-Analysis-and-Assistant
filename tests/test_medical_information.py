import unittest
from pathlib import Path
from unittest.mock import patch

from ui.app import (
    answer_medical_information_question,
    load_medical_information,
    search_medical_information,
)


class MedicalInformationLookupTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "medicine_name": "Alpha Tablet",
                "uses": "Test use",
                "precautions": "Test precaution",
                "side_effects": "Test side effect",
                "source": "MedlinePlus",
            },
            {
                "medicine_name": "Alpha Extended Release",
                "uses": "Test use",
                "precautions": "Test precaution",
                "side_effects": "Test side effect",
                "source": "MedlinePlus",
            },
        ]

    def test_exact_match_has_priority_over_broader_match(self):
        result = search_medical_information(self.records, "alpha tablet")

        self.assertEqual("exact", result["match_type"])
        self.assertEqual(1, result["total_matches"])
        self.assertEqual("Alpha Tablet", result["records"][0]["medicine_name"])

    def test_exact_match_normalizes_case_and_whitespace(self):
        result = search_medical_information(self.records, "  ALPHA    TABLET  ")

        self.assertEqual("exact", result["match_type"])
        self.assertEqual([self.records[0]], result["records"])

    def test_explicit_alias_is_supported_without_inference(self):
        records = [dict(self.records[0], aliases="Alpha Brand|Alpha-T")]

        result = search_medical_information(records, "alpha brand")

        self.assertEqual("explicit_alias", result["match_type"])
        self.assertEqual([records[0]], result["records"])

    def test_substring_match_is_a_labelled_fallback(self):
        result = search_medical_information(self.records, "extended")

        self.assertEqual("substring", result["match_type"])
        self.assertEqual([self.records[1]], result["records"])

    def test_no_matches_returns_empty_result(self):
        result = search_medical_information(self.records, "unknown medicine")

        self.assertEqual("none", result["match_type"])
        self.assertEqual([], result["records"])
        self.assertEqual(0, result["total_matches"])

    def test_result_limit_preserves_total_count(self):
        records = [
            dict(self.records[0], medicine_name=f"Alpha {index}")
            for index in range(12)
        ]

        result = search_medical_information(records, "alpha", limit=10)

        self.assertEqual("substring", result["match_type"])
        self.assertEqual(12, result["total_matches"])
        self.assertEqual(10, len(result["records"]))

    def test_missing_file_raises_for_ui_to_handle(self):
        missing_path = Path(__file__).parent / "fixtures" / "missing_medlineplus_lookup.csv"
        load_medical_information.clear()

        with self.assertRaises(FileNotFoundError):
            load_medical_information(str(missing_path))

    def test_question_helper_passes_retrieval_to_answerer(self):
        retrieval_result = {"retrieval_status": "insufficient_evidence", "records": []}
        answer_result = {"status": "insufficient_evidence", "answer": "Not enough evidence."}
        with patch("ui.app.retrieve_medical_information", return_value=retrieval_result) as retrieve, patch(
            "ui.app.generate_medical_information_answer", return_value=answer_result
        ) as answer:
            actual_retrieval, actual_answer = answer_medical_information_question(
                "What is Alpha Tablet used for?", self.records
            )

        retrieve.assert_called_once_with("What is Alpha Tablet used for?", self.records)
        answer.assert_called_once_with("What is Alpha Tablet used for?", retrieval_result)
        self.assertEqual(retrieval_result, actual_retrieval)
        self.assertEqual(answer_result, actual_answer)


if __name__ == "__main__":
    unittest.main()
