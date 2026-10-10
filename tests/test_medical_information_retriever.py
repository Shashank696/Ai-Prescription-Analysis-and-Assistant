import unittest

from src.rag.medical_information_retriever import retrieve_medical_information


class MedicalInformationRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "medicine_name": "Alpha Tablet",
                "uses": "Alpha Tablet is used to treat test condition.",
                "precautions": "Tell a doctor before taking Alpha Tablet.",
                "side_effects": "Alpha Tablet may cause headache.",
                "source": "MedlinePlus",
            },
            {
                "medicine_name": "Beta Capsule",
                "uses": "Beta Capsule is used for a different condition.",
                "precautions": "Avoid Beta Capsule when instructed by a clinician.",
                "side_effects": "Beta Capsule may cause dizziness.",
                "source": "MedlinePlus",
            },
        ]

    def test_exact_medicine_question_retrieves_uses(self):
        result = retrieve_medical_information(
            "What is Alpha Tablet used for?", self.records
        )

        self.assertEqual("ready", result["retrieval_status"])
        self.assertEqual("Alpha Tablet", result["records"][0]["matched_medicine_name"])
        self.assertEqual(["uses"], result["records"][0]["retrieved_field_names"])
        self.assertTrue(result["records"][0]["evidence_relevant"])

    def test_case_whitespace_and_punctuation_are_normalized(self):
        result = retrieve_medical_information(
            "  what precautions for ALPHA-TABLET? ", self.records
        )

        self.assertEqual("ready", result["retrieval_status"])
        self.assertEqual(
            {"precautions"}, set(result["records"][0]["retrieved_field_names"])
        )

    def test_precautions_question_returns_precautions_only(self):
        result = retrieve_medical_information(
            "What warnings apply to Alpha Tablet?", self.records
        )

        self.assertEqual(
            {"precautions"}, set(result["records"][0]["retrieved_field_names"])
        )

    def test_side_effect_question_returns_side_effects(self):
        result = retrieve_medical_information(
            "What side effects can Alpha Tablet cause?", self.records
        )

        self.assertEqual(
            {"side_effects"}, set(result["records"][0]["retrieved_field_names"])
        )

    def test_cause_question_is_recognized_as_side_effect_intent(self):
        result = retrieve_medical_information(
            "Can Alpha Tablet cause headache?", self.records
        )

        self.assertEqual("ready", result["retrieval_status"])
        self.assertEqual(
            {"side_effects"}, set(result["records"][0]["retrieved_field_names"])
        )

    def test_unknown_medicine_is_insufficient_evidence(self):
        result = retrieve_medical_information("What is Gamma used for?", self.records)

        self.assertEqual("insufficient_evidence", result["retrieval_status"])
        self.assertEqual([], result["records"])

    def test_unsupported_question_about_matched_name_is_insufficient_evidence(self):
        result = retrieve_medical_information(
            "What is the weather near Alpha Tablet?", self.records
        )

        self.assertEqual("insufficient_evidence", result["retrieval_status"])

    def test_clear_decision_requests_are_insufficient_evidence(self):
        questions = [
            "Should I discontinue Alpha Tablet?",
            "Can I skip a dose of Alpha Tablet?",
            "Can I double my Alpha Tablet dose?",
            "Should I take more Alpha Tablet?",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = retrieve_medical_information(question, self.records)
                self.assertEqual("insufficient_evidence", result["retrieval_status"])
                self.assertIn("cannot support", result["message"])

    def test_informational_question_with_dosage_is_not_decision_request(self):
        result = retrieve_medical_information(
            "What precautions should I know about Alpha Tablet dosage?", self.records
        )

        self.assertEqual("ready", result["retrieval_status"])
        self.assertEqual(
            {"precautions"}, set(result["records"][0]["retrieved_field_names"])
        )

    def test_multiple_medicines_are_ambiguous(self):
        result = retrieve_medical_information(
            "Compare the side effects of Alpha Tablet and Beta Capsule.", self.records
        )

        self.assertEqual("ambiguous", result["retrieval_status"])
        self.assertEqual([], result["records"])

    def test_missing_requested_field_is_not_evidence(self):
        records = [dict(self.records[0], precautions="", side_effects="")]

        result = retrieve_medical_information(
            "What precautions apply to Alpha Tablet?", records
        )

        self.assertEqual("insufficient_evidence", result["retrieval_status"])

    def test_non_string_field_values_do_not_crash_and_are_preserved(self):
        records = [dict(self.records[0], side_effects=123)]

        result = retrieve_medical_information(
            "Can Alpha Tablet cause headache?", records
        )

        self.assertEqual("ready", result["retrieval_status"])
        self.assertEqual(123, result["records"][0]["retrieved_fields"]["side_effects"])

    def test_ordering_and_limit_are_deterministic(self):
        duplicate_records = [
            dict(self.records[0], source="Z source"),
            dict(self.records[0], source="A source"),
        ]

        first = retrieve_medical_information(
            "What is Alpha Tablet used for?", duplicate_records, limit=1
        )
        second = retrieve_medical_information(
            "What is Alpha Tablet used for?", duplicate_records, limit=1
        )

        self.assertEqual(first, second)
        self.assertEqual(1, len(first["records"]))
        self.assertEqual("A source", first["records"][0]["source"])

    def test_source_attribution_and_original_field_value_are_preserved(self):
        result = retrieve_medical_information(
            "What side effects can Alpha Tablet cause?", self.records
        )
        record = result["records"][0]

        self.assertEqual("MedlinePlus", record["source"])
        self.assertEqual(
            self.records[0]["side_effects"], record["retrieved_fields"]["side_effects"]
        )

    def test_empty_question_is_insufficient_evidence(self):
        result = retrieve_medical_information("   ", self.records)

        self.assertEqual("insufficient_evidence", result["retrieval_status"])
        self.assertEqual([], result["records"])


if __name__ == "__main__":
    unittest.main()
