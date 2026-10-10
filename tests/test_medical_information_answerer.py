import json
import socket
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from src.rag.medical_information_answerer import (
    generate_medical_information_answer,
)


class MockHttpResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8")

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class MedicalInformationAnswererTests(unittest.TestCase):
    def setUp(self):
        self.question = "What side effects can Alpha Tablet cause?"
        self.ready_result = {
            "retrieval_status": "ready",
            "message": "Retrieved dataset information.",
            "records": [{
                "matched_medicine_name": "Alpha Tablet",
                "retrieved_fields": {"side_effects": "Alpha Tablet may cause headache."},
                "source": "MedlinePlus",
            }],
        }

    def test_valid_evidence_generates_answer_and_prompt_is_grounded(self):
        response = {"message": {"content": "Headache is listed as a side effect."}}
        with patch(
            "src.rag.medical_information_answerer.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ) as mock_urlopen:
            result = generate_medical_information_answer(self.question, self.ready_result)

        self.assertEqual("answered", result["status"])
        self.assertIn("Headache", result["answer"])
        self.assertIn("Alpha Tablet", result["answer"])
        request_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        prompt = request_payload["messages"][0]["content"]
        self.assertIn(self.question, prompt)
        self.assertIn("Alpha Tablet may cause headache.", prompt)
        self.assertIn("Do not invent or infer facts", prompt)
        self.assertIn("Do not recommend changing", prompt)
        self.assertEqual(0, request_payload["options"]["num_gpu"])

    def test_ambiguous_result_does_not_call_ollama(self):
        with patch("src.rag.medical_information_answerer.urlopen") as mock_urlopen:
            result = generate_medical_information_answer(
                self.question, {"retrieval_status": "ambiguous", "message": "Choose one."}
            )

        self.assertEqual("ambiguous", result["status"])
        mock_urlopen.assert_not_called()

    def test_insufficient_evidence_does_not_call_ollama(self):
        with patch("src.rag.medical_information_answerer.urlopen") as mock_urlopen:
            result = generate_medical_information_answer(
                self.question,
                {"retrieval_status": "insufficient_evidence", "message": "Not enough data."},
            )

        self.assertEqual("insufficient_evidence", result["status"])
        self.assertEqual("Not enough data.", result["answer"])
        mock_urlopen.assert_not_called()

    def test_unknown_status_does_not_call_ollama(self):
        with patch("src.rag.medical_information_answerer.urlopen") as mock_urlopen:
            result = generate_medical_information_answer(
                self.question, {"retrieval_status": "unknown", "records": []}
            )

        self.assertEqual("error", result["status"])
        mock_urlopen.assert_not_called()

    def test_empty_or_unusable_evidence_does_not_call_ollama(self):
        invalid_results = [
            {"retrieval_status": "ready", "records": []},
            {"retrieval_status": "ready", "records": [{"matched_medicine_name": "Alpha Tablet", "retrieved_fields": {}}]},
        ]
        for retrieval_result in invalid_results:
            with self.subTest(retrieval_result=retrieval_result), patch(
                "src.rag.medical_information_answerer.urlopen"
            ) as mock_urlopen:
                result = generate_medical_information_answer(self.question, retrieval_result)
                self.assertEqual("insufficient_evidence", result["status"])
                mock_urlopen.assert_not_called()

    def test_malformed_retrieval_result_fails_safely_without_ollama(self):
        with patch("src.rag.medical_information_answerer.urlopen") as mock_urlopen:
            result = generate_medical_information_answer(self.question, ["not", "a", "mapping"])

        self.assertEqual("error", result["status"])
        mock_urlopen.assert_not_called()

    def test_empty_model_output_is_an_error(self):
        with patch(
            "src.rag.medical_information_answerer.urlopen",
            return_value=MockHttpResponse(json.dumps({"message": {"content": " "}})),
        ):
            result = generate_medical_information_answer(self.question, self.ready_result)

        self.assertEqual("error", result["status"])
        self.assertIn("no answer", result["answer"])

    def test_connection_and_timeout_fail_safely(self):
        for error in (URLError("connection refused"), socket.timeout()):
            with self.subTest(error=type(error).__name__), patch(
                "src.rag.medical_information_answerer.urlopen", side_effect=error
            ):
                result = generate_medical_information_answer(self.question, self.ready_result)
                self.assertEqual("error", result["status"])
                self.assertTrue(result["error"])

    def test_http_and_malformed_response_fail_safely(self):
        http_error = HTTPError("http://127.0.0.1:11434/api/chat", 500, "error", None, None)
        for response in (http_error, MockHttpResponse("not-json")):
            with self.subTest(response=type(response).__name__), patch(
                "src.rag.medical_information_answerer.urlopen", return_value=response
            ) as mock_urlopen:
                if isinstance(response, HTTPError):
                    mock_urlopen.side_effect = response
                result = generate_medical_information_answer(self.question, self.ready_result)
                self.assertEqual("error", result["status"])

    def test_sources_are_limited_to_retrieved_evidence(self):
        result_data = dict(self.ready_result)
        result_data["records"] = self.ready_result["records"] + [{
            "matched_medicine_name": "Ignored Tablet",
            "retrieved_fields": {},
            "source": "Unrelated source",
        }]
        with patch(
            "src.rag.medical_information_answerer.urlopen",
            return_value=MockHttpResponse(json.dumps({"message": {"content": "Evidence says headache."}})),
        ):
            result = generate_medical_information_answer(self.question, result_data)

        self.assertEqual(
            [{"medicine_name": "Alpha Tablet", "source": "MedlinePlus"}],
            result["sources"],
        )


if __name__ == "__main__":
    unittest.main()
