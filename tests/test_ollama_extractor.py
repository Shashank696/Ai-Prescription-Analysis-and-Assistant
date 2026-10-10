import json
import io
import importlib.util
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch
from urllib.error import HTTPError, URLError

from src.extraction.prescription_extractor import (
    ExtractionProviderError,
    OLLAMA_MAX_IMAGE_DIMENSION,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    extraction_prescription,
    extract_with_ollama,
    prepare_ollama_image,
)
from ui.app import process_with_retry


class MockHttpResponse:
    def __init__(self, payload):
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class OllamaExtractorTests(unittest.TestCase):
    def setUp(self):
        temporary_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        )
        temporary_file.write(b"test-image")
        temporary_file.close()
        self.image_path = Path(temporary_file.name)

    def tearDown(self):
        self.image_path.unlink(missing_ok=True)

    def test_local_ollama_response_validates_without_huggingface(self):
        model_response = {
            "message": {
                "content": json.dumps({
                    "patient_name": None,
                    "medicines": [{
                        "medicine_name": "naproxen",
                        "strength": "500 MG",
                    }],
                }),
            }
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(model_response)),
        ) as mock_urlopen, patch(
            "src.extraction.prescription_extractor.extract_with_huggingface"
        ) as mock_huggingface:
            prescription = extraction_prescription(str(self.image_path))

        self.assertEqual(prescription.medicines[0].medicine_name, "naproxen")
        self.assertEqual(prescription.medicines[0].strength, "500 MG")
        mock_huggingface.assert_not_called()
        request = mock_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["model"], OLLAMA_MODEL)
        self.assertEqual(request_payload["format"], "json")
        self.assertEqual(
            request_payload["options"]["num_ctx"],
            OLLAMA_NUM_CTX,
        )
        self.assertEqual(
            request_payload["options"]["num_predict"],
            OLLAMA_NUM_PREDICT,
        )
        self.assertTrue(request_payload["messages"][0]["images"][0])

    def test_num_predict_defaults_to_128_without_environment_override(self):
        import src.extraction.prescription_extractor as extractor_module

        spec = importlib.util.spec_from_file_location(
            "isolated_prescription_extractor",
            extractor_module.__file__,
        )
        isolated_module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {}, clear=True), patch(
            "dotenv.load_dotenv", return_value=False
        ):
            spec.loader.exec_module(isolated_module)

        self.assertEqual(isolated_module.OLLAMA_NUM_PREDICT, 128)

    def test_num_gpu_defaults_to_cpu(self):
        model_response = {"message": {"content": "{}"}}

        with patch.dict(os.environ, {}, clear=True), patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(model_response)),
        ) as mock_urlopen:
            extract_with_ollama("image", "prompt")

        request = mock_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["options"]["num_gpu"], 0)

    def test_num_gpu_respects_environment_override(self):
        model_response = {"message": {"content": "{}"}}

        with patch.dict(os.environ, {"OLLAMA_NUM_GPU": "2"}), patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(model_response)),
        ) as mock_urlopen:
            extract_with_ollama("image", "prompt")

        request = mock_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["options"]["num_gpu"], 2)

    def test_bounded_image_uses_original_bytes_without_reencoding(self):
        image = Mock()
        image.shape = (1239, 1754, 3)
        original_bytes = b"original-image"

        with open(self.image_path, "wb") as image_file:
            image_file.write(original_bytes)

        with patch(
            "src.extraction.prescription_extractor.cv2.imread",
            return_value=image,
        ), patch(
            "src.extraction.prescription_extractor.cv2.imencode",
        ) as mock_imencode:
            prepared = prepare_ollama_image(str(self.image_path))

        self.assertEqual(prepared, original_bytes)
        mock_imencode.assert_not_called()

    def test_large_image_is_resized_before_local_inference(self):
        image = Mock()
        image.shape = (2000, 4000, 3)
        resized = Mock()
        encoded_image = Mock()
        encoded_image.tobytes.return_value = b"bounded-image"

        with patch(
            "src.extraction.prescription_extractor.cv2.imread",
            return_value=image,
        ), patch(
            "src.extraction.prescription_extractor.cv2.resize",
            return_value=resized,
        ) as mock_resize, patch(
            "src.extraction.prescription_extractor.cv2.imencode",
            return_value=(True, encoded_image),
        ):
            prepared = prepare_ollama_image(str(self.image_path))

        self.assertEqual(prepared, b"bounded-image")
        mock_resize.assert_called_once_with(
            image,
            (OLLAMA_MAX_IMAGE_DIMENSION, OLLAMA_MAX_IMAGE_DIMENSION // 2),
            interpolation=ANY,
        )

    def test_missing_model_is_a_permanent_error(self):
        http_error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            404,
            "Not Found",
            None,
            None,
        )

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            side_effect=http_error,
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_model_missing")
        self.assertFalse(caught.exception.retryable)

    def test_http_rejection_preserves_status_and_safe_provider_detail(self):
        http_error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":"model does not support vision"}'),
        )

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            side_effect=http_error,
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_request_rejected")
        self.assertEqual(caught.exception.http_status, 400)
        self.assertIn("HTTP 400", caught.exception.message)
        self.assertIn("model does not support vision", caught.exception.message)
        self.assertFalse(caught.exception.retryable)

    def test_memory_allocation_error_is_permanent_and_actionable(self):
        http_error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":"std::bad_alloc"}'),
        )

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            side_effect=http_error,
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_memory_allocation_failed")
        self.assertEqual(caught.exception.http_status, 400)
        self.assertIn("could not allocate enough memory", caught.exception.message)
        self.assertFalse(caught.exception.retryable)

    def test_unavailable_service_is_a_permanent_error(self):
        with patch(
            "src.extraction.prescription_extractor.urlopen",
            side_effect=URLError("connection refused"),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_unavailable")
        self.assertFalse(caught.exception.retryable)

    def test_timeout_is_retryable(self):
        with patch(
            "src.extraction.prescription_extractor.urlopen",
            side_effect=socket.timeout(),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_timeout")
        self.assertTrue(caught.exception.retryable)

    def test_malformed_ollama_response_is_retryable(self):
        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse("not-json"),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_response_not_json")
        self.assertEqual(caught.exception.http_status, 200)
        self.assertTrue(caught.exception.retryable)

    def test_generation_limit_is_reported_without_retry(self):
        response = {
            "done": True,
            "done_reason": "length",
            "message": {"content": "{"},
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_output_truncated")
        self.assertIn("done_reason=length", caught.exception.message)
        self.assertFalse(caught.exception.retryable)

    def test_http_success_error_envelope_is_not_misreported_as_empty_content(self):
        response = {
            "done": False,
            "error": {"message": "runner stopped unexpectedly"},
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_response_error")
        self.assertIn("runner stopped unexpectedly", caught.exception.message)
        self.assertFalse(caught.exception.retryable)

    def test_non_complete_response_includes_safe_metadata(self):
        response = {
            "done": False,
            "done_reason": "not_provided",
            "message": {"content": "partial"},
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_incomplete_response")
        self.assertIn("done=False", caught.exception.message)
        self.assertIn("content_length=7", caught.exception.message)
        self.assertTrue(caught.exception.retryable)

    def test_empty_completed_response_is_categorized(self):
        response = {
            "done": True,
            "done_reason": "stop",
            "message": {"content": ""},
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extract_with_ollama("image", "prompt")

        self.assertEqual(caught.exception.code, "ollama_empty_content")
        self.assertIn("done_reason=stop", caught.exception.message)
        self.assertTrue(caught.exception.retryable)

    def test_incomplete_extraction_json_is_categorized(self):
        response = {
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"patient_name": '},
        }

        with patch(
            "src.extraction.prescription_extractor.urlopen",
            return_value=MockHttpResponse(json.dumps(response)),
        ), self.assertRaises(ExtractionProviderError) as caught:
            extraction_prescription(str(self.image_path))

        self.assertEqual(caught.exception.code, "ollama_invalid_json")
        self.assertIn("content_length=16", caught.exception.message)
        self.assertTrue(caught.exception.retryable)

    def test_retry_helper_stops_for_permanent_errors(self):
        permanent_error = ExtractionProviderError(
            "ollama_model_missing",
            "model missing",
            retryable=False,
        )
        process_prescription = Mock(side_effect=permanent_error)

        with self.assertRaises(ExtractionProviderError):
            process_with_retry(process_prescription, "image.jpg")

        self.assertEqual(process_prescription.call_count, 1)

    def test_retry_helper_limits_transient_errors_to_three_attempts(self):
        transient_error = ExtractionProviderError(
            "ollama_timeout",
            "timed out",
            retryable=True,
        )
        process_prescription = Mock(side_effect=transient_error)

        with self.assertRaises(ExtractionProviderError):
            process_with_retry(process_prescription, "image.jpg")

        self.assertEqual(process_prescription.call_count, 3)


if __name__ == "__main__":
    unittest.main()
