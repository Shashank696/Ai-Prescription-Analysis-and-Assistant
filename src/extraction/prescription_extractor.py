# ============================================================
# 1. Imports
# ============================================================

import sys
import os
import re

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional, List
from dotenv import load_dotenv

import os
import base64
import json
import socket
import cv2
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen



# ============================================================
# 2. Load Environment Variables
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat",
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:3b",
)
OLLAMA_TIMEOUT_SECONDS = 120
# Keep the local allocation within the available memory of modest GPUs. The
# extraction prompt and image are reduced before being sent so this context is
# sufficient for the structured response.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "128"))
OLLAMA_MAX_IMAGE_DIMENSION = int(os.getenv("OLLAMA_MAX_IMAGE_DIMENSION", "1920"))


class ExtractionProviderError(RuntimeError):
    """A safe, user-facing error from an extraction provider."""

    def __init__(self, code, message, retryable, http_status=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status


def get_error_status_code(error):
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) or getattr(
        error,
        "status_code",
        None,
    )


def get_safe_ollama_error_detail(error):
    """Read only a short structured provider diagnostic from an HTTP error."""

    try:
        body = error.read().decode("utf-8")
        detail = json.loads(body).get("error", "")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return ""

    detail = re.sub(r"\s+", " ", str(detail)).strip()
    return detail[:240]


def format_ollama_response_metadata(response_data, http_status):
    """Return safe response metadata without including model or image content."""

    message = response_data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    content_length = len(content) if isinstance(content, str) else 0
    done = response_data.get("done")
    done_reason = response_data.get("done_reason")
    return (
        f"HTTP {http_status}; done={done}; "
        f"done_reason={done_reason or 'not_provided'}; "
        f"content_length={content_length}"
    )


def get_safe_ollama_response_error(response_data):
    """Return a short provider error from a successful HTTP response."""

    detail = response_data.get("error")
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or detail.get("code")
    if detail is None:
        return ""

    detail = re.sub(r"\s+", " ", str(detail)).strip()
    return detail[:240]


def prepare_ollama_image(image_path):
    """Return a bounded JPEG for local vision inference, or the original bytes.

    Ollama decodes the supplied image before vision processing. Restricting a
    very large phone photograph avoids a request-sized memory spike while
    retaining enough detail for a prescription image.
    """

    with open(image_path, "rb") as image_file:
        original_bytes = image_file.read()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return original_bytes

    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= OLLAMA_MAX_IMAGE_DIMENSION:
        return original_bytes

    scale = OLLAMA_MAX_IMAGE_DIMENSION / largest_dimension
    image = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )

    encoded, jpeg = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    return jpeg.tobytes() if encoded else original_bytes


# ============================================================
# 3. Pydantic Models
# ============================================================

class Medicine(BaseModel):

    medicine_name: Optional[str] = None
    quantity: Optional[str] = None
    strength: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instruction: Optional[str] = None

    @field_validator(
        "medicine_name",
        "quantity",
        "strength",
        "frequency",
        "duration",
        "instruction",
        mode="before"
    )
    @classmethod
    def convert_to_string(cls, value):

        if value is not None:
            return str(value)

        return None
    


class Prescription(BaseModel):

    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    prescription_date: Optional[str] = None

    doctor_name: Optional[str] = None
    doctor_registration_number: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    clinic_phone: Optional[str] = None

    medicines: List[Medicine] = Field(default_factory=list)

    global_instruction: Optional[str] = None

    @field_validator(
        "patient_name",
        "patient_age",
        "patient_gender",
        "prescription_date",
        "doctor_name",
        "doctor_registration_number",
        "clinic_name",
        "clinic_address",
        "clinic_phone",
        "global_instruction",
        mode="before"
    )
    @classmethod
    def convert_to_string(cls, value):

        if value is not None:
            return str(value)

        return None


# ============================================================
# 4. Create Hugging Face Client
# ============================================================

client = None


def get_huggingface_client():
    """Return the legacy hosted client only when it is explicitly requested."""

    global client

    if not HF_TOKEN:
        raise ExtractionProviderError(
            "huggingface_credentials_missing",
            "Hosted Hugging Face inference is unavailable because its "
            "account credentials are not configured.",
            retryable=False,
        )

    if client is None:
        client = InferenceClient(
            provider="featherless-ai",
            api_key=HF_TOKEN,
        )

    return client


def extract_with_huggingface(image_url, prompt):
    """Legacy hosted extraction path. It is retained but not the default."""

    try:
        result = get_huggingface_client().chat.completions.create(
            model="Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
    except ExtractionProviderError:
        raise
    except Exception as error:
        status_code = get_error_status_code(error)
        if status_code in (401, 402, 403):
            raise ExtractionProviderError(
                "huggingface_account_unavailable",
                "Hosted Hugging Face inference is unavailable because of an "
                "account, authorization, or provider billing issue.",
                retryable=False,
            )
        raise ExtractionProviderError(
            "huggingface_request_failed",
            "Hosted Hugging Face inference could not complete the "
            "extraction request.",
            retryable=True,
        )

    return result.choices[0].message.content


def extract_with_ollama(image_base64, prompt):
    """Run local vision extraction through Ollama's chat API."""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_base64],
        }],
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "0")),
        },
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            http_status = getattr(response, "status", None) or 200
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = get_safe_ollama_error_detail(error)
        detail_suffix = f" Provider detail: {detail}" if detail else ""
        if error.code == 404:
            raise ExtractionProviderError(
                "ollama_model_missing",
                f"The local Ollama model '{OLLAMA_MODEL}' is unavailable. "
                "Pull the model in Ollama before trying again."
                + detail_suffix,
                retryable=False,
                http_status=error.code,
            )
        if error.code >= 500:
            raise ExtractionProviderError(
                "ollama_server_error",
                f"The local Ollama service reported HTTP {error.code}."
                + detail_suffix,
                retryable=True,
                http_status=error.code,
            )
        if "std::bad_alloc" in detail.lower():
            raise ExtractionProviderError(
                "ollama_memory_allocation_failed",
                "Local Ollama could not allocate enough memory for this "
                "extraction request. Close other GPU-intensive applications "
                "and try again."
                + detail_suffix,
                retryable=False,
                http_status=error.code,
            )
        raise ExtractionProviderError(
            "ollama_request_rejected",
            f"Ollama rejected the local extraction request (HTTP {error.code}). "
            "Check the local model and service configuration."
            + detail_suffix,
            retryable=False,
            http_status=error.code,
        )
    except (socket.timeout, TimeoutError):
        raise ExtractionProviderError(
            "ollama_timeout",
            "Local Ollama inference timed out. You can try the request again.",
            retryable=True,
        )
    except URLError:
        raise ExtractionProviderError(
            "ollama_unavailable",
            "Local Ollama is unavailable at http://127.0.0.1:11434. "
            "Start the Ollama service and try again.",
            retryable=False,
        )

    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError:
        raise ExtractionProviderError(
            "ollama_response_not_json",
            f"Local Ollama returned a non-JSON API response (HTTP {http_status}). "
            "The request can be retried.",
            retryable=True,
            http_status=http_status,
        )

    if not isinstance(response_data, dict):
        raise ExtractionProviderError(
            "ollama_response_not_object",
            f"Local Ollama returned an unexpected API response (HTTP {http_status}). "
            "The request can be retried.",
            retryable=True,
            http_status=http_status,
        )

    metadata = format_ollama_response_metadata(response_data, http_status)
    response_error = get_safe_ollama_response_error(response_data)
    if response_error:
        error_code = (
            "ollama_memory_allocation_failed"
            if "std::bad_alloc" in response_error.lower()
            else "ollama_response_error"
        )
        raise ExtractionProviderError(
            error_code,
            "Local Ollama returned an error response "
            f"({metadata}). Provider detail: {response_error}",
            retryable=False,
            http_status=http_status,
        )

    message = response_data.get("message")
    raw_output = message.get("content") if isinstance(message, dict) else None
    done_reason = response_data.get("done_reason")

    if done_reason == "length":
        raise ExtractionProviderError(
            "ollama_output_truncated",
            "Local Ollama stopped before completing the extraction JSON "
            f"because its generation limit was reached ({metadata}). "
            "Reduce the requested information or increase OLLAMA_NUM_PREDICT "
            "only if sufficient memory is available.",
            retryable=False,
            http_status=http_status,
        )

    if response_data.get("done") is False:
        raise ExtractionProviderError(
            "ollama_incomplete_response",
            f"Local Ollama did not complete the extraction response ({metadata}). "
            "The request can be retried.",
            retryable=True,
            http_status=http_status,
        )

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ExtractionProviderError(
            "ollama_empty_content",
            f"Local Ollama completed without extraction content ({metadata}). "
            "The request can be retried.",
            retryable=True,
            http_status=http_status,
        )

    return raw_output

def extraction_prescription(image_path: str):
    """
    Extract prescription information from an image using local Ollama.

    Args:
        image_path (str): Path to the prescription image.
        
    Returns:
        Prescription: Extracted prescription information.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ============================================================
    # 6. Convert Image to Base64
    # ============================================================

    image_base64 = base64.b64encode(
        prepare_ollama_image(image_path)
    ).decode("utf-8")


    # ============================================================
    # 8. Prompt
    # ============================================================

    # Retained for reference. The local model uses the compact equivalent below
    # because this verbose form exceeds a 4,096-token context with an image.
    legacy_prompt = """
    Analyze this medical prescription image carefully.

    Your task is ONLY to visually extract information that is actually written
    or clearly visible in the image.

    This is a transcription task, NOT a medical reasoning task.

    Return ONLY valid JSON.
    Do not provide explanations, comments, markdown, or additional text.

    ============================================================
    IMPORTANT EXTRACTION RULES
    ============================================================

    1. DO NOT GUESS.

    Never replace an unclear handwritten word with a medically plausible
    medicine name.

    For example, if the handwriting looks like "Ampho..." but you cannot
    clearly determine the complete medicine name, return null instead of
    guessing "Amphotericin B".

    2. DO NOT USE MEDICAL KNOWLEDGE TO COMPLETE TEXT.

    Do not infer a medicine name, strength, dosage, duration, diagnosis,
    or instruction from general medical knowledge.

    Only extract what is visually present.

    3. PRESERVE THE ORIGINAL TEXT.

    Preserve medicine names, abbreviations, numbers, symbols, and dosage
    patterns as they appear in the prescription whenever possible.

    Examples:
    - "1-0-1" → "1-0-1"
    - "t.i.d." → "t.i.d."
    - "a.c." → "a.c."
    - "625mg" → "625mg"

    Do NOT expand:
    - t.i.d. → three times daily
    - a.c. → before meals

    Expansion/interpretation will be performed later by another component.

    4. HANDLE UNCLEAR TEXT CONSERVATIVELY.

    If a field is:
    - not visible
    - partially visible
    - unreadable
    - ambiguous
    - uncertain

    return null.

    Do NOT choose the most likely medical term.

    5. UNDERSTAND THE PRESCRIPTION LAYOUT.

    Identify the different sections of the prescription before extracting
    the information.

    Possible sections include:

    - Patient information
    - Doctor/clinic information
    - Prescription date
    - Medicine / Inscription section
    - Subscription section
    - Signa / instructions section

    Do not assume that text appearing in one section belongs to another section.

    ============================================================
    MEDICINE EXTRACTION RULES
    ============================================================

    6. MEDICINE NAME

    Extract the medicine name exactly as it is visibly written.

    Do not convert a brand name into a generic name.

    Do not correct spelling using medical knowledge.

    If the medicine name is unclear, return null.

    7. STRENGTH

    Only extract a value as "strength" when the prescription explicitly
    shows a medicine strength.

    Examples:
    - 500 mg
    - 625 mg
    - 10 mg/5 ml
    - 250 mg

    Do NOT treat a quantity or volume as strength.

    For example:

    "15 ml"
    "120 ml"

    should NOT automatically be considered medicine strength.

    8. QUANTITY

    Extract quantity/volume only when explicitly written.

    Examples:
    - 10 tablets
    - 15 ml
    - 120 ml
    - 30 capsules

    If a prescription has a separate column such as "gm or ml", values
    written in that column should be treated as quantity/volume unless
    the prescription explicitly indicates that they represent strength.

    9. FREQUENCY

    Extract the frequency only when it is explicitly associated with
    the medicine or clearly specified in the prescription.

    Preserve it exactly as written.

    Examples:
    - 1-0-1
    - 1-1-1
    - b.i.d.
    - t.i.d.

    Do not expand abbreviations.

    10. DURATION

    Extract duration only when explicitly written.

    Examples:
    - 5 days
    - 7 days
    - 1 week

    Do not calculate or infer duration from quantity and frequency.

    11. INSTRUCTIONS

    Extract medicine-specific instructions only when they are explicitly
    associated with that medicine.

    Examples:
    - after meals
    - before meals
    - at bedtime

    Do not automatically assign a general prescription instruction to
    every medicine.

    ============================================================
    PRESCRIPTION-LEVEL INSTRUCTIONS
    ============================================================

    12. GLOBAL INSTRUCTION

    If the prescription contains an instruction in a general Signa/Sig
    section that applies to the prescription as a whole, preserve that
    instruction separately in "global_instruction".

    Example:

    "Sig: 5 ml t.i.d. a.c."

    should be extracted as:

    "global_instruction": "5 ml t.i.d. a.c."

    Do NOT automatically copy this instruction into every medicine.

    ============================================================
    PATIENT / DOCTOR INFORMATION
    ============================================================

    13. Extract patient and doctor information only when clearly visible.

    Do not infer:
    - age
    - gender
    - doctor name
    - registration number
    - clinic information

    If unclear or absent, return null.

    ============================================================
    FINAL SAFETY RULE
    ============================================================

    If you are uncertain between two possible medicine names, DO NOT choose
    one.

    Return null.

    Accuracy is more important than completeness.

    It is better to return null than to return an incorrect medicine name.

    ============================================================
    INFORMATION TO EXTRACT
    ============================================================

    PATIENT / PRESCRIPTION DETAILS:

    - patient_name
    - patient_age
    - patient_gender
    - prescription_date

    DOCTOR / CLINIC DETAILS:

    - doctor_name
    - doctor_registration_number
    - clinic_name
    - clinic_address
    - clinic_phone

    MEDICINE INFORMATION:

    For every clearly identifiable medicine extract:

    - medicine_name
    - frequency
    - quantity
    - strength
    - duration
    - instruction

    PRESCRIPTION-LEVEL INFORMATION:

    - global_instruction

    ============================================================
    RETURN EXACTLY THIS JSON STRUCTURE
    ============================================================

    {
        "patient_name": null,
        "patient_age": null,
        "patient_gender": null,
        "prescription_date": null,

        "doctor_name": null,
        "doctor_registration_number": null,
        "clinic_name": null,
        "clinic_address": null,
        "clinic_phone": null,

        "medicines": [
            {
                "medicine_name": null,
                "frequency": null,
                "quantity": null,
                "strength": null,
                "duration": null,
                "instruction": null
            }
        ],

        "global_instruction": null
    }
    """

    # This preserves every extraction field and safety rule from the original
    # prompt while leaving room for image tokens and a JSON response locally.
    prompt = """
Analyze this prescription image and return only valid JSON. Transcribe only
text that is clearly visible; this is not medical reasoning.

Never guess, correct, expand, or infer text using medical knowledge. Preserve
medicine names, abbreviations, numbers, symbols, and dosage patterns exactly
as written. For any absent, unclear, partial, unreadable, ambiguous, or
uncertain value, use null. Do not turn brand names into generic names. Do not
interpret quantities or volumes as strengths. Do not copy prescription-wide
instructions into each medicine.

Extract only clearly visible patient_name, patient_age, patient_gender,
prescription_date, doctor_name, doctor_registration_number, clinic_name,
clinic_address, clinic_phone, and global_instruction. For each clearly
identifiable medicine extract medicine_name, frequency, quantity, strength,
duration, and instruction. Keep global Signa/Sig instructions in
global_instruction.

Return exactly this structure:
{
  "patient_name": null, "patient_age": null, "patient_gender": null,
  "prescription_date": null, "doctor_name": null,
  "doctor_registration_number": null, "clinic_name": null,
  "clinic_address": null, "clinic_phone": null,
  "medicines": [{"medicine_name": null, "frequency": null,
    "quantity": null, "strength": null, "duration": null,
    "instruction": null}],
  "global_instruction": null
}
"""


    # ============================================================
    # 9. Send Image to Qwen
    # ============================================================

    print("\nAnalyzing prescription with local Ollama...\n")

    # Ollama is the active provider. The retained Hugging Face helper is never
    # selected here and there is no automatic fallback to hosted inference.
    raw_output = extract_with_ollama(
        image_base64,
        prompt,
    )


    # ============================================================
    # 10. Get Qwen Response
    # ============================================================

    print("=" * 60)
    print("OLLAMA RESPONSE RECEIVED")
    print("=" * 60)


    # ============================================================
    # 11. Clean JSON Response
    # ============================================================

    clean_output = raw_output.strip()


    # Remove opening markdown code block
    if clean_output.startswith("```json"):

        clean_output = clean_output[7:]


    elif clean_output.startswith("```"):

        clean_output = clean_output[3:]


    # Remove closing markdown code block
    if clean_output.endswith("```"):

        clean_output = clean_output[:-3]


    clean_output = clean_output.strip()


    # ============================================================
    # 12. Convert JSON String to Python Dictionary
    # ============================================================

    try:

        data = json.loads(clean_output)

        print("\nJSON parsing successful.")

    except json.JSONDecodeError:
        raise ExtractionProviderError(
            "ollama_invalid_json",
            "Local Ollama returned invalid extraction JSON "
            f"(content_length={len(clean_output)}). The request can be retried.",
            retryable=True,
        )


    # ============================================================
    # 13. Pydantic Validation
    # ============================================================

    if data is not None:

        try:

            prescription = Prescription.model_validate(data)

            print("\nPydantic validation successful.")
            
            return prescription

        except ValidationError:
            raise ExtractionProviderError(
                "ollama_invalid_schema",
                "Local Ollama returned extraction data that does not match "
                f"the prescription format (content_length={len(clean_output)}). "
                "The request can be retried.",
                retryable=True,
            )
        

if __name__ == "__main__":

    image_path = input(
        "Enter prescription image path: "
    ).strip()

    prescription = extraction_prescription(image_path)

    if prescription is None:
        print("Prescription extraction failed.")
        exit()

    print("\n" + "=" * 70)
    print("EXTRACTED MEDICINES")
    print("=" * 70)

    for medicine in prescription.medicines:
        print(
            f"Medicine: {medicine.medicine_name}"
        )

    verified, unverified = verify_extracted_medicines(
        prescription
    )

    print("\n" + "=" * 70)
    print("VERIFIED MEDICINES")
    print("=" * 70)

    for medicine in verified:
        print(
            f"Extracted : {medicine['extracted_name']}"
        )
        print(
            f"Verified  : {medicine['verified_name']}"
        )
        print(
            f"Match     : {medicine['match_type']}"
        )
        print("-" * 50)

    print("\n" + "=" * 70)
    print("UNVERIFIED MEDICINES")
    print("=" * 70)

    for medicine in unverified:
        print(
            f"{medicine['extracted_name']} "
            f"-> NOT FOUND"
        )
    
    print("\n" + "=" * 70)
    print("DEBUG DATASET NAMES")
    print("=" * 70)
