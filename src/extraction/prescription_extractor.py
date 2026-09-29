# ============================================================
# 1. Imports
# ============================================================

import sys
import os

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
import mimetypes



# ============================================================
# 2. Load Environment Variables
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Please add HF_TOKEN to your .env file."
    )


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

client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN
)

def extraction_prescription(image_path: str):
    """
    Extract prescription information from an image using Hugging Face Inference API.

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

    with open(image_path, "rb") as image_file:

        image_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")


    # ============================================================
    # 7. Detect MIME Type
    # ============================================================

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"


    image_url = (
        f"data:{mime_type};base64,{image_base64}"
    )


    # ============================================================
    # 8. Prompt
    # ============================================================

    prompt = """
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


    # ============================================================
    # 9. Send Image to Qwen
    # ============================================================

    print("\nAnalyzing prescription...\n")
    
    try:

        result = client.chat.completions.create(

            model="Qwen/Qwen3-VL-4B-Instruct",

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": image_url
                            }
                        },

                        {
                            "type": "text",
                            "text": prompt
                        }

                    ]
                }
            ],

            max_tokens=2000
        )
    
    except Exception as e:
        
        print("\n❌ Qwen API request failed.")
        print("Error:", e)

        return None


    # ============================================================
    # 10. Get Qwen Response
    # ============================================================

    raw_output = result.choices[0].message.content

    print("=" * 60)
    print("RAW QWEN OUTPUT")
    print("=" * 60)

    print(raw_output)


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

        print("\n✅ JSON Parsing Successful!")

    except json.JSONDecodeError as e:

        print("\n❌ Qwen did not return valid JSON.")

        print("JSON Error:", e)

        data = None


    # ============================================================
    # 13. Pydantic Validation
    # ============================================================

    if data is not None:

        try:

            prescription = Prescription.model_validate(data)

            print("\n✅ Pydantic Validation Successful!")

            print("\n" + "=" * 60)
            print("VALIDATED PRESCRIPTION")
            print("=" * 60)

            print(
                prescription.model_dump_json(indent=4)
            )
            
            return prescription

        except ValidationError as e:

            print("\n❌ Pydantic Validation Failed:")

            print(e)
            
            return None
        

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
