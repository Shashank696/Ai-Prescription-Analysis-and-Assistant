from src.extraction.image_preprocessing import (
    preprocess_image
)

from src.extraction.prescription_extractor import (
    extraction_prescription
)

from src.verification.medicine_verifier import (
    MedicineVerifier
)


MEDICINE_DATASET = (
    "data/cleaned_medical_drug_master.csv"
)


verifier = MedicineVerifier(
    MEDICINE_DATASET
)


def process_prescription(
    image_path: str
):

    # ------------------------------------------------------------
    # 1. Preprocess image
    # ------------------------------------------------------------

    # processed_image = preprocess_image(
    #     image_path
    # )

    # ------------------------------------------------------------
    # 2. Extract prescription using ONE VLM
    # ------------------------------------------------------------

    prescription = extraction_prescription(
        image_path #change it to processed_image if you want to use the preprocessed image
    )

    if prescription is None:
        return None

    # ------------------------------------------------------------
    # 3. Verify medicines against local dataset
    # ------------------------------------------------------------

    (
        verified_medicines,
        candidate_medicines,
        uncertain_medicines,
        unverified_medicines,
    ) = verifier.get_verified_medicines(
        prescription
    )

    # ------------------------------------------------------------
    # 4. Final result
    # ------------------------------------------------------------

    result = {

        "prescription": prescription,

        "verified_medicines":
            verified_medicines,

        "candidate_medicines":
            candidate_medicines,

        "uncertain_medicines":
            uncertain_medicines,

        "unverified_medicines":
            unverified_medicines
    }

    return result