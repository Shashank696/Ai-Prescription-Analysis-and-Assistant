from src.pipeline.prescription_pipeline import (
    process_prescription
)
from src.extraction.prescription_extractor import (
    ExtractionProviderError
)


def main():

    image_path = input(
        "Enter prescription image path: "
    ).strip()

    try:
        result = process_prescription(
            image_path
        )
    except ExtractionProviderError as error:
        print(f"\nExtraction failed: {error.message}")
        return

    if result is None:

        print(
            "\n❌ Prescription processing failed."
        )

        return

    prescription = result[
        "prescription"
    ]

    verified = result[
        "verified_medicines"
    ]

    candidates = result.get(
        "candidate_medicines",
        [],
    )

    uncertain = result.get(
        "uncertain_medicines",
        [],
    )

    unverified = result[
        "unverified_medicines"
    ]

    # ============================================================
    # EXTRACTED MEDICINES
    # ============================================================

    print("\n" + "=" * 70)
    print("EXTRACTED MEDICINES")
    print("=" * 70)

    for medicine in prescription.medicines:

        print(
            f"Medicine: "
            f"{medicine.medicine_name}"
        )

    verification_groups = [
        ("EXACT MATCHES", verified),
        ("CANDIDATE MATCHES", candidates),
        ("UNCERTAIN MATCHES", uncertain),
        ("UNMATCHED MEDICINES", unverified),
    ]

    for heading, medicines in verification_groups:
        print("\n" + "=" * 70)
        print(heading)
        print("=" * 70)

        if not medicines:
            print("None")
            continue

        for medicine in medicines:
            print(f"Extracted : {medicine.get('extracted_name')}")
            print(f"Matched   : {medicine.get('verified_name') or 'Not available'}")
            print(f"Status    : {medicine.get('verification_status')}")
            print(f"Match     : {medicine.get('match_type')}")
            print(f"Strength  : {medicine.get('strength_check')}")
            print(f"Form      : {medicine.get('dosage_form_check')}")
            print(f"Release   : {medicine.get('release_check')}")
            print(f"Candidates: {medicine.get('candidate_count', 0)}")
            print(f"Reason    : {medicine.get('message') or medicine.get('reason')}")
            print("-" * 50)

    # ============================================================
    # RAG INPUT
    # ============================================================

    print("\n" + "=" * 70)
    print("MEDICINES READY FOR RAG")
    print("=" * 70)

    for medicine in verified:

        print(
            f"- {medicine['verified_name']}"
        )


if __name__ == "__main__":
    main()
