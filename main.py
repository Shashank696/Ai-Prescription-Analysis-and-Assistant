from src.pipeline.prescription_pipeline import (
    process_prescription
)


def main():

    image_path = input(
        "Enter prescription image path: "
    ).strip()

    result = process_prescription(
        image_path
    )

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

    # ============================================================
    # VERIFIED
    # ============================================================

    print("\n" + "=" * 70)
    print("VERIFIED MEDICINES")
    print("=" * 70)

    for medicine in verified:

        print(
            f"Extracted : "
            f"{medicine['extracted_name']}"
        )

        print(
            f"Verified  : "
            f"{medicine['verified_name']}"
        )

        print(
            f"Match     : "
            f"{medicine['match_type']}"
        )

        print("-" * 50)

    # ============================================================
    # UNVERIFIED
    # ============================================================

    print("\n" + "=" * 70)
    print("UNVERIFIED MEDICINES")
    print("=" * 70)

    if not unverified:

        print("None")

    else:

        for medicine in unverified:

            print(
                f"{medicine['extracted_name']} "
                f"-> "
                f"{medicine['reason']}"
            )

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