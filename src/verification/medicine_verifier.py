import re
from pathlib import Path

import pandas as pd
import re


class MedicineVerifier:

    # Salt forms that may appear after the base medicine name.
    SALT_FORMS = {
        "hydrochloride",
        "sodium",
        "calcium",
        "potassium",
        "acetate",
        "sulfate",
        "succinate",
        "tartrate",
        "maleate",
        "fumarate",
        "citrate",
        "mesylate",
        "besylate",
    }

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        self.df = None

        self.drug_name_lookup = {}
        self.rxstring_lookup = {}
        self.fda_name_lookup = {}
        self.medicine_name_lookup = {}
        self.base_medicine_lookup = {}
        self.brand_name_lookup = {}

        self.load_dataset()
        self.build_search_indexes()

    # ============================================================
    # LOAD DATASET
    # ============================================================

    def load_dataset(self):

        print("Loading medicine dataset...")

        self.df = pd.read_csv(
            self.dataset_path,
            low_memory=False
        )

        if "drug_name" not in self.df.columns:
            raise ValueError(
                "Dataset must contain 'drug_name' column."
            )

        self.df = self.df[
            self.df["drug_name"].notna()
        ].copy()

    # ============================================================
    # NORMALIZE NAME
    # ============================================================

    def normalize_name(self, name: str) -> str:

        if name is None:
            return ""

        name = str(name).lower().strip()

        name = re.sub(
            r"[^a-z0-9\s/]",
            " ",
            name
        )

        name = re.sub(
            r"\s+",
            " ",
            name
        )

        return name.strip()

    # ============================================================
    # EXTRACT CLEAN MEDICINE NAME
    # ============================================================

    def extract_search_name(self, drug_name: str) -> str:

        if not drug_name:
            return ""

        name = str(drug_name).lower().strip()

        # Remove brand name
        name = re.sub(
            r"\[[^\]]*\]",
            "",
            name
        )

        # Remove leading volume
        name = re.sub(
            r"^\d+(?:\.\d+)?\s*"
            r"(?:ml|l|mg|g|mcg|µg)\s+",
            "",
            name
        )

        # Remove strength expressions
        name = re.sub(
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:mg|mcg|µg|g|kg|ml|l)"
            r"(?:\s*/\s*"
            r"(?:\d+(?:\.\d+)?\s*)?"
            r"(?:mg|mcg|µg|g|kg|ml|l))?\b",
            "",
            name
        )

        dosage_forms = [
            "oral tablet",
            "oral capsule",
            "oral solution",
            "oral suspension",
            "extended release",
            "delayed release",
            "film coated",
            "chewable tablet",
            "disintegrating tablet",
            "pen injector",
            "injection",
            "injectable",
            "tablet",
            "capsule",
            "solution",
            "suspension",
            "cream",
            "ointment",
            "gel",
            "lotion",
            "patch",
            "spray",
            "powder",
            "inhalation",
            "suppository",
        ]

        for form in dosage_forms:

            name = re.sub(
                rf"\b{re.escape(form)}\b",
                "",
                name
            )

        name = re.sub(
            r"\s+",
            " ",
            name
        ).strip()

        name = name.strip("/-,")

        return name

    # ============================================================
    # EXTRACT BASE MEDICINE NAME
    # ============================================================

    def extract_base_medicine_name(
        self,
        medicine_name: str
    ) -> str:

        name = self.normalize_name(
            medicine_name
        )

        if not name:
            return ""

        # Do not create base names from combinations.
        if "/" in name:
            return ""

        parts = name.split()

        if len(parts) == 1:
            return parts[0]

        # Example:
        #
        # benazepril hydrochloride
        #        ↓
        # benazepril
        #
        # but:
        #
        # amlodipine valsartan
        #        ↓
        # no base match

        if len(parts) == 2:

            if parts[1] in self.SALT_FORMS:
                return parts[0]

        return ""

    # ============================================================
    # BUILD SEARCH INDEXES
    # ============================================================

    def build_search_indexes(self):

        print("Building medicine search indexes...")

        self.df["normalized_drug_name"] = (
            self.df["drug_name"]
            .apply(self.normalize_name)
        )

        if "primary_rxstring" in self.df.columns:

            self.df["normalized_rxstring"] = (
                self.df["primary_rxstring"]
                .fillna("")
                .apply(self.normalize_name)
            )

        else:

            self.df["normalized_rxstring"] = ""

        if "fda_proprietary_name" in self.df.columns:

            self.df["normalized_fda_name"] = (
                self.df["fda_proprietary_name"]
                .fillna("")
                .apply(self.normalize_name)
            )

        else:

            self.df["normalized_fda_name"] = ""

        # --------------------------------------------------------
        # drug_name lookup
        # --------------------------------------------------------

        drug_df = self.df[
            self.df["normalized_drug_name"] != ""
        ].drop_duplicates(
            subset=["normalized_drug_name"]
        )

        self.drug_name_lookup = {
            row["normalized_drug_name"]:
                row.to_dict()

            for _, row in drug_df.iterrows()
        }

        # --------------------------------------------------------
        # RxNorm string lookup
        # --------------------------------------------------------

        rx_df = self.df[
            self.df["normalized_rxstring"] != ""
        ].drop_duplicates(
            subset=["normalized_rxstring"]
        )

        self.rxstring_lookup = {
            row["normalized_rxstring"]:
                row.to_dict()

            for _, row in rx_df.iterrows()
        }

        # --------------------------------------------------------
        # FDA proprietary name lookup
        # --------------------------------------------------------

        fda_df = self.df[
            self.df["normalized_fda_name"] != ""
        ].drop_duplicates(
            subset=["normalized_fda_name"]
        )

        self.fda_name_lookup = {
            row["normalized_fda_name"]:
                row.to_dict()

            for _, row in fda_df.iterrows()
        }

        # --------------------------------------------------------
        # Clean medicine name lookup
        # --------------------------------------------------------

        for _, row in self.df.iterrows():

            search_name = self.extract_search_name(
                row["drug_name"]
            )

            if not search_name:
                continue

            normalized = self.normalize_name(
                search_name
            )

            self.medicine_name_lookup.setdefault(
                normalized,
                []
            ).append(
                row.to_dict()
            )

            # ----------------------------------------------------
            # Base medicine lookup
            # ----------------------------------------------------

            base_name = self.extract_base_medicine_name(
                normalized
            )

            if base_name:

                self.base_medicine_lookup.setdefault(
                    base_name,
                    []
                ).append(
                    row.to_dict()
                )
                
            drug_name = str(row["drug_name"])

            brand_match = re.search(r"\[([^\]]+)\]", drug_name)

            if brand_match:
                brand_name = self.normalize_name(brand_match.group(1))

                if brand_name:
                    self.brand_name_lookup.setdefault(
                        brand_name,
                        []
                    ).append(row)
                    
    def clean_extracted_medicine_name(self, medicine_name):
        if not medicine_name:
            return ""

        name = self.normalize_name(medicine_name)

        name = re.sub(
            r"^(tab|tablet|cap|capsule|inj|injection|syr|syrup)\.?\s+",
            "",
            name
        )

        return name.strip()

    # ============================================================
    # SEARCH MEDICINE NAME
    # ============================================================

    def search_medicine_name(
        self,
        query: str
    ):

        query = self.normalize_name(
            query
        )

        return self.medicine_name_lookup.get(
            query,
            []
        )

    # ============================================================
    # EXACT SEARCH
    # ============================================================

    def search_exact(
        self,
        medicine_name: str
    ):

        query = self.normalize_name(
            medicine_name
        )

        if not query:
            return []

        results = []

        if query in self.drug_name_lookup:

            results.append({
                "record":
                    self.drug_name_lookup[query],
                "matched_field":
                    "drug_name",
                "score":
                    100
            })

        if query in self.rxstring_lookup:

            results.append({
                "record":
                    self.rxstring_lookup[query],
                "matched_field":
                    "primary_rxstring",
                "score":
                    100
            })

        if query in self.fda_name_lookup:

            results.append({
                "record":
                    self.fda_name_lookup[query],
                "matched_field":
                    "fda_proprietary_name",
                "score":
                    100
            })

        return results

    # ============================================================
    # VERIFY MEDICINE
    # ============================================================

    def verify_medicine(
        self,
        medicine_name: str
    ):

        if not medicine_name:

            return {
                "verified": False,
                "match_type": "none",
                "medicine_name": None,
                "records": []
            }

        # --------------------------------------------------------
        # 1. Exact dataset match
        # --------------------------------------------------------

        exact_results = self.search_exact(
            medicine_name
        )

        if exact_results:

            return {
                "verified": True,
                "match_type": "exact",
                "medicine_name":
                    self.normalize_name(
                        medicine_name
                    ),
                "records": [
                    result["record"]
                    for result in exact_results
                ]
            }

        # --------------------------------------------------------
        # 2. Clean medicine name
        # --------------------------------------------------------

        search_name = self.extract_search_name(
            medicine_name
        )

        if search_name:

            name_results = self.search_medicine_name(
                search_name
            )

            if name_results:

                return {
                    "verified": True,
                    "match_type":
                        "medicine_name",
                    "medicine_name":
                        self.normalize_name(
                            search_name
                        ),
                    "records":
                        name_results
                }
                
        # --------------------------------------------------------
        # 3. Brand-name match
        # --------------------------------------------------------
                
        cleaned_name = self.clean_extracted_medicine_name(medicine_name)

        brand_matches = self.brand_name_lookup.get(cleaned_name, [])

        if brand_matches:
            return {
                "verified": True,
                "medicine_name": cleaned_name,
                "match_type": "brand_name",
                "records": brand_matches[0]
            }

        # --------------------------------------------------------
        # 4. Base medicine name
        # --------------------------------------------------------

        normalized_name = self.normalize_name(
            medicine_name
        )

        base_results = (
            self.base_medicine_lookup.get(
                normalized_name,
                []
            )
        )

        if base_results:

            return {
                "verified": True,
                "match_type":
                    "base_medicine",
                "medicine_name":
                    normalized_name,
                "records":
                    base_results
            }

        # --------------------------------------------------------
        # 5. Not found
        # --------------------------------------------------------

        return {
            "verified": False,
            "match_type": "none",
            "medicine_name":
                medicine_name,
            "records": []
        }

    # ============================================================
    # GET VERIFIED MEDICINES
    # ============================================================

    def get_verified_medicines(
        self,
        prescription
    ):

        verified_medicines = []
        unverified_medicines = []

        for medicine in prescription.medicines:

            if not medicine.medicine_name:

                unverified_medicines.append({
                    "extracted_name": None,
                    "reason":
                        "Medicine name could not be extracted."
                })

                continue

            result = self.verify_medicine(
                medicine.medicine_name
            )

            if result["verified"]:

                verified_medicines.append({

                    "extracted_name":
                        medicine.medicine_name,

                    "verified_name":
                        result["medicine_name"],

                    "match_type":
                        result["match_type"],

                    "prescription_data":
                        medicine.model_dump(),

                    "database_records":
                        result["records"]
                })

            else:

                unverified_medicines.append({

                    "extracted_name":
                        medicine.medicine_name,

                    "reason":
                        "Not found in medicine dataset."
                })

        return (
            verified_medicines,
            unverified_medicines
        )