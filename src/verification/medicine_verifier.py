import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

VERIFICATION_EXACT = "exact_match"
VERIFICATION_CANDIDATE = "candidate_match"
VERIFICATION_UNCERTAIN = "uncertain"
VERIFICATION_UNMATCHED = "unmatched"

STRENGTH_ALIGNED = "aligned"
STRENGTH_CONFLICT = "conflict"
STRENGTH_MISSING_PRESCRIPTION = "missing_in_prescription"
STRENGTH_MISSING_DATASET = "missing_in_dataset"
STRENGTH_NOT_APPLICABLE = "not_applicable"

CHARACTERISTIC_ALIGNED = "aligned"
CHARACTERISTIC_CONFLICT = "conflict"
CHARACTERISTIC_NOT_CHECKED = "not_checked"


class MedicineVerifier:

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

    DOSAGE_FORM_MARKERS = [
        "oral tablet",
        "oral capsule",
        "oral suspension",
        "oral solution",
        "topical spray",
        "medicated liquid soap",
        "extended release oral tablet",
        "film coated",
        "tablet",
        "capsule",
        "suspension",
        "solution",
        "spray",
        "cream",
        "ointment",
        "gel",
        "injection",
        "injectable",
    ]

    RELEASE_MARKERS = [
        ("extended release", "extended_release"),
        ("12 hr", "extended_release"),
        ("24 hr", "extended_release"),
        ("delayed release", "delayed_release"),
    ]

    STRENGTH_COMPONENT_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|l|%)",
        re.IGNORECASE,
    )

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

    def load_dataset(self):

        print("Loading medicine dataset...")

        self.df = pd.read_csv(
            self.dataset_path,
            low_memory=False,
        )

        if "drug_name" not in self.df.columns:
            raise ValueError(
                "Dataset must contain 'drug_name' column."
            )

        self.df = self.df[
            self.df["drug_name"].notna()
        ].copy()

    @staticmethod
    def _record_to_dict(record: Any) -> Dict[str, Any]:
        if record is None:
            return {}
        if isinstance(record, dict):
            return record
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(record)

    @staticmethod
    def _unique_records(records: List[Any]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for record in records:
            row = MedicineVerifier._record_to_dict(record)
            primary_rxcui = row.get("primary_rxcui")
            has_primary_rxcui = primary_rxcui is not None and not (
                isinstance(primary_rxcui, float) and pd.isna(primary_rxcui)
            ) and str(primary_rxcui).strip()
            if has_primary_rxcui:
                product_fields = (
                    "drug_name",
                    "primary_rxstring",
                    "available_strength",
                    "fda_proprietary_name",
                    "fda_dosage_form",
                    "ingredient_rxcui",
                    "active_moiety_rxcui",
                )
                key = (
                    "rxnorm_product",
                    str(primary_rxcui).strip(),
                    tuple(str(row.get(field) or "").strip() for field in product_fields),
                )
            else:
                key = ("label_record", row.get("SETID") or row.get("drug_name"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    def normalize_name(self, name: str) -> str:

        if name is None:
            return ""

        name = str(name).lower().strip()

        name = re.sub(
            r"[^a-z0-9\s/]",
            " ",
            name,
        )

        name = re.sub(
            r"\s+",
            " ",
            name,
        )

        return name.strip()

    def extract_search_name(self, drug_name: str) -> str:

        if not drug_name:
            return ""

        name = str(drug_name).lower().strip()

        name = re.sub(
            r"\[[^\]]*\]",
            "",
            name,
        )

        name = re.sub(
            r"^\d+(?:\.\d+)?\s*"
            r"(?:ml|l|mg|g|mcg|µg)\s+",
            "",
            name,
        )

        name = re.sub(
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:mg|mcg|µg|g|kg|ml|l)"
            r"(?:\s*/\s*"
            r"(?:\d+(?:\.\d+)?\s*)?"
            r"(?:mg|mcg|µg|g|kg|ml|l))?\b",
            "",
            name,
        )

        dosage_forms = [
            "oral tablet",
            "oral capsule",
            "oral solution",
            "oral suspension",
            "extended release",
            "delayed release",
            "immediate release",
            "12 hr",
            "24 hr",
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
                name,
            )

        name = re.sub(
            r"\s+",
            " ",
            name,
        ).strip()

        return name.strip("/-,")

    def extract_base_medicine_name(
        self,
        medicine_name: str,
    ) -> str:

        name = self.normalize_name(
            medicine_name,
        )

        if not name:
            return ""

        if "/" in name:
            return ""

        parts = name.split()

        if len(parts) == 1:
            return parts[0]

        if len(parts) == 2:
            if parts[1] in self.SALT_FORMS:
                return parts[0]

        return ""

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

        for _, row in self.df.iterrows():
            row_dict = row.to_dict()

            normalized_drug_name = row_dict["normalized_drug_name"]
            if normalized_drug_name:
                self.drug_name_lookup.setdefault(
                    normalized_drug_name,
                    [],
                ).append(row_dict)

            normalized_rxstring = row_dict["normalized_rxstring"]
            if normalized_rxstring:
                self.rxstring_lookup.setdefault(
                    normalized_rxstring,
                    [],
                ).append(row_dict)

            normalized_fda_name = row_dict["normalized_fda_name"]
            if normalized_fda_name:
                self.fda_name_lookup.setdefault(
                    normalized_fda_name,
                    [],
                ).append(row_dict)

            search_name = self.extract_search_name(
                row["drug_name"],
            )

            if search_name:
                normalized = self.normalize_name(
                    search_name,
                )

                self.medicine_name_lookup.setdefault(
                    normalized,
                    [],
                ).append(
                    row_dict,
                )

                base_name = self.extract_base_medicine_name(
                    normalized,
                )

                if base_name:
                    self.base_medicine_lookup.setdefault(
                        base_name,
                        [],
                    ).append(
                        row_dict,
                    )

            drug_name = str(row["drug_name"])
            brand_match = re.search(
                r"\[([^\]]+)\]",
                drug_name,
            )

            if brand_match:
                brand_name = self.normalize_name(
                    brand_match.group(1),
                )

                if brand_name:
                    self.brand_name_lookup.setdefault(
                        brand_name,
                        [],
                    ).append(
                        row_dict,
                    )

    def clean_extracted_medicine_name(
        self,
        medicine_name: str,
    ) -> str:
        if not medicine_name:
            return ""

        name = self.normalize_name(medicine_name)

        name = re.sub(
            r"^(tab|tablet|cap|capsule|inj|injection|syr|syrup)\.?\s+",
            "",
            name,
        )

        return name.strip()

    def search_medicine_name(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        query = self.normalize_name(
            query,
        )

        return [
            self._record_to_dict(record)
            for record in self.medicine_name_lookup.get(
                query,
                [],
            )
        ]

    def _exact_line_records(
        self,
        medicine_name: str,
    ) -> List[Dict[str, Any]]:

        query = self.normalize_name(
            medicine_name,
        )

        if not query:
            return []

        records = []

        if query in self.drug_name_lookup:
            records.extend(
                self.drug_name_lookup[query],
            )

        if query in self.rxstring_lookup:
            records.extend(
                self.rxstring_lookup[query],
            )

        return self._unique_records(records)

    def _fda_proprietary_records(
        self,
        medicine_name: str,
    ) -> List[Dict[str, Any]]:

        query = self.normalize_name(
            medicine_name,
        )

        if not query:
            return []

        if query in self.fda_name_lookup:
            return self._unique_records(
                self.fda_name_lookup[query],
            )

        return []

    def parse_strength_components(
        self,
        text: Optional[str],
    ) -> List[Tuple[float, str]]:

        if text is None or (
            isinstance(text, float) and pd.isna(text)
        ):
            return []

        components = []

        for match in self.STRENGTH_COMPONENT_PATTERN.finditer(
            str(text),
        ):
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == "l":
                unit = "ml"
            components.append((value, unit))

        return components

    def compare_strength(
        self,
        prescribed: Optional[str],
        available: Any,
    ) -> str:

        if prescribed is None or not str(prescribed).strip():
            return STRENGTH_MISSING_PRESCRIPTION

        if available is None or (
            isinstance(available, float) and pd.isna(available)
        ) or not str(available).strip():
            return STRENGTH_MISSING_DATASET

        prescribed_components = self.parse_strength_components(
            prescribed,
        )
        available_components = self.parse_strength_components(
            available,
        )

        if not prescribed_components:
            return STRENGTH_NOT_APPLICABLE

        if not available_components:
            return STRENGTH_MISSING_DATASET

        if len(prescribed_components) != len(
            available_components,
        ):
            return STRENGTH_CONFLICT

        for (p_value, p_unit), (a_value, a_unit) in zip(
            prescribed_components,
            available_components,
        ):
            if p_unit != a_unit:
                return STRENGTH_CONFLICT
            if abs(p_value - a_value) > 0.01:
                return STRENGTH_CONFLICT

        return STRENGTH_ALIGNED

    def _release_profile(
        self,
        text: str,
    ) -> set:

        normalized = self.normalize_name(text)
        profile = set()

        for marker, label in self.RELEASE_MARKERS:
            if marker in normalized:
                profile.add(label)

        if re.search(r"\ber\b", normalized):
            profile.add("extended_release")

        if re.search(r"\bxr\b", normalized):
            profile.add("extended_release")

        return profile

    def _dosage_form_profile(
        self,
        text: str,
        fda_dosage_form: Any = None,
    ) -> set:

        normalized = self.normalize_name(text)
        profile = set()

        for marker in self.DOSAGE_FORM_MARKERS:
            if marker in normalized:
                profile.add(marker)

        if fda_dosage_form is not None and not (
            isinstance(fda_dosage_form, float)
            and pd.isna(fda_dosage_form)
        ):
            fda_normalized = self.normalize_name(
                str(fda_dosage_form),
            )
            for marker in self.DOSAGE_FORM_MARKERS:
                if marker in fda_normalized:
                    profile.add(marker)

        return profile

    def _characteristic_checks(
        self,
        query_text: str,
        record: Dict[str, Any],
    ) -> Tuple[str, str]:

        drug_name = str(
            record.get("drug_name") or "",
        )
        fda_form = record.get("fda_dosage_form")

        query_release = self._release_profile(query_text)
        record_release = self._release_profile(drug_name)

        release_check = CHARACTERISTIC_NOT_CHECKED
        if query_release:
            release_check = CHARACTERISTIC_ALIGNED
            if query_release != record_release:
                release_check = CHARACTERISTIC_CONFLICT

        query_forms = self._dosage_form_profile(query_text)
        record_forms = self._dosage_form_profile(
            drug_name,
            fda_form,
        )

        form_check = CHARACTERISTIC_NOT_CHECKED
        if query_forms and record_forms:
            form_check = CHARACTERISTIC_ALIGNED
            if query_forms.isdisjoint(record_forms):
                form_check = CHARACTERISTIC_CONFLICT

        return form_check, release_check

    def _strength_for_record(
        self,
        prescribed_strength: Optional[str],
        medicine_name: str,
        record: Dict[str, Any],
    ) -> str:

        available = record.get("available_strength")

        if prescribed_strength and str(
            prescribed_strength,
        ).strip():
            return self.compare_strength(
                prescribed_strength,
                available,
            )

        embedded = self.parse_strength_components(
            medicine_name,
        )
        if embedded:
            return self.compare_strength(
                medicine_name,
                available,
            )

        return STRENGTH_MISSING_PRESCRIPTION

    def _filter_records_by_checks(
        self,
        records: List[Dict[str, Any]],
        medicine_name: str,
        prescribed_strength: Optional[str],
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        str,
        str,
    ]:

        records = self._unique_records(records)

        if not records:
            return (
                [],
                [],
                STRENGTH_NOT_APPLICABLE,
                CHARACTERISTIC_NOT_CHECKED,
                CHARACTERISTIC_NOT_CHECKED,
            )

        aligned = []
        strength_results = []
        form_results = []
        release_results = []

        for record in records:
            form_check, release_check = self._characteristic_checks(
                medicine_name,
                record,
            )
            form_results.append(form_check)
            release_results.append(release_check)

            if CHARACTERISTIC_CONFLICT in (
                form_check,
                release_check,
            ):
                continue

            strength_result = self._strength_for_record(
                prescribed_strength,
                medicine_name,
                record,
            )
            strength_results.append(strength_result)

            if strength_result in (
                STRENGTH_ALIGNED,
                STRENGTH_MISSING_PRESCRIPTION,
                STRENGTH_MISSING_DATASET,
                STRENGTH_NOT_APPLICABLE,
            ):
                aligned.append(record)

        overall_strength = STRENGTH_NOT_APPLICABLE
        if prescribed_strength and str(
            prescribed_strength,
        ).strip():
            if any(
                result == STRENGTH_ALIGNED
                for result in strength_results
            ):
                overall_strength = STRENGTH_ALIGNED
            elif any(
                result == STRENGTH_CONFLICT
                for result in strength_results
            ):
                overall_strength = STRENGTH_CONFLICT
            else:
                overall_strength = STRENGTH_MISSING_DATASET
        elif strength_results:
            if all(
                result == STRENGTH_MISSING_PRESCRIPTION
                for result in strength_results
            ):
                overall_strength = STRENGTH_MISSING_PRESCRIPTION
            elif any(
                result == STRENGTH_ALIGNED
                for result in strength_results
            ):
                overall_strength = STRENGTH_ALIGNED

        overall_form = CHARACTERISTIC_NOT_CHECKED
        if CHARACTERISTIC_CONFLICT in form_results:
            overall_form = CHARACTERISTIC_CONFLICT
        elif CHARACTERISTIC_ALIGNED in form_results:
            overall_form = CHARACTERISTIC_ALIGNED

        overall_release = CHARACTERISTIC_NOT_CHECKED
        if CHARACTERISTIC_CONFLICT in release_results:
            overall_release = CHARACTERISTIC_CONFLICT
        elif CHARACTERISTIC_ALIGNED in release_results:
            overall_release = CHARACTERISTIC_ALIGNED

        return (
            aligned,
            records,
            overall_strength,
            overall_form,
            overall_release,
        )

    def _build_result(
        self,
        verification_status: str,
        match_type: str,
        matched_name: Optional[str],
        records: List[Any],
        strength_check: str,
        dosage_form_check: str,
        release_check: str,
        message: str,
    ) -> Dict[str, Any]:

        record_dicts = self._unique_records(records)

        return {
            "verification_status": verification_status,
            "match_type": match_type,
            "matched_name": matched_name,
            "medicine_name": matched_name,
            "verified": verification_status == VERIFICATION_EXACT,
            "records": record_dicts,
            "candidate_count": len(record_dicts),
            "strength_check": strength_check,
            "dosage_form_check": dosage_form_check,
            "release_check": release_check,
            "message": message,
        }

    def verify_medicine(
        self,
        medicine_name: str,
        strength: Optional[str] = None,
    ) -> Dict[str, Any]:

        if medicine_name is None or not str(
            medicine_name,
        ).strip():
            return self._build_result(
                VERIFICATION_UNMATCHED,
                "none",
                None,
                [],
                STRENGTH_NOT_APPLICABLE,
                CHARACTERISTIC_NOT_CHECKED,
                CHARACTERISTIC_NOT_CHECKED,
                "Medicine name could not be extracted.",
            )

        cleaned = self.clean_extracted_medicine_name(
            medicine_name,
        )

        line_records = self._exact_line_records(
            cleaned,
        )
        if not line_records:
            line_records = self._exact_line_records(
                medicine_name,
            )

        if line_records:
            (
                aligned,
                all_records,
                strength_check,
                dosage_form_check,
                release_check,
            ) = (
                self._filter_records_by_checks(
                    line_records,
                    medicine_name,
                    strength,
                )
            )

            conflicts = []
            if strength_check == STRENGTH_CONFLICT:
                conflicts.append("strength")
            if dosage_form_check == CHARACTERISTIC_CONFLICT:
                conflicts.append("dosage form")
            if release_check == CHARACTERISTIC_CONFLICT:
                conflicts.append("release characteristic")

            if conflicts:
                return self._build_result(
                    VERIFICATION_UNCERTAIN,
                    "full_line_conflict",
                    self.normalize_name(cleaned or medicine_name),
                    all_records,
                    strength_check,
                    dosage_form_check,
                    release_check,
                    "The product name matches the dataset, but the "
                    "prescription conflicts with its "
                    f"{', '.join(conflicts)}.",
                )

            if len(aligned) == 1 and strength_check in (
                STRENGTH_ALIGNED,
                STRENGTH_MISSING_PRESCRIPTION,
            ):
                if strength_check == STRENGTH_MISSING_PRESCRIPTION:
                    embedded = self.parse_strength_components(
                        medicine_name,
                    )
                    if not embedded:
                        return self._build_result(
                            VERIFICATION_UNCERTAIN,
                            "full_line_missing_strength",
                            self.normalize_name(
                                cleaned or medicine_name,
                            ),
                            all_records,
                            strength_check,
                            dosage_form_check,
                            release_check,
                            "The prescription line matches a dataset "
                            "product name, but strength could not be "
                            "confirmed from the available fields.",
                        )

                return self._build_result(
                    VERIFICATION_EXACT,
                    "full_line",
                    self.normalize_name(
                        cleaned or medicine_name,
                    ),
                    aligned,
                    strength_check,
                    dosage_form_check,
                    release_check,
                    "The extracted medicine line matches a dataset "
                    "product without conflicting strength, form, or "
                    "release characteristics.",
                )

            if aligned:
                return self._build_result(
                    VERIFICATION_CANDIDATE,
                    "full_line_multiple",
                    self.normalize_name(
                        cleaned or medicine_name,
                    ),
                    aligned,
                    strength_check,
                    dosage_form_check,
                    release_check,
                    "Multiple dataset products match this medicine "
                    "line; further confirmation is required.",
                )

        ingredient_query = self.extract_search_name(
            cleaned or medicine_name,
        )
        ingredient_records = self.search_medicine_name(
            ingredient_query,
        ) if ingredient_query else []

        brand_records = self.brand_name_lookup.get(
            cleaned,
            [],
        )
        brand_records = self._unique_records(brand_records)

        fda_records = self._fda_proprietary_records(
            cleaned or medicine_name,
        )

        base_name = self.extract_base_medicine_name(
            cleaned or medicine_name,
        )
        base_records = self.base_medicine_lookup.get(
            base_name,
            [],
        ) if base_name else []
        base_records = self._unique_records(base_records)

        if brand_records:
            (
                _,
                _,
                strength_check,
                dosage_form_check,
                release_check,
            ) = self._filter_records_by_checks(
                brand_records,
                medicine_name,
                strength,
            )
            status = VERIFICATION_CANDIDATE
            message = (
                "The extracted name matches a brand name in the "
                "dataset. This is a possible match and requires "
                "confirmation against strength and product form."
            )
            if strength_check == STRENGTH_CONFLICT:
                status = VERIFICATION_UNCERTAIN
                message = (
                    "The brand name matches the dataset, but the "
                    "prescribed strength does not align with the "
                    "available brand products."
                )
            elif CHARACTERISTIC_CONFLICT in (
                dosage_form_check,
                release_check,
            ):
                status = VERIFICATION_UNCERTAIN
                message = (
                    "The brand name matches the dataset, but the "
                    "prescription conflicts with the available product "
                    "form or release characteristic."
                )
            return self._build_result(
                status,
                "brand_name",
                cleaned,
                brand_records,
                strength_check,
                dosage_form_check,
                release_check,
                message,
            )

        if ingredient_records:
            (
                aligned,
                all_ingredient,
                strength_check,
                dosage_form_check,
                release_check,
            ) = (
                self._filter_records_by_checks(
                    ingredient_records,
                    medicine_name,
                    strength,
                )
            )
            has_strength = strength is not None and str(
                strength,
            ).strip()

            if not aligned and (
                strength_check == STRENGTH_CONFLICT
                or dosage_form_check == CHARACTERISTIC_CONFLICT
                or release_check == CHARACTERISTIC_CONFLICT
            ):
                conflict_parts = []
                if strength_check == STRENGTH_CONFLICT:
                    conflict_parts.append("strength")
                if dosage_form_check == CHARACTERISTIC_CONFLICT:
                    conflict_parts.append("dosage form")
                if release_check == CHARACTERISTIC_CONFLICT:
                    conflict_parts.append("release characteristic")
                return self._build_result(
                    VERIFICATION_UNCERTAIN,
                    "medicine_name_conflict",
                    self.normalize_name(ingredient_query),
                    all_ingredient,
                    strength_check,
                    dosage_form_check,
                    release_check,
                    "Ingredient-style matches were found, but the "
                    "prescription conflicts with the available "
                    f"product {', '.join(conflict_parts)}.",
                )

            if has_strength and len(aligned) == 1 and (
                strength_check == STRENGTH_ALIGNED
            ):
                return self._build_result(
                    VERIFICATION_EXACT,
                    "ingredient_with_strength",
                    self.normalize_name(ingredient_query),
                    aligned,
                    strength_check,
                    dosage_form_check,
                    release_check,
                    "The generic name and prescribed strength align "
                    "with one dataset product.",
                )

            records = aligned or all_ingredient
            status = VERIFICATION_CANDIDATE
            message = (
                "The extracted name matches ingredient-style dataset "
                "entries. This is a possible match, not proof of a "
                "specific product."
            )

            if not has_strength:
                status = VERIFICATION_UNCERTAIN
                message = (
                    "The extracted generic or ingredient name matches "
                    "multiple possible products; strength and form were "
                    "not sufficient to identify one product."
                )
            elif len(aligned) > 1:
                status = VERIFICATION_CANDIDATE
                message = (
                    "Multiple dataset products match this generic name "
                    "and prescribed strength; further confirmation is "
                    "required."
                )

            return self._build_result(
                status,
                "medicine_name",
                self.normalize_name(ingredient_query),
                records,
                strength_check,
                dosage_form_check,
                release_check,
                message,
            )

        if fda_records:
            (
                _,
                _,
                strength_check,
                dosage_form_check,
                release_check,
            ) = self._filter_records_by_checks(
                fda_records,
                medicine_name,
                strength,
            )
            return self._build_result(
                VERIFICATION_CANDIDATE,
                "fda_proprietary_name",
                cleaned,
                fda_records,
                strength_check,
                dosage_form_check,
                release_check,
                "The extracted name matches an FDA proprietary name "
                "in the dataset, not a specific product identity.",
            )

        if base_records:
            (
                _,
                _,
                strength_check,
                dosage_form_check,
                release_check,
            ) = self._filter_records_by_checks(
                base_records,
                medicine_name,
                strength,
            )
            return self._build_result(
                VERIFICATION_UNCERTAIN,
                "base_medicine",
                base_name,
                base_records,
                strength_check,
                dosage_form_check,
                release_check,
                "The extracted name matches a base ingredient in the "
                "dataset, but product strength and form are not "
                "confirmed.",
            )

        return self._build_result(
            VERIFICATION_UNMATCHED,
            "none",
            self.normalize_name(cleaned or medicine_name),
            [],
            STRENGTH_NOT_APPLICABLE,
            CHARACTERISTIC_NOT_CHECKED,
            CHARACTERISTIC_NOT_CHECKED,
            "No matching medicine was found in the dataset.",
        )

    def _build_verification_item(
        self,
        medicine,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        matched_name = result.get("matched_name")

        return {
            "extracted_name": medicine.medicine_name,
            "matched_name": matched_name,
            "verified_name": matched_name,
            "verification_status": result["verification_status"],
            "match_type": result["match_type"],
            "strength_check": result["strength_check"],
            "dosage_form_check": result["dosage_form_check"],
            "release_check": result["release_check"],
            "message": result["message"],
            "prescription_data": medicine.model_dump(),
            "database_records": result["records"],
            "candidate_count": result["candidate_count"],
        }

    def get_verified_medicines(
        self,
        prescription,
    ):
        exact_matches = []
        candidate_matches = []
        uncertain_matches = []
        unmatched_medicines = []

        for medicine in prescription.medicines:

            if not medicine.medicine_name:

                unmatched_medicines.append({
                    "extracted_name": None,
                    "matched_name": None,
                    "verified_name": None,
                    "verification_status": VERIFICATION_UNMATCHED,
                    "match_type": "none",
                    "strength_check": STRENGTH_NOT_APPLICABLE,
                    "dosage_form_check": CHARACTERISTIC_NOT_CHECKED,
                    "release_check": CHARACTERISTIC_NOT_CHECKED,
                    "message": (
                        "Medicine name could not be extracted."
                    ),
                    "reason": (
                        "Medicine name could not be extracted."
                    ),
                    "prescription_data": medicine.model_dump(),
                    "database_records": [],
                    "candidate_count": 0,
                })
                continue

            result = self.verify_medicine(
                medicine.medicine_name,
                medicine.strength,
            )

            item = self._build_verification_item(
                medicine,
                result,
            )

            status = result["verification_status"]

            if status == VERIFICATION_EXACT:
                exact_matches.append(item)
            elif status == VERIFICATION_CANDIDATE:
                candidate_matches.append(item)
            elif status == VERIFICATION_UNCERTAIN:
                uncertain_matches.append(item)
            else:
                item["reason"] = result["message"]
                unmatched_medicines.append(item)

        return (
            exact_matches,
            candidate_matches,
            uncertain_matches,
            unmatched_medicines,
        )
