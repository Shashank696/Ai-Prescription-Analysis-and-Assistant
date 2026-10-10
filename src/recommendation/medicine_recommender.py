import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


TIER_SAME_PRODUCT = "same_product"
TIER_POSSIBLE_EQUIVALENT = "possible_equivalent"
TIER_WEAKER_MATCH = "weaker_match"

CHECK_ALIGNED = "aligned"
CHECK_CONFLICT = "conflict"
CHECK_MISSING = "missing"


class MedicineRecommendationEngine:
    """Find dataset product candidates for an already identified product."""

    FORM_MARKERS = (
        "oral tablet",
        "oral capsule",
        "oral suspension",
        "oral solution",
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
    )
    RELEASE_MARKERS = (
        ("extended release", "extended_release"),
        ("12 hr", "extended_release"),
        ("24 hr", "extended_release"),
        ("delayed release", "delayed_release"),
    )
    STRENGTH_COMPONENT = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|%)"
        r"(?:\s*/\s*(\d+(?:\.\d+)?)?\s*(ml|l))?",
        re.IGNORECASE,
    )

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.df = pd.read_csv(self.dataset_path, low_memory=False)
        required_columns = {
            "SETID",
            "drug_name",
            "primary_rxcui",
            "primary_rxstring",
            "ingredient_rxcui",
            "active_moiety_rxcui",
            "available_strength",
            "fda_dosage_form",
        }
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(
                "Recommendation dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        self.records = self.df.where(pd.notna(self.df), None).to_dict("records")

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @classmethod
    def _normalized_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", cls._text(value).lower()).strip()

    @classmethod
    def _identifier_set(cls, value: Any) -> set:
        return {
            identifier.strip()
            for identifier in cls._text(value).split("|")
            if identifier.strip()
        }

    @classmethod
    def _strength_components(
        cls,
        value: Any,
    ) -> Optional[List[Tuple[float, str, Optional[float], Optional[str]]]]:
        text = cls._normalized_text(value)
        if not text:
            return None

        components = []
        consumed = []
        for match in cls.STRENGTH_COMPONENT.finditer(text):
            value_number = float(match.group(1))
            unit = match.group(2).lower()
            denominator_number = (
                float(match.group(3)) if match.group(3) else None
            )
            denominator_unit = match.group(4).lower() if match.group(4) else None
            if denominator_unit == "l":
                denominator_unit = "ml"
            components.append(
                (value_number, unit, denominator_number, denominator_unit)
            )
            consumed.append(match.group(0))

        if not components:
            return None

        remainder = text
        for part in consumed:
            remainder = remainder.replace(part, "", 1)
        if re.sub(r"[\s,/]+", "", remainder):
            return None

        return components

    @classmethod
    def _strength_check(cls, source: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        source_components = cls._strength_components(
            source.get("available_strength")
        )
        candidate_components = cls._strength_components(
            candidate.get("available_strength")
        )
        if source_components is None or candidate_components is None:
            return CHECK_MISSING
        if source_components != candidate_components:
            return CHECK_CONFLICT
        return CHECK_ALIGNED

    @classmethod
    def _form_profile(cls, record: Dict[str, Any]) -> set:
        text = cls._normalized_text(
            f"{cls._text(record.get('drug_name'))} "
            f"{cls._text(record.get('fda_dosage_form'))}"
        )
        return {marker for marker in cls.FORM_MARKERS if marker in text}

    @classmethod
    def _form_check(cls, source: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        source_forms = cls._form_profile(source)
        candidate_forms = cls._form_profile(candidate)
        if not source_forms or not candidate_forms:
            return CHECK_MISSING
        if source_forms.isdisjoint(candidate_forms):
            return CHECK_CONFLICT
        return CHECK_ALIGNED

    @classmethod
    def _release_profile(cls, record: Dict[str, Any]) -> Optional[str]:
        text = cls._normalized_text(
            f"{cls._text(record.get('drug_name'))} "
            f"{cls._text(record.get('fda_dosage_form'))}"
        )
        if not text:
            return None
        labels = {
            label for marker, label in cls.RELEASE_MARKERS if marker in text
        }
        if len(labels) > 1:
            return None
        return next(iter(labels), "standard_release")

    @classmethod
    def _release_check(cls, source: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        source_release = cls._release_profile(source)
        candidate_release = cls._release_profile(candidate)
        if source_release is None or candidate_release is None:
            return CHECK_MISSING
        if source_release != candidate_release:
            return CHECK_CONFLICT
        return CHECK_ALIGNED

    @classmethod
    def _same_product(cls, source: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        source_rxcui = cls._text(source.get("primary_rxcui"))
        candidate_rxcui = cls._text(candidate.get("primary_rxcui"))
        if source_rxcui or candidate_rxcui:
            return bool(source_rxcui and source_rxcui == candidate_rxcui)
        source_rxstring = cls._normalized_text(source.get("primary_rxstring"))
        candidate_rxstring = cls._normalized_text(candidate.get("primary_rxstring"))
        return bool(source_rxstring and source_rxstring == candidate_rxstring)

    @classmethod
    def _candidate_result(
        cls,
        source: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        source_ingredients = cls._identifier_set(source.get("ingredient_rxcui"))
        candidate_ingredients = cls._identifier_set(candidate.get("ingredient_rxcui"))
        source_moieties = cls._identifier_set(source.get("active_moiety_rxcui"))
        candidate_moieties = cls._identifier_set(candidate.get("active_moiety_rxcui"))

        ingredient_match = bool(
            source_ingredients
            and candidate_ingredients
            and source_ingredients == candidate_ingredients
        )
        active_moiety_match = bool(
            source_moieties
            and candidate_moieties
            and source_moieties == candidate_moieties
        )

        if not ingredient_match and not active_moiety_match:
            return None

        checks = {
            "ingredients": CHECK_ALIGNED if ingredient_match else CHECK_CONFLICT,
            "strength": cls._strength_check(source, candidate),
            "dosage_form": cls._form_check(source, candidate),
            "release": cls._release_check(source, candidate),
        }
        reasons = []
        missing = []
        conflicts = []

        if ingredient_match:
            reasons.append("Exact ingredient identifier set matches.")
        else:
            reasons.append(
                "Active-moiety identifiers match, but ingredient identifiers differ."
            )
            conflicts.append("Ingredient identifiers differ.")

        for attribute in ("strength", "dosage_form", "release"):
            result = checks[attribute]
            label = attribute.replace("_", " ")
            if result == CHECK_ALIGNED:
                reasons.append(f"{label.capitalize()} matches.")
            elif result == CHECK_MISSING:
                missing.append(f"{label.capitalize()} could not be compared.")
            else:
                conflicts.append(f"{label.capitalize()} conflicts.")

        if cls._same_product(source, candidate):
            tier = TIER_SAME_PRODUCT
        elif ingredient_match and all(
            checks[attribute] == CHECK_ALIGNED
            for attribute in ("strength", "dosage_form", "release")
        ):
            tier = TIER_POSSIBLE_EQUIVALENT
        else:
            tier = TIER_WEAKER_MATCH

        return {
            "recommendation_tier": tier,
            "record": candidate,
            "match_reasons": reasons,
            "missing_attributes": missing,
            "conflicts": conflicts,
            "attribute_checks": checks,
        }

    def recommend_equivalents(
        self,
        source_record: Dict[str, Any],
        limit: int = 10,
    ) -> Dict[str, Any]:
        source = dict(source_record)
        source_ingredients = self._identifier_set(source.get("ingredient_rxcui"))
        source_strength = self._strength_components(source.get("available_strength"))
        source_forms = self._form_profile(source)
        source_release = self._release_profile(source)

        missing_source_data = []
        if not source_ingredients:
            missing_source_data.append("ingredient identifiers")
        if source_strength is None:
            missing_source_data.append("strength")
        if not source_forms:
            missing_source_data.append("dosage form")
        if source_release is None:
            missing_source_data.append("release characteristic")

        if missing_source_data:
            return {
                "source_product": source,
                "recommendation_status": "insufficient_source_data",
                "candidates": [],
                "message": (
                    "Possible equivalent products cannot be evaluated because "
                    "the source product is missing "
                    + ", ".join(missing_source_data)
                    + "."
                ),
            }

        source_setid = self._text(source.get("SETID"))
        candidates = []
        for candidate in self.records:
            if source_setid and self._text(candidate.get("SETID")) == source_setid:
                continue
            result = self._candidate_result(source, candidate)
            if result is not None:
                candidates.append(result)

        tier_order = {
            TIER_SAME_PRODUCT: 0,
            TIER_POSSIBLE_EQUIVALENT: 1,
            TIER_WEAKER_MATCH: 2,
        }
        candidates.sort(
            key=lambda item: (
                tier_order[item["recommendation_tier"]],
                len(item["conflicts"]),
                len(item["missing_attributes"]),
                self._text(item["record"].get("drug_name")),
            )
        )

        return {
            "source_product": source,
            "recommendation_status": "ready",
            "candidates": candidates[:limit],
            "message": (
                "Candidates are dataset matches only. They are not clinical "
                "substitution advice."
            ),
        }
