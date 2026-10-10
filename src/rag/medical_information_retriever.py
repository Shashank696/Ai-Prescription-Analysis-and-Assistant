"""Deterministic retrieval over supplied MedlinePlus-style records."""

import re


FIELD_NAMES = ("uses", "precautions", "side_effects")
STOP_WORDS = {
    "a", "about", "an", "and", "are", "for", "how", "i", "in", "is",
    "me", "of", "on", "please", "tell", "the", "to", "what", "with",
}
FIELD_INTENTS = {
    "uses": {"use", "uses", "used", "treat", "treats", "treatment", "indication", "purpose"},
    "precautions": {"avoid", "before", "caution", "cautions", "contraindication", "interaction", "precaution", "precautions", "warning", "warnings"},
    "side_effects": {"adverse", "cause", "caused", "causes", "effect", "effects", "side", "symptom", "symptoms"},
}
UNSUPPORTED_DECISION_TERMS = {"discontinue", "discontinued", "discontinuing", "stop"}


def normalize_text(value):
    """Normalize case, whitespace, and punctuation for deterministic matching."""

    if value is None:
        return ""
    return " ".join(re.sub(r"[^\w]+", " ", str(value)).casefold().split())


def _tokens(value):
    return set(normalize_text(value).split())


def _has_value(value):
    return bool(normalize_text(value))


def _is_unsupported_decision_request(question_tokens):
    """Identify clear treatment-change requests without blocking dosage information."""

    if question_tokens & UNSUPPORTED_DECISION_TERMS:
        return True
    if "skip" in question_tokens and {"dose", "doses", "medication", "medicine"} & question_tokens:
        return True
    if "double" in question_tokens and {"dose", "doses"} & question_tokens:
        return True
    if "extra" in question_tokens and {"dose", "doses"} & question_tokens:
        return True
    if {"more", "less"} & question_tokens and "take" in question_tokens:
        return True
    if {"increase", "decrease", "change"} & question_tokens and {"dose", "doses", "medication", "medicine"} & question_tokens:
        return True
    if {"replace", "substitute", "switch"} & question_tokens:
        return True
    return False


def _contains_phrase(question, medicine_name):
    return f" {medicine_name} " in f" {question} "


def _identified_medicines(question, records):
    candidates = []
    for record in records:
        normalized_name = normalize_text(record.get("medicine_name"))
        if normalized_name and _contains_phrase(question, normalized_name):
            candidates.append((normalized_name, record))

    # A fully stated combination product takes precedence over a contained
    # single-ingredient record, for example "abacavir ... lamivudine" over
    # "abacavir" when both are supplied records.
    names = {name for name, _ in candidates}
    return [
        (name, record)
        for name, record in candidates
        if not any(name != other and _contains_phrase(other, name) for other in names)
    ]


def _requested_fields(question_tokens):
    requested = [
        field_name
        for field_name, intent_terms in FIELD_INTENTS.items()
        if question_tokens & intent_terms
    ]
    return requested or list(FIELD_NAMES)


def _result(status, message, records=None):
    return {
        "retrieval_status": status,
        "message": message,
        "records": records or [],
    }


def retrieve_medical_information(question, records, limit=3):
    """Retrieve relevant MedlinePlus fields from the supplied records only.

    The function never infers aliases or medicine equivalence. It returns no
    retrieved records when a medicine is absent, ambiguous, clinically
    decision-oriented, or unsupported by a non-empty relevant dataset field.
    """

    normalized_question = normalize_text(question)
    if not normalized_question:
        return _result(
            "insufficient_evidence",
            "Enter a medicine-specific question to search the available medical information.",
        )

    identified = _identified_medicines(normalized_question, records)
    unique_names = sorted({name for name, _ in identified})
    if not unique_names:
        return _result(
            "insufficient_evidence",
            "No medicine name from the available dataset was identified in the question.",
        )
    if len(unique_names) > 1:
        return _result(
            "ambiguous",
            "More than one medicine was identified. Ask about one medicine at a time.",
        )

    question_tokens = _tokens(normalized_question)
    if _is_unsupported_decision_request(question_tokens):
        return _result(
            "insufficient_evidence",
            "The dataset retrieval cannot support dose, stopping, switching, or substitution decisions.",
        )

    matched_name = unique_names[0]
    name_tokens = _tokens(matched_name)
    content_tokens = question_tokens - name_tokens - STOP_WORDS
    requested_fields = _requested_fields(question_tokens)
    retrieved = []

    for _, record in identified:
        field_values = {
            field_name: record.get(field_name)
            for field_name in requested_fields
            if _has_value(record.get(field_name))
        }
        if not field_values:
            continue

        overlap = sum(
            len(content_tokens & _tokens(value))
            for value in field_values.values()
        )
        has_requested_intent = requested_fields != list(FIELD_NAMES)
        evidence_relevant = has_requested_intent or overlap > 0
        if not evidence_relevant:
            continue

        retrieved.append({
            "matched_medicine_name": record.get("medicine_name"),
            "retrieved_field_names": list(field_values),
            "retrieved_fields": field_values,
            "source": record.get("source") or "Dataset record",
            "retrieval_score": 100 + (10 * overlap) + (5 * len(field_values)),
            "retrieval_reason": (
                "Explicit medicine-name match; requested fields contain available dataset information."
                if has_requested_intent
                else "Explicit medicine-name match with overlapping question terms in available dataset fields."
            ),
            "evidence_relevant": True,
        })

    if not retrieved:
        return _result(
            "insufficient_evidence",
            "The matched medicine record does not contain relevant information for this question.",
        )

    retrieved.sort(
        key=lambda item: (
            -item["retrieval_score"],
            normalize_text(item["matched_medicine_name"]),
            normalize_text(item["source"]),
        )
    )
    return _result(
        "ready",
        "Retrieved dataset information for the identified medicine.",
        retrieved[:max(0, limit)],
    )
