"""
FHIR JSON anonymization handler.

Strategy: pure structural field replacement — no NER model invoked.

FHIR field names carry explicit semantics defined by the specification.
We map each PII-bearing field directly to its entity type and replace the
value with a typed placeholder.  Free-form narrative fields (div, comment)
are wholly replaced with [REDACTED] since their content cannot be typed
precisely but is known to be patient-level information.

For unstructured clinical prose that requires NER, use /api/v1/anonymize/text.
"""

from typing import Any, Dict, List, Set, Tuple

# ─── Field registries ────────────────────────────────────────────────────────

# Field name → entity_type: structurally certain PII → direct typed replacement.
_DIRECT_PII: Dict[str, str] = {
    "birthDate":        "date_of_birth",
    "deceasedDate":     "date_of_birth",
    "deceasedDateTime": "date_of_birth",
    "family":    "name",
    "prefix":    "name",
    "suffix":    "name",
    "city":      "address",
    "state":     "address",
    "district":  "address",
    "postalCode": "postal_code",
    "country":   "country",
    "display":   "name",   # Reference.display (always a person/org name)
}

# String-array fields: each element is typed PII.
_ARRAY_PII: Dict[str, str] = {
    "given": "name",    # HumanName.given
    "line":  "address", # Address.line
}

# Free-form narrative fields: content is patient-level but type is unknown.
# Replaced wholly with [REDACTED] — no NER needed or performed.
_REDACT_FIELDS: Set[str] = {"div", "comment"}

# ─── Public API ──────────────────────────────────────────────────────────────


def anonymize_fhir(
    resource: Any,
    model_key: str,       # unused — no NER; kept for API compatibility
    min_confidence: float,  # unused — no NER; kept for API compatibility
) -> Tuple[Any, List[Dict]]:
    """Anonymize PII fields in a FHIR resource via pure structural replacement.

    No NER inference is performed.
    Returns (anonymized_resource, deduplicated_entities).
    """
    collector: List[Dict] = []
    result = _walk(resource, collector)

    seen: set = set()
    unique: List[Dict] = []
    for e in collector:
        key = (e["entity_type"], e["text"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return result, unique


# ─── Internal ────────────────────────────────────────────────────────────────


def _walk(obj: Any, collector: List[Dict]) -> Any:
    if isinstance(obj, dict):
        return _walk_dict(obj, collector)
    if isinstance(obj, list):
        return [_walk(item, collector) for item in obj]
    return obj


def _walk_dict(obj: dict, collector: List[Dict]) -> dict:
    result = {}
    for key, value in obj.items():
        if key in _DIRECT_PII:
            result[key] = _replace(value, _DIRECT_PII[key], collector)
        elif key in _ARRAY_PII:
            result[key] = _replace_list(value, _ARRAY_PII[key], collector)
        elif key == "value":
            # ContactPoint.value: type inferred from URI scheme (tel: → phone, mailto: → email).
            result[key] = _contact_value(value, collector)
        elif key == "text":
            # HumanName.text / Address.text → string [NAME].
            # Narrative.text → dict containing 'div' → recurse so div gets redacted.
            result[key] = _text_field(value, collector)
        elif key in _REDACT_FIELDS:
            result[key] = _redact_whole(value)
        else:
            result[key] = _walk(value, collector) if isinstance(value, (dict, list)) else value
    return result


# ─── Replacement helpers ──────────────────────────────────────────────────────


def _replace(value: Any, entity_type: str, collector: List[Dict]) -> Any:
    if isinstance(value, str) and value:
        collector.append({"entity_type": entity_type, "text": value, "start": 0, "end": len(value), "score": 1.0})
        return f"[{entity_type.upper()}]"
    if isinstance(value, list):
        return [_replace(item, entity_type, collector) for item in value]
    return value


def _replace_list(value: Any, entity_type: str, collector: List[Dict]) -> Any:
    if isinstance(value, list):
        return [_replace(item, entity_type, collector) for item in value]
    return _replace(value, entity_type, collector)


def _contact_value(value: Any, collector: List[Dict]) -> Any:
    if isinstance(value, str) and value:
        entity_type = "email" if value.lower().startswith("mailto:") else "phone_number"
        collector.append({"entity_type": entity_type, "text": value, "start": 0, "end": len(value), "score": 1.0})
        return f"[{entity_type.upper()}]"
    if isinstance(value, list):
        return [_contact_value(item, collector) for item in value]
    return value


def _text_field(value: Any, collector: List[Dict]) -> Any:
    """HumanName.text / Address.text are strings → [NAME].
    Narrative.text is a dict → recurse so its 'div' gets redacted."""
    if isinstance(value, str) and value:
        collector.append({"entity_type": "name", "text": value, "start": 0, "end": len(value), "score": 1.0})
        return "[NAME]"
    if isinstance(value, (dict, list)):
        return _walk(value, collector)
    return value


def _redact_whole(value: Any) -> Any:
    """Replace a free-form text field wholesale with [REDACTED]."""
    if isinstance(value, str):
        return "[REDACTED]"
    if isinstance(value, list):
        return ["[REDACTED]" if isinstance(item, str) else item for item in value]
    return value
