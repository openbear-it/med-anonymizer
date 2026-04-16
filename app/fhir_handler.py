"""
FHIR JSON anonymization handler.

Strategy: recursively walk the FHIR resource tree; for every string value
call the NER anonymizer and replace PII with placeholders.
Non-string values (numbers, booleans, null, nested objects/arrays) are
traversed but left structurally intact.
"""

from typing import Any, Dict, List, Tuple

from .anonymizer import anonymizer


def anonymize_fhir(
    resource: Any,
    model_key: str,
    min_confidence: float,
) -> Tuple[Any, List[Dict]]:
    """Anonymize all string fields in a FHIR resource recursively.

    Returns (anonymized_resource, deduplicated_entities).
    """
    collector: List[Dict] = []
    result = _walk(resource, model_key, min_confidence, collector)

    # Deduplicate entities by (type, text) to avoid inflating counts
    # from repeated identical values (e.g. patient name in multiple fields).
    seen: set = set()
    unique: List[Dict] = []
    for e in collector:
        key = (e["entity_type"], e["text"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return result, unique


# ─── Internal ────────────────────────────────────────────────────────────────


def _walk(obj: Any, model_key: str, min_conf: float, collector: List[Dict]) -> Any:
    if isinstance(obj, str):
        return _anonymize_str(obj, model_key, min_conf, collector)
    if isinstance(obj, dict):
        return {k: _walk(v, model_key, min_conf, collector) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(item, model_key, min_conf, collector) for item in obj]
    # numbers, booleans, None — return as-is
    return obj


def _anonymize_str(
    text: str,
    model_key: str,
    min_conf: float,
    collector: List[Dict],
) -> str:
    """Anonymize a single string value and collect entities."""
    if not text or len(text.strip()) < 3:
        return text
    redacted, entities = anonymizer.anonymize(
        text,
        model_key=model_key,
        placeholder_format="tag",
        min_confidence=min_conf,
    )
    collector.extend(entities)
    return redacted
