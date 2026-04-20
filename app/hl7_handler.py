"""
HL7 v2.x anonymization handler.

Strategy: pure structural segment/field replacement — no NER model invoked.

HL7 v2 field positions carry explicit semantics defined by the standard.
Known PII-bearing fields receive a typed placeholder; fields containing
free-form clinical text (OBX-5, NTE-3) are wholly replaced with [REDACTED]
since their content is patient-level but the PII type cannot be determined
from the field position alone.

For unstructured clinical prose that requires NER, use /api/v1/anonymize/text.
"""

from typing import Dict, List, Set, Tuple

# ─── Field registries ────────────────────────────────────────────────────────

# {segment: {1-based field index: entity_type}} — direct typed replacement.
_DIRECT_FIELDS: Dict[str, Dict[int, str]] = {
    "PID": {
        3:  "medical_record_number",
        5:  "name",
        6:  "name",
        7:  "date_of_birth",
        8:  "gender",
        9:  "name",
        11: "address",
        13: "phone_number",
        14: "phone_number",
        18: "unique_identifier",
        19: "ssn",
        20: "unique_identifier",
    },
    "NK1": {
        2: "name",
        4: "address",
        5: "phone_number",
        6: "phone_number",
    },
    "PV1": {
        7:  "name",
        8:  "name",
        9:  "name",
        17: "name",
    },
    "IN1": {
        16: "name",
    },
    "GT1": {
        3: "name",
        5: "address",
        6: "phone_number",
    },
}

# {segment: {1-based field indices}} — free-text fields that may contain
# patient-level information of unknown type → wholly replaced with [REDACTED].
_REDACT_FIELDS: Dict[str, Set[int]] = {
    "OBX": {5},   # Observation value (could be free-form text or structured result)
    "NTE": {3},   # Notes and comments
}

# ─── Public API ──────────────────────────────────────────────────────────────


def anonymize_hl7(
    message: str,
    model_key: str,         # unused — no NER; kept for API compatibility
    placeholder_format: str,
    min_confidence: float,  # unused — no NER; kept for API compatibility
) -> Tuple[str, List[Dict]]:
    """Anonymize an HL7 v2.x message via pure structural field replacement.

    No NER inference is performed.
    Normalises line endings to \\n before processing, restores \\r on return.
    Returns (anonymized_message, detected_entities).
    """
    normalised = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalised.split("\n")

    all_entities: List[Dict] = []
    result_lines: List[str] = []

    for line in lines:
        if not line.strip():
            result_lines.append(line)
            continue
        processed, entities = _process_segment(line, placeholder_format)
        result_lines.append(processed)
        all_entities.extend(entities)

    return "\n".join(result_lines).replace("\n", "\r"), all_entities


# ─── Internal helpers ────────────────────────────────────────────────────────


def _process_segment(
    segment: str,
    placeholder_format: str,
) -> Tuple[str, List[Dict]]:
    fields = segment.split("|")
    seg_name = fields[0]
    entities: List[Dict] = []

    # Direct typed replacement
    if seg_name in _DIRECT_FIELDS:
        for field_idx, entity_type in _DIRECT_FIELDS[seg_name].items():
            if field_idx < len(fields) and fields[field_idx]:
                original = fields[field_idx]
                fields[field_idx] = _typed_placeholder(entity_type, placeholder_format)
                entities.append({
                    "entity_type": entity_type,
                    "text": original,
                    "start": 0,
                    "end": len(original),
                    "score": 1.0,
                })

    # Whole-field redaction for free-text segments
    if seg_name in _REDACT_FIELDS:
        for field_idx in _REDACT_FIELDS[seg_name]:
            if field_idx < len(fields) and fields[field_idx]:
                original = fields[field_idx]
                fields[field_idx] = "[REDACTED]"
                entities.append({
                    "entity_type": "free_text",
                    "text": original,
                    "start": 0,
                    "end": len(original),
                    "score": 1.0,
                })

    return "|".join(fields), entities


def _typed_placeholder(entity_type: str, fmt: str) -> str:
    # 'pseudonym' on structured fields falls back to typed placeholder.
    return "[REDACTED]" if fmt == "redacted" else f"[{entity_type.upper()}]"
