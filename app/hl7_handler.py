"""
HL7 v2.x anonymization handler.

Strategy: run NER over the full HL7 message text (which is plain ASCII/UTF-8
with pipe/caret delimiters) and redact detected PII spans in-place.
HL7 segment structure is preserved; only free-text values are redacted.
"""

from typing import Dict, List, Tuple

from .anonymizer import anonymizer, _redact


def anonymize_hl7(
    message: str,
    model_key: str,
    placeholder_format: str,
    min_confidence: float,
) -> Tuple[str, List[Dict]]:
    """Anonymize an HL7 v2.x message.

    Normalises line endings to \\n before NER, then restores \\r (HL7 standard).
    Returns (anonymized_message, detected_entities).
    """
    # HL7 messages use \r as segment terminator; normalise for NER
    normalised = message.replace("\r\n", "\n").replace("\r", "\n")

    entities = anonymizer.detect(normalised, model_key=model_key, min_confidence=min_confidence)
    redacted = _redact(normalised, entities, placeholder_format)

    # Restore HL7-standard carriage-return segment terminator
    redacted_hl7 = redacted.replace("\n", "\r")
    return redacted_hl7, entities
