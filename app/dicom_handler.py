"""
DICOM metadata anonymization handler.

Strategy:
- Known structured PII tags (PatientName, PatientID, …) are replaced directly
  with a typed placeholder — no NER needed, the tag keyword already tells us
  the entity type.
- Free-text tags (PatientComments, StudyDescription, …) are run through the
  NER anonymizer so contextual PII is also caught.

Returns a JSON-serialisable report of all modified tags plus the bytes of the
anonymized DICOM file.

Requires: pydicom>=2.4.0  (optional dependency — 503 returned if missing)
"""

import io
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pydicom
    import pydicom.datadict as _dd

    PYDICOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYDICOM_AVAILABLE = False

from .anonymizer import anonymizer

# ─── PII tag registry ────────────────────────────────────────────────────────
# Maps DICOM keyword → entity_type string used for the placeholder.
# None means "run NER on the field text instead of direct replacement".

_STRUCTURED_PII: Dict[str, str] = {
    "PatientName": "name",
    "PatientID": "medical_record_number",
    "PatientBirthDate": "date_of_birth",
    "PatientBirthTime": "time",
    "PatientAge": "age",
    "PatientSex": "gender",
    "PatientAddress": "address",
    "PatientTelephoneNumbers": "phone_number",
    "PatientMotherBirthName": "name",
    "OtherPatientNames": "name",
    "OtherPatientIDs": "medical_record_number",
    "PatientInsurancePlanCodeSequence": "unique_identifier",
    "InstitutionName": "organization",
    "InstitutionAddress": "address",
    "InstitutionalDepartmentName": "organization",
    "ReferringPhysicianName": "name",
    "ConsultingPhysicianName": "name",
    "PerformingPhysicianName": "name",
    "OperatorsName": "name",
    "RequestingPhysician": "name",
    "ScheduledPerformingPhysicianName": "name",
    "ResponsiblePerson": "name",
}

_FREE_TEXT_PII = {
    "PatientComments",
    "AdditionalPatientHistory",
    "StudyDescription",
    "SeriesDescription",
    "RequestedProcedureDescription",
    "PerformedProcedureStepDescription",
    "ImageComments",
    "StudyComments",
}


# ─── Public API ──────────────────────────────────────────────────────────────


def anonymize_dicom(
    file_bytes: bytes,
    model_key: str,
    min_confidence: float,
    placeholder_format: str = "tag",
) -> Tuple[List[Dict], int, bytes]:
    """Anonymize PII fields in a DICOM file.

    Returns (modified_tags_report, ner_entities_found, anonymized_dcm_bytes).
    modified_tags_report is a list of dicts with keys:
        tag, name, original, anonymized
    """
    ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    modified_tags: List[Dict] = []
    ner_entities_total = 0

    # ── Structured PII tags ──────────────────────────────────────────────────
    for keyword, entity_type in _STRUCTURED_PII.items():
        result = _process_structured(ds, keyword, entity_type, placeholder_format)
        if result:
            modified_tags.append(result)

    # ── Free-text PII tags ───────────────────────────────────────────────────
    for keyword in _FREE_TEXT_PII:
        result, n_ents = _process_freetext(ds, keyword, model_key, min_confidence, placeholder_format)
        if result:
            modified_tags.append(result)
            ner_entities_total += n_ents

    # ── Serialize modified DICOM ─────────────────────────────────────────────
    out = io.BytesIO()
    ds.save_as(out)
    return modified_tags, ner_entities_total, out.getvalue()


# ─── Internal helpers ────────────────────────────────────────────────────────


def _get_str(ds, keyword: str) -> Optional[str]:
    """Return string representation of a DICOM element, or None if absent/empty."""
    if not hasattr(ds, keyword):
        return None
    try:
        val = getattr(ds, keyword)
    except Exception:
        return None
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _set_str(ds, keyword: str, value: str) -> None:
    """Set a DICOM element to a plain string value."""
    try:
        setattr(ds, keyword, value)
    except Exception as exc:
        logger.debug("Impossibile impostare %s: %s", keyword, exc)


def _tag_str(ds, keyword: str) -> str:
    """Return the tag address as '(GGGG,EEEE)' string."""
    try:
        tag = _dd.tag_for_keyword(keyword)
        return f"({tag >> 16:04X},{tag & 0xFFFF:04X})"
    except Exception:
        return "(????,????)"


def _placeholder(entity_type: str, fmt: str) -> str:
    return f"[{entity_type.upper()}]" if fmt == "tag" else "[REDACTED]"


def _process_structured(ds, keyword: str, entity_type: str, fmt: str) -> Optional[Dict]:
    original = _get_str(ds, keyword)
    if original is None:
        return None
    anon = _placeholder(entity_type, fmt)
    _set_str(ds, keyword, anon)
    return {
        "tag": _tag_str(ds, keyword),
        "name": keyword,
        "original": original,
        "anonymized": anon,
    }


def _process_freetext(
    ds,
    keyword: str,
    model_key: str,
    min_confidence: float,
    fmt: str,
) -> Tuple[Optional[Dict], int]:
    original = _get_str(ds, keyword)
    if original is None:
        return None, 0
    anon_str, entities = anonymizer.anonymize(
        original, model_key=model_key, placeholder_format=fmt, min_confidence=min_confidence
    )
    if anon_str == original:
        return None, 0
    _set_str(ds, keyword, anon_str)
    return {
        "tag": _tag_str(ds, keyword),
        "name": keyword,
        "original": original,
        "anonymized": anon_str,
    }, len(entities)
