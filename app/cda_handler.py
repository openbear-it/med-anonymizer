"""
CDA (HL7 Clinical Document Architecture) / IHE XDS anonymization handler.

Strategy: XPath-targeted structural replacement — no NER model invoked.

CDA is a fully-typed XML format: the element/attribute name and its position
in the document hierarchy determine precisely what kind of PII it contains.

Attribute replacement (direct typed):
  patient/birthTime[@value]                     → [DATE_OF_BIRTH]
  patient/administrativeGenderCode[@code/@displayName] → [GENDER]
  patientRole/id[@extension]                    → [MEDICAL_RECORD_NUMBER]
  patientRole/telecom[@value]                   → [PHONE_NUMBER] / [EMAIL]
  assignedAuthor/telecom[@value]                → [PHONE_NUMBER] / [EMAIL]
  assignedAuthor/id[@extension]                 → [UNIQUE_IDENTIFIER]

Element text replacement (direct typed):
  patient/name (+ sub-elements)                 → [NAME]
  patientRole/addr (+ sub-elements)             → [ADDRESS]
  assignedPerson/name (+ sub-elements)          → [NAME]
  representedOrganization/name                  → [ORGANIZATION]
  representedCustodianOrganization/name         → [ORGANIZATION]

Whole-subtree redaction (patient-level but type unknown):
  section/text                                  → [REDACTED]

For unstructured clinical prose that requires NER, use /api/v1/anonymize/text.

Requires: lxml>=5.0.0  (optional dependency — 503 returned if missing)
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from lxml import etree

    LXML_AVAILABLE = True
except ImportError:  # pragma: no cover
    LXML_AVAILABLE = False


# ─── Public API ──────────────────────────────────────────────────────────────


def anonymize_cda(
    document_xml: str,
    model_key: str,         # unused — no NER; kept for API compatibility
    placeholder_format: str,
    min_confidence: float,  # unused — no NER; kept for API compatibility
) -> Tuple[str, List[Dict]]:
    """Anonymize PII in a CDA XML document via pure structural XPath replacement.

    No NER inference is performed.
    Returns (anonymized_xml_string, deduplicated_entity_list).
    Preserves XML declaration, namespaces, attributes, and document structure.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
        root = etree.fromstring(document_xml.encode("utf-8"), parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"XML CDA non valido: {exc}") from exc

    collector: List[Dict] = []
    _anonymize_targeted(root, placeholder_format, collector)

    anonymized_xml = etree.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")

    # Deduplicate entities by (type, text)
    seen: set = set()
    unique: List[Dict] = []
    for e in collector:
        key = (e["entity_type"], e["text"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return anonymized_xml, unique


# ─── Targeted anonymization ─────────────────────────────────────────────────


def _anonymize_targeted(
    root,
    placeholder_format: str,
    collector: List[Dict],
) -> None:
    """Apply XPath-targeted PII replacement to the document tree."""

    # ── Attribute replacements (structured, no NER) ───────────────────────────

    # patient/birthTime @value
    for el in _find(root, "patient", "birthTime"):
        _replace_attr(el, "value", "date_of_birth", placeholder_format, collector)

    # patient/administrativeGenderCode @code and @displayName
    for el in _find(root, "patient", "administrativeGenderCode"):
        _replace_attr(el, "code",        "gender", placeholder_format, collector)
        _replace_attr(el, "displayName", "gender", placeholder_format, collector)

    # patientRole/id @extension
    for el in _find(root, "patientRole", "id"):
        _replace_attr(el, "extension", "medical_record_number", placeholder_format, collector)

    # patientRole/telecom @value
    for el in _find(root, "patientRole", "telecom"):
        _replace_telecom_attr(el, placeholder_format, collector)

    # assignedAuthor/telecom @value
    for el in _find(root, "assignedAuthor", "telecom"):
        _replace_telecom_attr(el, placeholder_format, collector)

    # assignedAuthor/id @extension (author identifier)
    for el in _find(root, "assignedAuthor", "id"):
        _replace_attr(el, "extension", "unique_identifier", placeholder_format, collector)

    # ── Element text replacements (structured, no NER) ────────────────────────

    # patient/name
    for el in _find(root, "patient", "name"):
        _replace_element_texts(el, "[NAME]", "name", collector)

    # patientRole/addr
    for el in _find(root, "patientRole", "addr"):
        _replace_element_texts(el, "[ADDRESS]", "address", collector)

    # assignedPerson/name (author / practitioner)
    for el in _find(root, "assignedPerson", "name"):
        _replace_element_texts(el, "[NAME]", "name", collector)

    # representedOrganization/name and representedCustodianOrganization/name
    for parent_tag in ("representedOrganization", "representedCustodianOrganization"):
        for el in _find(root, parent_tag, "name"):
            _replace_element_texts(el, "[ORGANIZATION]", "organization", collector)

    # ── Whole-subtree redaction for clinical narratives ───────────────────────
    for el in _find(root, "section", "text"):
        _redact_subtree_text(el)


# ─── XPath helpers ────────────────────────────────────────────────────────────


def _find(root, parent_local: str, child_local: str):
    """Return all <child_local> elements that are direct children of any
    <parent_local> element, regardless of XML namespace."""
    return root.xpath(
        f"//*[local-name()='{parent_local}']/*[local-name()='{child_local}']"
    )


# ─── Element / attribute manipulation ────────────────────────────────────────


def _replace_attr(
    el,
    attr_name: str,
    entity_type: str,
    fmt: str,
    collector: List[Dict],
) -> None:
    """Replace an XML attribute value with a placeholder."""
    original = el.get(attr_name)
    if not original:
        return
    placeholder = f"[{entity_type.upper()}]" if fmt == "tag" else "[REDACTED]"
    el.set(attr_name, placeholder)
    collector.append({
        "entity_type": entity_type,
        "text": original,
        "start": 0,
        "end": len(original),
        "score": 1.0,
    })


def _replace_telecom_attr(el, fmt: str, collector: List[Dict]) -> None:
    """Replace a telecom @value attribute, choosing entity type by URI scheme."""
    original = el.get("value")
    if not original:
        return
    lower = original.lower()
    if lower.startswith("mailto:"):
        entity_type = "email"
    else:
        entity_type = "phone_number"
    placeholder = f"[{entity_type.upper()}]" if fmt == "tag" else "[REDACTED]"
    el.set("value", placeholder)
    collector.append({
        "entity_type": entity_type,
        "text": original,
        "start": 0,
        "end": len(original),
        "score": 1.0,
    })


def _replace_element_texts(
    el,
    placeholder: str,
    entity_type: str,
    collector: List[Dict],
) -> None:
    """Replace all non-empty text nodes within an element (and its descendants)
    with *placeholder*.  Collects the original values for reporting."""
    for node in el.iter():
        if node.text and node.text.strip():
            original = node.text.strip()
            collector.append({
                "entity_type": entity_type,
                "text": original,
                "start": 0,
                "end": len(original),
                "score": 1.0,
            })
            node.text = node.text.replace(original, placeholder)
        # tail belongs to the parent element, skip it to avoid stomping sibling text
        if node is not el and node.tail and node.tail.strip():
            original = node.tail.strip()
            collector.append({
                "entity_type": entity_type,
                "text": original,
                "start": 0,
                "end": len(original),
                "score": 1.0,
            })
            node.tail = node.tail.replace(original, placeholder)


def _redact_subtree_text(el) -> None:
    """Replace all text content in a subtree with [REDACTED]."""
    for node in el.iter():
        if node.text and node.text.strip():
            node.text = "[REDACTED]"
        if node is not el and node.tail and node.tail.strip():
            node.tail = "[REDACTED]"
