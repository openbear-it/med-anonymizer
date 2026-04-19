"""
CDA (HL7 Clinical Document Architecture) / IHE XDS anonymization handler.

Strategy: parse the CDA XML with lxml, walk every text node (element.text
and element.tail) and run the NER anonymizer on any non-trivial content.
The document structure, namespaces, attributes, and element tree are preserved;
only string text content is modified.

In the IHE XDS / XCA context this handler acts as a de-identification
middleware: the caller retrieves the CDA document from the repository, passes
it here, and submits the anonymized version back to a research or
secondary-use store.

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

from .anonymizer import anonymizer, _pseudonymize

# Minimum characters for a text node to be worth sending to NER
_MIN_TEXT_LEN = 3


# ─── Public API ──────────────────────────────────────────────────────────────


def anonymize_cda(
    document_xml: str,
    model_key: str,
    placeholder_format: str,
    min_confidence: float,
) -> Tuple[str, List[Dict]]:
    """Anonymize all text nodes in a CDA XML document.

    Returns (anonymized_xml_string, deduplicated_entity_list).
    Preserves XML declaration, namespaces, attributes, and structure.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
        root = etree.fromstring(document_xml.encode("utf-8"), parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"XML CDA non valido: {exc}") from exc

    collector: List[Dict] = []

    if placeholder_format == "pseudonym":
        # Collect all text first, then pseudonymize in a second pass
        _walk_collect(root, model_key, min_confidence, collector)
        _, _, mapping = anonymizer.pseudonymize(
            _extract_full_text(root), model_key, min_confidence
        )
        _walk_apply_pseudonym(root, mapping)
    else:
        _walk_anonymize(root, model_key, placeholder_format, min_confidence, collector)

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


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _walk_anonymize(element, model_key, placeholder_format, min_confidence, collector):
    """Recursively anonymize element.text and child.tail in place."""
    if element.text and len(element.text.strip()) >= _MIN_TEXT_LEN:
        anon, ents = anonymizer.anonymize(
            element.text,
            model_key=model_key,
            placeholder_format=placeholder_format,
            min_confidence=min_confidence,
        )
        element.text = anon
        collector.extend(ents)

    for child in element:
        _walk_anonymize(child, model_key, placeholder_format, min_confidence, collector)
        if child.tail and len(child.tail.strip()) >= _MIN_TEXT_LEN:
            anon, ents = anonymizer.anonymize(
                child.tail,
                model_key=model_key,
                placeholder_format=placeholder_format,
                min_confidence=min_confidence,
            )
            child.tail = anon
            collector.extend(ents)


def _walk_collect(element, model_key, min_confidence, collector):
    """Collect entities from all text nodes (for pseudonym pre-pass)."""
    if element.text and len(element.text.strip()) >= _MIN_TEXT_LEN:
        ents = anonymizer.detect(element.text, model_key=model_key, min_confidence=min_confidence)
        collector.extend(ents)
    for child in element:
        _walk_collect(child, model_key, min_confidence, collector)
        if child.tail and len(child.tail.strip()) >= _MIN_TEXT_LEN:
            ents = anonymizer.detect(child.tail, model_key=model_key, min_confidence=min_confidence)
            collector.extend(ents)


def _extract_full_text(element) -> str:
    """Concatenate all text nodes with newline separators (for pseudonym mapping)."""
    parts: List[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.extend(_extract_full_text(child).split("\n"))
        if child.tail:
            parts.append(child.tail)
    return "\n".join(p for p in parts if p.strip())


def _apply_mapping(text: str, mapping: Dict[str, str]) -> str:
    """Replace all occurrences of mapping keys in text."""
    for original, pseudonym in mapping.items():
        text = text.replace(original, pseudonym)
    return text


def _walk_apply_pseudonym(element, mapping: Dict[str, str]):
    """Apply the pseudonym mapping to all text nodes in place."""
    if element.text:
        element.text = _apply_mapping(element.text, mapping)
    for child in element:
        _walk_apply_pseudonym(child, mapping)
        if child.tail:
            child.tail = _apply_mapping(child.tail, mapping)
