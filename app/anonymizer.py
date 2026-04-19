"""
Core anonymization engine — singleton wrapper around HuggingFace NER pipelines.

Supports multiple OpenMed PII models with automatic text chunking for
documents exceeding the model's max sequence length (~384 tokens ≈ 1 200 chars).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline

logger = logging.getLogger(__name__)

# ─── Model registry ──────────────────────────────────────────────────────────
# Tutti i modelli sono ottimizzati per l'italiano, valutati su AI4Privacy (subset IT).
# Base model: DeBERTa-v3 (SuperClinical) e XLM-RoBERTa-large (BigMed).

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "italian-superclinical-base": {
        "model_id": "OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1",
        "name": "Italian SuperClinical Base 184M",
        "description": (
            "DeBERTa-v3-base fine-tuned su testi italiani (AI4Privacy). "
            "Modello leggero e veloce, ideale per uso quotidiano in produzione. "
            "Rileva 54 categorie PII ottimizzate per il contesto italiano."
        ),
        "parameters": "184M",
        "f1_score": 0.9596,
    },
    "italian-bigmed-large": {
        "model_id": "OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1",
        "name": "Italian BigMed Large 560M",
        "description": (
            "XLM-RoBERTa-large fine-tuned su testi italiani (AI4Privacy). "
            "Ottimo equilibrio tra precisione e copertura multilingue. "
            "Consigliato per note cliniche e referti ospedalieri."
        ),
        "parameters": "560M",
        "f1_score": 0.9671,
    },
    "italian-superclinical-large": {
        "model_id": "OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1",
        "name": "Italian SuperClinical Large 434M",
        "description": (
            "DeBERTa-v3-large fine-tuned su testi italiani (AI4Privacy). "
            "Miglior F1 tra tutti i modelli italiani OpenMed (0.9728). "
            "Ideale per documenti critici dove la precisione è prioritaria."
        ),
        "parameters": "434M",
        "f1_score": 0.9728,
    },
}

DEFAULT_MODEL = "italian-superclinical-large"

# Characters per chunk (≈ 384 tokens at ~3.5 chars/token, conservative margin)
_MAX_CHUNK_CHARS = 1_200


# ─── Anonymizer singleton ────────────────────────────────────────────────────


class Anonymizer:
    """Thread-safe singleton that manages NER pipelines and text de-identification."""

    _instance: Optional["Anonymizer"] = None

    def __new__(cls) -> "Anonymizer":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._pipelines: Dict[str, Any] = {}
            cls._instance = inst
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def preload(self, model_key: str = DEFAULT_MODEL) -> None:
        """Eagerly load a model (called at startup to avoid cold-start latency)."""
        logger.info("Pre-caricamento modello '%s'…", model_key)
        self._get_pipeline(model_key)
        logger.info("Modello '%s' pronto.", model_key)

    def detect(
        self,
        text: str,
        model_key: str = DEFAULT_MODEL,
        min_confidence: float = 0.70,
    ) -> List[Dict]:
        """Run NER on *text* and return entities with score ≥ min_confidence."""
        ner = self._get_pipeline(model_key)
        raw = self._run_chunked(ner, text)
        return [e for e in raw if e["score"] >= min_confidence]

    def pseudonymize(
        self,
        text: str,
        model_key: str = DEFAULT_MODEL,
        min_confidence: float = 0.70,
    ) -> Tuple[str, List[Dict], Dict[str, str]]:
        """Detect PII and replace each unique surface form with a deterministic pseudonym.

        Returns (pseudonymized_text, entities, pseudonym_map).
        pseudonym_map maps each original string → its assigned pseudonym label.
        """
        entities = self.detect(text, model_key, min_confidence)
        pseudonymized, mapping = _pseudonymize(text, entities)
        return pseudonymized, entities, mapping

    def anonymize(
        self,
        text: str,
        model_key: str = DEFAULT_MODEL,
        placeholder_format: str = "tag",
        min_confidence: float = 0.70,
    ) -> Tuple[str, List[Dict]]:
        """Detect PII entities and replace them with placeholders.

        Returns (anonymized_text, entities).
        """
        entities = self.detect(text, model_key, min_confidence)
        redacted = _redact(text, entities, placeholder_format)
        return redacted, entities

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_pipeline(self, model_key: str) -> Any:
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modello '{model_key}' non supportato. "
                f"Disponibili: {list(AVAILABLE_MODELS)}"
            )
        if model_key not in self._pipelines:
            model_id = AVAILABLE_MODELS[model_key]["model_id"]
            logger.info("Caricamento HuggingFace model: %s", model_id)
            self._pipelines[model_key] = pipeline(
                "ner",
                model=model_id,
                aggregation_strategy="simple",
                device=-1,  # CPU — compatibile con AMD64 e ARM64
            )
        return self._pipelines[model_key]

    def _run_chunked(self, ner_pipeline: Any, text: str) -> List[Dict]:
        """Run NER on text, splitting into chunks when necessary."""
        if len(text) <= _MAX_CHUNK_CHARS:
            return [_normalize(e) for e in ner_pipeline(text)]

        all_entities: List[Dict] = []
        for chunk_text, chunk_start in _chunk_text(text):
            for raw_ent in ner_pipeline(chunk_text):
                ent = _normalize(raw_ent)
                ent["start"] += chunk_start
                ent["end"] += chunk_start
                all_entities.append(ent)
        return all_entities


# ─── Module-level singleton ───────────────────────────────────────────────────

anonymizer = Anonymizer()


# ─── Pure helpers (no state) ─────────────────────────────────────────────────


def _normalize(raw: Dict) -> Dict:
    return {
        "entity_type": raw.get("entity_group") or raw.get("entity", "UNKNOWN"),
        "text": raw.get("word", ""),
        "start": int(raw.get("start", 0)),
        "end": int(raw.get("end", 0)),
        "score": round(float(raw.get("score", 0.0)), 4),
    }


def _pseudonymize(text: str, entities: List[Dict]) -> Tuple[str, Dict[str, str]]:
    """Replace each unique entity surface form with a typed counter pseudonym.

    Example: 'Mario Rossi' → '[NAME_001]', '055-123456' → '[PHONE_NUMBER_001]'
    Entities with the same text always receive the same pseudonym (intra-document
    consistency). Returns (pseudonymized_text, {original_text: pseudonym}).
    """
    type_counters: Dict[str, int] = {}
    mapping: Dict[str, str] = {}

    # First pass: assign pseudonyms in document order (left-to-right)
    for ent in sorted(entities, key=lambda e: e["start"]):
        key = ent["text"]
        if key not in mapping:
            t = ent["entity_type"].upper()
            type_counters[t] = type_counters.get(t, 0) + 1
            mapping[key] = f"[{t}_{type_counters[t]:03d}]"

    # Second pass: substitute right-to-left to preserve offsets
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        ph = mapping[ent["text"]]
        text = text[: ent["start"]] + ph + text[ent["end"] :]

    return text, mapping


def _redact(text: str, entities: List[Dict], placeholder_format: str) -> str:
    """Replace entity spans in *text* from right to left to preserve offsets."""
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        ph = (
            f"[{ent['entity_type'].upper()}]"
            if placeholder_format == "tag"
            else "[REDACTED]"
        )
        text = text[: ent["start"]] + ph + text[ent["end"] :]
    return text


def _chunk_text(text: str) -> List[Tuple[str, int]]:
    """Split *text* at paragraph/sentence boundaries into ≤ _MAX_CHUNK_CHARS chunks.

    Each item is (chunk_text, start_offset_in_original).
    """
    # Step 1: split on paragraph breaks
    para_pattern = re.compile(r"\n{2,}")
    paragraphs: List[Tuple[str, int]] = []
    prev = 0
    for m in para_pattern.finditer(text):
        para = text[prev : m.start()]
        if para.strip():
            paragraphs.append((para, prev))
        prev = m.end()
    if text[prev:].strip():
        paragraphs.append((text[prev:], prev))
    if not paragraphs:
        paragraphs = [(text, 0)]

    # Step 2: merge small paragraphs / split large ones
    chunks: List[Tuple[str, int]] = []
    buf, buf_start = "", 0

    for para, para_start in paragraphs:
        if len(para) > _MAX_CHUNK_CHARS:
            # Flush buffer first
            if buf.strip():
                chunks.append((buf.strip(), buf_start))
                buf, buf_start = "", 0
            # Split paragraph into sentences
            chunks.extend(_chunk_by_sentences(para, para_start))
        elif len(buf) + len(para) + 2 > _MAX_CHUNK_CHARS and buf:
            chunks.append((buf.strip(), buf_start))
            buf, buf_start = para, para_start
        else:
            if buf:
                buf += "\n\n" + para
            else:
                buf, buf_start = para, para_start

    if buf.strip():
        chunks.append((buf.strip(), buf_start))

    return chunks or [(text[:_MAX_CHUNK_CHARS], 0)]


def _chunk_by_sentences(text: str, base_offset: int) -> List[Tuple[str, int]]:
    """Split text at sentence boundaries; last resort for very long paragraphs."""
    sent_end = re.compile(r"(?<=[.!?])\s+")
    sentences: List[Tuple[str, int]] = []
    prev = 0
    for m in sent_end.finditer(text):
        sentences.append((text[prev : m.start() + 1], base_offset + prev))
        prev = m.end()
    if text[prev:].strip():
        sentences.append((text[prev:], base_offset + prev))
    if not sentences:
        return [(text[:_MAX_CHUNK_CHARS], base_offset)]

    chunks: List[Tuple[str, int]] = []
    buf, buf_start = "", 0
    for sent, sent_start in sentences:
        if len(buf) + len(sent) + 1 > _MAX_CHUNK_CHARS and buf:
            chunks.append((buf.strip(), buf_start))
            buf, buf_start = sent, sent_start
        else:
            if buf:
                buf += " " + sent
            else:
                buf, buf_start = sent, sent_start
    if buf.strip():
        chunks.append((buf.strip(), buf_start))
    return chunks
