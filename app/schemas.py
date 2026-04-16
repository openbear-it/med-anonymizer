from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field


class ModelKey(str, Enum):
    italian_superclinical_base = "italian-superclinical-base"
    italian_bigmed_large = "italian-bigmed-large"
    italian_superclinical_large = "italian-superclinical-large"


class PlaceholderFormat(str, Enum):
    tag = "tag"
    redacted = "redacted"


# ─── Requests ────────────────────────────────────────────────────────────────


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        description="Testo clinico da anonimizzare (note, referti, lettere di dimissione…)",
        min_length=1,
        max_length=100_000,
        examples=["Paziente Mario Rossi, nato il 15/03/1965, SSN: 123-45-6789, tel. 055-123456."],
    )
    model: ModelKey = Field(
        ModelKey.italian_superclinical_base,
        description="Modello NER OpenMed da utilizzare per il rilevamento PII",
    )
    placeholder_format: PlaceholderFormat = Field(
        PlaceholderFormat.tag,
        description="'tag' sostituisce con [ENTITY_TYPE], 'redacted' usa [REDACTED]",
    )
    min_confidence: float = Field(
        0.70,
        description="Soglia minima di confidenza del modello (0.0–1.0)",
        ge=0.0,
        le=1.0,
    )


class FhirRequest(BaseModel):
    resource: Any = Field(
        ...,
        description="Risorsa FHIR R4 in formato JSON (Patient, Observation, Bundle…)",
        examples=[
            {
                "resourceType": "Patient",
                "name": [{"family": "Rossi", "given": ["Mario"]}],
                "birthDate": "1965-03-15",
            }
        ],
    )
    model: ModelKey = Field(ModelKey.italian_superclinical_base)
    min_confidence: float = Field(0.70, ge=0.0, le=1.0)


class Hl7Request(BaseModel):
    message: str = Field(
        ...,
        description="Messaggio HL7 v2.x da anonimizzare (ADT, ORU, ORM…)",
        min_length=1,
        examples=[
            "MSH|^~\\&|HIS|OSP|LAB|LAB|20240410||ADT^A01|001|P|2.5\rPID|1||12345^^^MRN||Rossi^Mario||19650315|M|||Via Roma 42^^FI"
        ],
    )
    model: ModelKey = Field(ModelKey.italian_superclinical_base)
    placeholder_format: PlaceholderFormat = Field(PlaceholderFormat.tag)
    min_confidence: float = Field(0.70, ge=0.0, le=1.0)


class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        description="Testo su cui rilevare entità PII (senza anonimizzare)",
        min_length=1,
        max_length=100_000,
    )
    model: ModelKey = Field(ModelKey.italian_superclinical_base)
    min_confidence: float = Field(0.70, ge=0.0, le=1.0)


# ─── Responses ───────────────────────────────────────────────────────────────


class Entity(BaseModel):
    entity_type: str = Field(..., description="Tipo di entità PII rilevata (es. name, ssn, date_of_birth)")
    text: str = Field(..., description="Testo originale dell'entità")
    start: int = Field(..., description="Indice di inizio nel testo (0-based)")
    end: int = Field(..., description="Indice di fine nel testo (0-based, esclusivo)")
    score: float = Field(..., description="Confidenza del modello (0.0–1.0)")


class AnonymizeResponse(BaseModel):
    anonymized: str = Field(..., description="Testo con le entità PII sostituite da placeholder")
    entities: List[Entity] = Field(..., description="Lista delle entità PII rilevate")
    entity_count: int = Field(..., description="Numero totale di entità rilevate")
    model_used: str = Field(..., description="ID HuggingFace del modello utilizzato")
    processing_time_ms: float = Field(..., description="Tempo di elaborazione in millisecondi")


class DetectResponse(BaseModel):
    entities: List[Entity]
    entity_count: int
    model_used: str
    processing_time_ms: float


class FhirAnonymizeResponse(BaseModel):
    anonymized_resource: Any = Field(..., description="Risorsa FHIR con tutti i campi stringa anonimizzati")
    total_entities: int = Field(..., description="Numero totale di entità PII trovate in tutta la risorsa")
    model_used: str
    processing_time_ms: float


class ModelInfo(BaseModel):
    key: str
    name: str
    description: str
    parameters: str
    f1_score: float
    is_default: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    version: str
