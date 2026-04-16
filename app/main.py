"""
Med-Anonymizer — FastAPI application entry point.

Exposes REST endpoints for clinical text, FHIR and HL7 v2 de-identification
powered by OpenMed NER models (https://huggingface.co/OpenMed).
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .anonymizer import AVAILABLE_MODELS, DEFAULT_MODEL, anonymizer
from .fhir_handler import anonymize_fhir
from .hl7_handler import anonymize_hl7
from .schemas import (
    AnonymizeResponse,
    DetectRequest,
    DetectResponse,
    Entity,
    FhirAnonymizeResponse,
    FhirRequest,
    HealthResponse,
    Hl7Request,
    ModelInfo,
    ModelsResponse,
    TextRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
STATIC_DIR = Path(__file__).parent / "static"


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Med-Anonymizer v%s starting — preloading default model…", APP_VERSION)
    anonymizer.preload(DEFAULT_MODEL)
    logger.info("Service ready.")
    yield
    logger.info("Med-Anonymizer shutting down.")


# ─── FastAPI application ──────────────────────────────────────────────────────


app = FastAPI(
    title="Med-Anonymizer",
    description=(
        "## De-identificazione clinica AI · powered by [OpenMed](https://huggingface.co/OpenMed)\n\n"
        "Servizio REST per l'anonimizzazione automatica di informazioni personali sensibili (PII) "
        "da testi clinici, risorse FHIR R4 e messaggi HL7 v2.x.\n\n"
        "I modelli NER OpenMed rilevano **54 categorie di PII** incluse nome, data di nascita, "
        "SSN, MRN, indirizzo, telefono, email, e molte altre.\n\n"
        "### Conformità normativa\n"
        "Supporta flussi di de-identificazione conformi a **HIPAA Safe Harbor** e **GDPR Art. 4(5)**.\n\n"
        "### Pagine utili\n"
        "- [🔧 Interfaccia di test](/ui)\n"
        "- [ℹ️ Informazioni sul servizio](/info)\n"
        "- [📖 ReDoc](/redoc)\n"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "OpenMed", "url": "https://huggingface.co/OpenMed"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Utility ─────────────────────────────────────────────────────────────────


def _to_response(
    anonymized: str,
    entities: list,
    model_key: str,
    elapsed_ms: float,
) -> AnonymizeResponse:
    return AnonymizeResponse(
        anonymized=anonymized,
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used=AVAILABLE_MODELS[model_key]["model_id"],
        processing_time_ms=round(elapsed_ms, 2),
    )


# ─── HTML pages ───────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False, summary="Interfaccia di test")
async def ui_page():
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/info", response_class=HTMLResponse, include_in_schema=False, summary="Informazioni")
async def info_page():
    return HTMLResponse(content=(STATIC_DIR / "info.html").read_text(encoding="utf-8"))


# ─── System endpoints ─────────────────────────────────────────────────────────


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Health check",
    description="Verifica lo stato del servizio e i modelli attualmente caricati in memoria.",
)
async def health():
    return HealthResponse(
        status="ok",
        loaded_models=list(anonymizer._pipelines.keys()),
        version=APP_VERSION,
    )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    tags=["Sistema"],
    summary="Lista modelli disponibili",
    description="Restituisce tutti i modelli OpenMed PII disponibili con metadati e metriche.",
)
async def list_models():
    return ModelsResponse(
        models=[
            ModelInfo(
                key=k,
                name=v["name"],
                description=v["description"],
                parameters=v["parameters"],
                f1_score=v["f1_score"],
                is_default=(k == DEFAULT_MODEL),
            )
            for k, v in AVAILABLE_MODELS.items()
        ],
        default=DEFAULT_MODEL,
    )


# ─── Anonymization endpoints ──────────────────────────────────────────────────


@app.post(
    "/api/v1/anonymize/text",
    response_model=AnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza testo libero",
    description=(
        "Rileva entità PII in un testo clinico (nota, referto, lettera di dimissione…) "
        "usando NER e le sostituisce con placeholder. "
        "Supporta documenti lunghi grazie al chunking automatico."
    ),
)
async def anonymize_text(req: TextRequest):
    t0 = time.perf_counter()
    try:
        redacted, entities = anonymizer.anonymize(
            req.text,
            model_key=req.model.value,
            placeholder_format=req.placeholder_format.value,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return _to_response(redacted, entities, req.model.value, elapsed)


@app.post(
    "/api/v1/anonymize/fhir",
    response_model=FhirAnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza una risorsa FHIR",
    description=(
        "Attraversa ricorsivamente ogni campo stringa di una risorsa FHIR R4 (JSON) "
        "e anonimizza i valori che contengono PII. "
        "Preserva la struttura e i tipi della risorsa originale."
    ),
)
async def anonymize_fhir_endpoint(req: FhirRequest):
    t0 = time.perf_counter()
    try:
        anon_resource, entities = anonymize_fhir(
            req.resource,
            model_key=req.model.value,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return FhirAnonymizeResponse(
        anonymized_resource=anon_resource,
        total_entities=len(entities),
        model_used=AVAILABLE_MODELS[req.model.value]["model_id"],
        processing_time_ms=round(elapsed, 2),
    )


@app.post(
    "/api/v1/anonymize/hl7",
    response_model=AnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza un messaggio HL7 v2",
    description=(
        "Analizza un messaggio HL7 v2.x (ADT, ORU, ORM…) con NER e redige le entità PII "
        "rilevate nei campi di testo libero. La struttura del messaggio (segmenti, delimitatori) "
        "viene preservata."
    ),
)
async def anonymize_hl7_endpoint(req: Hl7Request):
    t0 = time.perf_counter()
    try:
        redacted, entities = anonymize_hl7(
            req.message,
            model_key=req.model.value,
            placeholder_format=req.placeholder_format.value,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return _to_response(redacted, entities, req.model.value, elapsed)


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    tags=["Analisi"],
    summary="Rileva entità PII (solo analisi)",
    description=(
        "Esegue il rilevamento NER e restituisce le entità trovate con posizione e confidenza, "
        "**senza modificare il testo originale**. Utile per ispezione e audit."
    ),
)
async def detect_entities(req: DetectRequest):
    t0 = time.perf_counter()
    try:
        entities = anonymizer.detect(
            req.text,
            model_key=req.model.value,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return DetectResponse(
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used=AVAILABLE_MODELS[req.model.value]["model_id"],
        processing_time_ms=round(elapsed, 2),
    )
