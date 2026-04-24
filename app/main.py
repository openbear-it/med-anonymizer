"""
Med-Anonymizer — FastAPI application entry point.

Exposes REST endpoints for clinical text, FHIR and HL7 v2 de-identification
powered by OpenMed NER models (https://huggingface.co/OpenMed).
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .anonymizer import AVAILABLE_MODELS, DEFAULT_MODEL, PRIVACY_FILTER_MODEL_ID, anonymizer
from .cda_handler import LXML_AVAILABLE, anonymize_cda
from .dicom_handler import PYDICOM_AVAILABLE, anonymize_dicom
from .fhir_handler import anonymize_fhir
from .hl7_handler import anonymize_hl7
from .schemas import (
    AnonymizeResponse,
    CdaAnonymizeResponse,
    CdaRequest,
    DetectRequest,
    DetectResponse,
    DicomAnonymizeResponse,
    DicomFieldResult,
    Entity,
    FhirAnonymizeResponse,
    FhirRequest,
    HealthResponse,
    Hl7Request,
    ModelInfo,
    ModelKey,
    ModelsResponse,
    PlaceholderFormat,
    TextRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

APP_VERSION = "1.3.0"
STATIC_DIR = Path(__file__).parent / "static"


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Med-Anonymizer v%s starting — preloading default model…", APP_VERSION)
    anonymizer.preload(DEFAULT_MODEL)
    anonymizer.preload_privacy_filter()
    logger.info("Service ready.")
    yield
    logger.info("Med-Anonymizer shutting down.")


# ─── FastAPI application ──────────────────────────────────────────────────────


app = FastAPI(
    title="Med-Anonymizer",
    description=(
        "## De-identificazione clinica AI · powered by [OpenMed](https://huggingface.co/OpenMed) + [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)\n\n"
        "Servizio REST per l'anonimizzazione automatica di informazioni personali sensibili (PII) "
        "da testi clinici, risorse FHIR R4, messaggi HL7 v2.x, file DICOM e documenti CDA (IHE XDS/XCA).\n\n"
        "### 🤖 Pipeline NER dual-model (testo libero)\n"
        "Quando `use_privacy_filter=true` (default), **due modelli girano in parallelo**:\n\n"
        "1. **OpenMed** (italiano, 54 categorie PII) — DeBERTa-v3 / XLM-RoBERTa fine-tuned su AI4Privacy\n"
        "2. **OpenAI Privacy Filter** (multilingue, 8 categorie, 128K context) — bidirectional token classifier\n\n"
        "Le entità vengono unite (union) con deduplicazione degli span sovrapposti per score di confidenza. "
        "Impostare `use_privacy_filter=false` per usare solo OpenMed.\n\n"
        "### Funzionalità\n"
        "| Feature | Descrizione |\n"
        "|---------|-------------|\n"
        "| **Dual-model NER** | OpenMed + OpenAI Privacy Filter in parallelo — massima copertura |\n"
        "| **Anonimizzazione testo** | Sostituisce PII con `[ENTITY_TYPE]` o `[REDACTED]` tramite NER |\n"
        "| **Anonimizzazione strutturata** | FHIR, HL7, DICOM, CDA: sostituzione diretta per campo — nessuna inferenza AI |\n"
        "| **Pseudonimizzazione** | Assegna pseudonimi coerenti `[NAME_001]` con mappa reversibile (solo testo libero) |\n"
        "| **Revisione umana** | Endpoint `/detect` per highlight interattivo + annotazione manuale |\n"
        "| **Pipeline DICOM** | 21 tag strutturati a sostituzione diretta + campi liberi con NER |\n"
        "| **CDA / IHE** | De-identifica documenti HL7 CDA R2 XML (XDS, XCA, MHD) per sostituzione strutturale |\n\n"
        "### Confidenza minima (`min_confidence`)\n"
        "Filtra le predizioni NER (range 0.0–1.0). Default **0.70** – bilanciamento ottimale. "
        "Valori più bassi aumentano la sensitività; valori più alti la precisione.\n"
        "Applicabile solo agli endpoint `/anonymize/text`, `/detect` e `/anonymize/dicom` (campi liberi).\n\n"
        "### Conformità normativa\n"
        "Supporta flussi di de-identificazione conformi a **HIPAA Safe Harbor** e **GDPR Art.\u202f4(5)**.\n\n"
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
    openapi_tags=[
        {"name": "Anonimizzazione", "description": "Endpoint di de-identificazione per testo, FHIR, HL7, DICOM e CDA."},
        {"name": "NER / Rilevamento", "description": "Rilevamento entit\u00e0 PII senza anonimizzazione."},
        {"name": "Sistema", "description": "Health check, lista modelli, utilit\u00e0 di sistema."},
    ],
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


@app.get(
    "/api/v1/dicom/sample",
    tags=["Sistema"],
    summary="Scarica DICOM di esempio",
    description=(
        "Genera e restituisce un file DICOM sintetico con dati paziente fittizi, "
        "utile per testare il pipeline di anonimizzazione DICOM."
    ),
)
async def dicom_sample():
    if not PYDICOM_AVAILABLE:
        raise HTTPException(status_code=503, detail="pydicom non disponibile")
    import io

    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from fastapi.responses import Response

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta

    # Dati paziente fittizi
    ds.PatientName = "Rossi^Mario^Luigi"
    ds.PatientID = "MRN-2024-98765"
    ds.PatientBirthDate = "19650315"
    ds.PatientSex = "M"
    ds.PatientAddress = "Via Roma 42, 50100 Firenze FI"
    ds.PatientComments = "Paziente di test. CF: RSSMRA65C15D612Z. Tel: 055-123456"
    ds.AdditionalPatientHistory = (
        "Il paziente Mario Rossi riferisce dolori toracici ricorrenti. "
        "Contatto emergenza: Laura Rossi (moglie), tel. 333-9876543."
    )

    # Studio
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.StudyDate = "20240410"
    ds.StudyTime = "120000"
    ds.Modality = "CT"
    ds.StudyDescription = "TC Torace con contrasto - Dott.ssa Anna Ferrari"
    ds.SeriesDescription = "Torace HR"
    ds.InstitutionName = "Ospedale Santa Maria Nuova - Firenze"
    ds.ReferringPhysicianName = "Ferrari^Anna^^^Dott.ssa"
    ds.ImageComments = "Esame richiesto dal Prof. Bianchi, tel. 055-9876543"

    # Pixel data minimo (1×1 pixel CT)
    ds.Rows = 1
    ds.Columns = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = b"\x00\x00"

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/dicom",
        headers={"Content-Disposition": "attachment; filename=sample_paziente_rossi.dcm"},
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
        "Supporta documenti lunghi grazie al chunking automatico.\n\n"
        "Con `use_privacy_filter=true` (default) i modelli **OpenMed** e **OpenAI Privacy Filter** "
        "girano in parallelo e le loro entità vengono unite per massimizzare la copertura."
    ),
)
async def anonymize_text(req: TextRequest):
    t0 = time.perf_counter()
    pseudonym_map = None
    try:
        if req.placeholder_format == PlaceholderFormat.pseudonym:
            redacted, entities, pseudonym_map = anonymizer.pseudonymize(
                req.text,
                model_key=req.model.value,
                min_confidence=req.min_confidence,
                use_privacy_filter=req.use_privacy_filter,
            )
        else:
            redacted, entities = anonymizer.anonymize(
                req.text,
                model_key=req.model.value,
                placeholder_format=req.placeholder_format.value,
                min_confidence=req.min_confidence,
                use_privacy_filter=req.use_privacy_filter,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    model_used = AVAILABLE_MODELS[req.model.value]["model_id"]
    if req.use_privacy_filter:
        model_used = f"{model_used} + {PRIVACY_FILTER_MODEL_ID}"
    resp = AnonymizeResponse(
        anonymized=redacted,
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used=model_used,
        processing_time_ms=round(elapsed, 2),
    )
    resp.pseudonym_map = pseudonym_map
    return resp


@app.post(
    "/api/v1/anonymize/fhir",
    response_model=FhirAnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza una risorsa FHIR",
    description=(
        "Anonimizza i campi PII di una risorsa FHIR R4 (JSON) per **sostituzione strutturale** — "
        "nessun modello AI viene invocato. "
        "Ogni campo viene mappato direttamente al suo tipo PII secondo la specifica FHIR "
        "(`family`→`[NAME]`, `birthDate`→`[DATE_OF_BIRTH]`, `value` ContactPoint→`[PHONE_NUMBER]`/`[EMAIL]`, ecc.). "
        "Il contenuto narrativo (`div`, `comment`) viene sostituito integralmente con `[REDACTED]`. "
        "Preserva la struttura e i tipi della risorsa originale."
    ),
)
async def anonymize_fhir_endpoint(req: FhirRequest):
    t0 = time.perf_counter()
    try:
        anon_resource, entities = anonymize_fhir(
            req.resource,
            model_key="",
            min_confidence=0.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return FhirAnonymizeResponse(
        anonymized_resource=anon_resource,
        total_entities=len(entities),
        processing_time_ms=round(elapsed, 2),
    )


@app.post(
    "/api/v1/anonymize/hl7",
    response_model=AnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza un messaggio HL7 v2",
    description=(
        "Anonimizza un messaggio HL7 v2.x (ADT, ORU, ORM…) per **sostituzione strutturale** — "
        "nessun modello AI viene invocato. "
        "I campi PII noti (PID-3 MRN, PID-5 nome, PID-7 nascita, PID-11 indirizzo, PID-13/14 telefono, "
        "NK1, PV1, IN1, GT1…) ricevono un placeholder tipizzato direttamente dalla posizione di campo. "
        "I campi di testo libero (OBX-5, NTE-3) vengono sostituiti integralmente con `[REDACTED]`. "
        "La struttura del messaggio (segmenti, delimitatori) viene preservata."
    ),
)
async def anonymize_hl7_endpoint(req: Hl7Request):
    t0 = time.perf_counter()
    try:
        redacted, entities = anonymize_hl7(
            req.message,
            model_key="",
            placeholder_format=req.placeholder_format.value,
            min_confidence=0.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return AnonymizeResponse(
        anonymized=redacted,
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used="structural",
        processing_time_ms=round(elapsed, 2),
    )


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    tags=["NER / Rilevamento"],
    summary="Rileva entità PII (solo analisi)",
    description=(
        "Esegue il rilevamento NER e restituisce le entità trovate con posizione e confidenza, "
        "**senza modificare il testo originale**. Utile per ispezione e audit.\n\n"
        "Con `use_privacy_filter=true` (default) usa la pipeline dual-model "
        "(OpenMed + OpenAI Privacy Filter in parallelo)."
    ),
)
async def detect_entities(req: DetectRequest):
    t0 = time.perf_counter()
    try:
        entities = anonymizer.detect(
            req.text,
            model_key=req.model.value,
            min_confidence=req.min_confidence,
            use_privacy_filter=req.use_privacy_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    model_used = AVAILABLE_MODELS[req.model.value]["model_id"]
    if req.use_privacy_filter:
        model_used = f"{model_used} + {PRIVACY_FILTER_MODEL_ID}"
    return DetectResponse(
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used=model_used,
        processing_time_ms=round(elapsed, 2),
    )


@app.post(
    "/api/v1/anonymize/dicom",
    response_model=DicomAnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza metadati DICOM",
    description=(
        "Accetta un file DICOM (.dcm) come upload multipart e anonimizza i tag PII strutturati "
        "(PatientName, PatientID, ReferringPhysicianName…) con sostituzione diretta, "
        "e i campi di testo libero (PatientComments, StudyDescription…) con NER OpenMed. "
        "Restituisce un report JSON dei tag modificati più i bytes del file DICOM anonimizzato "
        "come campo base64 separato."
    ),
)
async def anonymize_dicom_endpoint(
    file: UploadFile = File(..., description="File DICOM (.dcm) da anonimizzare"),
    model: ModelKey = Form(ModelKey.italian_superclinical_base),
    min_confidence: float = Form(0.70),
    placeholder_format: PlaceholderFormat = Form(PlaceholderFormat.tag),
):
    if not PYDICOM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="pydicom non installato. Eseguire: pip install 'pydicom>=2.4.0'",
        )
    t0 = time.perf_counter()
    file_bytes = await file.read()
    try:
        modified_tags, ner_entities, _ = anonymize_dicom(
            file_bytes,
            model_key=model.value,
            min_confidence=min_confidence,
            placeholder_format=placeholder_format.value,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Errore lettura DICOM: {exc}")
    elapsed = (time.perf_counter() - t0) * 1000
    return DicomAnonymizeResponse(
        tags_modified=len(modified_tags),
        entities_found=ner_entities,
        modified_tags=[DicomFieldResult(**t) for t in modified_tags],
        model_used=AVAILABLE_MODELS[model.value]["model_id"],
        processing_time_ms=round(elapsed, 2),
    )


@app.post(
    "/api/v1/anonymize/cda",
    response_model=CdaAnonymizeResponse,
    tags=["Anonimizzazione"],
    summary="Anonimizza documento CDA / IHE XDS",
    description=(
        "Anonimizza un documento HL7 CDA R2 (Clinical Document Architecture) in formato XML "
        "per **sostituzione strutturale XPath** — nessun modello AI viene invocato. "
        "Gli elementi PII noti (patient/name, patientRole/addr, birthTime, telecom, id, assignedPerson/name, "
        "org names) vengono sostituiti direttamente con placeholder tipizzati. "
        "Il contenuto narrativo (`section/text`) viene sostituito integralmente con `[REDACTED]`. "
        "La struttura XML, i namespace e gli attributi vengono preservati. "
        "Compatibile con il profilo IHE XDS-SD e utilizzabile come middleware di "
        "de-identificazione prima dell'invio a repository XDS/XCA."
    ),
)
async def anonymize_cda_endpoint(req: CdaRequest):
    if not LXML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="lxml non installato. Eseguire: pip install 'lxml>=5.0.0'",
        )
    t0 = time.perf_counter()
    try:
        anon_xml, entities = anonymize_cda(
            req.document,
            model_key="",
            placeholder_format=req.placeholder_format.value,
            min_confidence=0.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return CdaAnonymizeResponse(
        anonymized_document=anon_xml,
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        processing_time_ms=round(elapsed, 2),
    )
