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

from .anonymizer import AVAILABLE_MODELS, DEFAULT_MODEL, anonymizer
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

APP_VERSION = "1.2.0"
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
        "da testi clinici, risorse FHIR R4, messaggi HL7 v2.x, file DICOM e documenti CDA (IHE XDS/XCA).\n\n"
        "I modelli NER OpenMed rilevano **54 categorie di PII** incluse nome, data di nascita, "
        "SSN, MRN, indirizzo, telefono, email, e molte altre.\n\n"
        "### Funzionalit\u00e0\n"
        "| Feature | Descrizione |\n"
        "|---------|-------------|\n"
        "| **Anonimizzazione** | Sostituisce PII con `[ENTITY_TYPE]` o `[REDACTED]` |\n"
        "| **Pseudonimizzazione** | Assegna pseudonimi coerenti `[NAME_001]` con mappa reversibile |\n"
        "| **Revisione umana** | Endpoint `/detect` per highlight interattivo + annotazione manuale |\n"
        "| **Pipeline DICOM** | 21 tag strutturati + campi testuali con NER |\n"
        "| **CDA / IHE** | De-identifica documenti HL7 CDA R2 XML (XDS, XCA, MHD) |\n"
        "| **Estrai struttura** | Converte testo in FHIR Patient R4 o HL7 ADT^A01 via `/detect` |\n\n"
        "### Confidenza minima (`min_confidence`)\n"
        "Filtra le predizioni NER (range 0.0\u20131.0). Default **0.70** \u2013 bilanciamento ottimale. "
        "Valori pi\u00f9 bassi aumentano la sensitivit\u00e0; valori pi\u00f9 alti la precisione.\n\n"
        "### Conformit\u00e0 normativa\n"
        "Supporta flussi di de-identificazione conformi a **HIPAA Safe Harbor** e **GDPR Art.\u202f4(5)**.\n\n"
        "### Pagine utili\n"
        "- [\U0001f527 Interfaccia di test](/ui)\n"
        "- [\u2139\ufe0f Informazioni sul servizio](/info)\n"
        "- [\U0001f4d6 ReDoc](/redoc)\n"
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
        "Supporta documenti lunghi grazie al chunking automatico."
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
            )
        else:
            redacted, entities = anonymizer.anonymize(
                req.text,
                model_key=req.model.value,
                placeholder_format=req.placeholder_format.value,
                min_confidence=req.min_confidence,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    resp = _to_response(redacted, entities, req.model.value, elapsed)
    resp.pseudonym_map = pseudonym_map
    return resp


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
    pseudonym_map = None
    try:
        if req.placeholder_format == PlaceholderFormat.pseudonym:
            normalised = req.message.replace("\r\n", "\n").replace("\r", "\n")
            redacted_norm, entities, pseudonym_map = anonymizer.pseudonymize(
                normalised,
                model_key=req.model.value,
                min_confidence=req.min_confidence,
            )
            redacted = redacted_norm.replace("\n", "\r")
        else:
            redacted, entities = anonymize_hl7(
                req.message,
                model_key=req.model.value,
                placeholder_format=req.placeholder_format.value,
                min_confidence=req.min_confidence,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    resp = _to_response(redacted, entities, req.model.value, elapsed)
    resp.pseudonym_map = pseudonym_map
    return resp


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    tags=["NER / Rilevamento"],
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
        "Anonimizza un documento HL7 CDA R2 (Clinical Document Architecture) in formato XML. "
        "Tutti i nodi di testo vengono analizzati con NER OpenMed e le entità PII sostituite. "
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
            model_key=req.model.value,
            placeholder_format=req.placeholder_format.value,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return CdaAnonymizeResponse(
        anonymized_document=anon_xml,
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
        model_used=AVAILABLE_MODELS[req.model.value]["model_id"],
        processing_time_ms=round(elapsed, 2),
    )
