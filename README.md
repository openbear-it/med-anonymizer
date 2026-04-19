# Med-Anonymizer

REST API per la **de-identificazione clinica AI** di testi, risorse FHIR R4, messaggi HL7 v2, file DICOM e documenti CDA.  
Powered by [OpenMed](https://huggingface.co/OpenMed) — modelli NER BERT fine-tuned su 54 categorie PII (HIPAA / GDPR).

---

## Avvio locale

```bash
# 1. Clona e posizionati nel progetto
cd med-anonymizer

# 2. Crea e attiva un virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Installa PyTorch CPU (leggero) + dipendenze
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Avvia il server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Al primo avvio il modello (`italian-superclinical-base`) viene scaricato da HuggingFace (~450 MB).  
I download successivi usano la cache locale in `~/.cache/huggingface/`.

**Pagine disponibili:**

| URL | Descrizione |
|-----|-------------|
| http://localhost:8000/ui | Interfaccia di test |
| http://localhost:8000/info | Informazioni sul servizio |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/redoc | ReDoc |

---

## Endpoint principali

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `POST` | `/api/v1/anonymize/text` | Testo clinico libero |
| `POST` | `/api/v1/anonymize/fhir` | Risorsa FHIR R4 (JSON) |
| `POST` | `/api/v1/anonymize/hl7` | Messaggio HL7 v2.x |
| `POST` | `/api/v1/anonymize/dicom` | File DICOM (multipart) |
| `POST` | `/api/v1/anonymize/cda` | Documento CDA XML (IHE XDS/XCA) |
| `POST` | `/api/v1/detect` | Solo NER, senza redazione |
| `GET`  | `/api/v1/dicom/sample` | Scarica DICOM sintetico per test |
| `GET`  | `/api/v1/models` | Lista modelli disponibili |
| `GET`  | `/api/v1/health` | Health check |

### Esempio curl

```bash
curl -X POST http://localhost:8000/api/v1/anonymize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paziente Mario Rossi, CF: RSSMRA65C15D612Z, tel. 055-123456",
    "model": "italian-superclinical-base",
    "placeholder_format": "tag",
    "min_confidence": 0.70
  }'
```

### Anonimizzazione DICOM

```bash
curl -X POST http://localhost:8000/api/v1/anonymize/dicom \
  -F "file=@scan.dcm" \
  -F "model=italian-superclinical-base" \
  -F "min_confidence=0.70" \
  -F "placeholder_format=tag"
```

---

## Funzionalità

| Feature | Descrizione |
|---------|-------------|
| **Anonimizzazione** | Sostituisce PII con `[ENTITY_TYPE]` o `[REDACTED]` |
| **Pseudonimizzazione** | Assegna pseudonimi coerenti (`[NAME_001]`) con mappa reversibile |
| **Revisione umana** | Mostra entità evidenziate, l'operatore include/esclude/annota manualmente |
| **Pipeline DICOM** | Anonimizza 21 tag strutturati + campi testuali liberi con NER |
| **CDA / IHE** | De-identifica documenti HL7 CDA R2 XML (XDS, XCA, MHD) |
| **Estrai struttura** | Converte testo libero in FHIR Patient R4 o HL7 ADT^A01 tramite NER |

---

## Modelli disponibili (italiano)

Tutti ottimizzati per l'italiano, valutati su [AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) (subset IT).

| Chiave | Parametri | F1 Micro | Note |
|--------|-----------|----------|------|
| `italian-superclinical-base` *(default)* | 184M | 0.960 | DeBERTa-v3-base · veloce |
| `italian-bigmed-large` | 560M | 0.967 | XLM-RoBERTa-large · note cliniche |
| `italian-superclinical-large` | 434M | 0.973 | DeBERTa-v3-large · #1 italiano |

Rilevano **54 categorie PII** in italiano (nome, CF, MRN, data nascita, indirizzo, email, telefono, ecc.) conformi a GDPR.

---

## Confidenza minima (`min_confidence`)

Il parametro `min_confidence` (0.0–1.0) filtra le predizioni NER:

| Valore | Comportamento |
|--------|--------------|
| 0.50 | Molto sensibile — cattura più entità, più falsi positivi |
| **0.70** | **Bilanciato — default consigliato** |
| 0.90 | Preciso — può perdere entità borderline |

Per massimizzare la qualità: usa un modello più grande + revisione umana + abbassa la soglia.

---

## Licenza

Apache 2.0 — modelli OpenMed su [huggingface.co/OpenMed](https://huggingface.co/OpenMed)


---

## Avvio locale (senza Docker)

```bash
# 1. Clona e posizionati nel progetto
cd med-anonymizer

# 2. Crea e attiva un virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Installa PyTorch CPU (leggero) + dipendenze
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Avvia il server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Al primo avvio il modello (`BiomedBERT-Base-110M`) viene scaricato da HuggingFace (~450 MB).  
I download successivi usano la cache locale in `~/.cache/huggingface/`.

**Pagine disponibili:**

| URL | Descrizione |
|-----|-------------|
| http://localhost:8000/ui | Interfaccia di test |
| http://localhost:8000/info | Informazioni sul servizio |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/redoc | ReDoc |

---

## Endpoint principali

| Metodo | Path | Input |
|--------|------|-------|
| `POST` | `/api/v1/anonymize/text` | Testo clinico libero |
| `POST` | `/api/v1/anonymize/fhir` | Risorsa FHIR R4 (JSON) |
| `POST` | `/api/v1/anonymize/hl7`  | Messaggio HL7 v2.x |
| `POST` | `/api/v1/detect`         | Solo NER, senza redazione |
| `GET`  | `/api/v1/models`         | Lista modelli disponibili |
| `GET`  | `/api/v1/health`         | Health check |

### Esempio curl

```bash
curl -X POST http://localhost:8000/api/v1/anonymize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paziente Mario Rossi, CF: RSSMRA65C15D612Z, tel. 055-123456",
    "model": "italian-superclinical-base",
    "placeholder_format": "tag",
    "min_confidence": 0.70
  }'
```

---

## Modelli disponibili (italiano)

Tutti ottimizzati per l'italiano, valutati su [AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) (subset IT).

| Chiave | Parametri | F1 Micro | Note |
|--------|-----------|----------|------|
| `italian-superclinical-base` *(default)* | 184M | 0.960 | DeBERTa-v3-base · veloce |
| `italian-bigmed-large` | 560M | 0.967 | XLM-RoBERTa-large · note cliniche |
| `italian-superclinical-large` | 434M | 0.973 | DeBERTa-v3-large · #1 italiano |

Rilevano **54 categorie PII** in italiano (nome, CF, MRN, data nascita, indirizzo, email, telefono, ecc.) conformi a GDPR.

---

## Licenza

Apache 2.0 — modelli OpenMed su [huggingface.co/OpenMed](https://huggingface.co/OpenMed)
