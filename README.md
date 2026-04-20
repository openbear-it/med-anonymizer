# Med-Anonymizer

REST API per la **de-identificazione clinica** di testi, risorse FHIR R4, messaggi HL7 v2, file DICOM e documenti CDA.  
Powered by [OpenMed](https://huggingface.co/OpenMed) — modelli NER fine-tuned su 54 categorie PII (HIPAA / GDPR).

---

## Avvio locale

```bash
# 1. Crea e attiva un virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Installa PyTorch CPU (leggero) + dipendenze
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Avvia il server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Al primo avvio il modello default (`italian-superclinical-large`) viene scaricato da HuggingFace (~900 MB).  
I download successivi usano la cache locale configurabile con `HF_HOME`.

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
| `POST` | `/api/v1/anonymize/text` | Testo clinico libero (NER) |
| `POST` | `/api/v1/anonymize/fhir` | Risorsa FHIR R4 — sostituzione strutturale |
| `POST` | `/api/v1/anonymize/hl7` | Messaggio HL7 v2.x — sostituzione strutturale |
| `POST` | `/api/v1/anonymize/dicom` | File DICOM (multipart) — tag strutturati + NER su campi liberi |
| `POST` | `/api/v1/anonymize/cda` | Documento CDA XML — sostituzione strutturale XPath |
| `POST` | `/api/v1/detect` | Solo NER, senza redazione |
| `GET`  | `/api/v1/dicom/sample` | Scarica DICOM sintetico per test |
| `GET`  | `/api/v1/models` | Lista modelli disponibili |
| `GET`  | `/api/v1/health` | Health check |

---

## Strategia di anonimizzazione

### Testo libero — `/anonymize/text` e `/detect`
Utilizza il modello NER OpenMed per rilevare le entità PII nel testo e sostituirle con placeholder.  
Supporta chunking automatico per documenti lunghi, pseudonimizzazione e revisione umana.

### Formati strutturati — FHIR, HL7, CDA
Sostituzione **strutturale pura**: nessun modello AI invocato.  
Ogni campo è mappato direttamente al suo tipo PII per nome di campo / posizione nel segmento / XPath.

| Formato | Strategia |
|---------|-----------|
| **FHIR** | `family`→`[NAME]`, `birthDate`→`[DATE_OF_BIRTH]`, `postalCode`→`[POSTAL_CODE]`, `value` (telecom)→`[PHONE_NUMBER]`/`[EMAIL]`, ecc. Il contenuto narrativo (`div`, `comment`) viene sostituito con `[REDACTED]`. |
| **HL7 v2** | PID-5→`[NAME]`, PID-7→`[DATE_OF_BIRTH]`, PID-11→`[ADDRESS]`, PID-19→`[SSN]`… OBX-5 e NTE-3 (testo libero)→`[REDACTED]`. |
| **CDA** | `patient/name`→`[NAME]`, `patientRole/addr`→`[ADDRESS]`, `birthTime@value`→`[DATE_OF_BIRTH]`, `telecom@value`→`[PHONE_NUMBER]`/`[EMAIL]`… `section/text` (narrativa)→`[REDACTED]`. |

### DICOM — `/anonymize/dicom`
Tag strutturati (PatientName, PatientID, ReferringPhysicianName…) sostituiti direttamente.  
Campi di testo libero (PatientComments, StudyDescription…) processati con NER.

---

## Esempi curl

### Testo libero (NER)
```bash
curl -X POST http://localhost:8000/api/v1/anonymize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paziente Mario Rossi, CF: RSSMRA65C15D612Z, tel. 055-123456",
    "model": "italian-superclinical-large",
    "placeholder_format": "tag",
    "min_confidence": 0.70
  }'
```

### FHIR (sostituzione strutturale — nessun parametro AI)
```bash
curl -X POST http://localhost:8000/api/v1/anonymize/fhir \
  -H "Content-Type: application/json" \
  -d '{
    "placeholder_format": "tag",
    "resource": {
      "resourceType": "Patient",
      "name": [{"family": "Rossi", "given": ["Mario"]}],
      "birthDate": "1965-03-15",
      "telecom": [{"system": "phone", "value": "055-123456"}]
    }
  }'
```

### DICOM (upload multipart)
```bash
curl -X POST http://localhost:8000/api/v1/anonymize/dicom \
  -F "file=@scan.dcm" \
  -F "model=italian-superclinical-large" \
  -F "min_confidence=0.70" \
  -F "placeholder_format=tag"
```

---

## Funzionalità

| Feature | Descrizione |
|---------|-------------|
| **Anonimizzazione testo** | NER → sostituisce PII con `[ENTITY_TYPE]` o `[REDACTED]` |
| **Anonimizzazione strutturata** | FHIR / HL7 / CDA: sostituzione diretta per campo, zero inferenza AI |
| **Pseudonimizzazione** | Pseudonimi coerenti `[NAME_001]` con mappa reversibile (solo testo libero) |
| **Revisione umana** | `/detect` per highlight interattivo + esclusione manuale delle entità |
| **Pipeline DICOM** | 21 tag strutturati a sostituzione diretta + NER su campi liberi |
| **CDA / IHE** | De-identifica HL7 CDA R2 XML per flussi XDS / XCA / MHD |

---

## Modelli disponibili (italiano)

Valutati su [AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) (subset IT).

| Chiave | Parametri | F1 | Note |
|--------|-----------|----|------|
| `italian-superclinical-base` | 184M | 0.960 | DeBERTa-v3-base · veloce |
| `italian-bigmed-large` | 560M | 0.967 | XLM-RoBERTa-large · note cliniche |
| `italian-superclinical-large` *(default)* | 434M | 0.973 | DeBERTa-v3-large · massima precisione |

Rilevano **54 categorie PII** conformi a GDPR / HIPAA Safe Harbor.

---

## Confidenza minima (`min_confidence`)

Applicabile solo a `/anonymize/text`, `/detect` e `/anonymize/dicom` (campi liberi).

| Valore | Comportamento |
|--------|--------------|
| 0.50 | Molto sensibile — più entità, più falsi positivi |
| **0.70** | **Bilanciato — default consigliato** |
| 0.90 | Preciso — può perdere entità borderline |

---

## Licenza

Apache 2.0 — modelli OpenMed su [huggingface.co/OpenMed](https://huggingface.co/OpenMed)
