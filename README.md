# Med-Anonymizer

REST API per la **de-identificazione clinica AI** di testi, risorse FHIR R4 e messaggi HL7 v2.  
Powered by [OpenMed](https://huggingface.co/OpenMed) — modelli NER BERT fine-tuned su 54 categorie PII (HIPAA / GDPR).

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

## Build Docker (multi-arch AMD64 + ARM64)

```bash
# Build e push su registry
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/yourorg/med-anonymizer:1.0.0 \
  --push .

# Deploy su Kubernetes
kubectl apply -f kubernetes/
```

---

## Licenza

Apache 2.0 — modelli OpenMed su [huggingface.co/OpenMed](https://huggingface.co/OpenMed)
