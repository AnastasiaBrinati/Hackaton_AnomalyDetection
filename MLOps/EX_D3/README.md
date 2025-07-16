# MLOps Object Detection System

Un sistema completo di Machine Learning Operations (MLOps) per l'object detection usando YOLOv8, implementato con un'architettura a microservizi basata su Docker, Flask, Celery e Redis.

## 🏗️ Architettura del Sistema

Il sistema è composto da 3 componenti principali:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Service   │    │   ML Worker     │    │   Redis Broker  │
│   (Flask)       │◄──►│   (Celery)      │◄──►│   (Message Queue)│
│   Port: 5001    │    │   YOLOv8        │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Componenti:

1. **API Service** (`api_services/`):
   - **Flask API** che espone gli endpoint REST
   - Riceve le immagini codificate in base64
   - Invia i task a Celery per l'elaborazione asincrona
   - Restituisce i risultati dell'object detection

2. **ML Worker** (`ml_worker/`):
   - **Worker Celery** che esegue l'inferenza
   - Carica il modello **YOLOv8 nano** con lazy loading
   - Elabora le immagini e restituisce i risultati
   - Gestisce le librerie grafiche necessarie per OpenCV

3. **Redis**:
   - **Message broker** per la comunicazione tra API e Worker
   - **Result backend** per memorizzare i risultati dei task

4. **Shared** (`shared/`):
   - **Configurazione condivisa** tra API e Worker
   - Definizione dell'app Celery e dei task

## 🚀 Avvio Rapido

### Prerequisiti
- Docker e Docker Compose installati
- Almeno 2GB di RAM libera (per il download del modello YOLOv8)

### Avvio del Sistema
```bash
# Clona il repository
git clone <repository-url>
cd EX_D3

# Avvia tutto il sistema
docker-compose up -d

# Verifica che tutti i servizi siano attivi
docker-compose ps
```

### Verifica del Sistema
```bash
# Test dell'endpoint di health check
curl -X GET http://localhost:5001/health
# Risposta attesa: {"status":"OK"}
```

## 📡 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```
**Risposta:**
```json
{
  "status": "OK"
}
```

#### 2. Predict (Object Detection)
```http
POST /predict
Content-Type: application/json

{
  "image": "<base64-encoded-image>"
}
```

**Risposta:**
```json
{
  "task_id": "8726b5a0-58ac-44b4-a75f-5096624780e1"
}
```

#### 3. Get Result
```http
GET /result/<task_id>
```

**Possibili risposte:**

**Task in elaborazione:**
```json
{
  "status": "PENDING"
}
```

**Task completato con successo:**
```json
{
  "status": "SUCCESS",
  "result": "[{\"class\": \"bear\", \"confidence\": 0.67, \"bbox\": [2599.6, 834.5, 3915.8, 3241.5]}]"
}
```

**Task fallito:**
```json
{
  "status": "FAILED",
  "error": "Descrizione dell'errore"
}
```

## 🧪 Testing del Sistema

### Test Completo con Immagine
```bash
# 1. Codifica un'immagine in base64
base64 -w 0 testImage.jpg > /tmp/image_b64.txt

# 2. Crea il JSON della richiesta
echo "{\"image\":\"$(cat /tmp/image_b64.txt)\"}" > /tmp/request.json

# 3. Invia la richiesta
curl -X POST -H "Content-Type: application/json" \
     -d @/tmp/request.json \
     http://localhost:5001/predict

# 4. Ottieni il task_id dalla risposta e controlla il risultato
curl "http://localhost:5001/result/<task_id>"
```

### Risultati Attesi
Il sistema rileverà gli oggetti nell'immagine e restituirà:
- **class**: Nome della classe dell'oggetto (es. "bear", "person", "car")
- **confidence**: Livello di confidenza (0.0 - 1.0)
- **bbox**: Bounding box [x1, y1, x2, y2] in pixel

## 🗂️ Struttura del Progetto

```
EX_D3/
├── api_services/
│   ├── api.py              # Flask API endpoints
│   ├── requirements.txt    # Dipendenze Python per l'API
│   └── Dockerfile         # Container per l'API
├── ml_worker/
│   ├── worker.py          # Worker Celery (attualmente non usato)
│   ├── requirements.txt   # Dipendenze Python per ML (YOLOv8, OpenCV)
│   └── Dockerfile        # Container per il Worker
├── shared/
│   ├── __init__.py       # Rende shared un modulo Python
│   ├── celery_app.py     # Configurazione base di Celery
│   └── celery_config.py  # Task definitions e logica ML
├── docker-compose.yml    # Orchestrazione dei servizi
└── README.md            # Questa documentazione
```

## ⚙️ Configurazione

### Variabili d'Ambiente
- `FLASK_ENV=development`: Modalità di sviluppo per l'API
- `REDIS_URL=redis://redis:6379/0`: URL del broker Redis

### Porte Esposte
- `5001`: API Flask
- `6379`: Redis (per debug)

## 🔧 Troubleshooting

### Problemi Comuni

#### 1. Container ml_worker si ferma
```bash
# Controlla i logs
docker-compose logs ml_worker

# Possibili cause:
# - Memoria insufficiente per il download del modello
# - Errori di import delle librerie
```

#### 2. Errore "No module named 'shared'"
```bash
# Ricostruisci i container
docker-compose down
docker-compose build
docker-compose up -d
```

#### 3. Task sempre in stato PENDING
```bash
# Verifica che il worker sia connesso
docker-compose logs ml_worker | grep "celery@"
# Dovresti vedere: "celery@<container_id> ready."
```

#### 4. Errore di permissions con i volumi
```bash
# Il sistema non usa più i volume mount per /shared
# Se hai problemi, rimuovi i volumi:
docker-compose down -v
docker-compose up -d
```

## 🚀 Funzionalità Avanzate

### Lazy Loading del Modello
Il modello YOLOv8 viene caricato solo quando necessario:
- **Primo avvio**: Download automatico del modello (può richiedere 1-2 minuti)
- **Richieste successive**: Modello già in memoria, inferenza veloce

### Gestione degli Errori
Il sistema gestisce automaticamente:
- Immagini malformate
- Errori di decodifica base64
- Timeout di connessione
- Errori di inferenza del modello

### Scalabilità
Per aumentare la capacità:
```bash
# Aumenta il numero di worker
docker-compose up -d --scale ml_worker=3
```

## 📈 Performance

### Metriche Tipiche
- **Primo task**: 30-60 secondi (download modello)
- **Task successivi**: 1-3 secondi per immagine
- **Memoria utilizzata**: ~2GB per container ml_worker
- **Throughput**: 10-20 immagini/minuto per worker

## 🔐 Sicurezza

### Considerazioni per la Produzione
- Implementare autenticazione per l'API
- Validare le dimensioni delle immagini
- Limitare il rate delle richieste
- Usare HTTPS per le comunicazioni
- Configurare firewall per Redis

## 📝 Sviluppo e Contributi

### Setup per Sviluppo
```bash
# Monta i volumi per lo sviluppo
docker-compose up -d

# I volumi sono già configurati per:
# - ./api_services:/usr/src/app/api_services
# - ./ml_worker:/usr/src/app/ml_worker
```

### Aggiunta di Nuovi Modelli
1. Modifica `shared/celery_config.py`
2. Aggiorna `ml_worker/requirements.txt`
3. Ricostruisci il container: `docker-compose build ml_worker`

### Logging
```bash
# Logs in real-time
docker-compose logs -f

# Logs di un servizio specifico
docker-compose logs -f ml_worker
```

## 🎯 Casi d'Uso

Questo sistema è adatto per:
- **Analisi di immagini in tempo reale**
- **Sistemi di sorveglianza automatica**
- **Classificazione di contenuti multimediali**
- **Prototipazione rapida di applicazioni ML**
- **Educazione e ricerca in MLOps**

## 📚 Tecnologie Utilizzate

- **Python 3.13**
- **Flask** - Web framework per l'API
- **Celery** - Task queue per elaborazione asincrona
- **Redis** - Message broker e result backend
- **YOLOv8** - Modello di object detection
- **OpenCV** - Elaborazione delle immagini
- **Docker** - Containerizzazione
- **Docker Compose** - Orchestrazione dei servizi

## 📄 Licenza

Questo progetto è sviluppato per scopi educativi nell'ambito del corso MLOps.

---

**Sviluppato con ❤️ per il corso MLOps** 