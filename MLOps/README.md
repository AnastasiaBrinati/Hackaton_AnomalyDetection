# MLOps - Monitoraggio con Prometheus e Grafana

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.1-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.0-orange.svg)
![Prometheus](https://img.shields.io/badge/Prometheus-latest-orange.svg)
![Grafana](https://img.shields.io/badge/Grafana-latest-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-required-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Questo progetto implementa un sistema di monitoraggio per modelli di Machine Learning utilizzando Prometheus e Grafana, come parte di un framework MLOps per il testing continuo.

## 📋 Prerequisiti

- **Sistema Operativo**: Linux (testato su Fedora 42)
- **Docker**: Per eseguire Prometheus e Grafana
- **Docker Compose**: Per orchestrare i container
- **Python 3.8+**: Per l'applicazione ML
- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Setup Iniziale

### 🐧 **Setup per Linux (Testato su Fedora 42)**

#### 1. Configurazione Docker

```bash
# Installa Docker (se non presente)
sudo dnf install docker docker-compose -y

# Avvia il servizio Docker
sudo systemctl start docker

# Abilita Docker all'avvio del sistema
sudo systemctl enable docker

# Aggiungi il tuo utente al gruppo docker (per evitare sudo)
sudo usermod -aG docker $USER
```

> ⚠️ **Importante**: Dopo aver aggiunto l'utente al gruppo docker, **riavvia il terminale** o fai **logout/login** per attivare i nuovi permessi.

#### 2. Risoluzione Problemi SELinux (Fedora/RHEL)

Su Fedora, SELinux può bloccare l'accesso ai file. Risolvi con:

```bash
# Cambia contesto SELinux per la directory prometheus
sudo chcon -Rt container_file_t prometheus/

# Permetti ai container di accedere ai file degli utenti
sudo setsebool -P container_use_cephfs on
```

#### 3. Configurazione Prometheus per Linux

Il file `prometheus/prometheus.yml` deve essere configurato con l'IP dell'host invece di `host.docker.internal`:

```bash
# Trova l'IP del tuo host
HOST_IP=$(hostname -I | awk '{print $1}')
echo "IP dell'host: $HOST_IP"

# Modifica la configurazione di Prometheus
sed -i "s/host.docker.internal:5000/${HOST_IP}:5000/" prometheus/prometheus.yml
```

Verifica il contenuto del file:
```bash
cat prometheus/prometheus.yml
```

Dovrebbe mostrare qualcosa come:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model-app'
    static_configs:
      - targets: ['192.168.10.66:5000']  # Il tuo IP
```

#### 4. Verifica installazione Docker

```bash
# Verifica che Docker sia attivo
systemctl status docker

# Testa Docker (dovrebbe funzionare senza sudo dopo il riavvio del terminale)
docker --version
docker-compose --version
```

### 🪟 **Setup per Windows (Windows 10/11)**

#### 1. Installazione Docker Desktop

1. **Scarica Docker Desktop** da [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Installa** seguendo il wizard di installazione
3. **Riavvia** il computer quando richiesto
4. **Avvia Docker Desktop** dal menu Start

#### 2. Configurazione WSL2 (Raccomandato)

```powershell
# Apri PowerShell come Amministratore e esegui:
wsl --install

# Riavvia il computer
# Dopo il riavvio, configura una distribuzione Linux (es. Ubuntu)
```

#### 3. Verifica installazione Docker

```powershell
# Apri PowerShell o CMD e testa:
docker --version
docker-compose --version

# Verifica che Docker sia in esecuzione
docker run hello-world
```

#### 4. Configurazione Prometheus per Windows

Su Windows, `host.docker.internal` funziona correttamente, quindi **NON modificare** il file `prometheus.yml`:

```yaml
# prometheus/prometheus.yml (lascia così su Windows)
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model-app'
    static_configs:
      - targets: ['host.docker.internal:5000']  # Funziona su Windows
```

#### 5. Installazione Python

1. **Scarica Python** da [python.org](https://www.python.org/downloads/)
2. **Durante l'installazione**, seleziona "Add Python to PATH"
3. **Verifica installazione**:
```cmd
python --version
pip --version
```

#### 6. Creazione Environment Virtuale

```cmd
# Crea environment virtuale
python -m venv MLOps

# Attiva environment (Windows CMD)
MLOps\Scripts\activate

# Attiva environment (PowerShell)
MLOps\Scripts\Activate.ps1

# Dovresti vedere (MLOps) nel prompt
```


## ✅ Configurazione Completata e Testata

Questo setup è stato completamente testato su **Fedora 42** con tutte le problematiche risolte:

### **Problemi Risolti:**
1. **Docker permissions** - Utente aggiunto al gruppo docker
2. **SELinux blocking** - Configurato per permettere ai container di accedere ai file
3. **Docker networking** - Configurato IP dell'host invece di host.docker.internal
4. **Prometheus connection** - Configurazione corretta per Linux
5. **Flask API** - Errori di sintassi corretti e librerie installate

### **Stato Attuale Verificato:**
- ✅ **Prometheus**: http://localhost:9090 (raccoglie metriche)
- ✅ **Grafana**: http://localhost:3000 (pronto per dashboard)
- ✅ **Flask API**: http://localhost:5000 (modello ML attivo)
- ✅ **Connessione**: Prometheus → Flask API (health: up)

## 🏗️ Struttura del Progetto

Il progetto è organizzato come segue:

### **📁 File Versionati (nel repository)**
```
MLOps/
├── README.md                 # Questa guida
├── .gitignore               # Esclusioni Git per dataset/modelli
├── docker-compose.yml        # Configurazione container Docker
├── Grafana.ipynb            # Notebook principale con tutorial
├── app.py                   # API Flask per il modello ML
└── prometheus/
    └── prometheus.yml       # Configurazione Prometheus
```

### **📁 File Generati/Scaricati (esclusi dal repository)**
```
MLOps/
├── creditcard.csv          # 🔽 Dataset da scaricare da Kaggle (150MB)
├── model.joblib            # 🤖 Modello addestrato (generato dal notebook)
├── scaler.joblib           # 📊 Scaler per normalizzazione (generato dal notebook)
├── MLOps/                  # 🐍 Environment virtuale Python (cartella)
├── logs/                   # 📝 File di log dell'applicazione
└── .ipynb_checkpoints/     # 📓 Checkpoint Jupyter Notebook
```

> 📝 **Nota Git**: I file grandi (dataset, modelli) e temporanei sono esclusi dal version control tramite `.gitignore` per mantenere il repository leggero e pulito.

## 🔧 Configurazione Servizi

### 1. Avvio Container

#### 🐧 **Linux:**
```bash
# Avvia Prometheus e Grafana
sudo docker-compose up -d

# Verifica che i container siano in esecuzione
sudo docker ps
```

#### 🪟 **Windows:**
```cmd
# Avvia Prometheus e Grafana
docker-compose up -d

# Verifica che i container siano in esecuzione
docker ps
```

**Output atteso (entrambi i sistemi):**
```
NAMES        STATUS          PORTS
grafana      Up X minutes    0.0.0.0:3000->3000/tcp
prometheus   Up X minutes    0.0.0.0:9090->9090/tcp
```

### 2. Verifica Connessione Prometheus → Flask

#### 🐧 **Linux:**
```bash
# Verifica che Prometheus veda l'app Flask
curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"[^"]*"'

# Dovrebbe restituire: "health":"up"
```

#### 🪟 **Windows (PowerShell):**
```powershell
# Verifica che Prometheus veda l'app Flask
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/targets" | Select-String '"health":"[^"]*"'

# Oppure usa curl se installato:
curl -s "http://localhost:9090/api/v1/targets" | findstr "health"
```

### 3. Accesso ai Servizi

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin` (ti chiederà di cambiarla al primo accesso)

- **Prometheus**: http://localhost:9090 ⚠️ **NON** `http://prometheus:9090`
  - Interfaccia web per query e monitoring
  - `prometheus:9090` funziona solo dall'interno dei container Docker

- **Flask API**: http://localhost:5000
  - Endpoint: `POST /predict` per predizioni
  - Endpoint: `GET /metrics` per metriche Prometheus

## 📊 Primo Accesso a Grafana

### 1. Login Iniziale
1. Vai su http://localhost:3000
2. Username: `admin`, Password: `admin`
3. Cambia la password quando richiesto

### 2. Configurazione Data Source
1. Menu laterale (⚙️) → **Data Sources**
2. **Add data source** → **Prometheus**
3. URL: `http://prometheus:9090`
4. **Save & Test**

### 3. Creazione Dashboard
1. Menu laterale (📊) → **Dashboards** → **New Dashboard**
2. **Add visualization**
3. Configura i pannelli come descritto nel notebook

## 🤖 Setup Modello ML

### 1. Download Dataset

⚠️ **Importante**: Il dataset non è incluso nel repository per motivi di dimensione e licenza.

1. **Registrati su Kaggle** (se non hai già un account)
2. **Scarica il dataset** da [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
3. **Estrai e posiziona** il file `creditcard.csv` nella cartella principale del progetto

```bash
# La struttura dovrebbe essere:
MLOps/
├── creditcard.csv          # ← File da scaricare
├── docker-compose.yml
├── Grafana.ipynb
├── app.py
├── prometheus/
└── ...
```

> 📝 **Nota**: Il file `creditcard.csv` è escluso dal version control tramite `.gitignore` perché è un file di grandi dimensioni (150MB) e ha restrizioni di licenza Kaggle.

### 2. Attivazione Environment Virtuale

#### 🐧 **Linux:**
```bash
# SEMPRE attiva l'environment virtuale prima di qualsiasi operazione Python
source MLOps/bin/activate

# Dovresti vedere (MLOps) nel prompt del terminale
```

#### 🪟 **Windows:**
```cmd
# CMD
MLOps\Scripts\activate

# PowerShell
MLOps\Scripts\Activate.ps1

# Dovresti vedere (MLOps) nel prompt
```

### 3. Installazione Dipendenze Python

**Entrambi i sistemi (con environment attivo):**
```bash
# Installa le librerie
pip install pandas scikit-learn flask prometheus-client requests joblib

# Verifica installazione
python -c "import pandas, flask, prometheus_client, joblib; print('✅ Tutte le librerie OK!')"
```

### 4. Correzione File app.py
Il file `app.py` deve avere le correzioni applicate:
- Linea 32: `df = pd.DataFrame([data])` (non `pd.DataFrame(data, index=False)`)
- Linea 38: `prediction = model.predict(data_scaled)[0]` (prendi primo elemento)

### 5. Esecuzione del Notebook
Apri e esegui il notebook `Grafana.ipynb` per:
- Addestrare il modello (crea `model.joblib` e `scaler.joblib`)
- Comprendere come funziona l'API Flask
- Simulare traffico per testare le dashboard

### 6. Avvio dell'API Flask
```bash
# Con environment attivo, avvia l'API
python app.py

# Dovresti vedere:
# * Running on all addresses (0.0.0.0)
# * Running on http://127.0.0.1:5000
```

### 7. Test dell'API

#### 🐧 **Linux:**
```bash
# Test endpoint metrics
curl http://localhost:5000/metrics

# Test predizione
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
```

#### 🪟 **Windows (PowerShell):**
```powershell
# Test endpoint metrics
Invoke-RestMethod -Uri "http://localhost:5000/metrics"

# Test predizione
$body = @{
    Time = 0; V1 = -1.36; V2 = -0.07; V3 = 2.54; V4 = 1.38; V5 = -0.34;
    V6 = 0.46; V7 = 0.24; V8 = 0.10; V9 = 0.36; V10 = 0.09; V11 = -0.55;
    V12 = -0.62; V13 = -0.99; V14 = -0.31; V15 = 1.47; V16 = -0.47;
    V17 = 0.21; V18 = 0.03; V19 = 0.40; V20 = 0.25; V21 = -0.02;
    V22 = 0.28; V23 = -0.11; V24 = 0.07; V25 = 0.13; V26 = -0.19;
    V27 = 0.13; V28 = -0.02; Amount = 149.62
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body -ContentType "application/json"
```

#### 🪟 **Windows (CMD con curl):**
```cmd
# Se hai curl installato su Windows
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"Time\": 0, \"V1\": -1.36, \"V2\": -0.07, \"V3\": 2.54, \"V4\": 1.38, \"V5\": -0.34, \"V6\": 0.46, \"V7\": 0.24, \"V8\": 0.10, \"V9\": 0.36, \"V10\": 0.09, \"V11\": -0.55, \"V12\": -0.62, \"V13\": -0.99, \"V14\": -0.31, \"V15\": 1.47, \"V16\": -0.47, \"V17\": 0.21, \"V18\": 0.03, \"V19\": 0.40, \"V20\": 0.25, \"V21\": -0.02, \"V22\": 0.28, \"V23\": -0.11, \"V24\": 0.07, \"V25\": 0.13, \"V26\": -0.19, \"V27\": 0.13, \"V28\": -0.02, \"Amount\": 149.62}"
```

## 🔍 Metriche Monitorate

Il sistema traccia le seguenti metriche:

1. **predictions_total**: Contatore delle predizioni totali (etichettate per classe)
2. **prediction_latency_seconds**: Istogramma dei tempi di risposta
3. **model_accuracy**: Gauge per monitorare l'accuratezza del modello

## 📈 Dashboard Grafana Suggerite

### Pannello 1: Latenza P95
```promql
histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
```

### Pannello 2: Throughput
```promql
sum(rate(predictions_total[5m]))
```

### Pannello 3: Distribuzione Predizioni
```promql
sum(rate(predictions_total[5m])) by (class_name)
```

## 🛠️ Comandi Utili

### Gestione Container
```bash
# Avvia i servizi
docker-compose up -d

# Ferma i servizi
docker-compose down

# Visualizza log
docker-compose logs -f

# Riavvia un servizio specifico
docker-compose restart prometheus
```

### Debugging
```bash
# Verifica configurazione
docker-compose config

# Stato dei container
docker ps

# Log di un container specifico
docker logs prometheus
```

## 🚨 Risoluzione Problemi

### 🐧 **Problemi Specifici Linux**

#### Docker daemon non in esecuzione
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

#### Permessi negati per Docker
```bash
sudo usermod -aG docker $USER
# Poi riavvia il terminale o fai logout/login
```

#### SELinux blocca Prometheus (Fedora/RHEL)
```bash
# Errore: "avc: denied { read } for comm="prometheus""
sudo chcon -Rt container_file_t prometheus/
sudo setsebool -P container_use_cephfs on
sudo docker-compose restart prometheus
```

### 🪟 **Problemi Specifici Windows**

#### Docker Desktop non si avvia
1. **Verifica WSL2**: Assicurati che WSL2 sia installato e funzionante
```powershell
wsl --list --verbose
```

2. **Riavvia Docker Desktop**: Dal system tray, fai clic destro su Docker → Restart
3. **Verifica Hyper-V**: Su Windows Pro/Enterprise, assicurati che Hyper-V sia abilitato

#### PowerShell Execution Policy
Se ricevi errori nell'esecuzione di script PowerShell:
```powershell
# Apri PowerShell come Amministratore
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Problemi con host.docker.internal
Su Windows, se `host.docker.internal` non funziona:
```cmd
# Usa localhost invece
# Modifica prometheus.yml sostituendo host.docker.internal con localhost
```

#### Python non trovato
```cmd
# Verifica installazione Python
python --version

# Se non funziona, prova:
py --version

# Assicurati che Python sia nel PATH
```

### 🔧 **Problemi Comuni (Linux & Windows)**

### Prometheus non si connette all'API Flask
**Problema**: Target health "down" in Prometheus

**Soluzione**: Configura IP dell'host invece di `host.docker.internal`
```bash
# 1. Trova IP dell'host
HOST_IP=$(hostname -I | awk '{print $1}')
echo "IP dell'host: $HOST_IP"

# 2. Modifica prometheus.yml
sed -i "s/host.docker.internal:5000/${HOST_IP}:5000/" prometheus/prometheus.yml

# 3. Riavvia Prometheus
sudo docker-compose restart prometheus

# 4. Verifica connessione
curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"[^"]*"'
# Dovrebbe restituire: "health":"up"
```

### Errori import librerie Python
**Problema**: "Import flask could not be resolved"

**Soluzione**: Attiva l'environment virtuale
```bash
source MLOps/bin/activate
pip install pandas scikit-learn flask prometheus-client requests joblib
```

### Errori di sintassi in app.py
**Problema**: `pd.DataFrame(data, index=False)` non funziona

**Soluzione**: Correzioni necessarie
```python
# Riga 32: 
df = pd.DataFrame([data])  # NON pd.DataFrame(data, index=False)

# Riga 38:
prediction = model.predict(data_scaled)[0]  # Prendi primo elemento
```

### Flask restituisce 404
**Normale**: Flask ha solo endpoint `/predict` (POST) e `/metrics` (GET)
- GET su `/` restituisce 404 (normale)
- Usa `POST /predict` per predizioni
- Usa `GET /metrics` per metriche

### Errore "File 'creditcard.csv' non trovato"
**Problema**: Il dataset non è incluso nel repository

**Soluzione**: Scarica manualmente il dataset
```bash
# Errore tipico nel notebook:
# FileNotFoundError: [Errno 2] No such file or directory: 'creditcard.csv'

# Soluzione:
# 1. Vai su https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 2. Registrati/accedi a Kaggle
# 3. Scarica il dataset (creditcard.csv)
# 4. Posizionalo nella cartella principale del progetto
```

### Modello non trovato (model.joblib/scaler.joblib)
**Problema**: I file del modello sono generati automaticamente

**Soluzione**: Esegui prima il notebook
```python
# Nel notebook Grafana.ipynb, esegui la cella 3:
# "Addestramento del Modello di Machine Learning"
# Questo creerà automaticamente:
# - model.joblib
# - scaler.joblib
```

### Container non raggiungibili
```bash
# Verifica stato container
sudo docker ps

# Verifica rete Docker
docker network ls
docker network inspect mlops_default

# Riavvia se necessario
sudo docker-compose down && sudo docker-compose up -d
```

### Prometheus non raccoglie metriche
1. Verifica che l'API Flask sia in esecuzione su porta 5000
2. Controlla http://localhost:5000/metrics
3. Verifica la configurazione in `prometheus/prometheus.yml`
4. Controlla targets in http://localhost:9090/targets

### Errore "Error returned querying the Prometheus API"
**Problema**: Grafana non si connette a Prometheus

**Soluzione**: Usa URL corretta nel data source
- ✅ **Corretta**: `http://prometheus:9090` (da Grafana)
- ❌ **Sbagliata**: `http://localhost:9090` (da Grafana)

## 📝 Note Importanti

### **🐧 Linux (Testato su Fedora 42)**
- **Tutte le configurazioni** sono state testate e funzionano correttamente
- **SELinux**: Su Fedora/RHEL, SELinux deve essere configurato per permettere l'accesso ai file
- **Docker networking**: Usa IP dell'host invece di `host.docker.internal`
- **Environment virtuale**: SEMPRE attiva `source MLOps/bin/activate` prima di operazioni Python
- **Sudo**: Richiesto per comandi Docker fino al riavvio del terminale

### **🪟 Windows (Windows 10/11)**
- **Docker Desktop**: Richiede WSL2 o Hyper-V
- **Docker networking**: `host.docker.internal` funziona correttamente
- **Environment virtuale**: Usa `MLOps\Scripts\activate` (CMD) o `MLOps\Scripts\Activate.ps1` (PowerShell)
- **PowerShell**: Potrebbe richiedere modifica Execution Policy
- **Python**: Assicurati che sia nel PATH durante l'installazione

### **🔧 Generale (Entrambi i Sistemi)**
- **Persistenza dati**: I dati di Grafana e Prometheus non sono persistenti. Per produzione, configura volumi Docker appropriati.
- **Sicurezza**: Le configurazioni sono per sviluppo/test. In produzione, configura autenticazione e HTTPS.
- **Monitoraggio**: Questo è un esempio educativo. In produzione, considera metriche aggiuntive come drift detection e model performance.
- **URL corrette**: 
  - Dal browser: `http://localhost:9090` (Prometheus), `http://localhost:3000` (Grafana)
  - Da Grafana: `http://prometheus:9090` (data source Prometheus)
- **Porte**: Assicurati che le porte 3000, 5000, e 9090 siano libere

## 🎯 Prossimi Passi

1. **Alerting**: Configura alert in Grafana per soglie critiche
2. **A/B Testing**: Implementa versioning del modello
3. **Model Drift**: Aggiungi monitoraggio per data drift
4. **Logging**: Integra logging strutturato
5. **CI/CD**: Automatizza deployment con pipeline

## 📚 Risorse Utili

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## ✅ Checklist di Verifica

Prima di considerare il setup completo, verifica che tutti questi elementi funzionino:

### **Container Docker**

#### 🐧 **Linux:**
```bash
sudo docker ps
# Dovresti vedere 'grafana' e 'prometheus' in esecuzione
```

#### 🪟 **Windows:**
```cmd
docker ps
# Dovresti vedere 'grafana' e 'prometheus' in esecuzione
```

### **Connessione Prometheus → Flask**

#### 🐧 **Linux:**
```bash
curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"[^"]*"'
# Dovrebbe restituire: "health":"up"
```

#### 🪟 **Windows:**
```powershell
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/targets" | Select-String "health"
# Dovrebbe mostrare: "health":"up"
```

### **API Flask**

#### 🐧 **Linux:**
```bash
# Test endpoint metrics
curl http://localhost:5000/metrics | head -5

# Test predizione
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
# Dovrebbe restituire: {"class_name":"non_fraud","prediction":0}
```

#### 🪟 **Windows:**
```powershell
# Test endpoint metrics
Invoke-RestMethod -Uri "http://localhost:5000/metrics"

# Test predizione (esempio semplificato)
$testData = '{"Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $testData -ContentType "application/json"
# Dovrebbe restituire: {"class_name":"non_fraud","prediction":0}
```

### **Accesso Web**
- ✅ **Grafana**: http://localhost:3000 (admin/admin)
- ✅ **Prometheus**: http://localhost:9090 (interfaccia web)

### **Configurazione Grafana**
1. Accedi a Grafana
2. Aggiungi data source Prometheus (`http://prometheus:9090`)
3. Verifica connessione con "Save & Test"

### **Metriche in Prometheus**
1. Vai su http://localhost:9090
2. Cerca `predictions_total` nella query
3. Dovresti vedere dati delle predizioni

**Se tutti questi test passano, il tuo sistema MLOps è completamente funzionante!** 🎉

---

**Autore**: Tutorial MLOps - Testing con Prometheus e Grafana  
**Data**: 2025  
**Testato su**: 
- 🐧 **Linux**: Fedora 42 con Docker e SELinux  
- 🪟 **Windows**: Windows 10/11 con Docker Desktop e WSL2

**Compatibilità**: Cross-platform (Linux/Windows) 
