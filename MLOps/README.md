# MLOps - Sistema di Monitoraggio Avanzato con Prometheus e Grafana

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![Prometheus](https://img.shields.io/badge/Prometheus-latest-orange.svg)
![Grafana](https://img.shields.io/badge/Grafana-latest-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-required-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Sistema MLOps completo per il monitoraggio avanzato di modelli di Machine Learning utilizzando Prometheus e Grafana, con **simulatore di traffico**, **20+ metriche** e **deployment online**.

## ✨ Nuove Funzionalità

- 🚀 **Simulatore di traffico automatico** - Genera dati realistici continuamente
- 📊 **20+ metriche avanzate** - Qualità modello, sistema, business, errori
- 🌐 **Deployment online** - Deploy gratuito su Railway + Grafana Cloud
- 🔧 **Endpoint multipli** - Health check, simulazione traffico, info modello
- 💡 **Monitoraggio intelligente** - Latenza variabile, errori simulati, diversi tipi utente
- 📈 **Dashboard preconfigurate** - Template Grafana pronti all'uso

## Prerequisiti

- **Sistema Operativo**: Linux (testato su Fedora 42) o Windows 10/11
- **Docker**: Per eseguire Prometheus e Grafana
- **Docker Compose**: Per orchestrare i container
- **Python 3.8+**: Per l'applicazione ML
- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Setup Iniziale

### Setup per Linux (Testato su Fedora 42)

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

### Setup per Windows (Windows 10/11)

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

## Configurazione Completata e Testata

Questo setup è stato completamente testato su **Fedora 42** con tutte le problematiche risolte:

### **Problemi Risolti:**
1. **Docker permissions** - Utente aggiunto al gruppo docker
2. **SELinux blocking** - Configurato per permettere ai container di accedere ai file
3. **Docker networking** - Configurato IP dell'host invece di host.docker.internal
4. **Prometheus connection** - Configurazione corretta per Linux
5. **Flask API** - Errori di sintassi corretti e librerie installate
6. **Flask API hanging** - Risolto problema di Flask che si blocca e causa target "down"
7. **Grafana connection errors** - Risolto errore "querying the Prometheus API"

### **Stato Attuale Verificato:**
- ✅ **Prometheus**: http://localhost:9090 (raccoglie metriche)
- ✅ **Grafana**: http://localhost:3000 (pronto per dashboard)
- ✅ **Flask API**: http://localhost:5000 (modello ML attivo)
- ✅ **Connessione**: Prometheus → Flask API (health: up)
- ✅ **Logs**: Container Prometheus (172.18.0.x) fa richieste /metrics con successo

> ⚠️ **Nota**: Se Flask API si blocca (target Prometheus "down"), riavviare con `kill <PID>` e `python app.py`

## Struttura del Progetto

### File Principali
```
MLOps/
├── README.md                    # Questa guida
├── app.py                       # 🚀 API Flask avanzata con simulatore
├── requirements.txt             # 📦 Dipendenze Python
├── Dockerfile                   # 🐳 Containerizzazione per deployment
├── docker-compose.yml           # 🔧 Configurazione servizi locali
├── start_mlops.sh               # 🎯 Script avvio locale
├── deploy_railway.sh            # 🌐 Script deployment online
├── grafana_dashboard.json       # 📊 Dashboard Grafana preconfigurata
├── deployment_options.md        # 🚀 Guida deployment online
├── grafana_cloud_setup.md       # ☁️ Setup Grafana Cloud
├── SETUP_GUIDE.md               # 📋 Guida completa setup
└── prometheus/
    └── prometheus.yml           # ⚙️ Configurazione Prometheus
```

### File Generati/Scaricati
```
MLOps/
├── creditcard.csv              # 🔽 Dataset da Kaggle (150MB)
├── model.joblib                # 🤖 Modello addestrato
├── scaler.joblib               # 📊 Scaler per normalizzazione
├── MLOps/                      # 🐍 Environment virtuale Python
└── logs/                       # 📝 File di log dell'applicazione
```

## 🚀 Avvio Rapido

### Opzione 1: Avvio Locale Automatico
```bash
# Installa dipendenze e avvia tutto
chmod +x start_mlops.sh
./start_mlops.sh
```

### Opzione 2: Deployment Online (Gratuito)
```bash
# Deploy su Railway + Grafana Cloud
chmod +x deploy_railway.sh
./deploy_railway.sh
```

### Opzione 3: Avvio Manuale

#### 1. Installazione Dipendenze
```bash
# Attiva environment virtuale
source MLOps/bin/activate  # Linux
# MLOps\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt
```

#### 2. Avvio Servizi
```bash
# Avvia Prometheus e Grafana
docker-compose up -d

# Avvia app Flask
python app.py
```

## 🎯 Accesso ai Servizi

| Servizio | URL Locale | Credenziali |
|----------|------------|-------------|
| **App Flask** | http://localhost:5000 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |

## 📊 API Endpoints Avanzati

### Nuovi Endpoint Disponibili

#### 🔍 **Health Check**
```bash
curl http://localhost:5000/health
```
Restituisce stato sistema, uptime, predizioni totali e risorse.

#### 🎯 **Predizione Frodi**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "age": 35, "account_balance": 10000, "transaction_frequency": 12, "location_risk": 0.3}'
```
Restituisce predizione con probabilità frode, tipo utente, livello rischio.

#### 📈 **Metriche Prometheus**
```bash
curl http://localhost:5000/metrics
```
Espone tutte le 20+ metriche per Prometheus.

#### ℹ️ **Informazioni Modello**
```bash
curl http://localhost:5000/model_info
```
Restituisce versione modello, features, importance, ultimo training.

#### 🚦 **Simulazione Traffico**
```bash
curl -X POST http://localhost:5000/simulate_traffic \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 100}'
```
Genera traffico simulato per test e demo.

## 📊 Metriche Monitorate (20+ Metriche)

### 🎯 **Metriche di Predizione**
- `predictions_total` - Contatore predizioni per classe/utente/endpoint
- `prediction_latency_seconds` - Istogramma latenza predizioni
- `fraud_detection_score` - Score di confidenza frodi

### 🤖 **Metriche Qualità Modello**
- `model_accuracy` - Accuratezza modello
- `model_precision` - Precisione modello
- `model_recall` - Recall modello
- `model_f1_score` - F1-score modello
- `model_version` - Versione modello corrente
- `model_last_training_timestamp` - Timestamp ultimo training
- `feature_importance` - Importanza features

### 💻 **Metriche Sistema**
- `system_cpu_usage_percent` - Utilizzo CPU
- `system_memory_usage_percent` - Utilizzo memoria
- `system_disk_usage_percent` - Utilizzo disco

### 🌐 **Metriche HTTP**
- `http_requests_total` - Richieste HTTP per metodo/endpoint/status
- `http_request_duration_seconds` - Durata richieste HTTP

### 💼 **Metriche Business**
- `transaction_value_euros` - Valore transazioni
- `throughput_requests_per_second` - Throughput sistema
- `queue_size` - Dimensione coda

### ⚠️ **Metriche Errori**
- `errors_total` - Contatore errori per tipo/endpoint

## 🎨 Dashboard Grafana Preconfigurate

### Import Dashboard
1. Vai su **Grafana** → **Dashboard** → **Import**
2. Carica il file `grafana_dashboard.json`
3. Seleziona data source Prometheus

### Pannelli Disponibili
- **Predictions/sec** - Frequenza predizioni in tempo reale
- **Fraud Detection Rate** - Percentuale frodi rilevate
- **Response Time** - Latenza P50, P95, P99
- **Model Quality** - Accuracy, Precision, Recall, F1
- **System Resources** - CPU, Memory, Disk
- **Transaction Values** - Distribuzione valori transazioni
- **Error Rate** - Frequenza errori
- **Queue Size** - Dimensione coda sistema

### Query Prometheus di Esempio
```promql
# Throughput predizioni
rate(predictions_total[5m])

# Latenza P95
histogram_quantile(0.95, prediction_latency_seconds_bucket)

# Tasso frodi
rate(predictions_total{class_name="fraud"}[5m]) / rate(predictions_total[5m]) * 100

# Errori per minuto
rate(errors_total[1m]) * 60
```

## 🌐 Deployment Online (Gratuito)

### Opzione 1: Railway (Raccomandato)
```bash
# Deploy automatico
./deploy_railway.sh

# Vantaggi:
# ✅ Free tier generoso
# ✅ Deploy automatico da GitHub
# ✅ HTTPS incluso
# ✅ Scaling automatico
```

### Opzione 2: Grafana Cloud
```bash
# Dashboard gestita gratuitamente
# Segui grafana_cloud_setup.md

# Vantaggi:
# ✅ 10k metriche/mese gratis
# ✅ Alerting incluso
# ✅ Storage storico
# ✅ Gestione zero
```

### Combinazione Perfetta
1. **Railway** per l'app Flask
2. **Grafana Cloud** per dashboard
3. **Risultato**: Sistema MLOps completamente online gratis!

## 🔧 Funzionalità Avanzate

### 🤖 **Simulatore di Traffico Automatico**
- Genera richieste automatiche ogni 1-5 secondi
- Simula diversi tipi di utenti (premium, standard, basic)
- Latenza variabile per realismo
- Errori simulati (2% failure rate)

### 📊 **Metriche Business Intelligenti**
- Classifica transazioni per valore (high/low)
- Livelli di rischio (high/medium/low)
- Feature importance dinamica
- Versioning modello automatico

### 🚨 **Gestione Errori Avanzata**
- Errori simulati realistici
- Categorizzazione per tipo
- Tracking per endpoint
- Recovery automatico

### 🎯 **Monitoraggio Predittivo**
- Calcolo accuratezza in tempo reale
- Drift detection simulato
- Performance trending
- Alerting configurabile

## 🛠️ Comandi Utili

### Gestione Container
```bash
# Avvia servizi
docker-compose up -d

# Ferma servizi
docker-compose down

# Visualizza log
docker-compose logs -f

# Riavvia servizio specifico
docker-compose restart prometheus
```

### Test e Debug
```bash
# Verifica stato completo
curl http://localhost:5000/health | jq

# Genera traffico per test
curl -X POST http://localhost:5000/simulate_traffic \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 50}'

# Verifica target Prometheus
curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"[^"]*"'

# Monitor metriche in tempo reale
watch "curl -s http://localhost:5000/metrics | grep predictions_total"
```

### Deployment
```bash
# Deploy locale
./start_mlops.sh

# Deploy online
./deploy_railway.sh

# Setup Grafana Cloud
# Segui grafana_cloud_setup.md
```

## 🐛 Risoluzione Problemi

### Diagnosi Rapida
```bash
# 1. Verifica container
docker ps

# 2. Verifica Flask API
curl -s http://localhost:5000/health

# 3. Verifica target Prometheus
curl -s "http://localhost:9090/api/v1/targets" | grep health

# 4. Verifica processo Flask
ps aux | grep python | grep app.py
```

### Problemi Comuni

#### Target Prometheus "down"
```bash
# Riavvia Flask API
pkill -f "python app.py"
python app.py
```

#### Porta 5000 occupata
```bash
# Trova processo
lsof -i :5000
kill -9 <PID>
```

#### Container non si avviano
```bash
# Riavvia servizi
docker-compose down
docker-compose up -d
```

## 🎯 Utilizzo Scenario Reali

### 1. **Sviluppo Locale**
```bash
# Avvio rapido per sviluppo
./start_mlops.sh
# Accedi: http://localhost:3000
```

### 2. **Demo e Presentazioni**
```bash
# Genera traffico per demo
curl -X POST http://localhost:5000/simulate_traffic \
  -d '{"num_requests": 200}'
# Dashboard si popola immediatamente
```

### 3. **Produzione Online**
```bash
# Deploy su Railway
./deploy_railway.sh
# Setup Grafana Cloud
# Risultato: URL pubblico condivisibile
```

### 4. **Testing e Validazione**
```bash
# Test automatici
curl http://localhost:5000/health
curl http://localhost:5000/model_info
# Verifica tutte le metriche
```

## 🚀 Prossimi Passi

1. **Personalizza dashboard** con metriche specifiche
2. **Configura alerting** per soglie critiche
3. **Integra modello reale** sostituendo la simulazione
4. **Aggiungi A/B testing** per versioni modello
5. **Implementa drift detection** per qualità dati
6. **Configura CI/CD** per deployment automatico

## 📚 Risorse e Documentazione

- **Setup Guide**: `SETUP_GUIDE.md` - Guida completa setup
- **Deployment**: `deployment_options.md` - Tutte le opzioni deployment
- **Grafana Cloud**: `grafana_cloud_setup.md` - Setup dashboard cloud
- **Dashboard**: `grafana_dashboard.json` - Template dashboard
- **Scripts**: `start_mlops.sh`, `deploy_railway.sh` - Automazione

### Collegamenti Utili
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Railway Documentation](https://docs.railway.app/)
- [Grafana Cloud](https://grafana.com/products/cloud/)

## 🏆 Caratteristiche di Produzione

### ✅ **Pronto per Produzione**
- Containerizzato con Docker
- Metriche complete per monitoring
- Health checks implementati
- Logging strutturato
- Gestione errori robusta

### ✅ **Scalabile**
- Deployment cloud supportato
- Metriche ottimizzate per performance
- Architettura microservizi ready
- Load balancing compatible

### ✅ **Monitoraggio Completo**
- 20+ metriche business e sistema
- Dashboard preconfigurate
- Alerting configurabile
- Trending e analytics

---

**🎉 Congratulazioni!** Hai un sistema MLOps completo con monitoring avanzato, deployment online e dashboard professionali!

**Testato e Funzionante su:**
- 🐧 **Linux**: Fedora 42 con Docker e SELinux
- 🪟 **Windows**: Windows 10/11 con Docker Desktop
- ☁️ **Cloud**: Railway + Grafana Cloud (gratuito)

**Supporto**: Consulta le guide specifiche per configurazione avanzata e troubleshooting. 
