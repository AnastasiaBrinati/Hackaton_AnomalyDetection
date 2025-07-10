# MLOps - Monitoraggio con Prometheus e Grafana

Questo progetto implementa un sistema di monitoraggio per modelli di Machine Learning utilizzando Prometheus e Grafana, come parte di un framework MLOps per il testing continuo.

## 📋 Prerequisiti

- **Sistema Operativo**: Linux (testato su Fedora 42)
- **Docker**: Per eseguire Prometheus e Grafana
- **Docker Compose**: Per orchestrare i container
- **Python 3.8+**: Per l'applicazione ML
- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Setup Iniziale

### 1. Configurazione Docker

Se Docker non è installato o configurato:

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

### 2. Verifica installazione Docker

```bash
# Verifica che Docker sia attivo
systemctl status docker

# Testa Docker (dovrebbe funzionare senza sudo dopo il riavvio del terminale)
docker --version
docker-compose --version
```


## ✅ Operazioni Già Completate

Durante la configurazione iniziale sono state già eseguite le seguenti operazioni:

1. **Docker avviato** e configurato per l'avvio automatico
2. **docker-compose.yml sistemato** (rimossa versione obsoleta)
3. **Utente aggiunto al gruppo docker** (richiede riavvio terminale)
4. **Container Prometheus e Grafana avviati** e funzionanti
5. **File di configurazione prometheus.yml** già presente

**Stato attuale**:
- ✅ Prometheus: http://localhost:9090
- ✅ Grafana: http://localhost:3000
- ✅ Container in esecuzione

## 🏗️ Struttura del Progetto

Il progetto è organizzato come segue:

```
MLOps/
├── README.md                 # Questo file
├── docker-compose.yml        # Configurazione container
├── Grafana.ipynb            # Notebook principale con tutorial
├── prometheus/
│   └── prometheus.yml       # Configurazione Prometheus
├── app.py                   # API Flask per il modello ML (da creare)
├── model.joblib            # Modello addestrato (generato automaticamente)
├── scaler.joblib           # Scaler per normalizzazione (generato automaticamente)
└── creditcard.csv          # Dataset (da scaricare)
```

## 🔧 Configurazione Servizi

### 1. Avvio Container

```bash
# Avvia Prometheus e Grafana
docker-compose up -d

# Verifica che i container siano in esecuzione
docker ps
```

### 2. Accesso ai Servizi

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin` (ti chiederà di cambiarla al primo accesso)

- **Prometheus**: http://localhost:9090
  - Interfaccia web per query e monitoring

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
Scarica il dataset da [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e posiziona il file `creditcard.csv` nella cartella principale.

### 2. Installazione Dipendenze Python
```bash
pip install pandas scikit-learn==1.3.2 flask prometheus-client requests joblib
```

### 3. Esecuzione del Notebook
Apri e esegui il notebook `Grafana.ipynb` per:
- Addestrare il modello
- Creare l'API Flask con metriche
- Generare traffico di test

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

### Docker daemon non in esecuzione
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Permessi negati per Docker
```bash
sudo usermod -aG docker $USER
# Poi riavvia il terminale
```

### Container non raggiungibili
Verifica che i container siano nella stessa rete:
```bash
docker network ls
docker network inspect mlops_default
```

### Prometheus non raccoglie metriche
1. Verifica che l'API Flask sia in esecuzione su porta 5000
2. Controlla http://localhost:5000/metrics
3. Verifica la configurazione in `prometheus/prometheus.yml`

## 📝 Note Importanti

- **Persistenza dati**: I dati di Grafana e Prometheus non sono persistenti. Per produzione, configura volumi Docker appropriati.
- **Sicurezza**: Le configurazioni sono per sviluppo/test. In produzione, configura autenticazione e HTTPS.
- **Monitoraggio**: Questo è un esempio educativo. In produzione, considera metriche aggiuntive come drift detection e model performance.

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

---

**Autore**: Tutorial MLOps - Testing con Prometheus e Grafana
**Data**: 2024 
