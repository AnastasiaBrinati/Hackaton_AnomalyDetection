# 🚀 Guida Setup MLOps - Fraud Detection

## Panoramica
Questo sistema MLOps completo include:
- 🤖 **App Flask** con simulatore di traffico
- 📊 **Prometheus** per la raccolta metriche
- 📈 **Grafana** per la visualizzazione
- 🔄 **Traffico simulato** per dati realistici

## 📋 Prerequisiti

### Sistema
- Docker e Docker Compose
- Python 3.8+
- pip

### Verifica dipendenze
```bash
docker --version
python --version
pip --version
```

## 🚀 Avvio Rapido

### Opzione 1: Script automatico
```bash
chmod +x start_mlops.sh
./start_mlops.sh
```

### Opzione 2: Manuale

1. **Installa dipendenze Python**
   ```bash
   pip install -r requirements.txt
   ```

2. **Avvia servizi Docker**
   ```bash
   docker-compose up -d
   ```

3. **Avvia app Flask**
   ```bash
   python app.py
   ```

## 🎯 Accesso ai Servizi

| Servizio | URL | Credenziali |
|----------|-----|-------------|
| **App Flask** | http://localhost:5000 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |

## 📊 Endpoint Disponibili

### App Flask
- `GET /health` - Health check
- `POST /predict` - Predizione frodi
- `GET /metrics` - Metriche Prometheus  
- `GET /model_info` - Info modello
- `POST /simulate_traffic` - Simula traffico

### Esempi di utilizzo
```bash
# Health check
curl http://localhost:5000/health

# Predizione manuale
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "age": 35, "account_balance": 10000, "transaction_frequency": 12, "location_risk": 0.3}'

# Simula 50 transazioni
curl -X POST http://localhost:5000/simulate_traffic \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 50}'
```

## 📈 Dashboard Grafana

### Setup iniziale
1. Vai su http://localhost:3000
2. Login: `admin` / `admin`
3. Aggiungi Data Source:
   - Type: **Prometheus**
   - URL: **http://prometheus:9090**
   - Save & Test

### Importa Dashboard
1. Vai su **Dashboard → Import**
2. Carica il file `grafana_dashboard.json`
3. Seleziona il data source Prometheus

### Metriche disponibili
- **Predictions/sec** - Frequenza predizioni
- **Fraud Rate** - Percentuale frodi
- **Latency** - Tempo di risposta
- **Model Quality** - Accuracy, Precision, Recall, F1
- **System Resources** - CPU, Memory, Disk
- **Transaction Values** - Distribuzione valori
- **Error Rate** - Frequenza errori
- **Queue Size** - Dimensione coda

## 🔧 Personalizzazione

### Modifica parametri di simulazione
Nel file `app.py`, modifica:
```python
# Frequenza traffico (linea ~290)
time.sleep(random.uniform(1, 5))  # 1-5 secondi

# Percentuale errori (linea ~171)
if random.random() < 0.02:  # 2% errori

# Tipi di utenti (linea ~185)
user_type = random.choice(['premium', 'standard', 'basic'])
```

### Aggiungi nuove metriche
```python
# Nuova metrica
NEW_METRIC = Counter('new_metric_total', 'Description', ['label'])

# Uso
NEW_METRIC.labels(label='value').inc()
```

## 🐛 Troubleshooting

### Problema: Grafana non vede dati
1. Verifica che l'app Flask sia attiva
2. Controlla http://localhost:5000/metrics
3. Verifica configurazione Prometheus
4. Riavvia servizi: `docker-compose restart`

### Problema: Errore caricamento modello
L'app funziona anche senza modello reale, usando simulazione

### Problema: Porte occupate
Cambia le porte in `docker-compose.yml`:
```yaml
ports:
  - "9091:9090"  # Prometheus
  - "3001:3000"  # Grafana
```

## 📝 Comandi Utili

```bash
# Verifica servizi
docker-compose ps

# Log servizi
docker-compose logs -f

# Riavvia servizi
docker-compose restart

# Ferma tutto
docker-compose down

# Verifica metriche
curl http://localhost:5000/metrics | grep predictions

# Monitoraggio in tempo reale
watch "curl -s http://localhost:5000/health | jq"
```

## 🎯 Prossimi Passi

1. **Personalizza dashboard** con metriche specifiche
2. **Aggiungi alerting** per soglie critiche
3. **Integra modello reale** sostituendo la simulazione
4. **Configura storage** per dati storici
5. **Aggiungi autenticazione** per ambiente produzione

## 🆘 Supporto

In caso di problemi:
1. Verifica prerequisiti
2. Controlla log: `docker-compose logs`
3. Riavvia servizi: `docker-compose restart`
4. Verifica firewall/antivirus

---

🎉 **Buon monitoraggio!** La dashboard si popolerà automaticamente con dati realistici. 