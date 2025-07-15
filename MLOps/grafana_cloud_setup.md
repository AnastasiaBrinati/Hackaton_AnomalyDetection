# 🌐 Setup Grafana Cloud per Dashboard MLOps

## 🎯 Panoramica
Dopo aver deployato la tua app MLOps su Railway, puoi usare **Grafana Cloud** (gratuito) per visualizzare le metriche online senza dover hostare Grafana tu stesso.

## 📋 Step 1: Registrati su Grafana Cloud

1. Vai su https://grafana.com/products/cloud/
2. Clicca "Get started for free"
3. Crea account (14 giorni trial, poi free tier)
4. Crea un nuovo workspace

## 🔧 Step 2: Configura Data Source

### Opzione A: Prometheus Remote Write (Raccomandato)

1. **Nel tuo workspace Grafana Cloud:**
   - Vai su `Connections` → `Data Sources`
   - Copia l'URL del tuo Prometheus endpoint
   - Copia username e password

2. **Modifica la tua app per remote write:**

```python
# Aggiungi all'app.py
import os
from prometheus_client import CollectorRegistry, push_to_gateway
from prometheus_client.gateway import PushGateway

# Configurazione Grafana Cloud
GRAFANA_CLOUD_URL = os.environ.get('GRAFANA_CLOUD_URL', '')
GRAFANA_CLOUD_USER = os.environ.get('GRAFANA_CLOUD_USER', '')
GRAFANA_CLOUD_PASSWORD = os.environ.get('GRAFANA_CLOUD_PASSWORD', '')

def push_to_grafana_cloud():
    """Push metriche a Grafana Cloud"""
    if GRAFANA_CLOUD_URL and GRAFANA_CLOUD_USER:
        gateway = PushGateway(
            GRAFANA_CLOUD_URL,
            handler=lambda url, method, timeout, headers, data: requests.post(
                url, data=data, headers=headers, 
                auth=(GRAFANA_CLOUD_USER, GRAFANA_CLOUD_PASSWORD)
            )
        )
        try:
            push_to_gateway(gateway, job='mlops-dashboard', registry=None)
        except Exception as e:
            print(f"Errore push a Grafana Cloud: {e}")

# Aggiungi al background_simulator
def background_simulator():
    while True:
        # ... codice esistente ...
        
        # Push metriche a Grafana Cloud ogni 30 secondi
        if time.time() % 30 == 0:
            push_to_grafana_cloud()
```

### Opzione B: Prometheus Pubblico

1. **Esponi /metrics pubblicamente** (già fatto nella tua app)
2. **Configura External Prometheus:**

```yaml
# prometheus-cloud.yml
global:
  scrape_interval: 30s
  external_labels:
    environment: 'production'
    service: 'mlops-dashboard'

remote_write:
  - url: 'https://prometheus-prod-XX-prod-XX.grafana.net/api/prom/push'
    basic_auth:
      username: 'USERNAME_FROM_GRAFANA_CLOUD'
      password: 'PASSWORD_FROM_GRAFANA_CLOUD'

scrape_configs:
  - job_name: 'mlops-dashboard'
    static_configs:
      - targets: ['YOUR_RAILWAY_APP_URL.railway.app:443']
    scheme: https
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## 🚀 Step 3: Deploy con Variabili Ambiente

### Su Railway:
```bash
# Imposta variabili ambiente
railway variables set GRAFANA_CLOUD_URL=https://prometheus-prod-XX-prod-XX.grafana.net/api/prom/push
railway variables set GRAFANA_CLOUD_USER=your_username
railway variables set GRAFANA_CLOUD_PASSWORD=your_password

# Redeploy
railway up
```

## 📊 Step 4: Importa Dashboard

### Dashboard JSON per Grafana Cloud:
```json
{
  "dashboard": {
    "title": "MLOps Dashboard - Production",
    "panels": [
      {
        "title": "Predictions Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      },
      {
        "title": "Fraud Detection Rate",
        "type": "stat", 
        "targets": [
          {
            "expr": "rate(predictions_total{class_name=\"fraud\"}[5m]) / rate(predictions_total[5m]) * 100",
            "legendFormat": "Fraud Rate %"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, prediction_latency_seconds_bucket)",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, prediction_latency_seconds_bucket)", 
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## 🎯 Step 5: Configura Alerting

### Alert Rules di esempio:
```yaml
groups:
  - name: mlops-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, prediction_latency_seconds_bucket) > 1.0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High prediction latency"
          
      - alert: LowThroughput
        expr: rate(predictions_total[5m]) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low prediction throughput"
```

## 🔧 Script di Setup Completo

```bash
#!/bin/bash
# setup_grafana_cloud.sh

echo "🌐 Setup Grafana Cloud per MLOps Dashboard"

# Chiedi credenziali Grafana Cloud
read -p "🔑 Inserisci Grafana Cloud URL: " GRAFANA_URL
read -p "👤 Inserisci Username: " GRAFANA_USER  
read -sp "🔒 Inserisci Password: " GRAFANA_PASSWORD
echo

# Configura su Railway
railway variables set GRAFANA_CLOUD_URL=$GRAFANA_URL
railway variables set GRAFANA_CLOUD_USER=$GRAFANA_USER
railway variables set GRAFANA_CLOUD_PASSWORD=$GRAFANA_PASSWORD

echo "✅ Variabili configurate su Railway"

# Redeploy
railway up

echo "🚀 Deploy completato con integrazione Grafana Cloud!"
echo "📊 Le tue metriche dovrebbero apparire su Grafana Cloud in pochi minuti"
```

## 🎯 Test e Verifica

### 1. Verifica che le metriche arrivino:
```bash
# Testa endpoint metriche
curl https://YOUR_APP.railway.app/metrics

# Genera traffico per test
curl -X POST https://YOUR_APP.railway.app/simulate_traffic \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 100}'
```

### 2. Controlla su Grafana Cloud:
- Vai su `Explore`
- Prova query: `rate(predictions_total[5m])`
- Dovrebbe mostrare dati delle predizioni

### 3. Importa dashboard:
- Vai su `Dashboard` → `New` → `Import`
- Carica il JSON della dashboard
- Salva e visualizza

## 💡 Tips per Produzione

1. **Ottimizza frequenza scraping** (30s invece di 5s)
2. **Usa labels significativi** per filtrare metriche
3. **Configura retention policy** per dati storici
4. **Imposta alerting** per metriche critiche
5. **Monitora costi** del free tier Grafana Cloud

---

🎉 **Congratulazioni!** Hai una dashboard MLOps completamente online e funzionante! 