# 🌐 Soluzione Semplice per Grafana Cloud

## Problema
Grafana Cloud non può accedere a `localhost:9090` perché è sul tuo PC locale.

## ✅ Soluzione Veloce: Usa un servizio di monitoraggio gratuito

### Step 1: Prometeus.io (Gratuito)
1. Vai su https://prometheus.io/download/
2. Scarica Prometheus per Windows/Linux
3. Estrai e configura per leggere dalla tua app Railway

### Step 2: Configura prometheus.yml
```yaml
global:
  scrape_interval: 30s

scrape_configs:
  - job_name: 'railway-mlops'
    static_configs:
      - targets: ['mlopsgrafana-production.up.railway.app:443']
    scheme: https
    metrics_path: '/metrics'
```

### Step 3: Avvia Prometheus
```bash
./prometheus --config.file=prometheus.yml
```

### Step 4: Usa ngrok per tunnel
```bash
ngrok http 9090
```
Otterrai un URL pubblico come: `https://xyz.ngrok.io`

### Step 5: Configura Grafana Cloud
- Data Source URL: `https://xyz.ngrok.io`

## 🎯 Alternativa: Metrics dirette (Limitata)

Se vuoi vedere subito alcune metriche, usa un servizio come:
- **Grafana Cloud Synthetic Monitoring** per fare ping alla tua app
- **UptimeRobot** per monitoraggio basic

## 📊 Query disponibili
Una volta collegato, potrai usare:
```promql
predictions_total
rate(predictions_total[5m])
histogram_quantile(0.95, prediction_latency_seconds_bucket)
``` 