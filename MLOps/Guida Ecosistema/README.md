# Guida Completa a Grafana e Prometheus

## Introduzione

**Prometheus** è un sistema di monitoraggio e alerting open-source che raccoglie metriche temporali da vari target. **Grafana** è una piattaforma di visualizzazione che trasforma questi dati in dashboard interattive e comprensibili.

## Architettura del Sistema

```
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│ Applicazioni│────▶│  Prometheus  │────▶│    Grafana    │
│   (Target)  │      │   (Storage)  │      │(Visualization)│
└─────────────┘      └──────────────┘      └───────────────┘
```

## Installazione

### 1. Installare Prometheus

**Docker:**
```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

**Linux (Binary):**
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*
./prometheus --config.file=prometheus.yml
```

### 2. Installare Grafana

**Docker:**
```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana-enterprise
```

**Linux (APT):**
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/enterprise/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana-enterprise
sudo systemctl start grafana-server
```

## Configurazione Prometheus

### File prometheus.yml Base

```yaml
global:
  scrape_interval: 15s      # Frequenza di raccolta metriche
  evaluation_interval: 15s  # Frequenza valutazione regole

scrape_configs:
  # Monitoraggio di Prometheus stesso
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Esempio: Node Exporter per metriche del sistema
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  # Esempio: Applicazione custom
  - job_name: 'my-app'
    static_configs:
      - targets: ['app1.example.com:8080', 'app2.example.com:8080']
```

## Exporters Comuni

### 1. Node Exporter (Metriche Sistema)
```bash
docker run -d \
  --name node-exporter \
  -p 9100:9100 \
  prom/node-exporter
```

**Metriche fornite:**
- CPU, memoria, disco
- Network I/O
- Filesystem
- Load average

### 2. MySQL Exporter
```bash
docker run -d \
  --name mysql-exporter \
  -p 9104:9104 \
  -e DATA_SOURCE_NAME="user:password@(hostname:3306)/" \
  prom/mysqld-exporter
```

### 3. Redis Exporter
```bash
docker run -d \
  --name redis-exporter \
  -p 9121:9121 \
  -e REDIS_ADDR="redis://localhost:6379" \
  oliver006/redis_exporter
```

### 4. Custom Application Metrics

Per esporre metriche custom dalla tua applicazione:

**Python (prometheus_client):**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Definisci metriche
request_count = Counter('app_requests_total', 'Total requests')
request_duration = Histogram('app_request_duration_seconds', 'Request duration')
active_users = Gauge('app_active_users', 'Active users')

# Usa le metriche nel codice
@request_duration.time()
def process_request():
    request_count.inc()
    # La tua logica qui
    
# Avvia server metriche
start_http_server(8000)
```

**Node.js (prom-client):**
```javascript
const client = require('prom-client');
const express = require('express');

// Crea metriche
const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_ms',
  help: 'Duration of HTTP requests in ms',
  labelNames: ['method', 'route', 'status_code']
});

// Middleware per tracciare richieste
app.use((req, res, next) => {
  const end = httpRequestDuration.startTimer();
  res.on('finish', () => {
    end({ method: req.method, route: req.path, status_code: res.statusCode });
  });
  next();
});

// Endpoint metriche
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', client.register.contentType);
  res.end(await client.register.metrics());
});
```

## Configurazione Grafana

### 1. Accesso Iniziale
- URL: http://localhost:3000
- Username: admin
- Password: admin (cambiare al primo accesso)

### 2. Aggiungere Prometheus come Data Source

1. Vai su Configuration → Data Sources
2. Clicca "Add data source"
3. Seleziona "Prometheus"
4. Configura:
   - URL: http://localhost:9090
   - Access: Server (default)
   - Scrape interval: 15s

### 3. Creare una Dashboard

#### Dashboard per Monitoraggio Sistema

1. Crea nuova dashboard
2. Aggiungi pannelli:

**CPU Usage:**
```promql
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

**Memory Usage:**
```promql
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
```

**Disk Usage:**
```promql
100 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100)
```

**Network Traffic:**
```promql
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])
```

#### Dashboard per Applicazione Web

**Request Rate:**
```promql
sum(rate(http_requests_total[5m])) by (status)
```

**Response Time (95th percentile):**
```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
```

**Error Rate:**
```promql
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
```

**Active Users:**
```promql
app_active_users
```

## Best Practices per Query PromQL

### 1. Aggregazioni Utili

**Media per gruppo:**
```promql
avg by (instance) (node_cpu_usage)
```

**Somma totale:**
```promql
sum(http_requests_total)
```

**Rate per counter:**
```promql
rate(http_requests_total[5m])
```

### 2. Filtri e Selettori

**Filtro per label:**
```promql
http_requests_total{status="200", method="GET"}
```

**Regex matching:**
```promql
http_requests_total{status=~"2.."}
```

**Negazione:**
```promql
http_requests_total{status!="200"}
```

## Alerting

### Configurazione Alert in Prometheus

**prometheus.yml:**
```yaml
rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

**alerts.yml:**
```yaml
groups:
  - name: system
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% (current value: {{ $value }}%)"

      - alert: LowDiskSpace
        expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Less than 10% disk space remaining"
```

### Alert in Grafana

1. Vai al pannello → Edit → Alert
2. Configura condizioni:
   - Evaluate every: 1m
   - For: 5m
   - Conditions: WHEN avg() OF query(A, 5m, now) IS ABOVE 80
3. Configura notifiche (email, Slack, etc.)

## Scegliere in Base ai Bisogni

### Monitoraggio Infrastruttura Base
- **Node Exporter**: CPU, memoria, disco
- **cAdvisor**: Per container Docker
- **Dashboard**: Node Exporter Full

### Applicazione Web
- **Metriche custom**: Request rate, latency, errors
- **Database exporters**: MySQL/PostgreSQL exporter
- **Dashboard**: Web application dashboard

### Microservizi
- **Service mesh metrics**: Istio, Linkerd
- **Distributed tracing**: Integrazione con Jaeger
- **Dashboard**: Service mesh dashboard

### IoT/Edge Computing
- **Pushgateway**: Per metriche batch/ephemeral
- **Remote write**: Per storage centralizzato
- **Dashboard**: Device monitoring

## Tips per Performance

### 1. Ottimizzazione Storage
```yaml
# prometheus.yml
global:
  scrape_interval: 30s  # Aumenta per ridurre storage
  
storage:
  tsdb:
    retention.time: 15d  # Mantieni dati per 15 giorni
    retention.size: 10GB # Limite storage
```

### 2. Query Efficienti
- Usa `rate()` invece di `delta()` per counter
- Limita il time range delle query
- Pre-aggrega dove possibile con recording rules

### 3. Recording Rules
```yaml
groups:
  - name: aggregations
    interval: 30s
    rules:
      - record: instance:node_cpu:rate5m
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

## Troubleshooting Comune

### Prometheus non raccoglie metriche
1. Verifica targets: http://localhost:9090/targets
2. Controlla firewall/network
3. Verifica formato metriche: curl http://target:port/metrics

### Grafana non mostra dati
1. Test query in Prometheus UI
2. Verifica time range
3. Controlla data source configuration

### Performance issues
1. Aumenta scrape_interval
2. Usa recording rules
3. Ottimizza query PromQL

## Conclusione

Questa guida fornisce le basi per implementare un sistema di monitoraggio completo. Ricorda di:
- Iniziare con metriche essenziali
- Aggiungere gradualmente complessità
- Documentare le tue dashboard
- Impostare alert significativi
- Mantenere il sistema aggiornato

Per approfondimenti, consulta la documentazione ufficiale di Prometheus e Grafana.
