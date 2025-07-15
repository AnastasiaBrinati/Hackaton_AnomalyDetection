#!/bin/bash

echo "🚀 Avvio dell'ambiente MLOps..."

# 1. Verifica che Docker sia in esecuzione
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker non è in esecuzione. Avvia Docker e riprova."
    exit 1
fi

# 2. Installa le dipendenze Python
echo "📦 Installazione delle dipendenze Python..."
pip install -r requirements.txt

# 3. Avvia i servizi Docker (Prometheus e Grafana)
echo "🐳 Avvio di Prometheus e Grafana..."
docker-compose up -d

# 4. Attendi che i servizi siano pronti
echo "⏳ Attesa che i servizi siano pronti..."
sleep 10

# 5. Verifica che i servizi siano attivi
echo "🔍 Verifica dei servizi:"
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus: http://localhost:9090"
else
    echo "❌ Prometheus non è accessibile"
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana: http://localhost:3000"
else
    echo "❌ Grafana non è accessibile"
fi

# 6. Avvia l'app Flask
echo "🎯 Avvio dell'app Flask..."
echo "📊 Metriche disponibili su: http://localhost:5000/metrics"
echo "🔍 Health check: http://localhost:5000/health"
echo "📈 Grafana Dashboard: http://localhost:3000"
echo "🎛️  Prometheus: http://localhost:9090"
echo ""
echo "🚀 L'app Flask inizierà ora e genererà automaticamente traffico simulato!"
echo "📊 Vai su Grafana (http://localhost:3000) per vedere le metriche in tempo reale"
echo "   User: admin, Password: admin"

python app.py 