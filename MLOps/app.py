import joblib
import pandas as pd
import time
import random
import threading
import psutil
import os
from datetime import datetime
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, make_wsgi_app
import numpy as np

# 1. Inizializza l'applicazione Flask
app = Flask(__name__)

# 2. Carica il modello e lo scaler addestrati
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    model = None
    scaler = None

# 3. Definisci le metriche di Prometheus

# Metriche principali delle predizioni
PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['class_name', 'endpoint', 'user_type']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time taken to process a prediction request',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Metriche di qualità del modello
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
MODEL_PRECISION = Gauge('model_precision', 'Current model precision')
MODEL_RECALL = Gauge('model_recall', 'Current model recall')
MODEL_F1_SCORE = Gauge('model_f1_score', 'Current model F1 score')

# Metriche di sistema
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')

# Metriche HTTP
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Metriche di business
TRANSACTION_VALUE = Histogram(
    'transaction_value_euros',
    'Transaction value in euros',
    ['transaction_type', 'risk_level']
)

FRAUD_DETECTION_SCORE = Histogram(
    'fraud_detection_score',
    'Fraud detection confidence score',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metriche di errori
ERRORS_TOTAL = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

# Metriche di coda e throughput
QUEUE_SIZE = Gauge('queue_size', 'Current queue size')
THROUGHPUT = Gauge('throughput_requests_per_second', 'Current throughput in requests per second')

# Metriche di modello
MODEL_VERSION = Gauge('model_version', 'Current model version')
MODEL_LAST_TRAINING = Gauge('model_last_training_timestamp', 'Last training timestamp')
FEATURE_IMPORTANCE = Gauge('feature_importance', 'Feature importance scores', ['feature_name'])

# 4. Variabili globali per simulazione
request_count = 0
start_time = time.time()
recent_predictions = []
model_version = 1.2

# Simulazione di feature importance
feature_names = ['amount', 'age', 'account_balance', 'transaction_frequency', 'location_risk']
feature_importance_values = [0.35, 0.25, 0.20, 0.15, 0.05]

# 5. Funzioni di utilità per simulazione
def generate_fake_transaction():
    """Genera una transazione fake per simulazione"""
    return {
        'amount': random.uniform(10, 5000),
        'age': random.randint(18, 80),
        'account_balance': random.uniform(100, 50000),
        'transaction_frequency': random.randint(1, 50),
        'location_risk': random.uniform(0, 1)
    }

def simulate_model_prediction(data):
    """Simula una predizione del modello"""
    if not MODEL_LOADED or model is None or scaler is None:
        # Simulazione semplice se il modello non è caricato
        fraud_prob = random.uniform(0, 1)
        if data['amount'] > 2000 or data['location_risk'] > 0.7:
            fraud_prob += 0.3
        return 1 if fraud_prob > 0.5 else 0, fraud_prob
    
    df = pd.DataFrame([data])
    data_scaled = scaler.transform(df)
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1] if hasattr(model, 'predict_proba') else random.uniform(0, 1)
    return prediction, prob

def update_model_metrics():
    """Aggiorna le metriche del modello con valori simulati"""
    MODEL_ACCURACY.set(random.uniform(0.85, 0.95))
    MODEL_PRECISION.set(random.uniform(0.82, 0.92))
    MODEL_RECALL.set(random.uniform(0.80, 0.90))
    MODEL_F1_SCORE.set(random.uniform(0.83, 0.91))
    MODEL_VERSION.set(model_version)
    MODEL_LAST_TRAINING.set(time.time() - random.randint(86400, 604800))  # 1-7 giorni fa
    
    # Aggiorna feature importance
    for feature, importance in zip(feature_names, feature_importance_values):
        FEATURE_IMPORTANCE.labels(feature_name=feature).set(importance + random.uniform(-0.05, 0.05))

def update_system_metrics():
    """Aggiorna le metriche di sistema"""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
    SYSTEM_DISK_USAGE.set(psutil.disk_usage('/').percent)

def calculate_throughput():
    """Calcola il throughput attuale"""
    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff > 0:
        THROUGHPUT.set(request_count / time_diff)

# 6. Endpoints dell'applicazione

@app.route('/predict', methods=['POST'])
def predict():
    global request_count, recent_predictions
    start_time_req = time.time()
    request_count += 1
    
    # Simula errori occasionali
    if random.random() < 0.02:  # 2% di errori
        ERRORS_TOTAL.labels(error_type='internal_error', endpoint='predict').inc()
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status_code='500').inc()
        return jsonify({'error': 'Internal server error'}), 500
    
    # Simula latenza variabile
    if random.random() < 0.1:  # 10% delle richieste più lente
        time.sleep(random.uniform(0.1, 0.5))
    
    try:
        # Prendi i dati JSON dalla richiesta o genera dati fake
        data = request.get_json()
        if not data:
            data = generate_fake_transaction()
        
        # Simula diversi tipi di utenti
        user_type = random.choice(['premium', 'standard', 'basic'])
        
        # Esegui la predizione
        prediction, fraud_prob = simulate_model_prediction(data)
        class_name = "fraud" if prediction == 1 else "non_fraud"
        
        # Aggiorna le metriche
        PREDICTIONS_TOTAL.labels(class_name=class_name, endpoint='predict', user_type=user_type).inc()
        
        # Calcola e registra la latenza
        latency = time.time() - start_time_req
        PREDICTION_LATENCY.observe(latency)
        HTTP_REQUEST_DURATION.labels(method='POST', endpoint='predict').observe(latency)
        
        # Metriche di business
        transaction_type = 'high_value' if data['amount'] > 1000 else 'low_value'
        risk_level = 'high' if fraud_prob > 0.7 else 'medium' if fraud_prob > 0.3 else 'low'
        
        TRANSACTION_VALUE.labels(transaction_type=transaction_type, risk_level=risk_level).observe(data['amount'])
        FRAUD_DETECTION_SCORE.observe(fraud_prob)
        
        # Salva predizione recente per calcoli di qualità
        recent_predictions.append({
            'prediction': prediction,
            'actual': random.choice([0, 1]),  # Simula label reale
            'timestamp': time.time()
        })
        
        # Mantieni solo le ultime 100 predizioni
        if len(recent_predictions) > 100:
            recent_predictions = recent_predictions[-100:]
        
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status_code='200').inc()
        
        return jsonify({
            'prediction': int(prediction),
            'class_name': class_name,
            'fraud_probability': round(fraud_prob, 3),
            'user_type': user_type,
            'transaction_value': data['amount'],
            'risk_level': risk_level
        })
        
    except Exception as e:
        ERRORS_TOTAL.labels(error_type='prediction_error', endpoint='predict').inc()
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status_code='400').inc()
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint di health check"""
    start_time_req = time.time()
    
    health_status = {
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'uptime_seconds': time.time() - start_time,
        'total_predictions': request_count,
        'system_info': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    }
    
    latency = time.time() - start_time_req
    HTTP_REQUEST_DURATION.labels(method='GET', endpoint='health').observe(latency)
    HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='health', status_code='200').inc()
    
    return jsonify(health_status)

@app.route('/simulate_traffic', methods=['POST'])
def simulate_traffic():
    """Endpoint per simulare traffico per test"""
    num_requests = request.get_json().get('num_requests', 10)
    
    for _ in range(num_requests):
        fake_data = generate_fake_transaction()
        with app.test_client() as client:
            client.post('/predict', json=fake_data)
    
    HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='simulate_traffic', status_code='200').inc()
    return jsonify({'message': f'Simulated {num_requests} requests'})

@app.route('/model_info', methods=['GET'])
def model_info():
    """Informazioni sul modello"""
    HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='model_info', status_code='200').inc()
    
    return jsonify({
        'model_version': model_version,
        'model_loaded': MODEL_LOADED,
        'feature_names': feature_names,
        'feature_importance': dict(zip(feature_names, feature_importance_values)),
        'last_training': datetime.fromtimestamp(time.time() - 86400).isoformat()
    })

# 7. Background tasks per simulazione continua
def background_simulator():
    """Simula traffico continuo in background"""
    while True:
        time.sleep(random.uniform(1, 5))  # Richiesta ogni 1-5 secondi
        
        # Simula richieste automatiche
        if random.random() < 0.7:  # 70% probabilità di generare richiesta
            fake_data = generate_fake_transaction()
            with app.test_client() as client:
                client.post('/predict', json=fake_data)
        
        # Aggiorna metriche di sistema
        update_system_metrics()
        update_model_metrics()
        calculate_throughput()
        
        # Simula dimensione coda
        QUEUE_SIZE.set(random.randint(0, 20))

# 8. Esponi l'endpoint `/metrics` per Prometheus
from werkzeug.middleware.dispatcher import DispatcherMiddleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# 9. Avvia l'applicazione
if __name__ == '__main__':
    # Inizializza le metriche del modello
    update_model_metrics()
    
    # Avvia il simulatore in background
    simulator_thread = threading.Thread(target=background_simulator, daemon=True)
    simulator_thread.start()
    
    # Configurazione porta per deployment
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 MLOps App avviata con simulatore di traffico!")
    print(f"📊 Metrics endpoint: http://localhost:{port}/metrics")
    print(f"🔍 Health check: http://localhost:{port}/health")
    print(f"🎯 Predict endpoint: http://localhost:{port}/predict")
    print(f"ℹ️  Model info: http://localhost:{port}/model_info")
    
    # Esegui su porta configurabile, accessibile da tutte le interfacce di rete
    app.run(host='0.0.0.0', port=port, debug=False)