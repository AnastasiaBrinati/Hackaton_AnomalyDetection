import joblib
import pandas as pd
import time
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, make_wsgi_app

# 1. Inizializza l'applicazione Flask
app = Flask(__name__)

# 2. Carica il modello e lo scaler addestrati
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# 3. Definisci le metriche di Prometheus
PREDICTIONS_TOTAL = Counter(
    'predictions_total', # Nome della metrica
    'Total number of predictions made',
    ['class_name'] # Etichetta per distinguere le classi
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time taken to process a prediction request'
)

# 4. Crea l'endpoint di predizione `/predict`
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    # Prendi i dati JSON dalla richiesta
    data = request.get_json()
    df = pd.DataFrame([data])  # Crea DataFrame con una riga

    # Applica lo scaler (DEVE essere lo stesso usato in addestramento)
    data_scaled = scaler.transform(df)

    # Esegui la predizione
    prediction = model.predict(data_scaled)[0]  # Prendi il primo (e unico) elemento
    class_name = "fraud" if prediction == 1 else "non_fraud"

    # Aggiorna il contatore delle predizioni
    PREDICTIONS_TOTAL.labels(class_name=class_name).inc()

    # Calcola e registra la latenza
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)

    return jsonify({'prediction': int(prediction), 'class_name': class_name})

# 5. Esponi l'endpoint `/metrics` per Prometheus
# La libreria prometheus_client si occupa di creare questo endpoint per noi
from werkzeug.middleware.dispatcher import DispatcherMiddleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# 6. Avvia l'applicazione
if __name__ == '__main__':
    # Esegui su porta 5000, accessibile da tutte le interfacce di rete ('0.0.0.0')
    app.run(host='0.0.0.0', port=5000)