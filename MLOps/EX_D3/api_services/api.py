# Funzione che contiene due endpoint:
'''
*POST `/predict`: Deve accettare un JSON con un'immagine codificata in base64. 
            Invece di eseguire l'inferenza, deve chiamare il task Celery in modo asincrono (`task.delay(...)`) 
            Restituire immediatamente un JSON con il `task_id` ricevuto.
* `GET /result/<task_id>`: Deve usare l'ID per interrogare Celery sullo stato del task. 
            Se il task è completato, restituisce i risultati; altrimenti, restituisce uno stato "PENDING"..
'''

from flask import Flask, request, jsonify
from shared.celery_app import celery_app
from celery.result import AsyncResult

app = Flask(__name__)

'''
Chiarezza sugli errori possibili:
- Input non valido oppure campo mancante 400
- Task non trovato 404
- Task in esecuzione 202
- Task completato 200
- Task fallito 500
'''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Input non valido oppure campo mancante'}), 400
    
    image_b64 = data['image']
    print(f"Image received: {image_b64}")
    # Accedo al task tramite celery_app invece che importarlo direttamente
    task = celery_app.send_task('detect_objects', args=[image_b64])
    return jsonify({'task_id': task.id}), 202

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    # Interrogare Celery per lo stato e il risultato del task
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        # Quando il task è stato completato, restituisce il risultato del task
        if task_result.successful():
            # Restituisce i risultati del task
            return jsonify({'status': 'SUCCESS', 
                            'result': task_result.get()
                            }), 200
        else:
            # Restituisce l'errore del task
            return jsonify({'status': 'FAILED', 
                            'error': str(task_result.info) # Vogliamo informazioni più dettagliate sull'errore
                            }), 500
    else:
        # Quando il task è in esecuzione, restituisce lo stato "PENDING"
        return jsonify({'status': 'PENDING'}), 202

@app.route('/health', methods=['GET'])
def health_checker():
    return jsonify({'status': 'OK'}), 200
