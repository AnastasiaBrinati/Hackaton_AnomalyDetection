# Funzione che prende in ingresso un'immagine nella stessa cartella e restituisce il json

import json
import base64

try:
    # Aprire l'immagine
    with open("testImage.jpg", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Creare il dict payload
    payload = {"image": encoded_image}

    # Aprire un file json in writing 
    with open("payload.json", "w") as json_file:
        json.dump(payload, json_file)
    # Conferma che tutto sia ok
    print("Payload creato con successo")

except FileNotFoundError:
    print("File di test non trovato. Assicurati di essere nella cartella giusta")