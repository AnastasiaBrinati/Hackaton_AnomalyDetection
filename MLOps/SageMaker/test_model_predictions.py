#!/usr/bin/env python3
"""
Test script per visualizzare predizioni del modello CNN su Fashion MNIST
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import boto3
from datetime import datetime

# Nomi delle classi Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_and_preprocess_data():
    """Carica e preprocessa i dati di test Fashion MNIST"""
    print("📊 Caricamento dataset Fashion MNIST...")
    
    # Carica i dati direttamente da TensorFlow
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Pre-processing
    x_test = x_test.astype('float32') / 255.0  # Normalizzazione
    x_test = np.expand_dims(x_test, -1)  # Aggiunge dimensione canale
    
    print(f"✅ Dati caricati: {x_test.shape[0]} immagini di test")
    print(f"   Dimensioni immagini: {x_test.shape[1:3]}")
    print(f"   Range valori: {x_test.min():.3f} - {x_test.max():.3f}")
    
    return x_test, y_test

def setup_aws_lambda_client():
    """Configura il client AWS Lambda"""
    print("🔧 Configurazione client AWS Lambda...")
    
    try:
        # Crea il client Lambda
        lambda_client = boto3.client('lambda')
        
        # Verifica le credenziali
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        print(f"✅ Client AWS configurato per account: {identity['Account']}")
        
        return lambda_client
    except Exception as e:
        print(f"❌ Errore configurazione AWS: {e}")
        print("💡 Suggerimento: Configura le credenziali AWS con 'aws configure'")
        return None

def test_single_image_lambda(lambda_client, lambda_function_name, x_test, y_test):
    """Testa una singola immagine usando AWS Lambda"""
    print(f"🎯 Test di una singola immagine su Lambda: {lambda_function_name}")
    
    # Prendiamo un'immagine casuale dal test set
    random_index = np.random.randint(0, x_test.shape[0])
    test_image = x_test[random_index]
    true_label_index = y_test[random_index]
    true_label_name = class_names[true_label_index]

    # L'input per il modello deve essere una lista flattened
    image_data_for_payload = test_image.flatten().tolist()

    # Costruiamo il payload per la Lambda
    payload = {
        "body": json.dumps({
            "image_data": image_data_for_payload
        })
    }

    try:
        # Invocazione
        print(f"📡 Invocazione Lambda con immagine di test (Vero: '{true_label_name}')...")
        response = lambda_client.invoke(
            FunctionName=lambda_function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        # Parsing della risposta
        response_payload = json.load(response['Payload'])
        
        if response_payload['statusCode'] == 200:
            response_body = json.loads(response_payload['body'])
            
            print("\n--- Risultato Test ---")
            print(f"Immagine di Test: Indice {random_index}, Etichetta Vera: {true_label_name} (classe {true_label_index})")
            print(f"Risposta dalla Lambda (Status Code): {response_payload['statusCode']}")
            print(f"Predizione del Modello: {response_body['predicted_class_name']} (classe {response_body['predicted_class_index']})")
            print("--------------------")

            # Visualizziamo l'immagine di test per un controllo visivo
            plt.figure(figsize=(6, 6))
            plt.imshow(test_image.squeeze(), cmap='gray')
            
            # Colore in base alla correttezza
            is_correct = true_label_index == response_body['predicted_class_index']
            color = 'green' if is_correct else 'red'
            accuracy_text = "✅ CORRETTO" if is_correct else "❌ SBAGLIATO"
            
            plt.title(f"Vero: {true_label_name}\nPredetto: {response_body['predicted_class_name']}\n{accuracy_text}", 
                     fontsize=12, color=color, fontweight='bold')
            plt.axis('off')
            
            # Salva l'immagine
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lambda_test_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Immagine salvata: {filename}")
            
            plt.show()
            
            return True, is_correct
        else:
            print(f"❌ Errore Lambda: {response_payload}")
            return False, False
            
    except Exception as e:
        print(f"❌ Errore durante invocazione Lambda: {e}")
        return False, False

def test_multiple_images_lambda(lambda_client, lambda_function_name, x_test, y_test, num_tests=5):
    """Testa multiple immagini usando AWS Lambda"""
    print(f"🎯 Test di {num_tests} immagini su Lambda: {lambda_function_name}")
    
    results = []
    correct_predictions = 0
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'🧪 Test Multiple Immagini via AWS Lambda - {lambda_function_name}', fontsize=16, fontweight='bold')
    
    for i in range(num_tests):
        # Seleziona immagine casuale
        random_index = np.random.randint(0, x_test.shape[0])
        test_image = x_test[random_index]
        true_label_index = y_test[random_index]
        true_label_name = class_names[true_label_index]
        
        # Prepara payload
        image_data_for_payload = test_image.flatten().tolist()
        payload = {
            "body": json.dumps({
                "image_data": image_data_for_payload
            })
        }
        
        try:
            print(f"📡 Test {i+1}/{num_tests} - Immagine {random_index} ({true_label_name})...")
            
            # Invoca Lambda
            response = lambda_client.invoke(
                FunctionName=lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            response_payload = json.load(response['Payload'])
            
            if response_payload['statusCode'] == 200:
                response_body = json.loads(response_payload['body'])
                predicted_class_name = response_body['predicted_class_name']
                predicted_class_index = response_body['predicted_class_index']
                
                is_correct = true_label_index == predicted_class_index
                if is_correct:
                    correct_predictions += 1
                
                # Visualizza risultato
                plt.subplot(2, 3, i + 1)
                plt.imshow(test_image.squeeze(), cmap='gray')
                plt.axis('off')
                
                color = 'green' if is_correct else 'red'
                accuracy_text = "✅" if is_correct else "❌"
                
                title = f'{accuracy_text} Vero: {true_label_name}\nPredetto: {predicted_class_name}'
                plt.title(title, fontsize=10, color=color, fontweight='bold')
                
                results.append({
                    'index': random_index,
                    'true_label': true_label_name,
                    'predicted_label': predicted_class_name,
                    'correct': is_correct
                })
                
            else:
                print(f"❌ Errore Lambda per immagine {i+1}: {response_payload}")
                
        except Exception as e:
            print(f"❌ Errore durante test {i+1}: {e}")
    
    plt.tight_layout()
    
    # Salva risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lambda_multiple_test_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Risultati salvati: {filename}")
    
    plt.show()
    
    # Mostra statistiche
    accuracy = correct_predictions / num_tests
    print(f"\n📊 Risultati Test Lambda:")
    print(f"🎯 Accuratezza: {accuracy:.2%} ({correct_predictions}/{num_tests})")
    print(f"✅ Predizioni corrette: {correct_predictions}")
    print(f"❌ Predizioni sbagliate: {num_tests - correct_predictions}")
    
    return results, accuracy

def find_lambda_function():
    """Trova automaticamente la funzione Lambda del progetto"""
    print("🔍 Ricerca funzione Lambda del progetto...")
    
    try:
        lambda_client = boto3.client('lambda')
        
        # Lista delle funzioni Lambda
        response = lambda_client.list_functions()
        
        # Cerca funzioni che potrebbero essere del nostro progetto
        project_keywords = ['mlops', 'fashion', 'classifier', 'exercise', 'invoker']
        
        for function in response['Functions']:
            function_name = function['FunctionName'].lower()
            for keyword in project_keywords:
                if keyword in function_name:
                    print(f"✅ Trovata funzione Lambda: {function['FunctionName']}")
                    return function['FunctionName']
        
        print("⚠️  Nessuna funzione Lambda del progetto trovata automaticamente")
        return None
        
    except Exception as e:
        print(f"❌ Errore durante ricerca Lambda: {e}")
        return None

def create_simple_model():
    """Crea un modello semplice per test (da usare se il modello principale non è disponibile)"""
    print("🔧 Creazione modello di fallback...")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("⚠️  Usando modello non addestrato per demo")
    return model

def load_trained_model():
    """Carica il modello addestrato se disponibile"""
    model_paths = [
        "model/saved_model/00000001",  # Path locale se esiste
        "model/model.h5",              # Path alternativo H5
        "trained_model"                # Path alternativo
    ]
    
    print("🔍 Ricerca modello addestrato...")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"📂 Tentativo caricamento da: {model_path}")
                if model_path.endswith('.h5'):
                    model = tf.keras.models.load_model(model_path)
                else:
                    model = tf.saved_model.load(model_path)
                print("✅ Modello addestrato caricato con successo!")
                return model, True
            except Exception as e:
                print(f"❌ Errore caricamento da {model_path}: {e}")
                continue
    
    print("⚠️  Modello addestrato non trovato, uso modello di fallback")
    return create_simple_model(), False

def make_predictions(model, x_test, is_trained_model=False):
    """Fa predizioni con il modello"""
    print("🎯 Esecuzione predizioni...")
    
    if is_trained_model and hasattr(model, 'signatures'):
        # Modello SavedModel
        predict_fn = model.signatures['serving_default']
        predictions = predict_fn(tf.convert_to_tensor(x_test))
        # Estrae i valori delle predizioni
        output_key = list(predictions.keys())[0]
        predictions = predictions[output_key].numpy()
    else:
        # Modello Keras standard
        predictions = model.predict(x_test, verbose=0)
    
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    return predicted_classes, confidence_scores

def plot_predictions(x_test, y_test, predictions, confidence_scores, num_images=8):
    """Visualizza le immagini con predizioni e etichette reali"""
    print(f"🎨 Creazione visualizzazione per {num_images} immagini...")
    
    # Seleziona immagini casuali
    indices = np.random.choice(len(x_test), num_images, replace=False)
    
    # Configura la griglia
    rows = 2
    cols = num_images // rows
    
    plt.figure(figsize=(15, 8))
    plt.suptitle('🧪 Test Predizioni Modello CNN - Fashion MNIST', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        
        # Mostra l'immagine
        image = x_test[idx].squeeze()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        # Prepara le etichette
        true_label = class_names[y_test[idx]]
        predicted_label = class_names[predictions[idx]]
        confidence = confidence_scores[idx]
        
        # Colore in base alla correttezza
        color = 'green' if y_test[idx] == predictions[idx] else 'red'
        
        # Titolo con vera etichetta e predizione
        title = f'Vero: {true_label}\nPredetto: {predicted_label}\nConfidenza: {confidence:.2%}'
        plt.title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # Salva l'immagine
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_predictions_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Immagine salvata: {filename}")
    
    plt.show()

def calculate_accuracy(y_test, predictions, num_test_samples=1000):
    """Calcola l'accuratezza su un campione di test"""
    print(f"📊 Calcolo accuratezza su {num_test_samples} campioni...")
    
    # Usa solo i primi num_test_samples per velocità
    y_sample = y_test[:num_test_samples]
    pred_sample = predictions[:num_test_samples]
    
    accuracy = np.mean(y_sample == pred_sample)
    print(f"🎯 Accuratezza: {accuracy:.2%}")
    
    # Mostra accuratezza per classe
    print("\n📈 Accuratezza per classe:")
    for i, class_name in enumerate(class_names):
        class_mask = y_sample == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(pred_sample[class_mask] == y_sample[class_mask])
            print(f"   {class_name}: {class_accuracy:.2%}")

def main():
    """Funzione principale"""
    print("🚀 === TEST MODELLO CNN FASHION MNIST ===")
    print(f"⏰ Avvio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # 1. Carica i dati di test
        x_test, y_test = load_and_preprocess_data()
        
        # 2. Chiedi all'utente quale tipo di test vuole fare
        print("\n🎯 Scegli il tipo di test:")
        print("1. Test con modello locale")
        print("2. Test con modello remoto (AWS Lambda)")
        print("3. Entrambi")
        
        choice = input("Inserisci la tua scelta (1-3): ").strip()
        
        if choice in ['1', '3']:
            print("\n--- TEST MODELLO LOCALE ---")
            # Carica il modello locale
            model, is_trained = load_trained_model()
            
            # Fa predizioni su un campione
            num_test_samples = 1000
            print(f"🎯 Test su {num_test_samples} campioni...")
            
            predictions, confidence_scores = make_predictions(
                model, x_test[:num_test_samples], is_trained
            )
            
            # Calcola accuratezza
            calculate_accuracy(y_test, predictions, num_test_samples)
            
            # Visualizza risultati
            plot_predictions(
                x_test[:num_test_samples], 
                y_test[:num_test_samples], 
                predictions, 
                confidence_scores,
                num_images=8
            )
        
        if choice in ['2', '3']:
            print("\n--- TEST MODELLO REMOTO (AWS LAMBDA) ---")
            
            # Configura client AWS
            lambda_client = setup_aws_lambda_client()
            
            if lambda_client:
                # Trova la funzione Lambda
                lambda_function_name = find_lambda_function()
                
                if not lambda_function_name:
                    lambda_function_name = input("Inserisci il nome della funzione Lambda: ").strip()
                
                if lambda_function_name:
                    # Test singola immagine
                    print("\n🎯 Test singola immagine:")
                    test_single_image_lambda(lambda_client, lambda_function_name, x_test, y_test)
                    
                    # Test multiple immagini
                    print("\n🎯 Test multiple immagini:")
                    test_multiple_images_lambda(lambda_client, lambda_function_name, x_test, y_test, num_tests=6)
                else:
                    print("❌ Nome funzione Lambda non fornito")
            else:
                print("❌ Impossibile configurare client AWS Lambda")
        
        print("=" * 50)
        print("✅ Test completato con successo!")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        print("💡 Suggerimento: Verifica che tutte le dipendenze siano installate")

if __name__ == "__main__":
    main() 