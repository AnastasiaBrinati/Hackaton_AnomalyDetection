#!/usr/bin/env python3
"""
MLOps Runner - Script per utilizzare il sistema MLOps SageMaker
Automatizza test, predizioni e visualizzazione dei risultati
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

# Import del nostro setup locale
try:
    from mlops_setup_local import MLOpsSetupLocal
except ImportError:
    print("❌ Errore: mlops_setup_local.py non trovato")
    print("💡 Assicurati di essere nella directory corretta")
    sys.exit(1)

class MLOpsRunner:
    """Classe per eseguire operazioni MLOps end-to-end"""
    
    def __init__(self):
        """Inizializza il runner"""
        self.setup = None
        self.config = None
        self.lambda_client = None
        self.test_data = None
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        print("🚀 MLOps Runner - Inizializzazione")
        print("=" * 50)
    
    def setup_environment(self):
        """Configura l'ambiente MLOps"""
        print("🔧 Configurazione ambiente MLOps...")
        
        try:
            # Inizializza setup locale
            self.setup = MLOpsSetupLocal()
            self.config = self.setup.setup_complete()
            
            # Configura client Lambda
            self.lambda_client = self.setup.boto_session.client('lambda')
            
            print("✅ Ambiente configurato con successo!")
            return True
            
        except Exception as e:
            print(f"❌ Errore configurazione: {e}")
            return False
    
    def get_docker_image_uri(self):
        """Ottiene l'URI dell'immagine Docker per ECR"""
        print("🐳 Generazione URI immagine Docker...")
        
        if not self.config:
            print("❌ Configurazione non disponibile")
            return None
        
        image_uri = f"{self.config['account_id']}.dkr.ecr.{self.config['aws_region']}.amazonaws.com/{self.config['ecr_repository_name']}:{self.config['image_tag']}"
        
        print(f"📋 URI dell'immagine Docker:")
        print(f"   {image_uri}")
        print(f"⚠️  Assicurati che l'immagine esista in ECR prima di procedere.")
        
        return image_uri
    
    def load_test_data(self):
        """Carica dati di test Fashion MNIST"""
        print("📊 Caricamento dati di test Fashion MNIST...")
        
        try:
            # Prova a caricare TensorFlow
            import tensorflow as tf
            
            # Carica Fashion MNIST
            (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            
            # Pre-processing
            x_test = x_test.astype('float32') / 255.0
            x_test = np.expand_dims(x_test, -1)
            
            self.test_data = {
                'x_test': x_test,
                'y_test': y_test
            }
            
            print(f"✅ Dati di test caricati: {x_test.shape[0]} immagini")
            return True
            
        except ImportError:
            print("⚠️  TensorFlow non disponibile, caricando dati dummy...")
            
            # Crea dati dummy per test
            self.test_data = {
                'x_test': np.random.rand(100, 28, 28, 1),
                'y_test': np.random.randint(0, 10, 100)
            }
            
            print("✅ Dati dummy caricati per test")
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento dati: {e}")
            return False
    
    def test_lambda_function(self, lambda_function_name, num_tests=1):
        """Testa la funzione Lambda con immagini casuali"""
        print(f"🔬 Test Lambda function: {lambda_function_name}")
        
        if not self.test_data:
            print("❌ Dati di test non disponibili")
            return False
        
        results = []
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            # Seleziona immagine casuale
            random_index = np.random.randint(0, self.test_data['x_test'].shape[0])
            test_image = self.test_data['x_test'][random_index]
            true_label_index = self.test_data['y_test'][random_index]
            true_label_name = self.class_names[true_label_index]
            
            # Prepara payload
            image_data_for_payload = test_image.flatten().tolist()
            payload = {
                "body": json.dumps({
                    "image_data": image_data_for_payload
                })
            }
            
            try:
                # Invoca Lambda
                print(f"📤 Invocazione Lambda (Vero: '{true_label_name}')...")
                response = self.lambda_client.invoke(
                    FunctionName=lambda_function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
                
                # Parsing risposta
                response_payload = json.load(response['Payload'])
                
                if response_payload['statusCode'] == 200:
                    response_body = json.loads(response_payload['body'])
                    
                    predicted_class_index = response_body['predicted_class_index']
                    predicted_class_name = response_body['predicted_class_name']
                    
                    # Risultato
                    is_correct = predicted_class_index == true_label_index
                    accuracy_symbol = "✅" if is_correct else "❌"
                    
                    print(f"📊 Risultato Test {i+1}:")
                    print(f"   Immagine: {random_index}")
                    print(f"   Vero: {true_label_name} (classe {true_label_index})")
                    print(f"   Predetto: {predicted_class_name} (classe {predicted_class_index})")
                    print(f"   Corretto: {accuracy_symbol}")
                    
                    # Salva risultato
                    results.append({
                        'test_index': i+1,
                        'image_index': random_index,
                        'true_label': true_label_name,
                        'true_class': true_label_index,
                        'predicted_label': predicted_class_name,
                        'predicted_class': predicted_class_index,
                        'correct': is_correct,
                        'image_data': test_image
                    })
                    
                else:
                    print(f"❌ Errore Lambda: Status Code {response_payload['statusCode']}")
                    print(f"   Risposta: {response_payload}")
                    
            except Exception as e:
                print(f"❌ Errore invocazione Lambda: {e}")
        
        # Mostra statistiche
        if results:
            correct_predictions = sum(1 for r in results if r['correct'])
            accuracy = correct_predictions / len(results) * 100
            
            print(f"\n📈 Statistiche Test:")
            print(f"   Test eseguiti: {len(results)}")
            print(f"   Predizioni corrette: {correct_predictions}")
            print(f"   Accuratezza: {accuracy:.1f}%")
        
        return results
    
    def visualize_results(self, results, save_plots=False):
        """Visualizza i risultati dei test"""
        print("📊 Visualizzazione risultati...")
        
        if not results:
            print("❌ Nessun risultato da visualizzare")
            return
        
        # Crea figura con subplot
        n_results = len(results)
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_results == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Mostra immagine
            ax.imshow(result['image_data'].squeeze(), cmap='gray')
            
            # Titolo con risultato
            color = 'green' if result['correct'] else 'red'
            title = f"Test {result['test_index']}\n"
            title += f"Vero: {result['true_label']}\n"
            title += f"Predetto: {result['predicted_label']}"
            
            ax.set_title(title, color=color, fontsize=10)
            ax.axis('off')
        
        # Rimuovi subplot vuoti
        for i in range(n_results, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                fig.delaxes(axes[col])
            else:
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mlops_test_results_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Grafico salvato: {filename}")
        
        plt.show()
    
    def run_full_pipeline(self, lambda_function_name, num_tests=3, save_plots=False):
        """Esegue pipeline completa di test"""
        print("🔄 Esecuzione pipeline completa...")
        
        # 1. Setup ambiente
        if not self.setup_environment():
            return False
        
        # 2. Mostra URI Docker
        image_uri = self.get_docker_image_uri()
        
        # 3. Carica dati test
        if not self.load_test_data():
            return False
        
        # 4. Test Lambda
        results = self.test_lambda_function(lambda_function_name, num_tests)
        
        # 5. Visualizza risultati
        if results:
            self.visualize_results(results, save_plots)
        
        print("\n🎉 Pipeline completata!")
        return True

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='MLOps Runner - Automatizza test e predizioni')
    parser.add_argument('--lambda-function', '-l', required=True, help='Nome della funzione Lambda')
    parser.add_argument('--num-tests', '-n', type=int, default=3, help='Numero di test da eseguire (default: 3)')
    parser.add_argument('--save-plots', '-s', action='store_true', help='Salva grafici dei risultati')
    parser.add_argument('--docker-uri-only', '-d', action='store_true', help='Mostra solo URI Docker')
    
    args = parser.parse_args()
    
    # Crea runner
    runner = MLOpsRunner()
    
    if args.docker_uri_only:
        # Solo URI Docker
        if runner.setup_environment():
            runner.get_docker_image_uri()
    else:
        # Pipeline completa
        success = runner.run_full_pipeline(
            lambda_function_name=args.lambda_function,
            num_tests=args.num_tests,
            save_plots=args.save_plots
        )
        
        if success:
            print("✅ Esecuzione completata con successo!")
        else:
            print("❌ Esecuzione fallita")
            sys.exit(1)

if __name__ == "__main__":
    main() 