# 🚀 MLOps Enterprise SageMaker Project

<!-- Badges -->
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11.0-orange)
![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Versioni Librerie -->
![boto3](https://img.shields.io/badge/boto3-1.34.34-lightblue)
![sagemaker](https://img.shields.io/badge/sagemaker-2.196.0-lightgreen)
![numpy](https://img.shields.io/badge/numpy-1.24.3-lightcoral)
![flask](https://img.shields.io/badge/flask-2.2.2-purple)

**Progetto MLOps enterprise-grade completo: Fashion MNIST → CNN Training → SageMaker Deploy → Lambda API → Production**

## 🎯 Obiettivi del Progetto

Questo progetto implementa un **pipeline MLOps enterprise completo** che dimostra:

- 📊 **Training CNN** su dataset Fashion MNIST (60K immagini)
- 🐳 **Containerizzazione** con Docker e BYOC (Bring Your Own Container)
- ⚡ **Training GPU** su SageMaker con ml.g4dn.xlarge
- 🔧 **Deploy endpoint** real-time per inferenza
- 🔗 **Integrazione Lambda** per API serverless
- 📈 **Monitoring** e gestione costi enterprise

## 🏗️ Architettura del Sistema

```
[Fashion MNIST] → [Docker CNN] → [SageMaker GPU] → [Endpoint] → [Lambda] → [API]
   60K images     TensorFlow      ml.g4dn.xlarge   Real-time    AWS        REST
```

## 📋 Prerequisiti

### 🖥️ Sistema
- **Python**: 3.8, 3.9, 3.10, o 3.11
- **Docker**: Desktop attivo e funzionante
- **Git**: Per clonare il repository
- **Jupyter**: Per eseguire il notebook
- **Account AWS**: Con credenziali attive

### 🔑 Permessi AWS
- **SageMaker**: FullAccess per training e deploy
- **ECR**: Push/pull immagini Docker
- **Lambda**: Creazione e invocazione funzioni
- **S3**: Lettura/scrittura bucket
- **IAM**: Gestione ruoli e policy

### 💰 Costi Stimati
- **Training**: ~$2-3 per sessione (GPU ml.g4dn.xlarge)
- **Endpoint**: ~$1/giorno (ml.t2.medium)
- **Lambda**: ~$0.0001 per invocazione
- **S3**: ~$0.02/GB/mese

## 🚀 Guida Rapida

### **Opzione 1: Notebook Completo (Consigliato)**
```bash
# 1. Clona repository
git clone <repository-url>
cd mlops-sagemaker-project

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Installa dipendenze aggiuntive per test script
pip install matplotlib boto3

# 4. Configura credenziali AWS
cp env.template .env
# Modifica .env con le tue credenziali

# 5. Avvia Jupyter
jupyter notebook aws_sagemaker_lambda_exercise.ipynb

# 6. Esegui tutte le celle in sequenza
```

### **Opzione 2: Test Script Diretto**
```bash
# Test immediato del modello con visualizzazioni
python test_model_predictions.py

# Scegli modalità:
# 1 = Test locale (fallback se modello non disponibile)
# 2 = Test remoto via AWS Lambda
# 3 = Test completo (entrambi)
```

### **Opzione 3: Pipeline Runner**
```bash
# Esegui pipeline completa: Build → Push → Train
python run_pipeline.py

# Questo script esegue:
# - Build immagine Docker
# - Push su ECR
# - Training su SageMaker
```

## �� Workflow Completo

### **Fase 1: Setup Ambiente**
- 🔧 Installazione dipendenze
- 🔑 Configurazione credenziali AWS
- 📊 Verifica connessione

### **Fase 2: Preparazione Dati**
- 📁 Download Fashion MNIST (60K immagini)
- 🔄 Pre-processing CNN (normalizzazione + reshape)
- 📤 Upload su S3 bucket

### **Fase 3: Containerizzazione**
- 📝 Creazione `train.py` (script training)
- 🌐 Creazione `serve.py` (script serving)
- 🐳 Build `Dockerfile` ottimizzato
- 🚢 Push su Amazon ECR

### **Fase 4: Training SageMaker**
- ⚡ Training CNN su GPU (ml.g4dn.xlarge)
- 📈 10 epochs con batch size 128
- 🎯 Accuracy target: ~90%
- 📤 Salvataggio modello su S3

### **Fase 5: Deploy Endpoint**
- 🚀 Deploy endpoint SageMaker
- 📊 Configurazione auto-scaling
- 🔍 Health check e monitoring
- ✅ Test inferenza

### **Fase 6: API Lambda**
- 🔗 Creazione funzione Lambda
- 🌐 Integrazione con endpoint SageMaker
- 📝 Gestione input/output JSON
- 🧪 Test end-to-end

### **Fase 7: Monitoring**
- 📊 CloudWatch logs
- 💰 Cost tracking
- 📈 Performance metrics
- 🔧 Troubleshooting

### **Fase 8: Cleanup**
- 🗑️ Eliminazione endpoint
- 🧹 Cleanup risorse AWS
- 💰 Ottimizzazione costi

## 🧪 Test Script Completo

### **📊 Panoramica del Test Script**

Il file `test_model_predictions.py` è uno script completo per testare il modello CNN sia in locale che tramite AWS Lambda. Offre un'interfaccia interattiva per visualizzare predizioni con immagini colorate in base alla correttezza.

### **🚀 Funzionalità Principali**

#### **🎯 Modalità di Test**
- **Test Locale**: Testa il modello addestrato localmente (se disponibile)
- **Test Remoto**: Testa il modello tramite AWS Lambda con endpoint SageMaker
- **Test Combinato**: Esegue entrambi i test per confrontare performance

#### **📈 Caratteristiche Avanzate**
- **Ricerca Automatica**: Trova automaticamente la funzione Lambda del progetto
- **Visualizzazione Intelligente**: Immagini colorate (verde=corretto, rosso=sbagliato)
- **Statistiche Dettagliate**: Accuratezza per classe e generale
- **Salvataggio Automatico**: Tutte le visualizzazioni salvate come PNG
- **Gestione Errori**: Fallback automatico se il modello non è disponibile

### **💻 Utilizzo del Test Script**

#### **🎮 Avvio Interattivo**
```bash
# Avvia il test script
python test_model_predictions.py

# Menu interattivo:
# 1. Test con modello locale
# 2. Test con modello remoto (AWS Lambda)  
# 3. Entrambi
```

#### **⚙️ Configurazione Prerequisiti**
```bash
# 1. Installa dipendenze aggiuntive
pip install matplotlib boto3

# 2. Configura AWS (se usi test remoto)
aws configure

# 3. Verifica prerequisiti automaticamente
python check_requirements.py

# 4. Verifica funzione Lambda esistente (manuale)
aws lambda list-functions | grep -i mlops
```

#### **📊 Test Modello Locale**
```python
# Lo script cerca automaticamente il modello addestrato in:
# - model/saved_model/00000001  (formato SavedModel)
# - model/model.h5              (formato H5)
# - trained_model/              (path alternativo)

# Se non trova il modello, usa un modello di fallback per demo
```

#### **🌐 Test Modello Remoto (AWS Lambda)**
```python
# Trova automaticamente funzioni Lambda che contengono:
# - mlops, fashion, classifier, exercise, invoker

# Testa con payload corretto:
payload = {
    "body": json.dumps({
        "image_data": image_data_flattened.tolist()
    })
}
```

### **🎨 Output e Visualizzazioni**

#### **📸 Immagini Salvate**
- `test_predictions_TIMESTAMP.png` - Test locale con griglia 2x4
- `lambda_test_TIMESTAMP.png` - Test singola immagine Lambda
- `lambda_multiple_test_TIMESTAMP.png` - Test multiple immagini Lambda

#### **📊 Statistiche Mostrate**
```
🎯 Accuratezza: 94.73% (947/1000)
✅ Predizioni corrette: 947
❌ Predizioni sbagliate: 53

📈 Accuratezza per classe:
   T-shirt/top: 89.2%
   Trouser: 98.1%
   Pullover: 87.4%
   Dress: 91.3%
   Coat: 88.7%
   Sandal: 98.9%
   Shirt: 78.2%
   Sneaker: 96.8%
   Bag: 98.7%
   Ankle boot: 96.4%
```

### **🔧 Funzioni Principali del Test Script**

#### **🎯 Test Singola Immagine**
```python
def test_single_image_lambda(lambda_client, lambda_function_name, x_test, y_test):
    """
    Testa una singola immagine casuale:
    - Seleziona immagine random dal test set
    - Invia payload a Lambda
    - Visualizza risultato con colori
    - Salva immagine con predizione
    """
```

#### **📊 Test Multiple Immagini**
```python
def test_multiple_images_lambda(lambda_client, lambda_function_name, x_test, y_test, num_tests=6):
    """
    Testa 6 immagini in una griglia:
    - Griglia 2x3 con immagini casuali
    - Statistiche accuracy complessiva
    - Salvataggio risultati
    """
```

#### **🔍 Ricerca Automatica Lambda**
```python
def find_lambda_function():
    """
    Cerca automaticamente funzioni Lambda del progetto:
    - Scansiona tutte le funzioni Lambda
    - Filtra per keywords: mlops, fashion, classifier
    - Restituisce nome funzione trovata
    """
```

### **📋 Esempi di Utilizzo**

#### **🎮 Test Completo Automatico**
```bash
# Esegui test completo
python test_model_predictions.py

# Scegli opzione "3" per test completo
# Output:
# - Statistiche modello locale
# - Confronto con modello remoto
# - Visualizzazioni salvate
```

#### **🎯 Test Solo Lambda**
```bash
# Solo test AWS Lambda
python test_model_predictions.py

# Scegli opzione "2"
# Il script trova automaticamente: mlops-exercise-invoker
# Testa 1 immagine singola + 6 immagini multiple
```

#### **📊 Test Solo Locale**
```bash
# Solo test modello locale
python test_model_predictions.py

# Scegli opzione "1"
# Se non trova modello addestrato, usa fallback per demo
```

### **🛠️ Troubleshooting Test Script**

#### **❌ Errore: "Modello non trovato"**
```bash
# Soluzione: Il modello non è stato scaricato localmente
# Usa test remoto (opzione 2) o addestra il modello
```

#### **❌ Errore: "AWS credentials not configured"**
```bash
# Soluzione: Configura credenziali AWS
aws configure
# Inserisci: Access Key, Secret Key, Region
```

#### **❌ Errore: "Lambda function not found"**
```bash
# Soluzione: Inserisci manualmente il nome della funzione
# Oppure verifica che la Lambda sia stata creata:
aws lambda list-functions | grep -i mlops
```

#### **❌ Errore: "matplotlib not installed"**
```bash
# Soluzione: Installa matplotlib
pip install matplotlib
```

### **🎯 Integrazione con il Workflow**

Il test script si integra perfettamente con il workflow MLOps:

1. **Dopo Training**: Testa il modello locale appena addestrato
2. **Dopo Deploy**: Testa l'endpoint via Lambda
3. **Monitoring**: Usa per test periodici di regressione
4. **Debug**: Visualizza predizioni sbagliate per migliorare il modello

### **📄 Esempio Output Completo**

Per vedere l'output completo del test script con esempi di visualizzazioni, consulta:
**➡️ [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md)**

Questo file mostra:
- 📱 Output console completo del test script
- 🎨 Esempi di visualizzazioni generate
- 📊 Interpretazione dei risultati
- 🔧 Troubleshooting dei problemi comuni

## 📁 Struttura File del Progetto

### **📄 File Essenziali**

| File | Descrizione | Utilizzo |
|------|-------------|----------|
| `aws_sagemaker_lambda_exercise.ipynb` | **Notebook principale** | Workflow completo step-by-step |
| `test_model_predictions.py` | **🧪 Test script completo** | Test locale e remoto con visualizzazioni |
| `check_requirements.py` | **✅ Verifica prerequisiti** | Controlla dipendenze e configurazione AWS |
| `model/train.py` | **Script training** | Training CNN su SageMaker |
| `model/serve.py` | **Script serving** | Serving endpoint Flask |
| `model/Dockerfile` | **Immagine Docker** | Container per SageMaker BYOC |
| `lambda_function/lambda_function.py` | **Codice Lambda** | API serverless per predizioni |
| `run_pipeline.py` | **Pipeline runner** | Script per eseguire pipeline completa |
| `requirements.txt` | **Dipendenze** | Librerie Python necessarie |
| `README.md` | **Documentazione** | Guida completa (questo file) |
| `EXAMPLE_OUTPUT.md` | **📊 Esempi output** | Output del test script con visualizzazioni |
| `.gitignore` | **Git ignore** | File da escludere dal repo |
| `env.template` | **Template credenziali** | Esempio configurazione AWS |

### **🗑️ File da Eliminare (Creati durante troubleshooting)**

| File | Perché Eliminare |
|------|-----------------|
| `cleanup_failed_training.py` | Script debug specifico - non educativo |
| `launch_training.py` | Duplica notebook - confonde workflow |
| `fix_sagemaker_role.py` | Troubleshooting IAM - dovrebbe essere setup corretto |
| `test.py` | Test Lambda ridondante - notebook ha già test |
| `create_lambda_function.py` | Creazione Lambda ridondante - notebook più completo |
| `change_Lambda.py` | Script non documentato - non fa parte del flow |
| `mlops_runner.py` | Troppo complesso per principianti - notebook è didattico |
| `USAGE.md` | Documentazione frammentata - tutto nel README |
| `main.py` | Preparazione dati ridondante - notebook fa tutto |
| `requirements-prod.txt` | Troppo specifico - un solo requirements.txt |
| `mlops_setup_local.py` | Setup complesso - notebook è più didattico |
| `Sage/` | Virtual environment - non dovrebbe essere nel repo |

## 🧹 Cleanup dei File Inutili

**Esegui questi comandi per pulire il progetto:**

```bash
# Elimina file di troubleshooting
rm -f cleanup_failed_training.py
rm -f launch_training.py
rm -f fix_sagemaker_role.py
rm -f test.py
rm -f create_lambda_function.py
rm -f change_Lambda.py
rm -f mlops_runner.py
rm -f USAGE.md
rm -f main.py
rm -f requirements-prod.txt
rm -f mlops_setup_local.py

# Elimina directory virtual environment
rm -rf Sage/

# Elimina dati generati (opzionale - saranno ricreati)
rm -rf data/

echo "✅ Pulizia completata!"
```

## 🔧 Troubleshooting

### **❌ Errori Comuni**

#### **Docker Build Error**
```bash
# Problema: Architettura incompatibile (Mac M1/M2/M3)
# Soluzione:
docker buildx build --platform linux/amd64 -t nome-immagine .
```

#### **SageMaker Role Error**
```bash
# Problema: Ruolo IAM non trovato
# Soluzione: Crea ruolo SageMaker nella AWS Console
# Policies necessarie:
# - AmazonSageMakerFullAccess
# - AmazonS3FullAccess
# - AmazonEC2ContainerRegistryFullAccess
```

#### **Lambda Function Error**
```bash
# Problema: Funzione Lambda non trovata
# Soluzione: Eseguire prima tutte le celle del notebook
# Il notebook crea automaticamente la funzione Lambda
```

### **🔍 Comandi di Debug**

```bash
# Verifica connessione AWS
aws sts get-caller-identity

# Lista training jobs
aws sagemaker list-training-jobs --max-results 5

# Lista endpoint
aws sagemaker list-endpoints

# Lista funzioni Lambda
aws lambda list-functions --max-items 10
```

## 📊 Metriche di Performance

### **Training Results**
- **Accuracy**: ~90% (Fashion MNIST)
- **Training Time**: ~5-8 minuti (GPU)
- **Model Size**: ~10-20MB
- **Inference Time**: ~50ms per predizione

### **Costs**
- **Training**: $0.50-2.00 per job
- **Endpoint**: $1.00/giorno (ml.t2.medium)
- **Lambda**: $0.0001 per invocazione
- **Storage**: $0.02/GB/mese

## 🏆 Risultati Attesi

Al completamento del progetto avrai:

- ✅ **CNN trainata** su 60K immagini Fashion MNIST
- ✅ **Endpoint SageMaker** funzionante e scalabile
- ✅ **API Lambda** serverless per predizioni
- ✅ **Pipeline completo** enterprise-grade
- ✅ **Monitoring** e cost management
- ✅ **Troubleshooting** expertise

## 🎯 Next Steps

### **Estensioni Consigliate**
1. **Custom Dataset**: Sostituire Fashion MNIST con dataset proprietario
2. **Model Optimization**: Quantizzazione e ottimizzazione performance
3. **CI/CD Pipeline**: GitLab/GitHub Actions per deployment automatico
4. **A/B Testing**: Confronto modelli in produzione
5. **Data Drift**: Monitoring qualità dati in produzione

### **Architetture Alternative**
1. **Batch Transform**: Per predizioni batch invece di real-time
2. **Multi-Model**: Hosting multipli modelli su stesso endpoint
3. **Edge Deploy**: Deploy su dispositivi edge con SageMaker Edge
4. **Streaming**: Integrazione con Kinesis per dati real-time

## 🤝 Contributing

Contributi benvenuti! Per contribuire:

1. Fork del repository
2. Crea feature branch (`git checkout -b feature/amazing-feature`)
3. Commit modifiche (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Apri Pull Request

## 📞 Support

Per domande o problemi:

- 📧 **Email**: support@mlops-project.com
- 📚 **Documentazione**: [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)

## 📜 License

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

---

**�� Buon MLOps! 🚀** 