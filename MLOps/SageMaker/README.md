# 🚀 MLOps SageMaker Project

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
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.2-purple)

**Progetto MLOps enterprise-grade per classificazione di immagini Fashion MNIST utilizzando AWS SageMaker.**

## 🎯 Obiettivi dell'Esercizio

Questo progetto ti guida attraverso la creazione di un **sistema MLOps completo** che include:

- 📊 **Training** di una CNN su Fashion MNIST con SageMaker
- 🐳 **Containerizzazione** con Docker personalizzato
- 🚀 **Deploy** di endpoint real-time per inferenza
- ⚡ **Integrazione** con AWS Lambda per API serverless
- 🔧 **Test** automatizzati end-to-end
- 📈 **Monitoring** e validazione del modello

## 📋 Prerequisiti

### 🖥️ Sistema
- **Python**: 3.8, 3.9, 3.10, o 3.11
- **Docker**: Installato e funzionante
- **Git**: Per clonare il repository
- **Account AWS**: Con credenziali di accesso

### 🔑 Permessi AWS
- **SageMaker**: Full access per training e deployment
- **ECR**: Push/pull immagini Docker
- **Lambda**: Creazione e invocazione funzioni
- **S3**: Lettura/scrittura bucket
- **IAM**: Gestione ruoli SageMaker

### 💰 Costi Stimati
- **Training**: ~$2-3 per sessione (GPU ml.g4dn.xlarge)
- **Endpoint**: ~$1/giorno (ml.t2.medium always-on)
- **Lambda**: ~$0.0001 per invocazione
- **S3**: ~$0.02/GB/mese

## 🚀 Guida all'Esercizio Passo-Passo

### **Fase 1: Setup Ambiente Locale**

#### 1.1 Clona e Configura Repository
```bash
git clone <repository-url>
cd mlops-sagemaker-project

# Crea e attiva virtual environment
python3 -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate    # Windows

# Installa dipendenze
pip install -r requirements.txt
```

#### 1.2 Configura Credenziali AWS
```bash
# Crea file .env dalle credenziali template
cp env.template .env

# Modifica .env con le tue credenziali AWS
nano .env
```

**Contenuto file .env:**
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xyz123...
AWS_DEFAULT_REGION=eu-west-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX
```

#### 1.3 Verifica Setup
```bash
# Testa configurazione
python mlops_setup_local.py

# Dovrebbe mostrare:
# ✅ Setup completato con successo!
# Account ID: 123456789012
# Bucket S3: sagemaker-fashion-mnist-123456789012
```

### **Fase 2: Training del Modello**

#### 2.1 Apri Jupyter Notebook
```bash
jupyter notebook aws_sagemaker_lambda_exercise.ipynb
```

#### 2.2 Esegui Celle di Training
1. **Setup e Configurazione** (Cella 1-2)
2. **Preparazione Dati** (Cella 3-4)
3. **Creazione Script Docker** (Cella 5-8)
4. **Build e Push Immagine** (Locale, seguire istruzioni)
5. **Training SageMaker** (Cella 9-10)

#### 2.3 Verifica Training
```bash
# Controlla logs su AWS Console
# Training dovrebbe completarsi in 10-15 minuti
```

### **Fase 3: Deploy e Test**

#### 3.1 Deploy Endpoint
```bash
# Esegui celle di deploy nel notebook
# Cella 11-12: Deploy endpoint
```

#### 3.2 Creazione Lambda
```bash
# Cella 13-14: Crea funzione Lambda
# Cella 15: Configurazione e test
```

#### 3.3 Test Automatizzato
```bash
# Test rapido
python quick_test.py

# Test completo
python mlops_runner.py -l mlops-fashion-classifier-invoker -n 5 -s
```

### **Fase 4: Monitoraggio e Validazione**

#### 4.1 Visualizza Risultati
```bash
# Esegui test con visualizzazione
python mlops_runner.py -l mlops-fashion-classifier-invoker -n 10 -s
```

#### 4.2 Monitoring AWS
- **CloudWatch**: Controlla metriche endpoint
- **SageMaker Console**: Stato training e endpoint
- **Lambda Console**: Logs e performance

#### 4.3 Pulizia Risorse
```bash
# Esegui ultima cella del notebook per eliminare:
# - Endpoint SageMaker
# - Funzione Lambda
# - Bucket S3
# - Repository ECR
```

## 🎯 Obiettivi di Apprendimento

Al completamento dell'esercizio avrai appreso:

✅ **BYOC (Bring Your Own Container)** con SageMaker  
✅ **GPU Training** per deep learning su cloud  
✅ **Real-time Inference** con endpoint gestiti  
✅ **API Serverless** con Lambda integration  
✅ **Docker** per ML containerization  
✅ **MLOps Pipeline** end-to-end  
✅ **Cost Management** per progetti ML  
✅ **Monitoring** e troubleshooting  

## 📊 Risultati Attesi

### Accuracy del Modello
- **Training**: ~95% su Fashion MNIST
- **Validation**: ~90% su dati di test
- **Tempo di inferenza**: <200ms per immagine

### Performance Sistema
- **Endpoint**: Ready in 5-8 minuti
- **Lambda**: Cold start <3 secondi
- **API**: Risposta completa <1 secondo

## 📁 Struttura Progetto

```
mlops-sagemaker-project/
├── 📄 File di Configurazione
│   ├── .env                          # Credenziali AWS (🚫 non in Git)
│   ├── .gitignore                    # File da ignorare
│   ├── requirements.txt              # Dipendenze complete
│   ├── requirements-prod.txt         # Solo produzione
│   └── env.template                  # Template credenziali
├── 📓 Notebook e Script
│   ├── aws_sagemaker_lambda_exercise.ipynb  # Notebook principale
│   ├── mlops_setup_local.py          # Setup automatico
│   ├── mlops_runner.py               # Script test avanzato
│   └── quick_test.py                 # Test rapido
├── 📚 Documentazione
│   ├── README.md                     # Guida principale
│   └── USAGE.md                      # Guida utilizzo script
└── 🏗️ Ambiente
    └── mlops_env/                    # Virtual environment (🚫 non in Git)
```

## 🛠️ Troubleshooting

### ❌ Errori Comuni e Soluzioni

#### **Errore: "TensorFlow not found"**
```bash
# Soluzione per sistema
pip install tensorflow==2.11.0

# Apple Silicon (M1/M2/M3)
pip install tensorflow-macos==2.11.0 tensorflow-metal

# Linux ARM64
pip install tensorflow-cpu==2.11.0
```

#### **Errore: "AWS credentials not found"**
```bash
# Verifica file .env
cat .env

# Verifica configurazione AWS
aws sts get-caller-identity

# Reconfigura se necessario
aws configure
```

#### **Errore: "SageMaker role not found"**
```bash
# Verifica ruolo esiste
aws iam get-role --role-name AmazonSageMaker-ExecutionRole-XXXXX

# Crea ruolo se necessario (via AWS Console)
```

#### **Errore: "Docker build failed"**
```bash
# Verifica Docker attivo
docker --version

# Su Mac M1/M2/M3
docker buildx build --platform linux/amd64 -t image-name .

# Permissions (Linux)
sudo usermod -aG docker $USER
```

#### **Errore: "Lambda function not found"**
```bash
# Lista funzioni esistenti
aws lambda list-functions --query 'Functions[].FunctionName'

# Verifica nome corretto nel codice
```

### 🔧 Comandi di Debug

#### **Verifica Sistema**
```bash
# Python version
python --version

# Dipendenze installate
pip list

# Configurazione AWS
aws configure list

# Docker funzionante
docker run hello-world
```

#### **Test Connessione AWS**
```bash
# Identity
aws sts get-caller-identity

# SageMaker
aws sagemaker list-training-jobs --max-results 5

# Lambda
aws lambda list-functions --max-items 5
```

#### **Log e Monitoring**
```bash
# CloudWatch logs SageMaker
aws logs describe-log-groups --log-group-name-prefix '/aws/sagemaker'

# Lambda logs
aws logs describe-log-groups --log-group-name-prefix '/aws/lambda'
```

## 🔒 Sicurezza e Best Practices

### 🛡️ Sicurezza
- ⚠️ **MAI** committare il file `.env` in Git
- 🔐 Usa ruoli IAM invece di credenziali hardcoded
- 🔄 Rotazione regolare delle credenziali AWS
- 🚫 Non condividere Access Keys via chat/email
- 🔒 Usa principio least privilege per ruoli IAM

### 💰 Gestione Costi
- 🛑 **Elimina sempre** le risorse dopo l'esercizio
- 📊 Monitora costi su AWS Cost Explorer
- ⏰ Imposta budget alerts
- 🔔 Usa CloudWatch per monitorare risorse

### 🏗️ Best Practices MLOps
- 📝 Versiona sempre i modelli
- 🧪 Testa sempre prima di produzione
- 📈 Monitora performance e drift
- 🔄 Automatizza pipeline CI/CD
- 📊 Logga metriche e predizioni

## 📞 Supporto e Risorse

### 🆘 Supporto Immediato
```bash
# Verifica configurazione completa
python mlops_setup_local.py

# Test veloce sistema
python quick_test.py

# Reset ambiente se necessario
deactivate
rm -rf mlops_env
python -m venv mlops_env
source mlops_env/bin/activate
pip install -r requirements.txt
```

### 📚 Documentazione AWS
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [Amazon ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/)

### 🎯 Checklist Completamento

#### ✅ **Fase 1: Setup**
- [ ] Python 3.8-3.11 installato
- [ ] Virtual environment attivo
- [ ] Dipendenze installate
- [ ] File .env configurato
- [ ] AWS credentials valide

#### ✅ **Fase 2: Training**
- [ ] Notebook avviato
- [ ] Dati Fashion MNIST caricati
- [ ] Script Docker creati
- [ ] Immagine Docker built e pushed
- [ ] Training SageMaker completato

#### ✅ **Fase 3: Deploy**
- [ ] Endpoint SageMaker attivo
- [ ] Funzione Lambda creata
- [ ] Test end-to-end riuscito
- [ ] Accuracy >80%

#### ✅ **Fase 4: Pulizia**
- [ ] Endpoint eliminato
- [ ] Funzione Lambda eliminata
- [ ] Bucket S3 svuotato
- [ ] Repository ECR eliminato
- [ ] Costi verificati

## 🎉 Congratulazioni!

Se hai completato tutti i passaggi, hai creato con successo un **sistema MLOps enterprise-grade** completo! 

**Competenze acquisite:**
- 🧠 **Machine Learning**: Training CNN su GPU cloud
- 🐳 **DevOps**: Containerizzazione e deployment
- ☁️ **Cloud**: Servizi AWS managed
- 🔧 **MLOps**: Pipeline automation e monitoring

**Prossimi passi:**
- Esplora altri dataset e modelli
- Implementa monitoring avanzato
- Aggiungi CI/CD pipeline
- Scala per produzione enterprise 