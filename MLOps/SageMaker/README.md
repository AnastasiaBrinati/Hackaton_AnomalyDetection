# 🚀 MLOps SageMaker Project

Progetto MLOps enterprise-grade per classificazione di immagini Fashion MNIST utilizzando AWS SageMaker.

## 📋 Prerequisiti

- Python 3.8-3.11
- Account AWS configurato
- Docker installato (per containerizzazione)
- Git

## 🔧 Setup Iniziale

### 1. Clona il Repository
```bash
git clone <repository-url>
cd mlops-sagemaker-project
```

### 2. Crea Virtual Environment
```bash
python3 -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate    # Windows
```

### 3. Installa Dipendenze
```bash
# Ambiente di sviluppo
pip install -r requirements.txt

# Solo produzione
pip install -r requirements-prod.txt
```

### 4. Configura Credenziali AWS
```bash
# Crea file .env
cp .env.example .env

# Modifica .env con le tue credenziali:
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_DEFAULT_REGION=eu-west-1
# SAGEMAKER_ROLE_ARN=arn:aws:iam::account:role/SageMakerRole
```

## 🚀 Utilizzo

### Setup Locale
```bash
python mlops_setup_local.py
```

### Jupyter Notebook
```bash
jupyter notebook aws_sagemaker_lambda_exercise.ipynb
```

## 📁 Struttura Progetto

```
mlops-sagemaker-project/
├── .env                    # Credenziali AWS (non in Git)
├── .gitignore              # File da ignorare
├── requirements.txt        # Dipendenze complete
├── requirements-prod.txt   # Solo produzione
├── mlops_setup_local.py    # Setup automatico
├── aws_sagemaker_lambda_exercise.ipynb  # Notebook principale
└── README.md               # Documentazione
```

## 🛠️ Troubleshooting

### Errore TensorFlow
```bash
# Per Apple Silicon
pip install tensorflow-macos tensorflow-metal

# Per Linux ARM64
pip install tensorflow-cpu
```

### Errore Credenziali AWS
```bash
# Verifica configurazione
aws sts get-caller-identity
```

## 🔒 Sicurezza

- ⚠️ **MAI** committare il file `.env`
- Usa ruoli IAM invece di credenziali hardcoded
- Rotazione regolare delle credenziali AWS

## 📞 Supporto

- Verifica la configurazione con: `python mlops_setup_local.py`
- Controlla i log AWS CloudWatch per errori SageMaker
- Consulta la documentazione AWS SageMaker 