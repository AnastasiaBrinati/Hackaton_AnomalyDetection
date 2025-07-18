# 📚 Guida all'Utilizzo degli Script MLOps

Questa guida spiega come utilizzare gli script per automatizzare test e operazioni MLOps.

## 🚀 Script Disponibili

### 1. `mlops_runner.py` - Script Completo
Script avanzato con parametri configurabili per test professionali.

### 2. `quick_test.py` - Test Rapido
Script semplificato per test veloci senza parametri.

### 3. `mlops_setup_local.py` - Configurazione
Script per configurare l'ambiente locale.

## 📋 Utilizzo Rapido

### Test Veloce
```bash
# Attiva ambiente
source mlops_env/bin/activate

# Esegui test rapido
python quick_test.py
```

### Test Personalizzato
```bash
# Test con parametri personalizzati
python mlops_runner.py --lambda-function mlops-fashion-classifier-invoker --num-tests 5 --save-plots

# Solo URI Docker
python mlops_runner.py --docker-uri-only --lambda-function nome-lambda
```

## 🔧 Parametri mlops_runner.py

| Parametro | Descrizione | Esempio |
|-----------|-------------|---------|
| `--lambda-function` `-l` | Nome funzione Lambda (obbligatorio) | `mlops-fashion-classifier-invoker` |
| `--num-tests` `-n` | Numero di test da eseguire | `3` (default) |
| `--save-plots` `-s` | Salva grafici dei risultati | Flag |
| `--docker-uri-only` `-d` | Mostra solo URI Docker | Flag |

## 📊 Esempi di Utilizzo

### Test Base
```bash
python mlops_runner.py -l mlops-fashion-classifier-invoker
```

### Test Multipli con Salvataggio
```bash
python mlops_runner.py -l mlops-fashion-classifier-invoker -n 10 -s
```

### Solo URI Docker
```bash
python mlops_runner.py -d -l mlops-fashion-classifier-invoker
```

## 📁 Output Generato

### Grafici
- File: `mlops_test_results_YYYYMMDD_HHMMSS.png`
- Formato: PNG ad alta risoluzione (300 DPI)
- Contenuto: Immagini test con predizioni

### Log Console
- Setup ambiente
- URI Docker
- Risultati test singoli
- Statistiche finali

## 🔧 Configurazione

### Variabili d'Ambiente (.env)
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-west-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::account:role/SageMakerRole
```

### Nome Funzione Lambda
Modifica la variabile `LAMBDA_FUNCTION_NAME` in `quick_test.py`:
```python
LAMBDA_FUNCTION_NAME = "tuo-nome-lambda-function"
```

## 📊 Interpretazione Risultati

### Output Console
```
📊 Risultato Test 1:
   Immagine: 1234
   Vero: T-shirt/top (classe 0)
   Predetto: T-shirt/top (classe 0)
   Corretto: ✅
```

### Statistiche
```
📈 Statistiche Test:
   Test eseguiti: 3
   Predizioni corrette: 2
   Accuratezza: 66.7%
```

### Grafici
- **Verde**: Predizione corretta
- **Rosso**: Predizione sbagliata
- **Titolo**: Mostra etichetta vera vs predetta

## 🛠️ Troubleshooting

### Errore "Lambda function not found"
```bash
# Verifica che la funzione Lambda esista
aws lambda list-functions --query 'Functions[?contains(FunctionName, `mlops`)]'
```

### Errore "Access denied"
```bash
# Verifica credenziali AWS
aws sts get-caller-identity
```

### Errore "TensorFlow not available"
```bash
# Installa TensorFlow
pip install tensorflow==2.11.0
```

### Errore "Matplotlib not available"
```bash
# Installa matplotlib
pip install matplotlib
```

## 🎯 Best Practices

### 1. Test Regolari
```bash
# Esegui test dopo ogni deploy
python quick_test.py
```

### 2. Batch Testing
```bash
# Test multipli per validazione
python mlops_runner.py -l function-name -n 20 -s
```

### 3. Monitoraggio
```bash
# Salva risultati per analisi
python mlops_runner.py -l function-name -n 10 -s
```

## 🔄 Workflow Consigliato

### 1. Setup Iniziale
```bash
# Configura ambiente
python mlops_setup_local.py

# Verifica configurazione
python mlops_runner.py -d -l function-name
```

### 2. Test Sviluppo
```bash
# Test rapidi durante sviluppo
python quick_test.py
```

### 3. Validazione
```bash
# Test estensivi pre-produzione
python mlops_runner.py -l function-name -n 20 -s
```

### 4. Monitoraggio Produzione
```bash
# Test periodici in produzione
python mlops_runner.py -l function-name -n 5
```

## 📈 Metriche di Successo

### Accuratezza
- **>90%**: Eccellente
- **80-90%**: Buona
- **70-80%**: Accettabile
- **<70%**: Richiede revisione

### Tempo di Risposta
- **<500ms**: Ottimo
- **500ms-1s**: Buono
- **1s-2s**: Accettabile
- **>2s**: Lento

### Errori
- **0%**: Ideale
- **<5%**: Accettabile
- **>5%**: Richiede attenzione

## 🆘 Supporto

### Log Debugging
```bash
# Attiva debug verbose
export DEBUG=true
python mlops_runner.py -l function-name
```

### Verifica Stato Sistema
```bash
# Controlla stato AWS
aws sts get-caller-identity
aws lambda list-functions
```

### Reset Ambiente
```bash
# Ricrea ambiente se necessario
deactivate
rm -rf mlops_env
python -m venv mlops_env
source mlops_env/bin/activate
pip install -r requirements.txt
``` 