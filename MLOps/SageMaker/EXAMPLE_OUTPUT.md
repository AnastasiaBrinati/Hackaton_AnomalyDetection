# 📊 Esempio Output Test Script

## 🚀 Esecuzione del Test Script

```bash
$ python test_model_predictions.py
```

## 📱 Output Completo

```
🚀 === TEST MODELLO CNN FASHION MNIST ===
⏰ Avvio: 2024-01-15 14:30:25
==================================================

📊 Caricamento dataset Fashion MNIST...
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
29515/29515 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26421880/26421880 [==============================] - 2s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
5148/5148 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4422102/4422102 [==============================] - 1s 0us/step

✅ Dati caricati: 10000 immagini di test
   Dimensioni immagini: (28, 28)
   Range valori: 0.000 - 1.000

🎯 Scegli il tipo di test:
1. Test con modello locale
2. Test con modello remoto (AWS Lambda)
3. Entrambi
Inserisci la tua scelta (1-3): 3

--- TEST MODELLO LOCALE ---
🔍 Ricerca modello addestrato...
📂 Tentativo caricamento da: model/saved_model/00000001
❌ Errore caricamento da model/saved_model/00000001: [Errno 2] No such file or directory
📂 Tentativo caricamento da: model/model.h5
❌ Errore caricamento da model/model.h5: [Errno 2] No such file or directory
📂 Tentativo caricamento da: trained_model
❌ Errore caricamento da trained_model: [Errno 2] No such file or directory
⚠️  Modello addestrato non trovato, uso modello di fallback
🔧 Creazione modello di fallback...
⚠️  Usando modello non addestrato per demo

🎯 Test su 1000 campioni...
🎯 Esecuzione predizioni...
📊 Calcolo accuratezza su 1000 campioni...
🎯 Accuratezza: 9.40%

📈 Accuratezza per classe:
   T-shirt/top: 5.6%
   Trouser: 12.1%
   Pullover: 8.9%
   Dress: 11.7%
   Coat: 7.3%
   Sandal: 15.2%
   Shirt: 6.8%
   Sneaker: 13.4%
   Bag: 8.7%
   Ankle boot: 10.3%

🎨 Creazione visualizzazione per 8 immagini...
💾 Immagine salvata: test_predictions_20240115_143045.png

--- TEST MODELLO REMOTO (AWS LAMBDA) ---
🔧 Configurazione client AWS Lambda...
✅ Client AWS configurato per account: 943398317602
🔍 Ricerca funzione Lambda del progetto...
✅ Trovata funzione Lambda: mlops-exercise-invoker

🎯 Test singola immagine:
🎯 Test di una singola immagine su Lambda: mlops-exercise-invoker
📡 Invocazione Lambda con immagine di test (Vero: 'Sneaker')...

--- Risultato Test ---
Immagine di Test: Indice 5847, Etichetta Vera: Sneaker (classe 7)
Risposta dalla Lambda (Status Code): 200
Predizione del Modello: Sneaker (classe 7)
--------------------
💾 Immagine salvata: lambda_test_20240115_143052.png

🎯 Test multiple immagini:
🎯 Test di 6 immagini su Lambda: mlops-exercise-invoker
📡 Test 1/6 - Immagine 2341 (T-shirt/top)...
📡 Test 2/6 - Immagine 7892 (Bag)...
📡 Test 3/6 - Immagine 1456 (Pullover)...
📡 Test 4/6 - Immagine 9123 (Ankle boot)...
📡 Test 5/6 - Immagine 3567 (Dress)...
📡 Test 6/6 - Immagine 6789 (Coat)...
💾 Risultati salvati: lambda_multiple_test_20240115_143058.png

📊 Risultati Test Lambda:
🎯 Accuratezza: 83.33% (5/6)
✅ Predizioni corrette: 5
❌ Predizioni sbagliate: 1

==================================================
✅ Test completato con successo!
```

## 🎨 Visualizzazioni Generate

### **📸 Test Predictions (Locale)**
File: `test_predictions_20240115_143045.png`

```
🧪 Test Predizioni Modello CNN - Fashion MNIST
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Vero: T-shirt   │ Vero: Trouser   │ Vero: Pullover  │ Vero: Dress     │
│ Predetto: Bag   │ Predetto: Ankle │ Predetto: Coat  │ Predetto: Shirt │
│ Confidenza: 15% │ Confidenza: 22% │ Confidenza: 18% │ Confidenza: 12% │
│ ❌ SBAGLIATO    │ ❌ SBAGLIATO    │ ❌ SBAGLIATO    │ ❌ SBAGLIATO    │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Vero: Coat      │ Vero: Sandal    │ Vero: Shirt     │ Vero: Sneaker   │
│ Predetto: Dress │ Predetto: Sandal│ Predetto: Bag   │ Predetto: Bag   │
│ Confidenza: 19% │ Confidenza: 45% │ Confidenza: 16% │ Confidenza: 23% │
│ ❌ SBAGLIATO    │ ✅ CORRETTO     │ ❌ SBAGLIATO    │ ❌ SBAGLIATO    │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### **🌐 Lambda Test (Singola Immagine)**
File: `lambda_test_20240115_143052.png`

```
┌─────────────────────────────────────┐
│          Immagine di Test           │
│                                     │
│     [Immagine 28x28 Sneaker]       │
│                                     │
│   Vero: Sneaker                     │
│   Predetto: Sneaker                 │
│   ✅ CORRETTO                       │
└─────────────────────────────────────┘
```

### **📊 Lambda Multiple Test**
File: `lambda_multiple_test_20240115_143058.png`

```
🧪 Test Multiple Immagini via AWS Lambda - mlops-exercise-invoker
┌─────────────────┬─────────────────┬─────────────────┐
│ ✅ Vero: T-shirt│ ✅ Vero: Bag    │ ❌ Vero: Pullover│
│ Predetto: T-shirt│ Predetto: Bag  │ Predetto: Coat  │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────┬─────────────────┬─────────────────┐
│ ✅ Vero: Ankle  │ ✅ Vero: Dress  │ ✅ Vero: Coat   │
│ Predetto: Ankle │ Predetto: Dress │ Predetto: Coat  │
└─────────────────┴─────────────────┴─────────────────┘
```

## 📋 Interpretazione Risultati

### **🎯 Modello Locale (Fallback)**
- **Accuratezza**: 9.40% (molto bassa perché non addestrato)
- **Scopo**: Solo per demo e testing della pipeline
- **Uso**: Per verificare che il codice funzioni senza modello addestrato

### **🌐 Modello Remoto (AWS Lambda)**
- **Accuratezza**: 83.33% (buona performance)
- **Scopo**: Test del modello reale addestrato su SageMaker
- **Uso**: Validazione del modello in produzione

### **📊 Differenze Performance**
- **Locale**: Casualità (~10% accuracy su 10 classi)
- **Remoto**: Modello addestrato reale (~85-95% accuracy)
- **Conclusione**: Il modello remoto è significativamente migliore

## 🔧 Troubleshooting Comuni

### **❌ "Modello non trovato" (Locale)**
```
📂 Tentativo caricamento da: model/saved_model/00000001
❌ Errore caricamento da model/saved_model/00000001: [Errno 2] No such file or directory
```
**Soluzione**: Normale se non hai addestrato il modello localmente. Usa test remoto (opzione 2).

### **✅ "Lambda trovata automaticamente"**
```
🔍 Ricerca funzione Lambda del progetto...
✅ Trovata funzione Lambda: mlops-exercise-invoker
```
**Significato**: Il sistema ha trovato automaticamente la funzione Lambda. Test remoto disponibile.

### **📊 "Accuratezza bassa con modello locale"**
```
🎯 Accuratezza: 9.40%
```
**Spiegazione**: Normale per modello non addestrato. È solo per testing della pipeline.

## 🎯 Prossimi Passi

1. **Dopo il test**: Analizza i risultati per capire performance del modello
2. **Migliorie**: Identifica classi con bassa accuratezza per migliorare training
3. **Produzione**: Usa metriche per decidere se deploy in produzione
4. **Monitoring**: Ripeti test periodicamente per rilevare degrado del modello 