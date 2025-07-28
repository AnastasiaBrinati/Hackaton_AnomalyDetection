# 🎵 SIAE Anomaly Detection Hackathon
## Rileva Anomalie nei Diritti d'Autore Musicali

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tracks](https://img.shields.io/badge/Tracks-4-purple.svg)
![Dataset](https://img.shields.io/badge/Dataset-95K%20samples-brightgreen.svg)

---

## 🎯 **IL TUO OBIETTIVO**

**Sviluppa algoritmi di Machine Learning per rilevare anomalie nei dati musicali SIAE.**

Ogni track ha un dataset di **training** e un dataset di **test**. Il tuo modello deve:
1. **Imparare dai dati di training** (che contengono le etichette delle anomalie)
2. **Predire le anomalie nel dataset di test** (senza vedere le etichette vere)
3. **Essere valutato automaticamente** comparando le tue predizioni con la ground truth nascosta

🏆 **VINCERE**: Chi ottiene il miglior **F1-Score** sui dati di test nascosti!

---

## 📊 **DOVE SONO I DATI**

### 🚀 **PASSO 1: Genera i Dataset (OBBLIGATORIO)**

**TUTTI i partecipanti devono usare gli stessi dataset identici:**

```bash
# Esegui questo comando una sola volta
python generate_datasets.py
```

**Questo crea la cartella `datasets/` con:**
```
datasets/
├── track1_live_events_train.csv          # 40,000 eventi per training
├── track1_live_events_test.csv           # 10,000 eventi per test  
├── track1_live_events_test_ground_truth.csv  # 🔒 NASCOSTO (per valutazione)
├── track2_documents_train.csv            # 4,000 documenti per training
├── track2_documents_test.csv             # 1,000 documenti per test
├── track2_documents_test_ground_truth.csv    # 🔒 NASCOSTO (per valutazione)
├── track3_music_train.csv                # 20,000 tracce per training
├── track3_music_test.csv                 # 5,000 tracce per test
├── track3_music_test_ground_truth.csv        # 🔒 NASCOSTO (per valutazione)
├── track4_copyright_train.csv            # 12,000 opere per training
├── track4_copyright_test.csv             # 3,000 opere per test
└── track4_copyright_test_ground_truth.csv    # 🔒 NASCOSTO (per valutazione)
```

### 🔒 **IMPORTANTE: Ground Truth Nascosta**
- I file `*_test_ground_truth.csv` contengono le **vere etichette** del test set
- Tu **NON puoi usarli** per il training
- Il sistema li usa **automaticamente** per valutare le tue predizioni
- **Fair play**: tutti lavorano solo sui dati di training!

---

## 📝 **COSA CONTENGONO I DATASET**

### 🎪 **Track 1: Live Events Anomaly Detection**

**Dati**: Eventi musicali live (concerti, festival, locali). Le anomalie sono combinazioni sospette di dati.

**File Training**: `datasets/track1_live_events_train.csv`
**File Test**: `datasets/track1_live_events_test.csv`

**Colonne principali**:
```python
event_id              # ID univoco evento
venue                 # Nome del locale/venue  
city                  # Città dell'evento
event_date            # Data dell'evento
attendance            # Numero partecipanti
capacity              # Capacità massima venue
n_songs_declared      # Numero brani dichiarati
total_revenue         # Ricavi totali evento
event_duration_hours  # Durata in ore
main_artist           # Artista principale
is_anomaly            # 🎯 TARGET: 0=normale, 1=anomalo (solo per valutare i modelli)
anomaly_type          # Tipo di anomalia (solo per valutare i modelli)
```

**🚨 Anomalie Sottili da rilevare**:
- **unlikely_combination**: Combinazioni di feature improbabili (es. superstar in un grande evento con ricavi irrisori).
- **anomalous_venue_behavior**: Comportamento anomalo aggregato per un `venue` (es. eventi sempre lunghissimi con pochissime canzoni).
- **hidden_cluster**: Piccoli gruppi di eventi con caratteristiche multivariate anomale (es. bassa affluenza, durata e ricavi).
- **subtle_revenue_fraud**: Ricavi leggermente ma costantemente inferiori alla norma attesa per quel tipo di evento.
- **impossible_tour_date**: Pattern temporali e geografici impossibili per il tour di un artista (es. stesso artista, stesso giorno, città distanti).

### 📄 **Track 2: Document Fraud Detection**

**Dati**: Metadati estratti da documenti SIAE. La frode risiede nell'incoerenza tra le caratteristiche.

**File Training**: `datasets/track2_documents_train.csv`
**File Test**: `datasets/track2_documents_test.csv`

**Colonne principali**:
```python
document_id           # ID univoco documento
creation_date         # Data di creazione
file_size_kb          # Dimensione del file in KB
resolution_dpi        # Risoluzione della scansione
text_confidence_avg   # Livello medio di confidenza OCR
pixel_noise_level     # Livello di rumore nell'immagine
edge_sharpness        # Nitidezza dei bordi del testo/immagini
metadata_consistency  # Coerenza dei metadati interni
submitter_id          # ID dell'utente che ha caricato il documento
is_fraudulent         # 🎯 TARGET: 0=autentico, 1=fraudolento (solo per valutare i modelli)
fraud_type            # Tipo di frode (solo per valutare i modelli)
```

**🚨 Frodi Sottili da rilevare**:
- **too_perfect_for_age**: Documenti dichiarati come molto vecchi ma con qualità digitale perfetta (zero rumore, alta risoluzione).
- **sophisticated_template_fraud**: Gruppi di documenti, da mittenti diversi, con caratteristiche di file quasi identiche (indizio di un template fraudolento condiviso).
- **internal_feature_inconsistency**: Incoerenza tra feature tecniche (es. alta nitidezza ma bassa confidenza del testo).
- **subtle_metadata_manipulation**: Discrepanze minime ma sospette nei metadati che richiedono un'analisi attenta.

### 🎵 **Track 3: Music Anomaly Detection**

**Dati**: Tracce musicali con metadati e feature audio. Le anomalie sono pattern di comportamento o tecniche.

**File Training**: `datasets/track3_music_train.csv`
**File Test**: `datasets/track3_music_test.csv`

**Colonne principali**:
```python
track_id                  # ID univoco traccia
artist_name               # Nome artista
track_listens             # Numero ascolti
track_favorites           # Numero preferiti
track_comments            # Numero commenti
artist_active_year_begin  # Anno inizio attività artista
bit_rate                  # Bit rate audio
spectral_bandwidth        # Larghezza di banda spettrale (feature audio)
listener_country_entropy  # Entropia geografica degli ascoltatori
is_anomaly                # 🎯 TARGET: 0=normale, 1=anomalo (solo per valutare i modelli)
anomaly_type              # Tipo di anomalia (solo per valutare i modelli)
```

**🚨 Anomalie Sottili da rilevare**:
- **sophisticated_bot_streaming**: Attività di bot con un rapporto ascolti/preferiti normale, ma con pattern geografici e di commenti innaturali.
- **artist_hijacking**: Artisti storicamente inattivi che pubblicano improvvisamente tracce di generi moderni (possibile furto di account o frode).
- **audio_quality_fraud**: Incoerenza tra la qualità dichiarata (`bit_rate`) e le reali caratteristiche spettrali del file (es. un file di bassa qualità "gonfiato").

### 🔒 **Track 4: Copyright Infringement Detection**

**Dati**: Opere creative e loro utilizzo. Le violazioni sono nascoste tramite tecniche di evasione.

**File Training**: `datasets/track4_copyright_train.csv`
**File Test**: `datasets/track4_copyright_test.csv`

**Colonne principali**:
```python
work_id                   # ID univoco opera
days_since_release        # Giorni dalla pubblicazione
play_count                # Conteggio totale ascolti
play_count_last_24h       # Conteggio ascolti nelle ultime 24h
audio_similarity_to_db    # Similarità audio con un database di opere protette
noise_floor_db            # Livello di rumore di fondo
file_hash                 # Hash univoco del file
is_infringement           # 🎯 TARGET: 0=legale, 1=violazione (solo per valutare i modelli)
infringement_type         # Tipo di violazione (solo per valutare i modelli)
```

**🚨 Violazioni Sottili da rilevare**:
- **evasion_by_modification**: Elusione del Content ID tramite lievi modifiche audio (velocità, intonazione) che cambiano l'hash ma non la similarità audio.
- **sleeper_infringement**: Opere caricate e tenute "dormienti", seguite da un picco di attività improvviso e innaturale per monetizzare rapidamente.
- **evasion_by_noise_masking**: Mascheramento di una traccia protetta tramite la sovrapposizione di un leggero rumore, che altera le feature tecniche.

---

## 🚀 **COME PARTECIPARE: GUIDA STEP-BY-STEP**

### **STEP 1: Setup Iniziale**

```bash
# 1. Clona il repository
git clone <repository-url>
cd "Anomaly Detection"

# 2. Installa dipendenze
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. 🚨 OBBLIGATORIO: Genera dataset identici per tutti
python generate_datasets.py
```

### **STEP 2: Scegli la Tua Track**

Puoi partecipare a **una o più track**. Ogni track è indipendente:

```bash
# Vai nella cartella della track che vuoi affrontare
cd Track1_Solution   # Live Events
# oppure
cd Track2_Solution   # Document Fraud  
# oppure
cd Track3_Solution   # Music Anomaly
# oppure
cd Track4_Solution   # Copyright Infringement
```

### **STEP 3: Personalizza il Tuo Team**

**🚨 IMPORTANTE**: Prima di eseguire lo script, apri il file Python e modifica:

```python
# Cerca queste righe nel file track*.py e modificale:
team_name = "IL_TUO_NOME_TEAM"           # 👈 CAMBIA QUI
members = ["Nome1", "Nome2", "Nome3"]    # 👈 CAMBIA QUI
```

### **STEP 4: Esegui lo Script Base**

```bash
# Esempio per Track 1
python track1_anomaly_detection.py
```

**Cosa succede**:
1. ✅ Carica i dati di training da `datasets/track1_live_events_train.csv`
2. ✅ Addestra un modello di machine learning
3. ✅ Carica i dati di test da `datasets/track1_live_events_test.csv`  
4. ✅ Fa predizioni sul test set
5. ✅ Genera il file di submission in `submissions/submission_[team]_track1.json`
6. ✅ Salva visualizzazioni e metriche

### **STEP 5: Verifica i Risultati**

```bash
# Controlla che sia stato generato il file di submission
ls submissions/

# Dovresti vedere file come:
# submission_il_tuo_team_track1.json
# submission_il_tuo_team_track2.json
# etc.
```

### **STEP 6: Fai la Submission**

```bash
# Aggiungi i file di submission
git add submissions/submission_*.json

# Commit con messaggio chiaro
git commit -m "Team [IL_TUO_NOME] - Submission Track 1,2,3,4"

# Push per aggiornare la leaderboard
git push origin main
```

**🎉 FATTO!** Il sistema valuta automaticamente le tue predizioni e aggiorna la leaderboard!

---

## 📄 **FORMATO FILE DI SUBMISSION**

Il tuo script deve generare un file JSON con questo formato **esatto**:

### **Esempio `submission_team_example_track1.json`**

```json
{
  "team_info": {
    "team_name": "Team Example",
    "members": ["Alice", "Bob", "Charlie"],
    "track": "Track1",
    "submission_time": "2024-01-15T14:30:00Z",
    "submission_number": 1
  },
  "model_info": {
    "algorithm": "Isolation Forest + DBSCAN",
    "features_used": ["attendance", "revenue_per_person", "occupancy_rate"],
    "hyperparameters": {
      "contamination": 0.08,
      "n_estimators": 200,
      "random_state": 42
    },
    "feature_engineering": [
      "revenue_per_person", "occupancy_rate", "songs_per_person"
    ]
  },
  "results": {
    "total_test_samples": 10000,
    "anomalies_detected": 950,
    "predictions": [0, 1, 0, 1, 0, 0, 1],
    "scores": [-0.1, 0.8, -0.2, 0.7, -0.3, 0.1, 0.9]
  },
  "metrics": {
    "precision": 0.75,
    "recall": 0.68,
    "f1_score": 0.71,
    "auc_roc": 0.85
  },
  "performance_info": {
    "training_time_seconds": 12.5,
    "prediction_time_seconds": 2.1,
    "memory_usage_mb": 128,
    "model_size_mb": 5.2
  }
}
```

### **🔧 Codice Template per Generare Submission**

```python
import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def generate_submission(df_test, predictions, scores, feature_cols, 
                       team_name="YOUR_TEAM", members=["Member1", "Member2"], 
                       track="Track1"):
    """
    Genera il file di submission nel formato corretto
    
    Parameters:
    - df_test: DataFrame di test
    - predictions: array di predizioni binarie (0/1)
    - scores: array di anomaly scores
    - feature_cols: lista delle features utilizzate
    - team_name: nome del team  
    - members: lista dei membri del team
    - track: nome della track ("Track1", "Track2", etc.)
    """
    
    # Metriche mock (il sistema calcolerà quelle reali con la ground truth)
    anomalies_detected = predictions.sum()
    anomaly_rate = anomalies_detected / len(predictions)
    
    # Stima approssimativa delle metriche (non reali!)
    precision = 0.70 + (anomaly_rate * 0.1)
    recall = 0.65 + (anomaly_rate * 0.15)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc_roc = 0.80 + (len(feature_cols) * 0.01)
    
    submission = {
        "team_info": {
            "team_name": team_name,
            "members": members,
            "track": track,
            "submission_time": datetime.now().isoformat() + "Z",
            "submission_number": 1
        },
        "model_info": {
            "algorithm": "Isolation Forest + Feature Engineering",
            "features_used": feature_cols.copy(),
            "hyperparameters": {
                "contamination": 0.08,
                "n_estimators": 200,
                "random_state": 42
            },
            "feature_engineering": [
                "ratio_features", "normalized_features", "categorical_encoding"
            ]
        },
        "results": {
            "total_test_samples": len(df_test),
            "anomalies_detected": int(anomalies_detected),
            "predictions": predictions.tolist(),  # 🚨 ARRAY COMPLETO RICHIESTO
            "scores": scores.tolist()             # 🚨 ARRAY COMPLETO RICHIESTO
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc_roc, 4)
        },
        "performance_info": {
            "training_time_seconds": 15.0,
            "prediction_time_seconds": 3.0,
            "memory_usage_mb": 256,
            "model_size_mb": 8.5
        }
    }
    
    # Salva il file
    filename = f"submissions/submission_{team_name.lower().replace(' ', '_')}_{track.lower()}.json"
    
    import os
    os.makedirs("submissions", exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Submission salvata: {filename}")
    return filename, submission

# ESEMPIO DI USO:
if __name__ == "__main__":
    # 1. Carica i dati di training
    df_train = pd.read_csv('datasets/track1_live_events_train.csv')
    df_test = pd.read_csv('datasets/track1_live_events_test.csv')
    
    # 2. Feature engineering
    def create_features(df):
        df['revenue_per_person'] = df['total_revenue'] / df['attendance']
        df['occupancy_rate'] = df['attendance'] / df['capacity']
        df['songs_per_person'] = df['n_songs'] / df['attendance']
        return df
    
    df_train = create_features(df_train)
    df_test = create_features(df_test)
    
    # 3. Prepara le features
    feature_cols = ['attendance', 'capacity', 'n_songs', 'total_revenue',
                   'revenue_per_person', 'occupancy_rate', 'songs_per_person']
    
    X_train = df_train[feature_cols].fillna(0)
    X_test = df_test[feature_cols].fillna(0)
    
    # 4. Addestra il modello
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    iso_forest.fit(X_train)
    
    # 5. Predizioni sul test set
    predictions = iso_forest.predict(X_test)
    predictions = (predictions == -1).astype(int)  # Converti -1/1 in 1/0
    scores = iso_forest.score_samples(X_test)
    
    # 6. Genera submission
    generate_submission(
        df_test=df_test,
        predictions=predictions,
        scores=scores,
        feature_cols=feature_cols,
        team_name="Team Example",  # 👈 CAMBIA QUI
        members=["Alice", "Bob"],   # 👈 CAMBIA QUI
        track="Track1"
    )
```

---

## 🏆 **SISTEMA DI VALUTAZIONE**

### **Come Funziona**

1. **Il tuo script genera** il file JSON con le predizioni
2. **Fai git push** del file di submission
3. **Il sistema automaticamente**:
   - Carica le tue predizioni dal JSON
   - Le confronta con la ground truth nascosta (`*_test_ground_truth.csv`)
   - Calcola le **metriche reali** (F1-Score, Precision, Recall, AUC-ROC)
   - Aggiorna la **leaderboard** in tempo reale

### **Scoring Formula**

```
Final Score = (F1-Score × 40%) + (AUC-ROC × 25%) + (Precision × 20%) + (Innovation × 15%)
```

Dove:
- **F1-Score**: Quanto bene rilevi le anomalie (principale)
- **AUC-ROC**: Robustezza del modello
- **Precision**: Accuratezza delle predizioni positive
- **Innovation**: Numero di features utilizzate e complessità

### **Leaderboard**

Controlla la tua posizione in tempo reale:
```bash
# Visualizza la leaderboard aggiornata
cat leaderboard.md
```

### **Premi**

- 🥇 **Vincitore Overall**: Miglior score complessivo
- 🏅 **Vincitore per Track**: Miglior score per ogni track
- 🌟 **Most Innovative**: Approccio più creativo
- 🚀 **Rising Star**: Team emergente con grande potenziale

---

## 💡 **CONSIGLI STRATEGICI**

### **1. Inizia Semplice**
Prima di costruire modelli complessi, fai un'analisi esplorativa dei dati (EDA). Usa `pairplot` e `boxplot` per visualizzare le differenze tra dati normali e anomali nel set di training.

### **2. Feature Engineering è la Chiave**
Le anomalie sono contestuali. Crea feature che catturino questo contesto.

**Esempi per Track 1 (Live Events):**
```python
# Analisi aggregata per venue o artista (aiuta a trovare comportamenti anomali)
df['avg_revenue_per_venue'] = df.groupby('venue')['total_revenue'].transform('mean')
df['artist_event_count'] = df.groupby('main_artist')['event_id'].transform('count')
```

**Esempi per Track 2 (Document Fraud):**
```python
# Feature di interazione
df['file_density'] = df['file_size_kb'] / df['page_count']
# Analisi aggregata per mittente (aiuta a trovare frodi di template)
df['submitter_doc_count'] = df.groupby('submitter_id')['document_id'].transform('count')
```

**Esempi per Track 3 (Music Anomaly):**
```python
# Feature di rapporto e comportamento
df['favorites_per_listen'] = df['track_favorites'] / (df['track_listens'] + 1)
df['artist_age'] = datetime.now().year - df['artist_active_year_begin']
```

**Esempi per Track 4 (Copyright):**
```python
# Feature che catturano pattern temporali
df['is_sleeper'] = (df['days_since_release'] > 180) & (df['play_count_last_24h'] > 1000)
# Rapporto tra similarità e rumore
df['similarity_to_noise_ratio'] = df['audio_similarity_to_db'] / (abs(df['noise_floor_db']) + 1)
```

### **3. Prova Algoritmi Diversi**
- **Non supervisionati (se non avessi le etichette):** `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`. Utili per trovare anomalie "sconosciute".
- **Supervisionati (approccio consigliato qui):** `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBoost`. Sfruttano le etichette nel training set per imparare a distinguere le anomalie note. Sono generalmente più potenti per questo tipo di problema.

### **4. Validazione Robusta**
Usa la validazione incrociata (`StratifiedKFold`) sul set di training per stimare le performance del tuo modello in modo affidabile prima di creare la submission.

---

## ❗ **ERRORI COMUNI DA EVITARE**

### **❌ Usare i dati di test per il training**
```python
# SBAGLIATO - NON FARE MAI!
df_all = pd.concat([df_train, df_test])  # ❌ CHEATING!
```

### **❌ Dimenticare di modificare team_name**
```python
# Ricorda di cambiare questi valori!
team_name = "YourTeam"  # 👈 CAMBIA QUI
members = ["Nome1"]     # 👈 CAMBIA QUI
```

### **❌ Predizioni mancanti o sbagliate**
```python
# Assicurati che le predizioni siano per TUTTO il test set
assert len(predictions) == len(df_test), "Predizioni incomplete!"
assert set(predictions) <= {0, 1}, "Predizioni devono essere 0 o 1!"
```

### **❌ JSON malformato**
```python
# Testa sempre il tuo JSON
import json
with open('submission_test.json') as f:
    data = json.load(f)  # Se fallisce, il JSON è malformato
```

---

## 🛠️ **TROUBLESHOOTING**

### **Problema: Dataset non trovato**
```bash
# Soluzione: Rigenera i dataset
python generate_datasets.py
```

### **Problema: Errore JSON nel submission**
```python
# Testa il JSON
python -m json.tool submissions/submission_team.json
```

### **Problema: Submission non appare in leaderboard**
```bash
# Verifica il commit
git status
git add submissions/
git commit -m "Submission update"
git push origin main
```

### **Problema: Performance troppo basse**
```python
# Controlla il bilanciamento delle classi
print(f"Anomalie nel training: {df_train['is_anomaly'].mean():.2%}")

# Prova parametri diversi
iso_forest = IsolationForest(contamination=0.05)  # Prova valori 0.05-0.15
```

---

## 📚 **RISORSE AGGIUNTIVE**

### **Esempi di Codice**
Tutti gli script nelle cartelle `Track*_Solution/` sono esempi funzionanti che puoi modificare e migliorare!

---

**🎯 Buona fortuna! Che vinca il miglior algoritmo di anomaly detection! 🚀**

---

*Ultima modifica: Luglio 2025* 