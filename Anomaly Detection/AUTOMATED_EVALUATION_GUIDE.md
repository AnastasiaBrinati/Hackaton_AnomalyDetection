# 🤖 Sistema di Valutazione Automatica SIAE Hackathon

## 🌟 Panoramica

Il sistema di valutazione è ora completamente automatizzato e supporta il nuovo approccio **train/test/ground_truth**:

- 👨‍💻 **Partecipanti** usano solo `dataset_train.csv` per sviluppare i modelli
- 🧪 **Sistema** testa automaticamente su `dataset_test.csv` (nascosto ai partecipanti)
- 🎯 **Valutazione** confronta le predizioni con `ground_truth.csv` (nascosto)
- 🏆 **Leaderboard** si aggiorna automaticamente ad ogni commit

## 🚀 Metodi di Automazione Disponibili

### 1. 🌐 GitHub Actions (Raccomandato)

**✅ Vantaggi:**
- Completamente in cloud, nessuna dipendenza locale
- Affidabile e scalabile per hackathon
- Log dettagliati e notifiche automatiche
- Funziona su tutti i repository GitHub

**📁 File:** `.github/workflows/auto-leaderboard.yml`

**🔧 Come funziona:**
1. Si attiva automaticamente quando vengono pushati file `submissions/submission_*.json`
2. Valida la presenza dei file ground truth
3. Esegue `evaluate_submissions.py` 
4. Confronta le predizioni con la ground truth reale
5. Aggiorna `leaderboard.md` automaticamente
6. Committa i cambiamenti

**📊 Trigger:**
```yaml
on:
  push:
    branches: [ main, master ]
    paths:
      - 'submissions/submission_*.json'
```

### 2. 🔧 Git Hooks (Sistema Locale)

**📁 Setup:** `python setup_auto_leaderboard.py`

**🔧 Come funziona:**
- Hook `post-commit` si attiva sui commit locali
- Controlla se ci sono nuovi file submission
- Esegue valutazione automaticamente
- Committa la leaderboard aggiornata

**⚠️ Limitazioni:**
- Funziona solo sulla macchina locale
- Richiede setup manuale per ogni contributore

### 3. 👀 File Watcher (Tempo Reale)

**📁 Script:** `python file_watcher.py`

**🔧 Come funziona:**
- Monitora la cartella `submissions/` in tempo reale
- Si attiva quando vengono modificati/creati file submission
- Aggiornamento immediato (< 5 secondi)

**⚠️ Limitazioni:**
- Richiede processo attivo
- Solo per testing/sviluppo locale

### 4. 📋 Aggiornamento Manuale

**📁 Script:** `./update_leaderboard_manual.sh`

**🔧 Uso:**
```bash
./update_leaderboard_manual.sh
```

## 🎯 Come Funziona la Nuova Valutazione

### 📚 Per i Partecipanti

1. **Caricano solo il training set:**
   ```python
   df_train = pd.read_csv('../datasets/track1_live_events_train.csv')
   ```

2. **Addestrano il modello sul training:**
   ```python
   model.fit(X_train, y_train)
   ```

3. **Fanno predizioni sul test set** (senza vedere la ground truth):
   ```python
   df_test = pd.read_csv('../datasets/track1_live_events_test.csv')
   predictions = model.predict(X_test)
   ```

4. **Generano submission con le predizioni:**
   ```python
   submission_data = {
       "results": {
           "predictions": predictions.tolist(),
           "scores": model.score_samples(X_test).tolist()
       }
   }
   ```

### 🤖 Per il Sistema di Valutazione

1. **Estrae le predizioni dalla submission:**
   ```python
   predictions = submission_data['results']['predictions']
   ```

2. **Carica la ground truth reale:**
   ```python
   df_gt = pd.read_csv('datasets/track1_live_events_test_ground_truth.csv')
   y_true = df_gt['is_anomaly'].values
   ```

3. **Calcola metriche reali:**
   ```python
   precision, recall, f1 = precision_recall_fscore_support(y_true, predictions)
   ```

4. **Aggiorna la leaderboard** con i risultati reali

## 📊 Struttura dei Dataset

```
datasets/
├── track1_live_events_train.csv          # Training data (per partecipanti)
├── track1_live_events_test.csv           # Test data (senza ground truth)
├── track1_live_events_test_ground_truth.csv  # Ground truth (nascosto)
├── track2_documents_train.csv
├── track2_documents_test.csv  
├── track2_documents_test_ground_truth.csv
├── track3_music_train.csv
├── track3_music_test.csv
├── track3_music_test_ground_truth.csv
├── track4_copyright_train.csv
├── track4_copyright_test.csv
└── track4_copyright_test_ground_truth.csv
```

## 🔧 Setup Completo

### 1. Per GitHub Actions (Raccomandato)

```bash
# 1. Il workflow è già configurato in .github/workflows/auto-leaderboard.yml
# 2. Nessun setup aggiuntivo richiesto
# 3. Si attiva automaticamente sui push di submission
```

### 2. Per Sistema Locale (Backup)

```bash
# Setup completo sistema locale
python setup_auto_leaderboard.py

# Test manuale
./update_leaderboard_manual.sh

# Monitoraggio in tempo reale (opzionale)
python file_watcher.py
```

## 📋 Formato Submission Aggiornato

I partecipanti devono includere le **predizioni effettive** nella submission:

```json
{
  "team_info": {
    "team_name": "Team Awesome",
    "members": ["Alice", "Bob"],
    "track": "Track1",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest",
    "features_used": ["feature1", "feature2"]
  },
  "results": {
    "predictions": [0, 1, 0, 1, 0],  # PREDIZIONI SUL TEST SET
    "scores": [-0.1, 0.8, -0.2, 0.7, -0.1],  # SCORES DI ANOMALIA
    "total_test_samples": 5,
    "anomalies_detected": 2
  },
  "metrics": {
    "precision": 0.85,  # Metriche calcolate su validation set locale
    "recall": 0.78,
    "f1_score": 0.81,
    "auc_roc": 0.89
  }
}
```

## 🎯 Vantaggi del Nuovo Sistema

### ✅ **Fairness**
- Tutti i team usano gli stessi dataset di training
- Valutazione su test set identico nascosto
- Metriche calcolate centralmente (no cheating)

### ✅ **Automatizzazione**
- Zero intervento manuale durante l'hackathon
- Leaderboard sempre aggiornata
- Log dettagliati per debug

### ✅ **Scalabilità**
- Supporta molti team simultanei
- GitHub Actions gestisce il carico
- Sistema robusto per eventi live

### ✅ **Trasparenza**
- Processo di valutazione completamente visibile
- Log pubblici delle valutazioni
- Statistiche dettagliate per track

## 🚨 Troubleshooting

### ❌ "Ground truth file not found"
```bash
# Assicurati che i file ground truth esistano
ls datasets/*_test_ground_truth.csv

# Se mancanti, rigenera i dataset
python generate_datasets.py
```

### ❌ "No predictions found in submission"
```python
# Assicurati che la submission includa le predizioni
submission_data = {
    "results": {
        "predictions": predictions.tolist(),  # NECESSARIO
        "scores": scores.tolist()  # OPZIONALE ma consigliato
    }
}
```

### ❌ "GitHub Actions not triggering"
1. Verifica che il file sia in `.github/workflows/auto-leaderboard.yml`
2. Controlla che il push includa file `submissions/submission_*.json`
3. Verifica i permessi del repository (Actions abilitate)

### ❌ "Dimension mismatch"
- Il sistema adatta automaticamente le dimensioni
- Assicurati che le predizioni abbiano almeno N elementi (dove N = dimensione test set)

## 📞 Supporto

Durante l'hackathon, controlla:

1. **GitHub Actions logs** nella tab "Actions" del repository
2. **File `submission_stats.json`** per statistiche real-time
3. **Directory `logs/`** per log dettagliati di valutazione
4. **Issue tracker** per segnalare problemi

## 🎉 Come Partecipare

1. **Modifica le soluzioni** per usare solo dataset_train.csv
2. **Fai predizioni** sul dataset_test.csv (senza vedere ground truth)
3. **Genera submission** con le predizioni incluse
4. **Committa e pusha** - la leaderboard si aggiorna automaticamente!

```bash
# Esempio completo
cd Track1_Solution
python track1_anomaly_detection.py
git add ../submissions/submission_team_track1.json
git commit -m "Team submission Track 1"
git push

# La leaderboard si aggiorna automaticamente in 2-3 minuti! 🚀
``` 