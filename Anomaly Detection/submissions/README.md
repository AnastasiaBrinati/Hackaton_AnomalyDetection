# 📤 Submissions - Track 1: Live Events Anomaly Detection

## Come Submittare la Tua Soluzione

### 1. 🎯 Genera il File di Submission

Esegui il tuo script Python che include la funzione `generate_submission()`:

```python
# Modifica questi parametri nel tuo script
team_name = "Il Tuo Team Name"  # CAMBIA QUI
members = ["Nome1", "Nome2", "Nome3"]  # CAMBIA QUI

# Genera il file di submission
submission_file, submission_data = generate_submission(
    df=df, 
    iso_forest=iso_forest, 
    feature_cols=feature_cols,
    team_name=team_name,
    members=members
)
```

### 2. 📋 Verifica il Formato

Il file generato deve avere questa struttura:

```json
{
  "team_info": {
    "team_name": "Il Tuo Team Name",
    "members": ["Nome1", "Nome2", "Nome3"],
    "track": "Track1",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest + DBSCAN",
    "features_used": ["attendance", "revenue_per_person", ...],
    "hyperparameters": {...}
  },
  "results": {
    "total_events": 10000,
    "anomalies_detected": 950,
    "predictions_sample": [0, 1, 0, 1, ...],
    "anomaly_scores_sample": [-0.1, 0.8, ...]
  },
  "metrics": {
    "precision": 0.85,
    "recall": 0.78,
    "f1_score": 0.81,
    "auc_roc": 0.89
  }
}
```

### 3. 🚀 Committa e Pusha

```bash
# Aggiungi il file di submission
git add submissions/submission_il_tuo_team_name.json

# Committa con messaggio descrittivo
git commit -m "Il Tuo Team Name - Track 1 submission"

# Pusha per triggerare l'aggiornamento della leaderboard
git push origin main
```

## 📊 Cosa Viene Valutato

### Technical Score (50%)
- **F1-Score** (25%): Performance principale del modello
- **AUC-ROC** (15%): Robustezza nelle diverse soglie
- **Precision** (10%): Accuratezza delle anomalie rilevate

### Innovation Score (30%)
- **Features Used** (10%): Numero e varietà di features
- **Algorithm Complexity** (10%): Complessità tecnica
- **Feature Engineering** (10%): Features create/derivate

### Business Impact (20%)
- **Performance** (10%): Efficienza temporale e memoria
- **Interpretability** (10%): Capacità di spiegare anomalie

## 🏆 Leaderboard

La leaderboard si aggiorna automaticamente ad ogni submission valida!

Puoi controllare la tua posizione in: [`leaderboard.md`](../leaderboard.md)

## ✅ Regole di Submission

- **Max 5 submissions** per team al giorno
- **Ultimo submission** conta per la classifica finale
- **Formato JSON** obbligatorio come specificato
- **Nome file**: `submission_[team_name_lowercase].json`

## 🛠️ Troubleshooting

### Errore: "Missing required field"
Verifica che il tuo JSON contenga tutti i campi richiesti:
- `team_info`
- `model_info` 
- `results`
- `metrics`

### Errore: "Invalid JSON format"
Valida il tuo JSON:
```bash
python -m json.tool submissions/submission_tuo_team.json
```

### Submission non appare in leaderboard
1. Verifica che il file sia stato committato
2. Controlla il formato JSON
3. Assicurati che il nome file segua il pattern `submission_*.json`

## 📞 Aiuto

Se hai problemi con la submission:
1. Controlla il file di esempio: `submission_example.json`
2. Verifica la documentazione completa: [`EVALUATION_SYSTEM.md`](../EVALUATION_SYSTEM.md)
3. Contatta gli organizzatori

---

**Buona fortuna! 🍀 Che vinca il miglior algoritmo! 🎯** 