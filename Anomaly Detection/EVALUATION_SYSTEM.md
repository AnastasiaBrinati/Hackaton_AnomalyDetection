# 🎯 Sistema di Valutazione Automatica SIAE Hackathon

## Panoramica

Questo sistema fornisce una valutazione automatica e una leaderboard in tempo reale per l'hackathon SIAE di anomaly detection. I partecipanti submittano file JSON standardizzati che vengono automaticamente valutati e classificati.

## 📁 Struttura File

```
├── submissions/                     # Cartella per i file di submission
│   ├── submission_example.json      # Esempio formato submission
│   └── submission_[TEAM_NAME].json  # File submission dei team
├── evaluate_submissions.py          # Script valutazione automatica
├── update_leaderboard.py           # Script aggiornamento leaderboard
├── leaderboard.md                   # Leaderboard live
└── Track1_Solution/                 # Soluzione di esempio
    └── track1_anomaly_detection.py  # Include generate_submission()
```

## 🚀 Setup per Organizzatori

### 1. Preparazione Iniziale

```bash
# Assicurati che tutti i file siano presenti
ls submissions/ evaluate_submissions.py update_leaderboard.py

# Testa il sistema di valutazione
python evaluate_submissions.py

# Verifica la leaderboard
cat leaderboard.md
```

### 2. Durante l'Hackathon

#### Modalità Automatica (Raccomandata)
```bash
# Monitora automaticamente nuovi commit ogni 30 secondi
python update_leaderboard.py --monitor
```

#### Modalità Manuale
```bash
# Aggiorna manualmente dopo ogni commit
python update_leaderboard.py
```

### 3. Verifica Sistema

```bash
# Genera submission di esempio
cd Track1_Solution
python track1_anomaly_detection.py

# Verifica che il file sia stato creato
ls ../submissions/submission_team_solution_example.json

# Aggiorna leaderboard
cd ..
python evaluate_submissions.py
```

## 👥 Istruzioni per i Partecipanti

### Per i Team

1. **Sviluppa** il modello di anomaly detection
2. **Modifica** `team_name` e `members` in `generate_submission()`
3. **Esegui** lo script per generare il file JSON
4. **Committa e pusha** il file nella cartella `submissions/`

### Esempio per i Partecipanti

```python
# Nel file track1_anomaly_detection.py
team_name = "Gli Algoritmi Vincenti"
members = ["Mario Rossi", "Laura Bianchi", "Giuseppe Verdi"]

submission_file, submission_data = generate_submission(
    df=df, 
    iso_forest=iso_forest, 
    feature_cols=feature_cols,
    team_name=team_name,
    members=members
)
```

```bash
# Comandi git per submission
git add submissions/submission_gli_algoritmi_vincenti.json
git commit -m "Gli Algoritmi Vincenti - Track 1 submission"
git push origin main
```

## 📊 Sistema di Scoring

### Componenti del Score (0-1 scale)

#### Technical Score (50%)
- **F1-Score** (25%): Performance principale
- **AUC-ROC** (15%): Robustezza del modello  
- **Precision** (10%): Accuratezza anomalie

#### Innovation Score (30%)
- **Feature Diversity** (10%): Numero e varietà features
- **Algorithm Complexity** (10%): Complessità tecnica
- **Feature Engineering** (10%): Features create

#### Business Score (20%)
- **Performance Efficiency** (10%): Tempo e memoria
- **Interpretability** (10%): Breakdown anomalie

### Formula Final Score
```
Final Score = (Technical × 0.5) + (Innovation × 0.3) + (Business × 0.2)
```

## 🔧 Configurazione Avanzata

### Personalizzazione Scoring

Modifica i pesi in `evaluate_submissions.py`:

```python
# In calculate_innovation_score()
feature_score = min(len(features) * 2, 40)  # Modifica punteggio per feature

# In calculate_business_score()
if train_time < 10:
    score += 20  # Modifica bonus per velocità
```

### Validazione Custom

Aggiungi validazioni in `validate_submission()`:

```python
# Esempio: verifica numero minimo features
if len(submission_data['model_info']['features_used']) < 5:
    errors.append("Minimum 5 features required")
```

## 📈 Monitoring e Logging

### Log delle Valutazioni

Il sistema stampa:
- ✅ Submission valide processate
- ❌ Errori di validazione  
- 🏆 Top 3 team
- 📊 Statistiche generali

### Notifiche

Estendi `send_notification()` per:
- Slack/Discord webhooks
- Email notifications
- Dashboard aggiornamenti

```python
def send_notification(team_name, score, rank):
    # Slack example
    import requests
    webhook_url = "YOUR_SLACK_WEBHOOK"
    message = f"🏆 New submission! {team_name} - Rank #{rank}"
    requests.post(webhook_url, json={"text": message})
```

## 🛠️ Troubleshooting

### Problemi Comuni

#### "No submission files found"
```bash
# Verifica cartella submissions
ls -la submissions/
# Dovrebbe contenere file submission_*.json
```

#### "Ground truth file not found"
```bash
# Verifica esistenza file di riferimento
ls Track1_Solution/live_events_with_anomalies.csv
# Se mancante, esegui prima la soluzione di esempio
```

#### "JSON format errors"
```bash
# Valida JSON file
python -m json.tool submissions/submission_team.json
# Dovrebbe restituire JSON formattato senza errori
```

### Reset Sistema

```bash
# Pulisci submissions per test
rm submissions/submission_*.json
# Mantieni solo submission_example.json

# Rigenera leaderboard vuota
python evaluate_submissions.py
```

## 📋 Checklist Pre-Hackathon

- [ ] ✅ Sistema di valutazione testato
- [ ] ✅ Leaderboard iniziale configurata  
- [ ] ✅ File di esempio funzionante
- [ ] ✅ Istruzioni per partecipanti chiare
- [ ] ✅ Monitoring automatico attivo
- [ ] ✅ Backup file importanti
- [ ] ✅ Permessi Git repository configurati

## 📋 Checklist Durante Hackathon

### Ogni Ora
- [ ] Verifica leaderboard aggiornata
- [ ] Controlla log errori
- [ ] Monitora performance sistema

### Ad Ogni Submission
- [ ] Valida formato JSON
- [ ] Verifica metriche ragionevoli
- [ ] Controlla posizione leaderboard

### Fine Giornata
- [ ] Backup submissions
- [ ] Report statistiche giornaliere
- [ ] Verifica integrità dati

## 🎯 Best Practices

1. **Comunicazione**: Informa i team su formato richiesto
2. **Monitoring**: Tieni attivo il monitoraggio automatico
3. **Backup**: Salva submissions regolarmente
4. **Trasparenza**: Leaderboard sempre visibile
5. **Support**: Help desk per problemi tecnici

## 📞 Support

Per problemi tecnici durante l'hackathon:
1. Verifica log degli errori
2. Controlla formato JSON submission
3. Testa manualmente valutazione
4. Contatta team tecnico se necessario 