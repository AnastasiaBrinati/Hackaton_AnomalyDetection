# Track 1: Anomaly Detection in Live Events (con FMA)

## Descrizione
Questo script implementa la Track 1 dell'hackathon SIAE, integrando i metadati FMA (Free Music Archive) per migliorare l'anomaly detection negli eventi live.

## 📊 Dataset
Il modello utilizza i dataset standardizzati dell'hackathon:
- **Training**: `../datasets/track1_live_events_train.csv` (40,000 eventi con ground truth)
- **Test**: `../datasets/track1_live_events_test.csv` (10,000 eventi senza ground truth)

### Dati degli Eventi
```python
event_id          # ID univoco evento
venue             # Nome del venue/locale
city              # Città dell'evento  
event_date        # Data e ora dell'evento
attendance        # Numero partecipanti effettivi
capacity          # Capacità massima del venue
n_songs           # Numero di brani eseguiti
total_revenue     # Ricavi totali dell'evento
is_anomaly        # 🎯 TARGET: 0=normale, 1=anomalo (solo training)
anomaly_type      # Tipo di anomalia specifica (solo training)
```

### 🚨 Anomalie da Rilevare
- **impossible_attendance**: Partecipanti > capacità venue
- **revenue_mismatch**: Ricavi impossibili per quel pubblico  
- **excessive_songs**: Troppi brani eseguiti (>40)
- **suspicious_timing**: Eventi in orari strani (2-6 AM)
- **duplicate_declaration**: Eventi dichiarati più volte

## 🚀 Come Usare

### STEP 1: Preparazione Dataset
```bash
# Dalla directory principale, genera i dataset (se non fatto già)
cd ..
python generate_datasets.py
cd Track1_Solution
```

### STEP 2: Personalizza Team
**🚨 IMPORTANTE**: Prima di eseguire, apri `track1_anomaly_detection.py` e modifica:

```python
# Cerca queste righe nel file e CAMBIA I VALORI:
team_name = "YourTeam"           # ← INSERISCI IL TUO NOME TEAM
members = ["Member1", "Member2"] # ← INSERISCI I MEMBRI DEL TEAM
```

### STEP 3: Esegui la pipeline
```bash
python track1_anomaly_detection.py
```

**Il sistema automaticamente**:
1. ✅ Carica dataset di training e test
2. ✅ Applica feature engineering avanzato
3. ✅ Addestra modello Isolation Forest
4. ✅ Genera predizioni sul test set
5. ✅ Crea visualizzazioni dettagliate
6. ✅ Salva file di submission JSON

### STEP 4: Verifica Risultati
```bash
# Controlla che sia stato generato il file di submission
ls ../submissions/submission_*_track1.json

# Dovresti vedere qualcosa come:
# submission_yourteam_track1.json
```

### STEP 5: Submit
```bash
git add ../submissions/submission_*.json
git commit -m "Team [TUO_NOME] - Track 1 submission"
git push origin main
```

## 🔧 Architettura del Modello

### Feature Engineering
Il modello crea **features avanzate** dai dati base:

**Features Base Derivate**:
```python
revenue_per_person = total_revenue / attendance
occupancy_rate = attendance / capacity  
songs_per_person = n_songs / attendance
avg_revenue_per_song = total_revenue / n_songs
```

**Features Temporali**:
```python
hour, day_of_week, month, is_weekend
```

**Features Categoriche**:
```python
venue_encoded, city_encoded  # Label encoding
```

**Indicatori di Anomalie**:
```python
is_over_capacity      # attendance > capacity
is_excessive_songs    # n_songs > 40
is_suspicious_timing  # ora 2-6 AM
is_low_revenue       # revenue_per_person < 5€
is_high_revenue      # revenue_per_person > 100€
```

**Statistiche per Venue**:
```python
venue_avg_attendance, venue_avg_revenue
attendance_vs_venue_avg, revenue_vs_venue_avg
```

### 🎯 Algoritmo: Approccio IBRIDO

Il modello usa una **strategia ibrida** che combina:

#### 1. **Regole Deterministiche** (Alta Precision)
```python
# ✅ Anomalie OVVIE con regole deterministiche:
attendance > capacity           # impossible_attendance  
n_songs > 40                   # excessive_songs
hour >= 2 AND hour <= 6        # suspicious_timing
stesso venue + stessa data     # duplicate_declaration
revenue_per_person < 1€        # extremely_low_revenue
```

**Vantaggi**: Precision = 100%, veloce, interpretabile

#### 2. **Isolation Forest** (Pattern Complessi)
```python
IsolationForest(
    contamination=0.08,    # 8% anomalie attese  
    n_estimators=200,      # 200 alberi per robustezza
    random_state=42        # Riproducibilità
)
```

**Vantaggi**: Rileva pattern multidimensionali sottili che le regole non catturano

#### 3. **Combinazione Ibrida**
```python
# Anomalia se:
has_rule_anomaly OR ml_prediction_anomaly

# Score combinato:
score = rule_score * 0.4 + ml_score * 0.6
```

**Perché questo approccio è migliore?**
- ✅ **Regole** catturano anomalie ovvie con 100% precision
- ✅ **ML** trova pattern complessi che sfuggono alle regole  
- ✅ **Combinazione** massimizza sia precision che recall
- ✅ **Interpretabile**: sappiamo sempre perché qualcosa è anomalo
- ✅ **Robusto**: non perde mai anomalie ovvie

## 📈 Output e Risultati

### File Generati
```bash
track1_results.png                    # Dashboard visualizzazioni
live_events_train_predictions.csv     # Predizioni su training
live_events_test_predictions.csv      # Predizioni su test
../submissions/submission_team_track1.json  # File di submission 
```

### Visualizzazioni
Il sistema genera **6 grafici informativi**:

1. **Anomaly Scores Training**: Distribuzione scores su training set
2. **Attendance vs Revenue Training**: Scatter plot eventi normali vs anomalie  
3. **Occupancy vs Revenue/Person Training**: Pattern di occupazione
4. **Anomaly Scores Test**: Distribuzione scores su test set
5. **Attendance vs Revenue Test**: Predizioni su test set
6. **Eventi per Città**: Distribuzione geografica anomalie

### Metriche di Performance
**Su Training Set** (dove abbiamo ground truth):
- Precision, Recall, F1-Score, AUC-ROC **REALI**
- Confusion Matrix per approccio ibrido
- Analisi per tipo di anomalia
- **Breakdown**: anomalie da regole vs ML

**Su Test Set** (per submission):
- Numero anomalie rilevate per categoria
- Breakdown: deterministiche vs ML
- Distribuzione scores combinati
- **Metriche REALI dal training** (usate nel file JSON)

**Output Dettagliato**:
```
📊 Risultati approccio ibrido:
   - Anomalie da regole deterministiche: 234
   - Anomalie da Isolation Forest: 187  
   - Anomalie totali (ibrido): 367
   - Overlap: 54
```

## 📄 Formato Submission

Il file JSON generato contiene:

```json
{
  "team_info": {
    "team_name": "YourTeam",
    "members": ["Member1", "Member2"],
    "track": "Track1"
  },
  "model_info": {
    "algorithm": "Isolation Forest with Feature Engineering",
    "features_used": [...],
    "hyperparameters": {...}
  },
  "results": {
    "total_test_samples": 10000,
    "anomalies_detected": 950,
    "predictions": [0,1,0,1,...],  // Array completo predizioni
    "scores": [-0.1,0.8,...]       // Array completo anomaly scores
  },
  "metrics": {
    "precision": 0.7234,    // ✅ REALE dal training set
    "recall": 0.6891,       // ✅ REALE dal training set  
    "f1_score": 0.7058,     // ✅ REALE dal training set
    "auc_roc": 0.8123       // ✅ REALE dal training set
  }
}
```

### 🎯 **Importante: Metriche Reali + Approccio Ibrido**
Il sistema ora implementa una **soluzione completa**:

#### ✅ **Risolve i Problemi Originali**:
- **Rileva duplicati**: controllo venue + data
- **Non solo matematica**: regole specifiche per ogni tipo di anomalia  
- **Combina regole + ML**: massimizza precision e recall
- **Metriche reali**: performance effettive dal training set

#### ✅ **Vantaggi dell'Approccio Ibrido**:
- **Regole deterministiche**: 100% precision su anomalie ovvie
- **Isolation Forest**: trova pattern multidimensionali complessi
- **Interpretabile**: sempre sappiamo perché qualcosa è anomalo  
- **Robusto**: non perde mai anomalie evidenti

#### ✅ **Quando Usare Cosa**:
- **Regole**: anomalie deterministiche (attendance > capacity)
- **ML**: pattern sottili (venue con pattern di revenue strani)
- **Ibrido**: combinazione per massimizzare F1-Score

## 🎛️ Personalizzazioni Avanzate

### Modificare Soglia Contamination
```python
# Nel train_isolation_forest(), cambia:
iso_forest = IsolationForest(contamination=0.05)  # Più strict
iso_forest = IsolationForest(contamination=0.12)  # Più permissivo
```

### Aggiungere Features Custom
```python
# Nel feature_engineering(), aggiungi:
df['custom_feature'] = df['total_revenue'] / df['capacity']
df['venue_efficiency'] = df['n_songs'] / df['capacity']

# Nel train_isolation_forest(), includi:
feature_cols.extend(['custom_feature', 'venue_efficiency'])
```

### Provare Altri Algoritmi
```python
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier

# Sostituisci Isolation Forest con:
model = OneClassSVM(nu=0.08)  # Support Vector Machine
# oppure usa ensemble
```

## 📊 Strategia di Ottimizzazione

### 1. Analizza i Risultati di Training
```python
# Dopo aver eseguito, controlla i tipi di anomalie meglio rilevati
anomaly_analysis = df_train[df_train['anomaly_type'].notna()].groupby('anomaly_type').agg({
    'is_anomaly_predicted': ['sum', 'count']
})
print(anomaly_analysis)
```

### 2. Ottimizza per Anomalie Specifiche
```python
# Crea features mirate per anomalie poco rilevate
df['revenue_capacity_ratio'] = df['total_revenue'] / df['capacity']
df['songs_attendance_ratio'] = df['n_songs'] / df['attendance']
```

### 3. Tuning Hyperparameters
```python
# Prova diversi valori:
contamination_values = [0.05, 0.08, 0.10, 0.12]
n_estimators_values = [100, 200, 300]

# Valuta con cross-validation sul training set
```

## ❗ Errori Comuni

### ❌ Non personalizzare team_name
```python
# SBAGLIATO - lascerà "YourTeam"
team_name = "YourTeam"  

# GIUSTO - inserisci il tuo nome
team_name = "TeamRossi"
```

### ❌ Dataset non trovati
```bash
# Se vedi errore "file non trovato", rigenera:
cd ..
python generate_datasets.py
cd Track1_Solution
```

### ❌ JSON malformato
```python
# Testa sempre il tuo JSON:
import json
with open('../submissions/submission_team_track1.json') as f:
    data = json.load(f)  # Deve funzionare senza errori
```

## 🏆 Tips per Vincere

### 1. **L'Approccio Ibrido è Vincente** 🎯
La combinazione regole + ML batte entrambi singolarmente:
- **Regole deterministiche**: garantiscono 100% precision su anomalie ovvie
- **Isolation Forest**: trova pattern complessi che sfuggono alle regole
- **Risultato**: massimi precision E recall

### 2. **Focus su F1-Score**
Il sistema valuta principalmente su F1-Score:
```python
# L'approccio ibrido bilancia automaticamente:
precision ↑  (regole deterministiche)
recall ↑     (ML trova pattern sottili)
→ F1-Score ↑
```

### 3. **Ottimizza i Pesi della Combinazione**
```python
# Esperimenta con diversi pesi:
score = rule_score * 0.3 + ml_score * 0.7  # Più peso al ML
score = rule_score * 0.5 + ml_score * 0.5  # Pesi uguali
```

### 4. **Aggiungi Regole Specifiche**
```python
# Crea regole per anomalie che ML sbaglia:
df['suspicious_revenue_pattern'] = (
    (df['revenue_per_person'] > 150) & 
    (df['n_songs'] < 5)
)
```

### 5. **Feature Engineering Avanzato**
```python
# Features che aiutano l'ML a trovare pattern sottili:
df['revenue_zscore_for_venue'] = ...
df['efficiency_ratio'] = df['total_revenue'] / (df['capacity'] * df['n_songs'])
```

---

**🎯 Buona fortuna! Che vinca il miglior algoritmo! 🚀**

---

*Ultimo aggiornamento: Luglio 2025* 