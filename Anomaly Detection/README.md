# 🎵 SIAE Anomaly Detection Hackathon
## Rilevamento Anomalie nei Diritti d'Autore e Utilizzi Musicali

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Tracks](https://img.shields.io/badge/Tracks-4-purple.svg)
![Duration](https://img.shields.io/badge/Duration-2%20days-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-95K%20samples-brightgreen.svg)
![Level](https://img.shields.io/badge/Level-Intermediate%20to%20Expert-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Participants](https://img.shields.io/badge/Teams-Unlimited-blue.svg)
![Evaluation](https://img.shields.io/badge/Evaluation-Automated-lightblue.svg)
![Leaderboard](https://img.shields.io/badge/Leaderboard-Real%20Time-ff69b4.svg)

### 📅 Durata: 2 giorni (14 ore totali)
- **Giorno 1**: 7 ore (9:30-13:00, 14:00-17:00)
- **Giorno 2**: 7 ore (9:30-13:00, 14:00-17:00)

---

## 🎯 In cosa consiste l'Hackathon

### 🎪 Obiettivo Principale
Sviluppare **sistemi di anomaly detection** per identificare comportamenti sospetti nei diritti d'autore musicali. I partecipanti devono creare algoritmi che rilevano automaticamente frodi, utilizzi anomali e pattern irregolari nei dati SIAE.

### 🏆 Come Funziona
1. **Scegli una Track** (o partecipa a multiple track)
2. **Esegui lo script** per generare dataset e modello di esempio
3. **Personalizza l'algoritmo** per migliorare le performance
4. **Submit the results** attraverso file JSON automatici
5. **Compete in the leaderboard** in tempo reale 

### 📊 Cosa Devi Calcolare
Per ogni track devi implementare un modello che:
- **Identifica anomalie** nei dati (binary classification: normale/anomalo)
- **Calcola metriche** di performance: F1-Score, Precision, Recall, AUC-ROC
- **Genera predizioni** su dataset di test
- **Produce visualization** dei risultati

### 📤 Come Submittare
1. **Modifica** il `team_name` e `members` nel codice
2. **Esegui** lo script della track scelta
3. **Verifica** che venga generato il file `submissions/submission_[team]_[track].json`
4. **Commit e push** - la leaderboard si aggiorna automaticamente!

### 🥇 Vincere l'Hackathon
- **Overall Winner**: Miglior score tra tutte le track
- **Track Winner**: Miglior score per singola track
- **Innovation Award**: Approccio più creativo

---

## 🚀 Quick Start Guide

### 📥 Setup Rapido (5 minuti)

```bash
# 1. Clona il repository
git clone <repository-url>
cd "Anomaly Detection"

# 2. Installa dipendenze base (opzionale per vedere esempi)
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. IMPORTANTE: Genera i dataset identici per tutti
python generate_datasets.py

# 4. Esegui una track di esempio
cd Track1_Solution
python track1_anomaly_detection.py
```

### 📊 Dataset Identici per Tutti i Partecipanti

**🎯 IMPORTANTE**: Per garantire performance comparabili, tutti i partecipanti devono usare gli stessi dataset.

#### 🚀 **Generazione Dataset Centralizzata**

```bash
# PASSO OBBLIGATORIO: Genera i dataset identici per tutti
python generate_datasets.py

# Questo comando crea:
# - datasets/track1_live_events.csv (50,000 eventi)
# - datasets/track2_documents.csv (5,000 documenti)  
# - datasets/track3_music.csv (25,000 tracce)
# - datasets/track4_copyright.csv (15,000 opere)
```

#### 🎪 **Track 1: Live Events Anomaly Detection**
![Isolation Forest](https://img.shields.io/badge/Algorithm-Isolation%20Forest-orange.svg)
![DBSCAN](https://img.shields.io/badge/Clustering-DBSCAN-blue.svg)
![Dataset Size](https://img.shields.io/badge/Dataset-50K%20events-green.svg)
![Execution Time](https://img.shields.io/badge/Runtime-2%20min-red.svg)

```bash
# Usa il dataset generato centralmente
df = pd.read_csv('datasets/track1_live_events.csv')
# ✅ 50,000 eventi live identici per tutti
# ✅ 5 tipi di anomalie: duplicate_declaration, impossible_attendance, revenue_mismatch, excessive_songs, suspicious_timing
# ⏱️ Tempo esecuzione: ~2 minuti
```

#### 📄 **Track 2: Document Fraud Detection**
![Computer Vision](https://img.shields.io/badge/Tech-Computer%20Vision-purple.svg)
![OCR](https://img.shields.io/badge/OCR-Tesseract-yellow.svg)
![Dataset Size](https://img.shields.io/badge/Dataset-5K%20documents-green.svg)
![Execution Time](https://img.shields.io/badge/Runtime-1%20min-red.svg)

```bash
# Usa il dataset generato centralmente
df = pd.read_csv('datasets/track2_documents.csv')
# ✅ 5,000 documenti SIAE identici per tutti
# ✅ 5 tipi di frodi: digital_alteration, signature_forgery, template_fraud, metadata_manipulation, quality_inconsistency
# ⏱️ Tempo esecuzione: ~1 minuto
```

#### 🎵 **Track 3: Music Anomaly Detection**
![FMA Dataset](https://img.shields.io/badge/Dataset-FMA%20Real-brightgreen.svg)
![Audio Features](https://img.shields.io/badge/Features-MFCC%20%2B%20Spectral-blue.svg)
![Dataset Size](https://img.shields.io/badge/Dataset-25K%20tracks-green.svg)
![Execution Time](https://img.shields.io/badge/Runtime-3%20min-red.svg)

```bash
# Usa il dataset generato centralmente
df = pd.read_csv('datasets/track3_music.csv')
# ✅ 25,000 tracce musicali identiche per tutti
# ✅ 5 tipi di anomalie: plagio_similarity, bot_streaming, metadata_manipulation, genre_mismatch, audio_quality_fraud
# ⏱️ Tempo esecuzione: ~3 minuti
```

#### 🔒 **Track 4: Copyright Infringement Detection**
![Guaranteed Clustering](https://img.shields.io/badge/Clustering-Guaranteed-brightgreen.svg)
![Advanced Features](https://img.shields.io/badge/Features-40%2B%20Dimensions-orange.svg)
![Dataset Size](https://img.shields.io/badge/Dataset-15K%20works-green.svg)
![Execution Time](https://img.shields.io/badge/Runtime-2%20min-red.svg)

```bash
# Usa il dataset generato centralmente
df = pd.read_csv('datasets/track4_copyright.csv')
# ✅ 15,000 opere creative identiche per tutti
# ✅ 5 tipi di violazioni: unauthorized_sampling, derivative_work, metadata_manipulation, cross_platform_violation, content_id_manipulation
# ⏱️ Tempo esecuzione: ~2 minuti
```

### 📈 Risultati Automatici

Ogni script produce automaticamente:
- **📊 Visualizzazioni** salvate come PNG
- **📋 Dataset CSV** con risultati completi  
- **📄 File submission JSON** per la leaderboard
- **📈 Metriche di performance** stampate a console

### 🏆 Submission alla Leaderboard

```bash
# Valuta tutte le submission
python evaluate_submissions.py

# Aggiorna leaderboard automaticamente
python update_leaderboard.py
```

### 🛠️ Tech Stack Completo

#### Core Technologies
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg?logo=scikit-learn)

#### Machine Learning & Analysis
![Isolation Forest](https://img.shields.io/badge/Isolation%20Forest-Anomaly%20Detection-red.svg)
![DBSCAN](https://img.shields.io/badge/DBSCAN-Clustering-blue.svg)
![PCA](https://img.shields.io/badge/PCA-Dimensionality%20Reduction-purple.svg)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-darkgreen.svg)

#### Visualization & Analysis
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-darkblue.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-brightgreen.svg?logo=plotly)

#### Audio & Computer Vision
![Audio Processing](https://img.shields.io/badge/Audio-MFCC%20%2B%20Spectral-yellow.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OCR%20%2B%20Features-purple.svg)
![Feature Engineering](https://img.shields.io/badge/Features-40%2B%20Dimensions-orange.svg)

### 📈 Performance Metrics

![Accuracy](https://img.shields.io/badge/Accuracy-85%25%2B-brightgreen.svg)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.80%2B-green.svg)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.85%2B-blue.svg)
![Precision](https://img.shields.io/badge/Precision-80%25%2B-lightgreen.svg)
![Recall](https://img.shields.io/badge/Recall-75%25%2B-yellow.svg)
![Processing Speed](https://img.shields.io/badge/Processing-1--3%20min-red.svg)

---

## 🎯 Obiettivo della Challenge

I partecipanti dovranno sviluppare sistemi di anomaly detection per identificare:
1. **Utilizzi anomali di opere musicali** (locali/eventi con pattern sospetti)
2. **Anomalie nelle dichiarazioni dei diritti** (ripartizioni insolite, duplicazioni)
3. **Pattern fraudolenti nelle riproduzioni** (streaming/radio con comportamenti anomali)
4. **Irregolarità documentali** (contratti o documenti con elementi sospetti)

---

## 📊 Dataset per Track - Dettagli Tecnici

### 🎪 Track 1: Live Events Anomaly Detection
**Come Ottenere i Dati**:
```bash
cd Track1_Solution
python track1_anomaly_detection.py
```

**Dataset Generati Automaticamente**:
- ✅ **50,000 eventi live sintetici** con 5 tipi di anomalie:
  - `duplicate_declaration` - Dichiarazioni duplicate
  - `impossible_attendance` - Partecipazione impossibile (>capacità)
  - `revenue_mismatch` - Revenue non coerente con audience
  - `excessive_songs` - Numero eccessivo di brani (50-100)
  - `suspicious_timing` - Orari sospetti (4:00 AM)

- ✅ **Metadati FMA** (opzionale):
  - Scaricati automaticamente da: `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`
  - Se download fallisce → dataset FMA sintetico generato (10,000 tracce)
  - Include generi, artisti, durate, popolarità

**Output**: `live_events_with_anomalies.csv` (50,000 righe)

### 📄 Track 2: Document Fraud Detection  
**Come Ottenere i Dati**:
```bash
cd Track2_Solution
python track2_document_fraud_detection.py
```

**Dataset Generati Automaticamente**:
- ✅ **5,000 documenti SIAE sintetici** con 5 tipi di frodi:
  - `digital_alteration` - Alterazioni digitali (noise elevato)
  - `signature_forgery` - Firme contraffatte (signature_regions = 0)
  - `template_fraud` - Template fraudolenti (watermark SIAE mancanti)
  - `metadata_manipulation` - Manipolazione metadati
  - `quality_inconsistency` - Qualità audio inconsistente

- ✅ **Features estratte automaticamente**:
  - Risoluzione, dimensioni file, confidence OCR
  - Regioni di firma, watermark, seal detection
  - Noise level, edge sharpness, consistency scores

**Output**: `documents_fraud_detection.csv` (5,000 righe)

### 🎵 Track 3: Music Anomaly Detection
**Come Ottenere i Dati**:
```bash
cd Track3_Solution
python track3_music.py
```

**Dataset Utilizzati**:
- ✅ **FMA (Free Music Archive) - Priorità 1**:
  - URL: `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip` (342MB)
  - Contenuto: 106,574 tracce reali, 16,341 artisti
  - Features: generi, date, durate, popolarità, coordinate artisti

- ✅ **Dataset FMA Sintetico - Fallback automatico**:
  - 25,000 tracce simulate se download FMA fallisce
  - Mantiene struttura e distribuzioni realistiche
  - Genera automaticamente 5 tipi di anomalie musicali

**Anomalie Musicali Simulate**:
- `plagio_similarity` - Similarità sospetta in features audio
- `bot_streaming` - Pattern innaturali di ascolto (like/play ratio anomalo)
- `metadata_manipulation` - Date future o inconsistenti
- `genre_mismatch` - Genere non corrispondente a features
- `audio_quality_fraud` - Qualità dichiarata vs dimensione file

**Output**: `music_anomaly_detection_results.csv` (25,000 righe)

### 🔒 Track 4: Copyright Infringement Detection
**Come Ottenere i Dati**:
```bash
cd Track4_Solution
python track4_copyright_infringement.py
```

**Dataset Generati Automaticamente**:
- ✅ **15,000 opere creative sintetiche** con 5 tipi di violazioni:
  - `unauthorized_sampling` - Campionamento non autorizzato (diviso in 3 cluster per tempo)
  - `derivative_work` - Opere derivate (divise per engagement alto/basso)
  - `metadata_manipulation` - Manipolazione metadati copyright
  - `cross_platform_violation` - Violazioni multi-piattaforma
  - `content_id_manipulation` - Elusione Content ID

- ✅ **Features Avanzate**:
  - Audio: tempo, tonalità, MFCC, spettro, chroma
  - Engagement: play/like/share counts, viral coefficient
  - Business: revenue, royalty rates, licensing
  - Technical: hash, fingerprint, compression, quality
  - Platform: distribuzione multi-piattaforma, geolocalizzazione

**Clustering Garantito**: Sistema ottimizzato che produce sempre 5-8 cluster visibili

**Output**: `copyright_infringement_detection_results.csv` (15,000 righe)

### 🔧 Fallback per Connessioni Limitate

Tutti i sistemi includono **fallback automatici**:
- Se download esterni falliscono → generazione sintetica locale
- Se librerie mancanti → versioni semplificate
- Se memoria limitata → dataset ridotti automaticamente
- **Nessun intervento manuale richiesto**

### 📈 Monitoraggio Download

```bash
# Verifica status download FMA
python -c "
import requests
try:
    r = requests.head('https://os.unil.cloud.switch.ch/fma/fma_metadata.zip', timeout=5)
    print(f'✅ FMA disponibile ({r.headers.get(\"content-length\", \"sconosciuto\")} bytes)')
except:
    print('❌ FMA non disponibile - useremo dataset sintetico')
"
```

---

## 📤 Guida Completa alla Submission

### 🎯 Passo 1: Preparazione del Team

**Prima di iniziare**:
1. **Forma il team** (massimo 4 persone)
2. **Scegli un nome team** (es: "DataDetectives", "AnomalyHunters")
3. **Decide la strategia**: una track o multi-track?

### 🚀 Passo 2: Esegui e Personalizza

Per ogni track che vuoi affrontare:

```bash
# Track 1: Live Events
cd Track1_Solution
# IMPORTANTE: Modifica questi parametri nel file prima di eseguire
# team_name = "Il Tuo Nome Team"
# members = ["Nome1", "Nome2", "Nome3"]
python track1_anomaly_detection.py

# Track 2: Document Fraud
cd Track2_Solution
# IMPORTANTE: Modifica team_name e members
python track2_document_fraud_detection.py

# Track 3: Music Anomaly
cd Track3_Solution
# IMPORTANTE: Modifica team_name e members
python track3_music.py

# Track 4: Copyright Infringement
cd Track4_Solution
# IMPORTANTE: Modifica team_name e members
python track4_copyright_infringement.py
```

### 📊 Passo 3: Verifica i Risultati

Ogni script genera automaticamente:
- **📋 Dataset CSV** con risultati completi
- **📄 File JSON** per submission nella cartella `submissions/`
- **📈 Grafici** di visualizzazione (salvati come PNG)
- **📊 Metriche** stampate a console

**Verifica che sia stato creato**:
```bash
# Controlla che i file siano stati generati
ls submissions/submission_*
# Dovresti vedere file come:
# submission_il_tuo_team_track1.json
# submission_il_tuo_team_track2.json
# etc.
```

### 🔧 Passo 4: Personalizza l'Algoritmo (Opzionale)

**Per migliorare il tuo score**:
1. **Modifica i parametri** dell'Isolation Forest:
   ```python
   iso_forest = IsolationForest(
       contamination=0.08,  # Prova 0.05-0.15
       n_estimators=200,    # Prova 100-500
       random_state=42
   )
   ```

2. **Aggiungi nuove features**:
   ```python
   # Esempio per Track 1
   df['custom_feature'] = df['attendance'] / df['capacity']
   df['revenue_efficiency'] = df['total_revenue'] / df['attendance']
   ```

3. **Cambia l'algoritmo**:
   ```python
   # Prova altri algoritmi
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import OneClassSVM
   ```

### 📤 Passo 5: Submission Finale

**Commit i tuoi risultati**:
```bash
# Aggiungi i file di submission
git add submissions/submission_il_tuo_team_*.json

# Commit con messaggio descrittivo
git commit -m "Il Tuo Team - Submission Track 1,2,3,4"

# Push per aggiornare la leaderboard
git push origin main
```

**🎉 FATTO!** La leaderboard si aggiorna automaticamente in pochi secondi.

### 🏆 Passo 6: Monitora la Leaderboard

**Controlla la tua posizione**:
- Apri `leaderboard.md` per vedere le classifiche
- La leaderboard mostra:
  - **Overall Ranking** (tutti i team)
  - **Track-specific Rankings**
  - **Detailed Scores** con breakdown

### 📋 Cosa Viene Valutato

**Il sistema calcola automaticamente**:

#### 📊 Technical Score (50%)
- **F1-Score** (25%): Quanto bene rilevi le anomalie
- **AUC-ROC** (15%): Robustezza del modello
- **Precision** (10%): Accuratezza delle tue predizioni

#### 🧠 Innovation Score (30%)
- **Numero Features** (10%): Varietà di features utilizzate
- **Complessità Algoritmo** (10%): Uso di tecniche avanzate
- **Feature Engineering** (10%): Features creative/derivate

#### 💼 Business Score (20%)
- **Performance** (10%): Velocità di esecuzione
- **Interpretabilità** (10%): Capacità di spiegare le anomalie

### 🔄 Regole di Submission

- **Maximum 5 submission** per team per giorno
- **L'ultima submission** conta per la valutazione finale
- **File size limit**: 50MB per submission
- **Formato obbligatorio**: JSON come negli esempi

### 🎯 Strategia Vincente

1. **Multi-track approach**: Partecipa a più track per massimizzare le opportunità
2. **Inizia semplice**: Usa la baseline e poi migliora
3. **Focus su F1-Score**: È il 25% del punteggio totale
4. **Feature engineering**: Crea features creative per l'innovation score
5. **Iterazione rapida**: Fai submission frequenti per testare miglioramenti

### 🚨 Troubleshooting

**Errore: "Missing required field"**
- Controlla che il JSON contenga tutti i campi richiesti
- Verifica che `team_name` e `members` siano stati modificati

**Errore: "Invalid JSON format"**
- Testa il JSON: `python -m json.tool submissions/submission_tuo_team.json`

**Submission non appare in leaderboard**
- Verifica che il file sia stato committato
- Controlla il nome file: deve essere `submission_[team]_[track].json`
- Assicurati che il campo `track` sia corretto ("Track1", "Track2", etc.)

### 📈 Parametri Specifici da Calcolare per Track

#### 🎪 Track 1: Live Events Anomaly Detection
**Devi calcolare**:
- **Predizioni binarie**: 0 (normale) o 1 (anomalo) per ogni evento
- **Anomaly scores**: Punteggio di anomalia (-1 a +1)
- **Metriche**: Precision, Recall, F1-Score, AUC-ROC
- **Contatori**: Total events, anomalies detected

**Anomalie da rilevare**:
- Eventi con attendance > capacity
- Revenue/attendance ratio anomalo
- Dichiarazioni duplicate (stesso venue+data)
- Numero eccessivo di brani (>40)
- Timing sospetti (eventi notturni 2-6 AM)

#### 📄 Track 2: Document Fraud Detection
**Devi calcolare**:
- **Predizioni binarie**: 0 (autentico) o 1 (fraudolento) per ogni documento
- **Fraud scores**: Punteggio di frode (-1 a +1)
- **Metriche**: Precision, Recall, F1-Score, AUC-ROC
- **Contatori**: Total documents, frauds detected

**Frodi da rilevare**:
- Alterazioni digitali (noise anomalo)
- Firme contraffatte/mancanti
- Template fraudolenti (watermark SIAE mancanti)
- Manipolazione metadati
- Inconsistenze qualità/formato

#### 🎵 Track 3: Music Anomaly Detection
**Devi calcolare**:
- **Predizioni binarie**: 0 (normale) o 1 (anomalo) per ogni traccia
- **Anomaly scores**: Punteggio di anomalia (-1 a +1)
- **Metriche**: Precision, Recall, F1-Score, AUC-ROC
- **Contatori**: Total tracks, anomalies detected

**Anomalie da rilevare**:
- Plagio (similarità elevata tra tracce)
- Bot streaming (like/play ratio innaturale)
- Manipolazione metadati (date future/inconsistenti)
- Genre mismatch (audio features vs genere)
- Audio quality fraud (qualità vs dimensione)

#### 🔒 Track 4: Copyright Infringement Detection
**Devi calcolare**:
- **Predizioni binarie**: 0 (legale) o 1 (violazione) per ogni opera
- **Infringement scores**: Punteggio di violazione (-1 a +1)
- **Metriche**: Precision, Recall, F1-Score, AUC-ROC
- **Contatori**: Total works, infringements detected

**Violazioni da rilevare**:
- Campionamento non autorizzato
- Opere derivate non autorizzate
- Manipolazione metadati copyright
- Violazioni cross-platform
- Elusione Content ID

### 🎯 Riassunto: Come Partecipare in 5 Passi

1. **📝 Prepara il Team**: Scegli nome e membri (max 4 persone)

2. **🗂️ Genera i Dataset Identici**: 
   ```bash
   python generate_datasets.py
   # OBBLIGATORIO: Crea dataset identici per tutti i partecipanti
   ```

3. **🚀 Esegui gli Script**: 
   ```bash
   cd Track1_Solution && python track1_anomaly_detection.py
   # Modifica prima team_name e members nel file!
   ```

4. **📊 Verifica i Risultati**: 
   ```bash
   ls submissions/submission_*
   # Controlla che i file JSON siano stati generati
   ```

5. **📤 Fai la Submission**: 
   ```bash
   git add submissions/submission_tuo_team_*.json
   git commit -m "Team Submission"
   git push origin main
   ```

6. **🏆 Monitora la Leaderboard**: 
   ```bash
   # Apri leaderboard.md per vedere la tua posizione
   ```

**🎉 È tutto! I dataset sono identici per tutti, le performance sono comparabili!**

### 🎯 Perché Dataset Identici?

#### ✅ **Vantaggi per i Partecipanti**
- **Performance comparabili**: Tutti i team lavorano sugli stessi dati
- **Fairness garantita**: Nessun vantaggio dovuto a differenze nei dataset
- **Benchmark affidabile**: Le tue metriche sono direttamente confrontabili
- **Riproducibilità**: Seed fisso (42) garantisce risultati identici

#### 🔧 **Vantaggi Tecnici**
- **Stesso numero di anomalie**: Stesse percentuali per tutti
- **Stesse distribuzioni**: Feature identiche tra partecipanti
- **Clustering garantito**: I cluster sono sempre visibili
- **Metriche standardizzate**: F1-Score, AUC-ROC, Precision confrontabili

#### 📊 **Caratteristiche Dataset**
- **Track 1**: 50,000 eventi, 5% anomalie, 5 tipologie
- **Track 2**: 5,000 documenti, 12% frodi, 5 tipologie
- **Track 3**: 25,000 tracce, 8% anomalie, 5 tipologie  
- **Track 4**: 15,000 opere, 7% violazioni, 5 tipologie

#### 💡 **Come Usare nei Tuoi Script**
```python
import pandas as pd

# Carica il dataset della track che vuoi affrontare
df_track1 = pd.read_csv('datasets/track1_live_events.csv')
df_track2 = pd.read_csv('datasets/track2_documents.csv')
df_track3 = pd.read_csv('datasets/track3_music.csv')
df_track4 = pd.read_csv('datasets/track4_copyright.csv')

# Ora puoi sviluppare il tuo modello sui dati identici per tutti!
```

---

## 🛠️ Setup Ambiente di Sviluppo

### Requisiti Base
```bash
# Creare ambiente virtuale
python -m venv hackathon_env
source hackathon_env/bin/activate  # Linux/Mac
# hackathon_env\Scripts\activate  # Windows

# Installare dipendenze base
pip install pandas numpy matplotlib seaborn scikit-learn
pip install jupyter notebook
pip install pyod  # Python Outlier Detection
pip install networkx  # Per analisi grafi
```

### Librerie Specifiche per Task
```bash
# Per analisi testuale
pip install nltk spacy transformers

# Per analisi immagini/documenti
pip install opencv-python pillow pytesseract

# Per serie temporali
pip install statsmodels prophet

# Per deep learning (opzionale)
pip install torch torchvision tensorflow
```

---

## 📋 Challenge Tracks

### 🎪 Track 1: Live Events Anomaly Detection (Livello: Intermedio)
**Soluzione Completa**: `Track1_Solution/track1_anomaly_detection.py`  
**Dataset**: Eventi live sintetici (50,000) + Metadati FMA (opzionale)

**Obiettivi**:
- ✅ Identificare eventi con attendance impossibile (>capacità venue)
- ✅ Rilevare revenue/attendance mismatch (guadagni anomali)
- ✅ Trovare dichiarazioni duplicate (stesso venue/data)
- ✅ Individuare timing sospetti (eventi notturni anomali)
- ✅ Rilevare numero eccessivo di brani eseguiti

**Tecniche implementate**:
- **Isolation Forest** per anomalie multivariate (contamination=0.1)
- **DBSCAN** per clustering venue con pattern simili
- **Feature engineering** avanzato (30+ features)
- **Visualizzazioni** complete con 6 grafici di analisi

### 📄 Track 2: Document Fraud Detection (Livello: Avanzato)  
**Soluzione Completa**: `Track2_Solution/track2_document_fraud_detection.py`  
**Dataset**: Documenti SIAE sintetici (5,000) con features di computer vision

**Obiettivi**:
- ✅ Rilevare alterazioni digitali (pixel noise anomalo)
- ✅ Identificare firme contraffatte o mancanti
- ✅ Trovare template fraudolenti (watermark SIAE mancanti)
- ✅ Individuare manipolazione metadati documento
- ✅ Rilevare inconsistenze qualità/formato

**Tecniche implementate**:
- **Isolation Forest** ottimizzato per fraud detection (contamination=0.12)
- **Computer Vision features**: noise level, edge sharpness, OCR confidence
- **Metadata analysis**: consistency scoring, temporal analysis
- **DBSCAN clustering** per pattern di frode simili

### 🎵 Track 3: Music Anomaly Detection (Livello: Esperto)
**Soluzione Completa**: `Track3_Solution/track3_music.py`  
**Dataset**: FMA reale (106K tracce) o sintetico (25K) con fallback automatico

**Obiettivi**:
- ✅ Rilevare plagio tramite similarità features audio (MFCC, tempo, chiave)
- ✅ Identificare bot streaming (pattern like/play innaturali)
- ✅ Trovare manipolazione metadati (date future, inconsistenze)
- ✅ Individuare genre mismatch (features audio vs genere dichiarato)
- ✅ Rilevare audio quality fraud (qualità vs dimensione file)

**Tecniche implementate**:
- **Isolation Forest** con 25+ features audio avanzate
- **PCA** per riduzione dimensionalità e clustering
- **FMA integration** con download automatico e fallback sintetico
- **Advanced audio features**: spectral analysis, chroma, MFCC

### 🔒 Track 4: Copyright Infringement Detection (Livello: Esperto)
**Soluzione Completa**: `Track4_Solution/track4_copyright_infringement.py`  
**Dataset**: Opere creative sintetiche (15,000) con clustering garantito

**Obiettivi**:
- ✅ Rilevare campionamento non autorizzato (similarità audio pattern)
- ✅ Identificare opere derivate (modifiche minori a opere esistenti)
- ✅ Trovare manipolazione metadati copyright (falsificazione info)
- ✅ Individuare violazioni cross-platform (distribuzione non autorizzata)
- ✅ Rilevare elusione Content ID (alterazioni per bypassare filtri)

**Tecniche implementate**:
- **Isolation Forest** con 40+ features multidimensionali
- **Clustering GARANTITO** rule-based (sempre 5-8 cluster visibili)
- **Multi-platform analysis**: engagement, revenue, fingerprinting
- **Advanced features**: audio complexity, viral coefficient, compression analysis

### 🚀 Tutte le Track includono:
- **🔄 Esecuzione automatica** (1-3 minuti per track)
- **📊 Visualizzazioni complete** (6 grafici per track)
- **📋 Dataset CSV** con risultati completi
- **📄 Submission JSON** automatica per leaderboard
- **📈 Metriche complete**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **🎯 Ground truth** per valutazione oggettiva

---

## 📅 Timeline Dettagliata

### Giorno 1
**9:00-10:00**: Kickoff e Presentazione
- Introduzione all'anomaly detection
- Presentazione dataset e challenge
- Formazione team (3-4 persone)
- Q&A tecnico

**10:00-13:00**: Esplorazione e Preparazione
- Download e setup dataset
- Analisi esplorativa (EDA)
- Definizione strategia
- Feature engineering iniziale

**14:00-17:00**: Sviluppo Modelli Base
- Implementazione primi modelli
- Baseline performance
- Iterazione e miglioramento
- Version control setup

### Giorno 2
**9:00-12:00**: Ottimizzazione e Advanced Techniques
- Ensemble methods
- Hyperparameter tuning
- Cross-validation
- Gestione sbilanciamento classi

**12:00-13:00**: Preparazione Presentazione
- Creazione slides
- Preparazione demo
- Documentazione risultati

**14:00-16:00**: Presentazioni Team
- 10 minuti per team
- Demo live
- Q&A

**16:00-17:00**: Premiazione e Wrap-up
- Valutazione giurie
- Premiazione vincitori
- Networking e feedback

---

## 📊 Sistema di Valutazione Multi-Track

### 🎯 Track Disponibili

#### Track 1: Live Events Anomaly Detection
- **Focus**: Rilevamento anomalie in eventi musicali live
- **Dataset**: Eventi live sintetici + metadati FMA
- **Script**: `Track1_Solution/track1_anomaly_detection.py`
- **Submission**: `submissions/submission_[team]_track1.json`

#### Track 2: Document Fraud Detection  
- **Focus**: Rilevamento frodi in documenti SIAE
- **Dataset**: Documenti digitali sintetici (contratti, licenze)
- **Script**: `Track2_Solution/track2_document_fraud_detection.py`
- **Submission**: `submissions/submission_[team]_track2.json`

#### Track 3: Music Anomaly Detection
- **Focus**: Rilevamento anomalie in tracce musicali FMA
- **Dataset**: FMA (Free Music Archive) con 25,000+ tracce
- **Script**: `Track3_Solution/track3_music_anomaly_detection.py`
- **Submission**: `submissions/submission_[team]_track3.json`

### 🚀 Partecipazione Multi-Track

Un team può partecipare a **uno o più track** simultaneamente:

```bash
# Solo Track 1 (Live Events)
cd Track1_Solution && python track1_anomaly_detection.py

# Solo Track 2 (Document Fraud)  
cd Track2_Solution && python track2_document_fraud_detection.py

# Solo Track 3 (Music Anomaly)
cd Track3_Solution && python track3_music.py

# Solo Track 4 (Copyright Infringement)
cd Track4_Solution && python track4_copyright_infringement.py

# Tutti i track (consigliato per massimizzare opportunità)
cd Track1_Solution && python track1_anomaly_detection.py
cd ../Track2_Solution && python track2_document_fraud_detection.py
cd ../Track3_Solution && python track3_music.py
cd ../Track4_Solution && python track4_copyright_infringement.py
```

### Come Funziona la Valutazione Automatica

I partecipanti devono generare un **file di submission** che verrà automaticamente valutato dal sistema. La leaderboard si aggiorna in tempo reale ad ogni submission per tutti i track.

### Formato File di Submission Multi-Track

Ogni team deve generare file JSON separati per ogni track:

#### Track 1: `submission_[TEAM_NAME]_track1.json`
```json
{
  "team_info": {
    "team_name": "Nome del Team",
    "members": ["Nome1", "Nome2", "Nome3"],
    "track": "Track1",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest + DBSCAN",
    "features_used": ["attendance", "revenue_per_person", "genre_encoded"],
    "hyperparameters": {
      "contamination": 0.1,
      "n_estimators": 100
    }
  },
  "results": {
    "total_events": 10000,
    "anomalies_detected": 950,
    "predictions": [0, 1, 0, 1, 0],
    "anomaly_scores": [-0.1, 0.8, -0.2, 0.7]
  },
  "metrics": {
    "precision": 0.85,
    "recall": 0.78,
    "f1_score": 0.81,
    "auc_roc": 0.89
  }
}
```

#### Track 2: `submission_[TEAM_NAME]_track2.json`
```json
{
  "team_info": {
    "team_name": "Nome del Team",
    "members": ["Nome1", "Nome2", "Nome3"],
    "track": "Track2",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest + Computer Vision Features",
    "features_used": ["text_confidence", "visual_integrity", "siae_authenticity"],
    "hyperparameters": {
      "contamination": 0.12,
      "n_estimators": 200
    }
  },
  "results": {
    "total_documents": 5000,
    "frauds_detected": 750,
    "predictions": [0, 1, 0, 1, 0],
    "fraud_scores": [-0.15, 0.92, -0.28, 0.88]
  },
  "metrics": {
    "precision": 0.82,
    "recall": 0.76,
    "f1_score": 0.79,
    "auc_roc": 0.87
  },
  "track2_specific": {
    "document_types_analyzed": 6,
    "avg_text_confidence": 0.847,
    "siae_watermark_detection_rate": 0.823
  }
}
```

#### Track 3: `submission_[TEAM_NAME]_track3.json`
```json
{
  "team_info": {
    "team_name": "Nome del Team",
    "members": ["Nome1", "Nome2", "Nome3"],
    "track": "Track3",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest + Advanced Music Features",
    "features_used": ["audio_complexity", "listens_vs_artist_avg", "genre_encoded"],
    "hyperparameters": {
      "contamination": 0.08,
      "n_estimators": 200
    }
  },
  "results": {
    "total_tracks": 25000,
    "anomalies_detected": 2000,
    "predictions": [0, 1, 0, 1, 0],
    "anomaly_scores": [-0.12, 0.95, -0.31, 0.82]
  },
  "metrics": {
    "precision": 0.80,
    "recall": 0.75,
    "f1_score": 0.77,
    "auc_roc": 0.85
  },
  "track3_specific": {
    "genres_analyzed": 18,
    "artists_analyzed": 2000,
    "avg_track_duration": 240.5,
    "avg_audio_complexity": 0.65,
    "suspicious_clusters": 8
  }
}
```

#### Track 4: `submission_[TEAM_NAME]_track4.json`
```json
{
  "team_info": {
    "team_name": "Nome del Team",
    "members": ["Nome1", "Nome2", "Nome3"],
    "track": "Track4",
    "submission_time": "2024-01-15T14:30:00Z"
  },
  "model_info": {
    "algorithm": "Isolation Forest + Guaranteed Clustering",
    "features_used": ["audio_complexity", "engagement_viral_coefficient", "revenue_per_stream"],
    "hyperparameters": {
      "contamination": 0.12,
      "n_estimators": 200
    }
  },
  "results": {
    "total_works": 15000,
    "infringements_detected": 1800,
    "predictions": [0, 1, 0, 1, 0],
    "infringement_scores": [-0.08, 0.91, -0.25, 0.87]
  },
  "metrics": {
    "precision": 0.84,
    "recall": 0.79,
    "f1_score": 0.81,
    "auc_roc": 0.88
  },
  "track4_specific": {
    "violation_types_detected": 5,
    "clusters_identified": 7,
    "avg_engagement_viral_coefficient": 0.73,
    "cross_platform_violations": 287,
    "content_id_manipulations": 195,
    "clustering_rate": 0.92
  }
}
```

### Criteri di Valutazione Automatica

#### Performance Tecnica (50%)
- **F1-Score** (25%): Media armonica di precision e recall
- **AUC-ROC** (15%): Robustezza del modello
- **Precision** (10%): Accuratezza delle anomalie rilevate

#### Innovazione e Tecnica (30%)
- **Numero di Features** (10%): Varietà di features utilizzate
- **Complessità Algoritmica** (10%): Uso di tecniche avanzate
- **Ensemble Methods** (10%): Combinazione di modelli

#### Business Metrics (20%)
- **Interpretabilità** (10%): Capacità di spiegare le anomalie
- **Scalabilità** (10%): Efficienza computazionale

### Come Submittare

1. **Genera il file di submission** usando il tuo script Python
2. **Fai commit** del file nella cartella `submissions/`
3. **Push** su GitHub - la leaderboard si aggiorna automaticamente
4. **Controlla** la tua posizione nella leaderboard

```bash
# Esempio di submission
git add submissions/submission_team_awesome.json
git commit -m "Team Awesome - Track 1 submission"
git push origin main
```

### Regole di Submission

- **Max 5 submissions** per team al giorno
- **Ultimo submission** conta per la valutazione finale
- **File size limit**: 50MB per submission
- **Formato obbligatorio**: JSON come specificato sopra

### Leaderboard Multi-Track

Il sistema genera **3 classifiche**:

#### 1. 🌟 Overall Leaderboard
- **Ranking globale** di tutti i team
- Basato sul **miglior score** ottenuto in qualsiasi track
- Determina il **vincitore assoluto** dell'hackathon

#### 2. 🎯 Track-Specific Leaderboards
- **Track 1**: Classifica dedicata Live Events Anomaly Detection
- **Track 2**: Classifica dedicata Document Fraud Detection
- **Track 3**: Classifica dedicata Music Anomaly Detection
- **Track 4**: Classifica dedicata Copyright Infringement Detection
- Competizione diretta tra team dello stesso track

#### 3. 📈 Detailed Performance Dashboard
- Metriche dettagliate per track
- Confronto algoritmi e tecniche
- Analisi performance temporale

La leaderboard è disponibile in tempo reale su:
- **File**: `leaderboard.md` (aggiornato automaticamente)
- **Dashboard**: Visualizzazione interattiva dei risultati
- **Metriche**: Ranking basato su score composito per track

### Scoring System

Il **Final Score** è calcolato come:
```
Final Score = (F1-Score × 0.25) + (AUC-ROC × 0.15) + (Precision × 0.10) + 
              (Innovation × 0.30) + (Business Impact × 0.20)
```

Dove:
- **Innovation Score**: Basato su features, algoritmi e approcci
- **Business Impact**: Valutato automaticamente su interpretabilità e scalabilità

---

## 🏆 Premi Multi-Track

### 🥇 Premio Overall (Vincitore Assoluto)
**Miglior team considerando tutti i track**:
- Buoni formazione avanzata (corsi online ML/AI)
- Mentorship con esperti del settore
- Possibilità stage/collaborazione SIAE
- Riconoscimento ufficiale SIAE

### 🎯 Premi per Track

#### 🏅 Track 1: Live Events Anomaly Detection
**1° Classificato**:
- Libri tecnici su anomaly detection
- Abbonamento piattaforme cloud (AWS/GCP)
- Certificato specializzato Track 1

#### 🏅 Track 2: Document Fraud Detection  
**1° Classificato**:
- Libri tecnici su computer vision e fraud detection
- Abbonamento piattaforme cloud (AWS/GCP)
- Certificato specializzato Track 2

#### 🏅 Track 3: Music Anomaly Detection
**1° Classificato**:
- Libri tecnici su music analytics e feature engineering
- Abbonamento piattaforme cloud (AWS/GCP)
- Certificato specializzato Track 3

### 🌟 Premi Speciali

#### "Most Innovative Team"
- Per l'approccio più creativo cross-track
- Gadget tech premium
- Certificato di innovazione

#### "Best Multi-Track Team"
- Per il team con migliori performance su più track
- Bonus formazione specializzata
- Riconoscimento versatilità

#### "Rising Star"
- Per il team emergente con approccio promettente
- Gadget tech
- Mentorship personalizzata

### 📊 Distribuzione Premi

- **1 vincitore overall** (premio principale)
- **3 vincitori track** (Track 1 + Track 2 + Track 3)
- **3 premi speciali** (innovazione, multi-track, rising star)
- **Certificati di partecipazione** per tutti i team validi

---

## 📚 Risorse Utili

### Tutorial e Guide
- [PyOD Documentation](https://pyod.readthedocs.io/): Libreria completa per outlier detection
- [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources)
- [Time Series Anomaly Detection](https://www.kaggle.com/learn/time-series)

### Paper Consigliati
- "Isolation Forest" (Liu et al., 2008)
- "LOF: Identifying Density-Based Local Outliers" (Breunig et al., 2000)
- "Anomaly Detection: A Survey" (Chandola et al., 2009)

### Notebook di Esempio
```python
# File: starter_notebook.ipynb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carica dataset
df = pd.read_csv('live_events_dataset.csv')

# Feature engineering base
df['revenue_per_person'] = df['total_revenue'] / df['attendance']
df['occupancy_rate'] = df['attendance'] / df['capacity']

# Prepara features
features = ['attendance', 'capacity', 'n_songs_declared', 
            'revenue_per_person', 'occupancy_rate']
X = df[features].fillna(0)

# Normalizza
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applica Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(X_scaled)

# Visualizza risultati
df['anomaly_score'] = iso_forest.score_samples(X_scaled)
df['is_anomaly'] = predictions == -1

print(f"Anomalie rilevate: {df['is_anomaly'].sum()}")
print(f"Precision: {(df['is_anomaly'] & df['anomaly_type'].notna()).sum() / df['is_anomaly'].sum():.2f}")
```

---

## 💡 Tips per i Partecipanti

1. **Start Simple**: Inizia con modelli base prima di tecniche complesse
2. **Feature Engineering**: Dedica tempo alla creazione di feature significative
3. **Validate Properly**: Usa strategie di validazione appropriate per time series
4. **Document Everything**: Commenta il codice e documenta le decisioni
5. **Think Business**: Considera sempre l'applicabilità reale

---

## 🤝 Supporto Durante l'Hackathon

### Mentori Disponibili
- **Data Scientists**: Per domande su algoritmi e modelli
- **Domain Experts**: Per capire il contesto SIAE
- **Tech Support**: Per problemi di setup e infrastruttura

### Canali di Comunicazione
- Slack/Discord dedicato
- Help desk fisico
- FAQ repository GitHub

---

## 📝 Deliverables Richiesti

### Submission Automatica (Obbligatoria)
1. **File di Submission JSON**
   - Formato standardizzato come specificato sopra
   - Nome file: `submission_[TEAM_NAME].json`
   - Posizionato nella cartella `submissions/`

2. **Codice Sorgente**
   - Script Python che genera il file di submission
   - Funzione `generate_submission()` implementata
   - README con istruzioni per riprodurre i risultati

### Presentazione Finale (Giorno 2)
3. **Presentazione** (max 10 slides)
   - Problema affrontato
   - Approccio tecnico
   - Risultati ottenuti (riferimento alla leaderboard)
   - Business impact

4. **Demo** (opzionale ma consigliata)
   - Dashboard interattiva
   - Visualizzazione anomalie in tempo reale
   - API REST per integration

---

## 🚀 Post-Hackathon

### Follow-up
- Repository pubblico con tutte le soluzioni
- Webinar con i vincitori
- Articolo tecnico sui migliori approcci
- Possibilità di continuare i progetti

### Community
- Gruppo LinkedIn alumni hackathon
- Newsletter con aggiornamenti
- Eventi futuri

---

## ⚖️ Note Legali e Privacy

- Tutti i dataset sono pubblici o sintetici
- Nessun dato reale SIAE viene utilizzato
- I partecipanti mantengono IP del proprio codice
- SIAE può richiedere licenza d'uso per soluzioni interessanti

---

## 📞 Contatti Organizzazione

**Email**: hackathon-anomaly@example.com  
**Website**: www.siae-hackathon-anomaly.it  
**Social**: #SIAEAnomalyHack

---

*Buona fortuna a tutti i partecipanti! Che vinca il miglior algoritmo! 🎯*
