# 🎵 Track 3: Music Anomaly Detection

## 🎯 Challenge Overview

**Obiettivo**: Sviluppare un sistema di rilevamento anomalie per tracce musicali utilizzando il dataset FMA (Free Music Archive) con tecniche avanzate di machine learning.

## 📋 Problema da Risolvere

La SIAE deve identificare anomalie nella distribuzione e utilizzo musicale:
- **Plagio e similarità fraudolente** tra tracce
- **Pattern di streaming innaturali** (bot, manipulation)
- **Metadata manipulation** (informazioni false o inconsistenti)
- **Genre mismatch** (generi non corrispondenti alle features audio)
- **Audio quality fraud** (qualità dichiarata non corrispondente)

## 🔧 Tecnologie Utilizzate

### Dataset
- **FMA (Free Music Archive)**: 25,000+ tracce musicali sintetiche
- **Metadati completi**: Generi, artisti, features audio, popolarità
- **Features audio**: Tempo, energy, danceability, valence, etc.

### Machine Learning
- **Isolation Forest**: Rilevamento anomalie multivariate
- **DBSCAN**: Clustering tracce sospette
- **Advanced Feature Engineering**: 35+ features composite

### Analisi Audio
- **Audio Features**: Caratteristiche musicali avanzate
- **Metadata Analysis**: Consistenza informazioni
- **Pattern Detection**: Rilevamento comportamenti anomali

## 📊 Tipi di Anomalie Rilevate

### 1. 🎭 Plagio Similarity (3%)
- Tracce con features audio troppo simili
- Stessi valori di tempo, energy, danceability
- Potenziale plagio musicale

### 2. 🤖 Bot Streaming (2.5%)
- Pattern di ascolto innaturali
- Molti listens ma pochi favorites
- Ratios sospetti engagement/popolarità

### 3. 📝 Metadata Manipulation (2%)
- Date di creazione impossibili (future)
- Informazioni inconsistenti
- Metadati alterati

### 4. 🎼 Genre Mismatch (1.5%)
- Genere dichiarato non corrispondente
- Classical con alta energy/danceability
- Features audio inconsistenti

### 5. 🔊 Audio Quality Fraud (1%)
- Qualità dichiarata non corrispondente
- File size troppo piccolo per bitrate
- Caratteristiche tecniche sospette

## 🚀 Come Eseguire la Soluzione

### 1. Setup Environment
```bash
cd Track3_Solution
pip install -r requirements.txt
```

### 2. Esecuzione Script
```bash
python track3_music_anomaly_detection.py
```

### 3. Output Generati
- **CSV**: `music_anomaly_detection_results.csv`
- **Analisi Generi**: `genre_anomaly_analysis.csv`
- **Artisti Sospetti**: `suspicious_artists_analysis.csv`
- **Visualizzazioni**: `music_anomaly_detection_results.png`
- **Submission**: `../submissions/submission_[team]_track3.json`

## 📈 Features Utilizzate (35+)

### Features Audio Base
- `tempo`, `loudness`, `energy`, `danceability`
- `valence`, `acousticness`, `instrumentalness`
- `speechiness`, `liveness`

### Features Composite
- `audio_complexity`: Combinazione energy+danceability+valence
- `mood_energy_combo`: Valence × Energy
- `acoustic_speech_balance`: Acousticness - Speechiness

### Features Popolarità
- `listens_to_favorites_ratio`: Ratio engagement
- `favorites_to_comments_ratio`: Ratio interazione
- `downloads_to_listens_ratio`: Ratio download
- `listens_per_day`: Popolarità normalizzata per età

### Features Qualità
- `quality_size_ratio`: File size vs bitrate
- `is_high_quality`: Qualità audio alta
- Bitrate, sample rate, file size

### Features Artista
- `artist_avg_listens`: Media ascolti artista
- `artist_genre_diversity`: Diversità generi
- `listens_vs_artist_avg`: Comparazione con media
- `artist_track_count`: Numero tracce artista

### Features Temporali
- `track_age_days`: Età traccia
- `artist_career_length`: Lunghezza carriera
- `is_recent_track`: Traccia recente

### Features Geografiche
- `artist_location`: Localizzazione artista
- `is_us_artist`, `is_european_artist`: Macro-regioni

## 🎯 Performance Attese

### Metriche Target
- **Precision**: 0.80+ (80% anomalie rilevate sono vere)
- **Recall**: 0.75+ (75% anomalie vere vengono rilevate)
- **F1-Score**: 0.77+ (Media armonica precision/recall)
- **AUC-ROC**: 0.85+ (Robustezza del modello)

### Interpretabilità
- **Clustering**: Raggruppamento tracce sospette
- **Feature Importance**: Quali caratteristiche sono più indicative
- **Anomaly Types**: Classificazione per tipo di anomalia

## 🔍 Analisi e Visualizzazioni

### 1. Distribuzione Anomaly Scores
Istogramma dei punteggi di anomalia per visualizzare la separazione

### 2. Scatter Plots Features Audio
- Energy vs Danceability
- Listens vs Favorites
- Correlazioni tra features principali

### 3. Analisi per Genere
- Boxplot anomaly scores per genere musicale
- Generi più sospetti

### 4. Heatmap Correlazioni
Matrice di correlazione tra features audio principali

### 5. Clustering Visualization
Visualizzazione cluster di tracce sospette

## 🏆 Strategia di Submission

### Parametri Ottimizzati
- **Contamination**: 0.08 (8% anomalie attese)
- **N_estimators**: 200 (Isolation Forest robusto)
- **Feature Selection**: 35+ features engineered

### Validazione
- Cross-validation per robustezza
- Analisi per genere musicale
- Clustering per interpretabilità

## 📝 Personalizzazione

### Modifica Team Info
```python
# In track3_music_anomaly_detection.py alla fine:
team_name = "Il Tuo Team Name"  # CAMBIA QUI
members = ["Nome1", "Nome2", "Nome3"]  # CAMBIA QUI
```

### Tuning Parametri
```python
# Isolation Forest
contamination = 0.08  # Percentuale anomalie attese
n_estimators = 200    # Numero alberi

# DBSCAN
eps = 0.5            # Distanza cluster
min_samples = 3      # Minimo campioni per cluster
```

## 🎵 Caratteristiche Innovative

### 1. **Multi-Level Feature Engineering**
- Features base, composite, temporali, geografiche
- Ratios e comparazioni con baseline artista

### 2. **Music-Specific Anomaly Detection**
- Adattato per caratteristiche musicali
- Considera generi, artisti, popolarità

### 3. **Comprehensive Analysis**
- Analisi per genere musicale
- Identificazione artisti sospetti
- Clustering pattern anomali

### 4. **Business-Oriented Outputs**
- Interpretabilità per operatori SIAE
- Classificazione per tipo di anomalia
- Prioritizzazione tracce sospette

---

## 🎯 Track 3 Goal

**Identificare e classificare anomalie musicali** per:
- Proteggere diritti d'autore
- Rilevare frodi distributive
- Monitorare qualità metadati
- Supportare decisioni SIAE

**🎵 La musica è dati, i dati raccontano storie, le anomalie rivelano segreti! 🎵** 