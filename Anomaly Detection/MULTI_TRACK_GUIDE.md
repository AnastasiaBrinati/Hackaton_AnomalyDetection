# 🏁 SIAE Hackathon - Sistema Multi-Track

## 🌟 Overview

Il sistema di valutazione SIAE supporta ora **multiple track** con leaderboard unificate e separate per ogni categoria di challenge.

## 🎯 Track Disponibili

### Track 1: Live Events Anomaly Detection
- **Focus**: Rilevamento anomalie in eventi musicali live
- **Dataset**: Eventi live sintetici + metadati FMA
- **Obiettivo**: Identificare eventi sospetti, revenue mismatch, attendance irregolari
- **Tecniche**: Isolation Forest, DBSCAN, Feature Engineering musicale

### Track 2: Document Fraud Detection  
- **Focus**: Rilevamento frodi in documenti SIAE
- **Dataset**: Documenti digitali sintetici (contratti, licenze, certificazioni)
- **Obiettivo**: Identificare alterazioni digitali, firme false, template non autorizzati
- **Tecniche**: Computer Vision, OCR, CNN, Autoencoder

### Track 3: Music Anomaly Detection
- **Focus**: Rilevamento anomalie in tracce musicali con dataset FMA
- **Dataset**: FMA (Free Music Archive) con 25,000+ tracce sintetiche
- **Obiettivo**: Identificare plagio, bot streaming, metadata manipulation, genre mismatch
- **Tecniche**: Advanced Feature Engineering, Isolation Forest, Music Analytics

### Track Futuri (In Preparazione)
- **Track 4**: Copyright Infringement Detection
- **Track 5**: Music Similarity Fraud
- **Track 6**: Streaming Pattern Analysis

## 🏆 Sistema di Classifiche

### 1. Classifica Generale (Overall)
- **Ranking globale** di tutti i team attraverso tutti i track
- Basato sul **miglior score** ottenuto in qualsiasi track
- Un team può partecipare a **più track** simultaneamente

### 2. Classifiche per Track
- **Ranking separato** per ogni track
- Competizione diretta tra team dello stesso track
- Metriche specifiche per ogni dominio

### 3. Premi Multi-Level
- **🥇 Overall Winner**: Miglior team generale
- **🏅 Track Winners**: Vincitori per ogni track
- **🌟 Most Innovative**: Approccio più creativo cross-track

## 📂 Struttura File System

```
SIAE_Hackathon/
├── Track1_Solution/
│   ├── track1_anomaly_detection.py
│   ├── requirements.txt
│   └── README.md
├── Track2_Solution/
│   ├── track2_document_fraud_detection.py
│   ├── requirements.txt
│   └── README.md
├── Track3_Solution/
│   ├── track3_music_anomaly_detection.py
│   ├── requirements.txt
│   └── README.md
├── submissions/
│   ├── submission_[team]_track1.json
│   ├── submission_[team]_track2.json
│   ├── submission_[team]_track3.json
│   ├── submission_example.json
│   └── submission_example_track2.json
├── evaluate_submissions.py    # Multi-track evaluator
├── leaderboard.md            # Unified leaderboard
└── setup_auto_leaderboard.py # Auto-update system
```

## 🚀 Come Partecipare Multi-Track

### Scelta 1: Single Track
```bash
# Solo Track 1
cd Track1_Solution
python track1_anomaly_detection.py
# Genera: submissions/submission_team_track1.json

# Solo Track 2  
cd Track2_Solution
python track2_document_fraud_detection.py
# Genera: submissions/submission_team_track2.json

# Solo Track 3
cd Track3_Solution
python track3_music_anomaly_detection.py
# Genera: submissions/submission_team_track3.json
```

### Scelta 2: Multi-Track (Consigliata)
```bash
# Team che partecipa a tutti i track
cd Track1_Solution
python track1_anomaly_detection.py  # Score Track 1

cd ../Track2_Solution  
python track2_document_fraud_detection.py  # Score Track 2

cd ../Track3_Solution
python track3_music_anomaly_detection.py  # Score Track 3

# La leaderboard mostrerà:
# - Best score overall (max tra Track1, Track2, Track3)
# - Posizione in tutte le classifiche separate
```

## 📊 Sistema di Scoring Multi-Track

### Unified Scoring Formula
Ogni track usa la stessa formula base:
```
Final Score = (Technical × 0.5) + (Innovation × 0.3) + (Business × 0.2)
```

### Track-Specific Adaptations

#### Track 1: Live Events
- **Technical**: F1-Score (anomaly detection), AUC-ROC, Precision
- **Innovation**: Features musicali, ensemble methods, FMA integration
- **Business**: Scalabilità eventi, interpretabilità anomalie

#### Track 2: Documents  
- **Technical**: F1-Score (fraud detection), AUC-ROC, Precision
- **Innovation**: Computer vision features, OCR integration, layout analysis
- **Business**: Processing speed, document type coverage

#### Track 3: Music Analytics
- **Technical**: F1-Score (music anomaly detection), AUC-ROC, Precision
- **Innovation**: Music feature engineering, FMA integration, clustering analysis
- **Business**: Scalability for music catalogs, genre/artist interpretability

## 🔧 Setup Multi-Track

### 1. Environment Setup
```bash
# Track 1 dependencies
cd Track1_Solution
pip install -r requirements.txt

# Track 2 dependencies  
cd Track2_Solution
pip install -r requirements.txt

# Track 3 dependencies
cd Track3_Solution
pip install -r requirements.txt

# Sistema di valutazione
pip install -r requirements_evaluation.txt
```

### 2. Auto-Update System
```bash
# Setup sistema automatico (supporta tutti i track)
python setup_auto_leaderboard.py

# Il sistema riconosce automaticamente:
# - Track1 submissions: track: "Track1"
# - Track2 submissions: track: "Track2"
# - Future tracks: track: "Track3", etc.
```

## 📝 Formato Submission Multi-Track

### Differenze per Track

#### Track 1 (Eventi Live)
```json
{
  "team_info": {"track": "Track1"},
  "results": {
    "total_events": 10000,
    "anomalies_detected": 950
  },
  "metrics": {...},
  "track1_specific": {
    "genre_analysis": {...},
    "venue_clusters": 8
  }
}
```

#### Track 2 (Document Fraud)
```json
{
  "team_info": {"track": "Track2"},
  "results": {
    "total_documents": 5000,
    "frauds_detected": 750
  },
  "metrics": {...},
  "track2_specific": {
    "document_types_analyzed": 6,
    "avg_text_confidence": 0.847,
    "siae_watermark_detection_rate": 0.823
  }
}
```

#### Track 3 (Music Anomaly)
```json
{
  "team_info": {"track": "Track3"},
  "results": {
    "total_tracks": 25000,
    "anomalies_detected": 2000
  },
  "metrics": {...},
  "track3_specific": {
    "genres_analyzed": 18,
    "artists_analyzed": 2000,
    "avg_track_duration": 240.5,
    "avg_audio_complexity": 0.65,
    "suspicious_clusters": 8
  }
}
```

## 🎪 Strategie Multi-Track

### Team Strategy 1: Specialization
- **Focus**: Diventare esperti in un singolo track
- **Vantaggio**: Massima expertise, soluzioni profonde
- **Rischio**: Limitati a una sola categoria di premi

### Team Strategy 2: Diversification  
- **Focus**: Partecipare a più track
- **Vantaggio**: Più opportunità di vincita, score overall migliore
- **Rischio**: Risorse diluite tra track

### Team Strategy 3: Innovation Focus
- **Focus**: Approcci creativi cross-track
- **Vantaggio**: Premio "Most Innovative"
- **Esempio**: Transfer learning tra Track1/Track3 (musica) e Track2 (documenti)

## 📈 Leaderboard Multi-Track

### Overall Top Teams
```
Rank | Team | Best Score | Track | Algorithm
-----|------|------------|-------|----------
1    | TeamA| 0.925      | Track2| CNN+OCR  
2    | TeamB| 0.890      | Track1| Ensemble
3    | TeamC| 0.875      | Track1| Deep Learning
```

### Track-Specific Rankings
- **Track 1**: Competizione diretta anomaly detection eventi live
- **Track 2**: Competizione diretta fraud detection documenti
- **Track 3**: Competizione diretta music anomaly detection
- **Future Tracks**: Rankings dedicati

## 🔮 Roadmap Track Futuri

### Track 4: Streaming Pattern Analysis
- **Dataset**: Pattern di streaming + FMA
- **Obiettivo**: Bot detection, playlist manipulation
- **Tecniche**: Time series, graph analysis

### Track 5: Copyright Infringement
- **Dataset**: Audio fingerprints + metadata
- **Obiettivo**: Unauthorized usage detection  
- **Tecniche**: Audio processing, similarity matching

### Track 6: Music Similarity Fraud
- **Dataset**: Audio features + copyright data
- **Obiettivo**: Plagio musicale detection
- **Tecniche**: Deep audio analysis, feature matching

## 💡 Best Practices Multi-Track

### Development
1. **Start Simple**: Implementa una baseline per ogni track
2. **Share Components**: Riusa feature engineering cross-track
3. **Track Metrics**: Monitora performance su entrambi
4. **Time Management**: Bilancia effort tra track

### Submission
1. **Test Separatamente**: Valida ogni track individualmente
2. **Commit Strategy**: Submissioni separate per track
3. **Documentation**: README specifici per track
4. **Version Control**: Branch/tag per track

### Competition
1. **Monitor All**: Segui leaderboard di tutti i track
2. **Learn Cross-Track**: Tecniche da altri track
3. **Collaborate**: Team specializzati possono collaborare
4. **Iterate Fast**: Quick wins su più track

---

**🚀 Il sistema multi-track amplifica le opportunità di successo e innovation! 🏆** 