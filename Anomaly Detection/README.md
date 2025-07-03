# 🎵 SIAE Anomaly Detection Hackathon
## Rilevamento Anomalie nei Diritti d'Autore e Utilizzi Musicali

### 📅 Durata: 2 giorni (14 ore totali)
- **Giorno 1**: 7 ore (9:30-13:00, 14:00-17:00)
- **Giorno 2**: 7 ore (9:30-13:00, 14:00-17:00)

---

## 🎯 Obiettivo della Challenge

I partecipanti dovranno sviluppare sistemi di anomaly detection per identificare:
1. **Utilizzi anomali di opere musicali** (locali/eventi con pattern sospetti)
2. **Anomalie nelle dichiarazioni dei diritti** (ripartizioni insolite, duplicazioni)
3. **Pattern fraudolenti nelle riproduzioni** (streaming/radio con comportamenti anomali)
4. **Irregolarità documentali** (contratti o documenti con elementi sospetti)

---

## 📊 Dataset Disponibili

### Dataset 1: MusicBrainz Database (Open Source)
**Descrizione**: Database aperto con informazioni su artisti, album, tracce e relazioni.

**Download**:
```bash
# Scaricare il dump PostgreSQL completo (circa 25GB compressi)
wget https://musicbrainz.org/doc/MusicBrainz_Database/Download

# O utilizzare subset più piccoli via API
pip install musicbrainzngs
```

**Contenuto**:
- Metadati di milioni di registrazioni musicali
- Relazioni tra artisti, opere e album
- Informazioni su pubblicazioni e distribuzioni

### Dataset 2: Free Music Archive (FMA)
**Descrizione**: Metadati di tracce musicali con informazioni su generi, artisti e utilizzi.

**Download**:
```bash
# Dataset metadati (342 MB)
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip

# Features audio pre-estratte (1GB) 
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```

**Contenuto**:
- 106,574 tracce di 16,341 artisti
- Metadati su generi, date, licenze
- Features audio pre-calcolate

### Dataset 3: Million Song Dataset (Subset)
**Descrizione**: Subset del famoso dataset con metadati di canzoni.

**Download**:
```bash
# Subset di 10,000 canzoni (280MB)
wget http://static.echonest.com/millionsongsubset_full.tar.gz
tar -xzf millionsongsubset_full.tar.gz

# Script Python per processare i file HDF5
pip install h5py pandas
```

### Dataset 4: Dataset Sintetico Eventi Live
**Descrizione**: Dataset generato per simulare dichiarazioni di eventi e concerti.

**Generazione** (fornito come script):
```python
# File: generate_live_events.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_live_events_dataset(n_events=50000):
    """
    Genera un dataset sintetico di eventi live con anomalie inserite
    """
    venues = [f"Venue_{i}" for i in range(1, 501)]
    cities = ["Milano", "Roma", "Napoli", "Torino", "Bologna", "Firenze", 
              "Palermo", "Genova", "Bari", "Venezia"]
    
    events = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_events):
        venue = random.choice(venues)
        city = random.choice(cities)
        
        # Genera evento normale
        event_date = start_date + timedelta(days=random.randint(0, 730))
        capacity = random.randint(50, 5000)
        attendance = random.randint(int(capacity * 0.3), capacity)
        
        # Numero di brani eseguiti (normale: 10-30)
        n_songs = random.randint(10, 30)
        
        # Revenue (normale: proporzionale all'attendance)
        base_revenue = attendance * random.uniform(15, 50)
        
        # Inserisci anomalie (10% dei casi)
        anomaly_type = None
        if random.random() < 0.1:
            anomaly_type = random.choice([
                "duplicate_declaration",
                "impossible_attendance", 
                "revenue_mismatch",
                "excessive_songs",
                "suspicious_timing"
            ])
            
            if anomaly_type == "duplicate_declaration":
                # Stesso venue, stessa data, orari vicini
                pass
            elif anomaly_type == "impossible_attendance":
                attendance = int(capacity * random.uniform(1.1, 1.5))
            elif anomaly_type == "revenue_mismatch":
                base_revenue = attendance * random.uniform(0.1, 5)
            elif anomaly_type == "excessive_songs":
                n_songs = random.randint(50, 100)
            elif anomaly_type == "suspicious_timing":
                # Eventi alle 4 del mattino
                event_date = event_date.replace(hour=4)
        
        events.append({
            'event_id': f'EVT_{i:06d}',
            'venue_id': venue,
            'city': city,
            'event_date': event_date,
            'capacity': capacity,
            'attendance': attendance,
            'n_songs_declared': n_songs,
            'total_revenue': round(base_revenue, 2),
            'anomaly_type': anomaly_type
        })
    
    return pd.DataFrame(events)

# Genera e salva il dataset
df = generate_live_events_dataset()
df.to_csv('live_events_dataset.csv', index=False)
print(f"Dataset generato con {len(df)} eventi")
print(f"Anomalie inserite: {df['anomaly_type'].notna().sum()}")
```

### Dataset 5: ISRC Database Sample
**Descrizione**: Codici ISRC (International Standard Recording Code) per tracciare registrazioni.

**Download**:
```bash
# Dataset esempio con ISRC codes
wget https://github.com/datasets/isrc/raw/main/data/isrc-sample.csv
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

### Track 1: Anomaly Detection in Live Events (Livello: Intermedio)
**Dataset**: Dataset Sintetico Eventi Live + FMA metadata

**Obiettivi**:
- Identificare eventi con attendance sospetta
- Rilevare venue con pattern di dichiarazione anomali
- Trovare discrepanze revenue/attendance

**Tecniche suggerite**:
- Isolation Forest per anomalie multivariate
- DBSCAN per clustering venue simili
- Time series analysis per pattern temporali

### Track 2: Copyright Declaration Anomalies (Livello: Avanzato)
**Dataset**: MusicBrainz + Million Song Dataset

**Obiettivi**:
- Identificare opere con attribuzioni sospette
- Rilevare pattern di plagio potenziale
- Trovare anomalie nelle catene di diritti

**Tecniche suggerite**:
- Graph Neural Networks per relazioni
- NLP per analisi similarità titoli
- Embedding musicali per confronti

### Track 3: Document Fraud Detection (Livello: Esperto)
**Dataset**: Generare documenti sintetici con anomalie

**Script generazione documenti**:
```python
# File: generate_contracts.py
from PIL import Image, ImageDraw, ImageFont
import random
import os

def generate_contract_images(n_contracts=1000):
    """Genera immagini di contratti con anomalie inserite"""
    
    for i in range(n_contracts):
        # Crea immagine base
        img = Image.new('RGB', (800, 1000), color='white')
        draw = ImageDraw.Draw(img)
        
        # Aggiungi testo standard
        draw.text((50, 50), "CONTRATTO DI CESSIONE DIRITTI", fill='black')
        draw.text((50, 100), f"Numero: {i:06d}", fill='black')
        
        # Inserisci anomalie casuali (10%)
        anomaly = None
        if random.random() < 0.1:
            anomaly = random.choice([
                "missing_signature",
                "altered_date",
                "duplicate_watermark",
                "suspicious_formatting"
            ])
            
            if anomaly == "missing_signature":
                # Non aggiungere area firma
                pass
            elif anomaly == "altered_date":
                # Data impossibile
                draw.text((50, 150), "Data: 31/02/2024", fill='red')
        
        # Salva immagine
        img.save(f'contracts/contract_{i:06d}.png')
        
        # Salva metadata
        with open('contracts_metadata.csv', 'a') as f:
            f.write(f"{i:06d},{anomaly}\n")
```

**Obiettivi**:
- Rilevare documenti alterati o falsificati
- Identificare firme mancanti o sospette
- Trovare inconsistenze nel layout

**Tecniche suggerite**:
- CNN per classificazione immagini
- Autoencoder per ricostruzione
- OCR + NLP per analisi testuale

### Track 4: Streaming Pattern Analysis (Livello: Intermedio)
**Dataset**: FMA + Dataset sintetico streaming

**Obiettivi**:
- Identificare pattern di ascolto bot/artificiali
- Rilevare playlist manipulation
- Trovare anomalie geografiche

**Tecniche suggerite**:
- LOF per pattern di ascolto
- Markov chains per sequenze
- Analisi geospaziale

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

## 📊 Criteri di Valutazione

### Performance Tecnica (40%)
- **Precision/Recall**: Bilanciamento tra falsi positivi e negativi
- **F1-Score**: Media armonica per dataset sbilanciati
- **AUC-ROC**: Robustezza a diverse soglie
- **Interpretabilità**: Capacità di spiegare le anomalie

### Innovazione (30%)
- Approcci creativi al problema
- Uso di tecniche avanzate
- Combinazione di metodologie

### Business Impact (20%)
- Applicabilità reale per SIAE
- Scalabilità della soluzione
- Costo computazionale

### Presentazione (10%)
- Chiarezza espositiva
- Qualità visualizzazioni
- Demo efficace

---

## 🏆 Premi Suggeriti

**1° Classificato**:
- Buoni formazione (corsi online ML/AI)
- Mentorship con esperti del settore
- Possibilità stage/collaborazione SIAE

**2° Classificato**:
- Libri tecnici su anomaly detection
- Abbonamento piattaforme cloud (AWS/GCP)

**3° Classificato**:
- Gadget tech
- Certificati di partecipazione speciali

**Premio Speciale "Most Innovative"**:
- Per l'approccio più creativo

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

1. **Codice Sorgente**
   - Repository GitHub pubblico
   - README con istruzioni
   - Requirements.txt

2. **Notebook Jupyter**
   - EDA documentata
   - Pipeline di training
   - Risultati e visualizzazioni

3. **Presentazione** (max 10 slides)
   - Problema affrontato
   - Approccio tecnico
   - Risultati ottenuti
   - Business impact

4. **Demo** (opzionale ma consigliata)
   - Dashboard interattiva
   - API REST
   - Web application

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
