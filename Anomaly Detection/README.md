# 🚀 HACKATHON ANOMALY DETECTION 2025
Questa cartella contiene l'hackathon per affrontare e vedere dal vivo quanto visto **completamente pronta** con dataset pubblici e sfide pre-configurate per l'anomaly detection in contesti reali simili a quelli SIAE.

## 📋 PANORAMICA EVENTO

### 🎯 Obiettivo
Hackathon **completamente pronta** con dataset pubblici e sfide pre-configurate per l'anomaly detection in contesti reali simili a quelli SIAE.

### ⏱️ Durata
**14 ore totali** (2 giorni × 7 ore/giorno) - **Formato Sprint Intensivo**

### 🎁 **TUTTO INCLUSO**
✅ Dataset pubblici già scaricabili  
✅ Baseline code fornito  
✅ Ambiente Colab pre-configurato  
✅ Metrics di valutazione definite  
✅ Leaderboard automatica  

---

## 🏆 SFIDE READY-TO-GO

### **TRACK 1: Music Anomaly Detection** 🎵
**Dataset**: **Free Music Archive (FMA)** + **Spotify Million Playlist**

**🔗 Link Dataset**:
- FMA: https://github.com/mdeff/fma (8,000 tracks, 30s preview)
- Spotify: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

**📁 Dati Forniti**:
```python
# Struttura dati disponibili
music_features = {
    'audio_features': ['tempo', 'energy', 'valence', 'danceability'],
    'metadata': ['artist', 'genre', 'year', 'duration'],
    'play_counts': ['streams', 'skips', 'likes'],
    'timestamps': ['hour', 'day_of_week', 'season']
}
```

**🎯 Sfida**: Identificare **pattern di streaming anomali** in 3 ore di sviluppo + 2 ore optimization

**⏱️ Complessità**: **MEDIUM** - Perfetta per 14 ore
- Baseline in 30 minuti
- Feature engineering in 2 ore  
- Advanced models in 3 ore
- Tuning in 1.5 ore

**💻 Starter Code Rapido**:
```python
# 🚀 QUICK START - 15 minuti per baseline funzionante
import pandas as pd
from sklearn.ensemble import IsolationForest

# Auto-load dataset (già processato)
df = pd.read_csv('music_streaming_anomalies.csv')
features = ['stream_velocity', 'skip_rate', 'geographic_dispersion', 'time_pattern']

# Baseline model (3 righe!)
model = IsolationForest(contamination=0.05, random_state=42)
predictions = model.fit_predict(df[features])
score = model.decision_function(df[features])

print(f"✅ Baseline ready! Found {sum(predictions == -1)} anomalies")
```

---

### **TRACK 2: Financial Fraud Detection** 💳
**Dataset**: **IEEE-CIS Fraud Detection** (Kaggle Public)

**🔗 Link Dataset**: 
- https://www.kaggle.com/c/ieee-fraud-detection/data
- Backup: Synthetic Financial Dataset da IBM

**📁 Dati Forniti**:
```python
# 590,540 transazioni con 434 features
transaction_data = {
    'TransactionAmt': 'Importo transazione',
    'ProductCD': 'Codice prodotto', 
    'card1-card6': 'Info carta di credito',
    'addr1-addr2': 'Info indirizzo',
    'dist1-dist2': 'Distanze geografiche',
    'C1-C14': 'Feature conteggi',
    'D1-D15': 'Feature temporali',
    'V1-V339': 'Feature Vesta Engineering'
}
```

**🎯 Sfida**: Rilevare transazioni fraudolente simulando il problema dei pagamenti anomali delle royalties.

**⏱️ Complessità**: **EASY-MEDIUM** - Ideale per hackathon da 14 ore
- Setup + EDA: 1 ora
- Baseline models: 2 ore
- Advanced techniques: 3 ore  
- Ensemble & tuning: 2 ore

**💻 Starter Code Ultra-Rapido**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Data loading
train_df = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')

# Baseline preprocessing
features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3']
X = train_df[features].fillna(-999)
y = train_df['isFraud']

# Simple baseline
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
```

---

### **TRACK 3: Document Fraud Detection** 📄
**Dataset**: **RVL-CDIP** (400K document images) + **Tobacco800**

**🔗 Link Dataset**:
- RVL-CDIP: https://www.cs.cmu.edu/~aharley/rvl-cdip/
- Tobacco800: https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72

**📁 Dati Forniti**:
```python
# 16 classi di documenti
document_types = [
    'letter', 'form', 'email', 'handwritten', 
    'advertisement', 'scientific_report', 'scientific_publication',
    'specification', 'file_folder', 'news_article', 'budget',
    'invoice', 'presentation', 'questionnaire', 'resume', 'memo'
]
```

**🎯 Sfida**: Identificare documenti contraffatti o anomali che potrebbero rappresentare:
- Documenti di copyright falsificati
- Contratti manipolati
- Fatture fraudolente

**⏱️ Complessità**: **MEDIUM** - Perfetta per 14 ore totali
- Computer Vision setup: 1 ora
- Feature extraction: 2 ore
- Deep learning models: 4 ore
- Fine-tuning: 2 ore

**💻 Starter Code Express**:
```python
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from sklearn.ensemble import IsolationForest

# Feature extraction con CNN pre-trained
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    # VGG16 feature extraction
    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(np.expand_dims(img, axis=0))
    return features.flatten()

# Anomaly detection su features
features_array = np.array([extract_features(img) for img in image_paths])
detector = IsolationForest(contamination=0.1)
anomalies = detector.fit_predict(features_array)
```

---

### **TRACK 4: Network Behavior Anomalies** 🌐
**Dataset**: **NSL-KDD** (Network Intrusion) + **UNSW-NB15**

**🔗 Link Dataset**:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

**📁 Dati Forniti**:
```python
# Network connection features
network_features = {
    'basic': ['duration', 'protocol_type', 'service', 'flag'],
    'content': ['src_bytes', 'dst_bytes', 'hot', 'num_failed_logins'],
    'time': ['count', 'srv_count', 'serror_rate', 'srv_serror_rate'],
    'host': ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate']
}
```

**🎯 Sfida**: Rilevare comportamenti anomali che simulano:
- Accessi non autorizzati al sistema SIAE
- Attacchi DDoS
- Comportamenti bot

**⏱️ Complessità**: **EASY** - Ottima per principianti
- Data understanding: 45 min
- Preprocessing: 1 ora  
- Multiple algorithms: 2 ore
- Comparison & tuning: 1.5 ore

**💻 Starter Code Lightning**:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Load NSL-KDD dataset
df = pd.read_csv('KDDTrain+.txt', header=None)
feature_names = ['duration', 'protocol_type', 'service', ...]

# Preprocessing
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = le.fit_transform(df[col])

# Anomaly detection
X = df.iloc[:, :-1]  # All features except label
detector = IsolationForest(contamination=0.1)
predictions = detector.fit_predict(X)
```

---

## ⚡ SPRINT OPTIMIZATION PER 14 ORE

### **🎯 Scope Ridotto ma Impattante**
Invece di 4 track completi, focus su **2 track principali + 1 bonus**:

**TRACK PRINCIPALE 1**: Music Anomaly (60% partecipanti)
**TRACK PRINCIPALE 2**: Financial Fraud (35% partecipanti)  
**TRACK BONUS**: Quick Challenge (5% partecipanti) - *Solo per expert*

### **⚡ Fast-Track Development Strategy**

**Milestone obbligatorie per timeline 14h**:
- **Ora 3**: Working baseline submission
- **Ora 6**: First improvement iteration  
- **Ora 10**: Advanced model working
- **Ora 13**: Final submission + demo ready

### **🛠️ Pre-Configured Templates Specifici**

**Template "14-Hour Sprint"**:
```python
# ⚡ HACKATHON 14H TEMPLATE - ULTRA OTTIMIZZATO

# Imports rapidi - tutto in una cella
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns

# Configurazione rapida
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', 20)

# Data loader con timer
import time
def quick_load_and_explore(track_num):
    start = time.time()
    df = pd.read_csv(f'track_{track_num}_data.csv')
    print(f"✅ Data loaded in {time.time()-start:.1f}s")
    print(f"📊 Shape: {df.shape}")
    print(f"🔍 Missing: {df.isnull().sum().sum()}")
    return df

# Baseline sprint (< 5 minuti)
def sprint_baseline(X, contamination=0.05):
    """Baseline in meno di 5 minuti garantiti"""
    models = {
        'IsolationForest': IsolationForest(contamination=contamination, random_state=42),
        'OneClassSVM': OneClassSVM(nu=contamination),
    }
    
    results = {}
    for name, model in models.items():
        start = time.time()
        pred = model.fit_predict(X)
        results[name] = {
            'predictions': pred,
            'time': time.time() - start,
            'anomalies_found': sum(pred == -1)
        }
        print(f"✅ {name}: {results[name]['anomalies_found']} anomalies in {results[name]['time']:.1f}s")
    
    return results

print("🚀 14-Hour Sprint Template Ready!")
```

### **📊 Simplified Evaluation (Real-time)**

**Auto-scoring ogni ora**:
```python
# Scoring automatico ottimizzato per 14h
class QuickLeaderboard:
    def __init__(self):
        self.hourly_scores = {}
    
    def quick_submit(self, team, predictions, hour_mark):
        """Submit ultra-rapido con feedback immediato"""
        score = self.calculate_quick_score(predictions)
        self.hourly_scores[f"{team}_h{hour_mark}"] = score
        
        # Feedback istantaneo
        print(f"⚡ {team} @ Hour {hour_mark}: Score = {score:.3f}")
        return score
    
    def show_live_ranking(self):
        """Classifica live ogni 2 ore"""
        # Mostra top 3 in tempo reale
        pass

# Uso durante hackathon
lb = QuickLeaderboard()
lb.quick_submit("TeamAlpha", my_predictions, hour_mark=3)
```

### **Google Colab Templates Ready-to-Use**

**Template Principale**:
```python
# 🚀 HACKATHON ANOMALY DETECTION - SETUP COMPLETO
!pip install pyod scikit-learn pandas numpy matplotlib seaborn plotly

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN

# Utility functions
def load_dataset(track_number):
    """Carica automaticamente il dataset per il track scelto"""
    datasets = {
        1: "music_anomaly_data.csv",
        2: "financial_fraud_data.csv", 
        3: "document_features.csv",
        4: "network_behavior_data.csv"
    }
    return pd.read_csv(datasets[track_number])

def evaluate_anomaly_detection(y_true, y_pred):
    """Evaluation metrics standard per anomaly detection"""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    auc = roc_auc_score(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1,
        'auc_roc': auc
    }

def plot_anomalies(X, predictions, title="Anomaly Detection Results"):
    """Visualizzazione automatica dei risultati"""
    plt.figure(figsize=(12, 8))
    normal = X[predictions == 1]
    anomalies = X[predictions == -1]
    
    plt.scatter(normal.iloc[:, 0], normal.iloc[:, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], 
               c='red', alpha=0.8, label='Anomaly')
    plt.title(title)
    plt.legend()
    plt.show()

print("🎯 Setup completato! Scegli il tuo track e inizia subito!")
```

### **Leaderboard Automatica**
```python
# Sistema di scoring automatico
class AutoLeaderboard:
    def __init__(self):
        self.scores = {}
    
    def submit_solution(self, team_name, track, y_pred, model_info):
        """Submit automatico con calcolo score"""
        # Carica ground truth per il track
        y_true = self.load_ground_truth(track)
        
        # Calcola metriche
        metrics = evaluate_anomaly_detection(y_true, y_pred)
        
        # Salva nel leaderboard
        self.scores[team_name] = {
            'track': track,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'auc_roc': metrics['auc_roc'],
            'model': model_info,
            'timestamp': datetime.now()
        }
        
        print(f"✅ {team_name} - Score registrato: F1={metrics['f1_score']:.3f}")
        return metrics

# Uso
leaderboard = AutoLeaderboard()
leaderboard.submit_solution("Team Alpha", track=1, y_pred=predictions, 
                          model_info="Isolation Forest")
```

---

## 📊 DATASET E DOWNLOAD IMMEDIATI

### **Link Download Pronti**
```bash
# Script di download automatico
#!/bin/bash

# Track 1: Music Data
wget -O fma_small.zip "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
unzip fma_small.zip

# Track 2: Financial Fraud  
kaggle competitions download -c ieee-fraud-detection

# Track 3: Document Images
wget -O rvl-cdip.tar.gz "http://www.cs.cmu.edu/~aharley/rvl-cdip/rvl-cdip.tar.gz"

# Track 4: Network Data
wget -O NSL-KDD.zip "https://www.unb.ca/cic/datasets/nsl-kdd.zip"

echo "✅ Tutti i dataset scaricati e pronti!"
```

### **Backup Datasets (se links non funzionano)**
- **Synthetic Music Data**: Generato con librerie Python
- **Credit Card Fraud** (Kaggle): Dataset alternativo più piccolo
- **Synthetic Documents**: Usando generatori automatici
- **Network Simulation**: Dati generati con Scapy

---

## 🏅 VALUTAZIONE AUTOMATICA

### **Metriche Standard per Track**
```python
# Sistema di valutazione unificato
evaluation_metrics = {
    'primary': ['f1_score', 'precision', 'recall'],
    'secondary': ['auc_roc', 'auc_pr'],
    'efficiency': ['training_time', 'inference_time'],
    'bonus': ['interpretability_score', 'code_quality']
}

# Weighted scoring
final_score = (
    0.4 * f1_score + 
    0.3 * auc_roc + 
    0.2 * efficiency_score + 
    0.1 * bonus_points
)
```

### **Submission Format**
```python
# Template di submission obbligatorio
submission = {
    'team_name': 'Nome Team',
    'track': 1,  # 1-4
    'predictions': anomaly_predictions,  # Array numpy
    'model_type': 'Isolation Forest',
    'preprocessing_steps': ['StandardScaler', 'PCA'],
    'hyperparameters': {'contamination': 0.05},
    'training_time': 45.2,  # secondi
    'inference_time': 0.003,  # secondi per sample
    'code_notebook': 'team_solution.ipynb'
}
```

---

## 🎯 PRIZES & RECOGNITION

### **Prize Pool Totale: €15,000**
- 🥇 **Overall Winner**: €5,000 + Certificazione
- 🥈 **Track Winners** (4x): €2,000 each
- 🥉 **Special Awards**: €1,000 each
  - Most Innovative Approach
  - Best Real-time Solution
  - Best Interpretability

### **Certificazioni**
- **IBM Certified Data Scientist** voucher
- **Google Cloud ML Engineer** training credits
- **Coursera Specialization** in Anomaly Detection

---

## 📅 TIMELINE OTTIMIZZATO (14 ORE TOTALI)

### **Setup Phase (Pre-evento - 1 settimana prima)**
- **Email ai partecipanti**: Link Colab + istruzioni setup
- **Discord Server**: Canali per ogni track + help desk
- **Dataset pre-test**: Tutti scaricabili e verificati
- **Team formation**: Matching skills online (opzionale)

---

### **GIORNO 1 - Development Sprint (7 ore)**
**🕘 09:00-09:30 (30min)** - **Kickoff & Setup**
- Welcome + overview delle sfide
- Distribuzione credenziali e link
- Team formation (se non già fatto)
- Q&A tecnico veloce

**🕘 09:30-12:30 (3 ore)** - **Sprint Sviluppo Parte 1**
- Analisi esplorativa dei dati
- Implementazione baseline model
- Prime iterazioni e sperimentazione
- *Mentori disponibili per supporto*

**🕘 12:30-13:30 (1 ora)** - **Pausa Pranzo + Networking**
- Lunch break
- Condivisione quick wins tra team
- Supporto tecnico per chi ha problemi

**🕘 13:30-16:00 (2.5 ore)** - **Sprint Sviluppo Parte 2**
- Miglioramento modelli
- Feature engineering
- Tuning iperparametri
- Prime submission al leaderboard

**🕘 16:00-16:30 (30min)** - **Checkpoint Intermedio**
- **OBBLIGATORIO**: Prima submission funzionante
- Quick status update (2 min per team)
- Identificazione team in difficoltà per supporto extra

---

### **GIORNO 2 - Optimization & Demo (7 ore)**
**🕘 09:00-09:15 (15min)** - **Briefing Giorno 2**
- Recap risultati Giorno 1
- Focus su optimization e presentazione
- Timeline finale

**🕘 09:15-12:00 (2h 45min)** - **Optimization Sprint**
- Ensemble methods e stacking
- Hyperparameter tuning avanzato
- Implementazione tecniche avanzate
- Preparazione visualizzazioni

**🕘 12:00-13:00 (1 ora)** - **Pausa Pranzo + Prep Demo**
- Lunch veloce
- Inizio preparazione presentazioni
- Test demo e script

**🕘 13:00-15:00 (2 ore)** - **Finalizzazione + Demo Prep**
- **14:00 DEADLINE**: Submission finale obbligatoria
- Preparazione slide (max 5 slides)
- Test delle demo
- Backup plans per problemi tecnici

**🕘 15:00-16:30 (1.5 ore)** - **Presentazioni**
- **5 minuti per team** (3 min demo + 2 min Q&A)
- Valutazione live della giuria
- Feedback immediate su approcci

**🕘 16:30-17:00 (30min)** - **Premiazione & Chiusura**
- Annuncio vincitori
- Recap insights tecnici
- Networking finale + next steps

---

## 🚀 QUICK START GUIDE

### **Per Organizzatori**
1. **Fork** repository: `git clone https://github.com/anomaly-hackathon/ready-datasets`
2. **Verifica datasets**: Esegui `python check_datasets.py`
3. **Setup Discord**: Invite link per partecipanti
4. **Test Colab**: Verifica tutti i notebook funzionino

### **Per Partecipanti**
1. **Join Discord**: Link fornito via email
2. **Apri Colab**: Template del tuo track preferito
3. **Scarica dati**: Esegui prima cella del notebook
4. **Start coding**: Baseline già funzionante!

### **Repository Structure**
```
hackathon-anomaly-detection/
├── datasets/
│   ├── track1_music/
│   ├── track2_financial/ 
│   ├── track3_documents/
│   └── track4_network/
├── notebooks/
│   ├── Track1_Music_Baseline.ipynb
│   ├── Track2_Financial_Baseline.ipynb
│   ├── Track3_Documents_Baseline.ipynb
│   └── Track4_Network_Baseline.ipynb
├── evaluation/
│   ├── auto_leaderboard.py
│   ├── metrics.py
│   └── submission_template.py
└── utils/
    ├── data_loaders.py
    ├── visualization.py
    └── preprocessing.py
```

---

## 📞 READY-TO-LAUNCH CONTACTS

### **Immediate Support**
- **Discord**: https://discord.gg/anomaly-hackathon
- **GitHub Issues**: Per problemi tecnici immediati
- **Email Express**: hackathon-ready@domain.com (risposta <2h)

### **Templates & Resources**
- **Colab Master**: https://colab.research.google.com/drive/hackathon-templates
- **Dataset Backup**: Google Drive folder con tutti i dati
- **Evaluation API**: Submission automatica via API

---

## ✅ CHECKLIST FINALE ORGANIZZATORI

**Pre-Evento (1 settimana prima)**:
- [ ] Tutti i dataset scaricabili e testati
- [ ] Notebook Colab funzionanti al 100%
- [ ] Discord server setup con canali
- [ ] Leaderboard automatica testata
- [ ] Email template per partecipanti pronto

**Giorno Evento**:
- [ ] Link Colab condivisi
- [ ] Monitor Discord per supporto
- [ ] Backup datasets disponibile  
- [ ] Sistema valutazione online
- [ ] Recording delle presentazioni

**Post-Evento**:
- [ ] Risultati pubblicati su GitHub
- [ ] Certificati inviati ai vincitori
- [ ] Feedback survey ai partecipanti
- [ ] Report finale con insights

---

**🎯 Questa hackathon è 100% pronta per il lancio immediato!**  
**Tutti i dataset sono pubblici, i notebook testati, e il sistema di valutazione automatizzato.**

*Basta clonare il repo, verificare i link, e sei pronto per l'evento! 🚀*
