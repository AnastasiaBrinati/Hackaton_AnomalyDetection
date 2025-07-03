# Track 1: Anomaly Detection in Live Events (con FMA)

## Descrizione
Questo script implementa la Track 1 dell'hackathon SIAE, integrando i metadati FMA (Free Music Archive) per migliorare l'anomaly detection negli eventi live.

## Caratteristiche
- **Dataset sintetico eventi live** con anomalie inserite
- **Integrazione metadati FMA** per informazioni musicali (generi, artisti, durata)
- **Isolation Forest** per anomaly detection
- **DBSCAN** per clustering venues
- **Visualizzazioni avanzate** con analisi dei generi musicali
- **Metriche di performance** dettagliate

## Installazione

### 1. Clona il repository e naviga nella directory
```bash
cd "Anomaly Detection"
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Esegui lo script
```bash
python track1_anomaly_detection.py
```

## Funzionalità principali

### Dataset Generation
- Genera eventi live sintetici con informazioni realistiche
- Integra metadati FMA per generi musicali, artisti e durata
- Inserisce anomalie di vari tipi:
  - `impossible_attendance`: Attendance > capacity
  - `revenue_mismatch`: Revenue non proporzionale all'attendance
  - `excessive_songs`: Numero anomalo di brani
  - `suspicious_timing`: Eventi in orari sospetti
  - `genre_mismatch`: Generi poco popolari con revenue alta
  - `artist_overload`: Troppi artisti per piccoli eventi

### Feature Engineering
- Features tradizionali: revenue_per_person, occupancy_rate, etc.
- Features musicali FMA:
  - Encoding dei generi musicali
  - Popolarità di generi e artisti
  - Durata stimata degli eventi
  - Numero di artisti per evento
  - Specializzazione dei venues

### Anomaly Detection
- **Isolation Forest** con features integrate FMA
- **DBSCAN** per clustering venues
- Valutazione performance con precision, recall, F1-score

### Visualizzazioni
- Distribuzione anomaly scores
- Scatter plots multi-dimensionali
- Analisi per genere musicale
- Clustering venues
- Confusion matrix
- Distribuzione generi musicali

## Output Files

Lo script genera i seguenti file:

1. **live_events_with_anomalies.csv**: Dataset completo con anomalie rilevate
2. **venue_clustering_results.csv**: Risultati clustering venues
3. **genre_analysis.csv**: Analisi anomalie per genere musicale
4. **anomaly_detection_results.png**: Dashboard visualizzazioni principali
5. **genre_distribution.png**: Distribuzione generi musicali

## Esempi di Utilizzo

### Eseguire con parametri personalizzati
```python
# Modifica la funzione main() per personalizzare
def main():
    # Genera più eventi
    df = generate_live_events_dataset(n_events=20000, music_data=music_data)
    
    # Cambia soglia contaminazione
    df, iso_forest, scaler, feature_cols = apply_isolation_forest(df, contamination=0.15)
```

### Analizzare specifici generi musicali
```python
# Filtra per genere specifico
rock_events = df[df['event_genre'] == 'Rock']
anomaly_rate = rock_events['is_anomaly_detected'].mean()
print(f"Tasso anomalie Rock: {anomaly_rate:.2%}")
```

## Metriche Performance

Il script fornisce metriche dettagliate:
- **Precision**: Percentuale di anomalie rilevate corrette
- **Recall**: Percentuale di anomalie vere identificate
- **F1-Score**: Media armonica di precision e recall
- **Analisi per tipo di anomalia**: Performance per ogni categoria
- **Analisi per genere musicale**: Tasso anomalie per genere

## Personalizzazione

### Aggiungere nuovi tipi di anomalie
```python
# Nel generate_live_events_dataset()
elif anomaly_type == "new_anomaly_type":
    # Implementa logica anomalia personalizzata
    pass
```

### Modificare features per anomaly detection
```python
# Nel apply_isolation_forest()
custom_features = ['feature1', 'feature2', 'feature3']
feature_cols.extend(custom_features)
```

## Problemi Comuni

### Download FMA fallisce
Se il download dei metadati FMA fallisce, lo script automaticamente:
- Crea un dataset FMA sintetico
- Continua l'analisi con dati simulati
- Mantiene la stessa struttura di output

### Errori di importazione
Assicurati di avere installato tutte le dipendenze:
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn
```

### Problemi di memoria
Per dataset grandi, riduci il numero di eventi:
```python
df = generate_live_events_dataset(n_events=5000)  # Invece di 50000
```

## Prossimi Sviluppi

- Integrazione con dati reali SIAE
- Algoritmi di ensemble per migliorare performance
- Analisi time series per pattern temporali
- API REST per deployment in produzione

## Contatti

Per domande o supporto sull'implementazione della Track 1. 