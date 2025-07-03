#!/usr/bin/env python3
"""
SIAE Anomaly Detection Hackathon - Track 1: Live Events
Anomaly Detection in Live Events using Isolation Forest and DBSCAN

This script implements the complete pipeline for Track 1:
1. Generate synthetic live events dataset
2. Perform anomaly detection using Isolation Forest
3. Cluster venues using DBSCAN
4. Visualize results and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os
import urllib.request
import zipfile
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def download_fma_metadata():
    """
    Scarica i metadati FMA se non già presenti
    """
    print("Verificando disponibilità metadati FMA...")
    
    if not os.path.exists('fma_metadata.zip') and not os.path.exists('fma_metadata'):
        print("Scaricando metadati FMA (342 MB)...")
        try:
            urllib.request.urlretrieve(
                'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',
                'fma_metadata.zip'
            )
            print("Download completato!")
        except Exception as e:
            print(f"Errore durante il download: {e}")
            print("Creando dataset FMA sintetico...")
            return create_synthetic_fma_data()
    
    if not os.path.exists('fma_metadata'):
        print("Estraendo metadati FMA...")
        try:
            with zipfile.ZipFile('fma_metadata.zip', 'r') as zip_ref:
                zip_ref.extractall('fma_metadata')
            print("Estrazione completata!")
        except Exception as e:
            print(f"Errore durante l'estrazione: {e}")
            return create_synthetic_fma_data()
    
    return load_fma_metadata()

def create_synthetic_fma_data():
    """
    Crea un dataset FMA sintetico quando il download non è disponibile
    """
    print("Creando dataset FMA sintetico...")
    
    genres = ['Electronic', 'Rock', 'Hip-Hop', 'Folk', 'Pop', 'Experimental', 
              'Jazz', 'Classical', 'Country', 'Blues', 'International', 'Ambient']
    
    artists = [f"Artist_{i}" for i in range(1, 1000)]
    
    tracks = []
    for i in range(10000):
        track = {
            'track_id': i,
            'artist_name': random.choice(artists),
            'track_title': f"Track_{i}",
            'genre_top': random.choice(genres),
            'track_duration': random.randint(120, 400),  # secondi
            'track_listens': random.randint(100, 100000),
            'track_favorites': random.randint(0, 1000),
            'track_date_created': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000)),
            'artist_active_year_begin': random.randint(1990, 2020),
            'artist_latitude': random.uniform(35.0, 70.0),  # Europa
            'artist_longitude': random.uniform(-10.0, 30.0)
        }
        tracks.append(track)
    
    return pd.DataFrame(tracks)

def load_fma_metadata():
    """
    Carica i metadati FMA reali dal file scaricato
    """
    try:
        # Carica il file principale dei metadati
        tracks_file = 'fma_metadata/tracks.csv'
        if os.path.exists(tracks_file):
            print("Caricando metadati FMA reali...")
            tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
            
            # Semplifica le colonne multi-level
            tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
            
            # Seleziona colonne rilevanti
            relevant_cols = []
            for col in tracks.columns:
                if any(keyword in col.lower() for keyword in ['genre', 'artist', 'track', 'title', 'duration']):
                    relevant_cols.append(col)
            
            if relevant_cols:
                tracks_simplified = tracks[relevant_cols].copy()
                tracks_simplified['track_id'] = tracks_simplified.index
                print(f"Caricati {len(tracks_simplified)} track con metadati FMA")
                return tracks_simplified
        
        # Fallback al dataset sintetico
        print("File FMA non trovato, usando dataset sintetico...")
        return create_synthetic_fma_data()
        
    except Exception as e:
        print(f"Errore nel caricamento FMA: {e}")
        return create_synthetic_fma_data()

def process_fma_for_events(fma_df):
    """
    Processa i metadati FMA per l'uso negli eventi live
    """
    print("Processando metadati FMA per eventi live...")
    
    # Estrai informazioni sui generi
    if 'genre_top' in fma_df.columns:
        genre_col = 'genre_top'
    else:
        # Cerca colonna genere
        genre_cols = [col for col in fma_df.columns if 'genre' in col.lower()]
        genre_col = genre_cols[0] if genre_cols else None
    
    if genre_col:
        genre_stats = fma_df[genre_col].value_counts()
        print(f"Generi musicali disponibili: {len(genre_stats)}")
        print(f"Top 5 generi: {genre_stats.head().index.tolist()}")
    
    # Crea un mapping semplificato per gli eventi
    event_music_data = {
        'genres': fma_df[genre_col].dropna().unique().tolist() if genre_col else [],
        'genre_popularity': fma_df[genre_col].value_counts().to_dict() if genre_col else {},
        'artists': fma_df.get('artist_name', pd.Series()).dropna().unique().tolist()[:500],  # Limita artisti
        'avg_track_duration': fma_df.get('track_duration', pd.Series()).mean() if 'track_duration' in fma_df.columns else 240
    }
    
    return event_music_data

def generate_live_events_dataset(n_events=50000, music_data=None):
    """
    Genera un dataset sintetico di eventi live con anomalie inserite
    Integra metadati FMA per informazioni musicali
    """
    print(f"Generando dataset con {n_events} eventi...")
    
    venues = [f"Venue_{i}" for i in range(1, 501)]
    cities = ["Milano", "Roma", "Napoli", "Torino", "Bologna", "Firenze", 
              "Palermo", "Genova", "Bari", "Venezia"]
    
    # Usa dati musicali FMA se disponibili
    if music_data:
        genres = music_data['genres']
        genre_popularity = music_data['genre_popularity']
        artists = music_data['artists']
        avg_duration = music_data['avg_track_duration']
    else:
        genres = ['Rock', 'Pop', 'Jazz', 'Electronic', 'Classical']
        genre_popularity = {genre: 1 for genre in genres}
        artists = [f"Artist_{i}" for i in range(1, 100)]
        avg_duration = 240
    
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
        
        # Informazioni musicali da FMA
        event_genre = random.choice(genres)
        primary_artist = random.choice(artists)
        # Generi popolari possono generare più revenue
        genre_multiplier = 1.0
        if event_genre in genre_popularity:
            popularity = genre_popularity[event_genre]
            # Normalizza la popolarità
            max_popularity = max(genre_popularity.values())
            genre_multiplier = 0.8 + (popularity / max_popularity) * 0.4
        
        base_revenue *= genre_multiplier
        
        # Durata media dei brani (in minuti)
        avg_song_duration = avg_duration / 60  # converti secondi in minuti
        estimated_event_duration = n_songs * avg_song_duration
        
        # Numero di artisti (la maggior parte eventi ha 1-3 artisti)
        n_artists = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
        
        # Inserisci anomalie (10% dei casi)
        anomaly_type = None
        if random.random() < 0.1:
            anomaly_type = random.choice([
                "duplicate_declaration",
                "impossible_attendance", 
                "revenue_mismatch",
                "excessive_songs",
                "suspicious_timing",
                "genre_mismatch",
                "artist_overload"
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
            elif anomaly_type == "genre_mismatch":
                # Genre non popolare con revenue molto alta
                unpopular_genres = [g for g in genres if genre_popularity.get(g, 0) < 
                                  np.percentile(list(genre_popularity.values()), 25)]
                if unpopular_genres:
                    event_genre = random.choice(unpopular_genres)
                    base_revenue *= 3  # Revenue anomalamente alta
            elif anomaly_type == "artist_overload":
                # Troppi artisti per un evento piccolo
                n_artists = random.randint(8, 15)
        
        events.append({
            'event_id': f'EVT_{i:06d}',
            'venue_id': venue,
            'city': city,
            'event_date': event_date,
            'capacity': capacity,
            'attendance': attendance,
            'n_songs_declared': n_songs,
            'total_revenue': round(base_revenue, 2),
            'event_genre': event_genre,
            'primary_artist': primary_artist,
            'n_artists': n_artists,
            'estimated_duration_minutes': round(estimated_event_duration, 1),
            'anomaly_type': anomaly_type
        })
    
    df = pd.DataFrame(events)
    print(f"Dataset generato con {len(df)} eventi")
    print(f"Anomalie inserite: {df['anomaly_type'].notna().sum()}")
    print(f"Generi musicali negli eventi: {df['event_genre'].nunique()}")
    return df

def feature_engineering(df):
    """
    Crea features aggiuntive per l'anomaly detection
    Include features basate sui metadati FMA
    """
    print("Eseguendo feature engineering...")
    
    # Features base
    df['revenue_per_person'] = df['total_revenue'] / df['attendance']
    df['occupancy_rate'] = df['attendance'] / df['capacity']
    df['songs_per_person'] = df['n_songs_declared'] / df['attendance']
    
    # Features temporali
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['hour'] = df['event_date'].dt.hour
    df['day_of_week'] = df['event_date'].dt.dayofweek
    df['month'] = df['event_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Features musicali (da FMA)
    if 'event_genre' in df.columns:
        # Encoding dei generi musicali
        genre_encoder = LabelEncoder()
        df['genre_encoded'] = genre_encoder.fit_transform(df['event_genre'])
        
        # Popolarità del genere
        genre_counts = df['event_genre'].value_counts()
        df['genre_popularity'] = df['event_genre'].map(genre_counts)
        
        # Features per numero di artisti
        df['artists_per_capacity'] = df['n_artists'] / df['capacity']
        df['is_single_artist'] = (df['n_artists'] == 1).astype(int)
        df['is_multiple_artists'] = (df['n_artists'] > 3).astype(int)
        
        # Features per durata
        if 'estimated_duration_minutes' in df.columns:
            df['duration_per_song'] = df['estimated_duration_minutes'] / df['n_songs_declared']
            df['duration_per_person'] = df['estimated_duration_minutes'] / df['attendance']
            df['is_long_event'] = (df['estimated_duration_minutes'] > 180).astype(int)  # >3 ore
        
        # Features per artista
        artist_counts = df['primary_artist'].value_counts()
        df['artist_popularity'] = df['primary_artist'].map(artist_counts)
        
        # Features combinati
        df['revenue_per_artist'] = df['total_revenue'] / df['n_artists']
        df['songs_per_artist'] = df['n_songs_declared'] / df['n_artists']
    
    # Features per venue
    venue_stats = df.groupby('venue_id').agg({
        'attendance': ['mean', 'std', 'count'],
        'total_revenue': 'mean',
        'capacity': 'mean'
    }).round(2)
    
    venue_stats.columns = ['venue_avg_attendance', 'venue_std_attendance', 
                          'venue_event_count', 'venue_avg_revenue', 'venue_avg_capacity']
    
    df = df.merge(venue_stats, left_on='venue_id', right_index=True)
    
    # Anomalie rispetto alla media del venue
    df['attendance_vs_venue_avg'] = df['attendance'] / df['venue_avg_attendance']
    df['revenue_vs_venue_avg'] = df['total_revenue'] / df['venue_avg_revenue']
    
    # Features per genere musicale e venue
    if 'event_genre' in df.columns:
        venue_genre_stats = df.groupby(['venue_id', 'event_genre']).size().reset_index(name='venue_genre_count')
        venue_genre_diversity = df.groupby('venue_id')['event_genre'].nunique().reset_index()
        venue_genre_diversity.columns = ['venue_id', 'venue_genre_diversity']
        
        df = df.merge(venue_genre_diversity, on='venue_id', how='left')
        df['venue_genre_diversity'] = df['venue_genre_diversity'].fillna(1)
        
        # Specializzazione del venue (alta diversità vs specializzazione)
        df['venue_specialization'] = 1 / df['venue_genre_diversity']
    
    print(f"Features create: {df.shape[1]} colonne totali")
    return df

def apply_isolation_forest(df, contamination=0.1):
    """
    Applica Isolation Forest per anomaly detection
    """
    print("Applicando Isolation Forest...")
    
    # Seleziona features numeriche per l'anomaly detection
    feature_cols = ['attendance', 'capacity', 'n_songs_declared', 
                   'revenue_per_person', 'occupancy_rate', 'songs_per_person',
                   'hour', 'day_of_week', 'month', 'is_weekend',
                   'venue_avg_attendance', 'venue_event_count',
                   'attendance_vs_venue_avg', 'revenue_vs_venue_avg']
    
    # Features musicali FMA
    fma_features = ['genre_encoded', 'genre_popularity', 'n_artists',
                   'artists_per_capacity', 'is_single_artist', 'is_multiple_artists',
                   'duration_per_song', 'duration_per_person', 'is_long_event',
                   'artist_popularity', 'revenue_per_artist', 'songs_per_artist',
                   'venue_genre_diversity', 'venue_specialization']
    
    # Aggiungi features FMA se disponibili
    feature_cols.extend([col for col in fma_features if col in df.columns])
    
    # Filtra features che esistono nel dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0)
    
    # Normalizza features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Applica Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination, 
        random_state=42,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly_detected'] = predictions == -1
    
    print(f"Anomalie rilevate: {df['is_anomaly_detected'].sum()}")
    print(f"Features utilizzate: {len(feature_cols)}")
    
    return df, iso_forest, scaler, feature_cols

def apply_dbscan_clustering(df):
    """
    Applica DBSCAN per clustering dei venue
    """
    print("Applicando DBSCAN per clustering venue...")
    
    # Aggrega dati per venue
    venue_features = df.groupby('venue_id').agg({
        'attendance': ['mean', 'std'],
        'total_revenue': 'mean',
        'capacity': 'mean',
        'n_songs_declared': 'mean',
        'occupancy_rate': 'mean',
        'is_anomaly_detected': 'sum'
    }).round(2)
    
    venue_features.columns = ['avg_attendance', 'std_attendance', 'avg_revenue', 
                             'avg_capacity', 'avg_songs', 'avg_occupancy', 'anomaly_count']
    
    # Normalizza per clustering
    scaler_venue = StandardScaler()
    venue_scaled = scaler_venue.fit_transform(venue_features.fillna(0))
    
    # Applica DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    venue_clusters = dbscan.fit_predict(venue_scaled)
    
    venue_features['cluster'] = venue_clusters
    
    print(f"Cluster identificati: {len(set(venue_clusters)) - (1 if -1 in venue_clusters else 0)}")
    print(f"Venue outlier (noise): {sum(venue_clusters == -1)}")
    
    return venue_features, dbscan

def evaluate_performance(df):
    """
    Valuta le performance dell'anomaly detection
    """
    print("\n=== VALUTAZIONE PERFORMANCE ===")
    
    # Crea ground truth
    y_true = df['anomaly_type'].notna().astype(int)
    y_pred = df['is_anomaly_detected'].astype(int)
    
    # Metriche
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Precision, Recall, F1
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Analisi per tipo di anomalia
    print("\n=== ANALISI PER TIPO DI ANOMALIA ===")
    anomaly_analysis = df[df['anomaly_type'].notna()].groupby('anomaly_type').agg({
        'is_anomaly_detected': ['sum', 'count']
    })
    anomaly_analysis.columns = ['detected', 'total']
    anomaly_analysis['detection_rate'] = anomaly_analysis['detected'] / anomaly_analysis['total']
    print(anomaly_analysis)
    
    return precision, recall, f1

def create_visualizations(df, venue_features):
    """
    Crea visualizzazioni dei risultati
    """
    print("Creando visualizzazioni...")
    
    # Setup matplotlib
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('SIAE Anomaly Detection - Track 1: Live Events Analysis (con FMA)', fontsize=16)
    
    # 1. Distribuzione anomaly scores
    axes[0, 0].hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df[df['is_anomaly_detected']]['anomaly_score'].max(), 
                      color='red', linestyle='--', label='Soglia anomalia')
    axes[0, 0].set_title('Distribuzione Anomaly Scores')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequenza')
    axes[0, 0].legend()
    
    # 2. Attendance vs Revenue (con anomalie evidenziate)
    normal_events = df[~df['is_anomaly_detected']]
    anomaly_events = df[df['is_anomaly_detected']]
    
    axes[0, 1].scatter(normal_events['attendance'], normal_events['total_revenue'], 
                      alpha=0.6, s=20, label='Eventi normali')
    axes[0, 1].scatter(anomaly_events['attendance'], anomaly_events['total_revenue'], 
                      alpha=0.8, s=30, color='red', label='Anomalie rilevate')
    axes[0, 1].set_title('Attendance vs Revenue')
    axes[0, 1].set_xlabel('Attendance')
    axes[0, 1].set_ylabel('Total Revenue')
    axes[0, 1].legend()
    
    # 3. Distribuzione generi musicali
    if 'event_genre' in df.columns:
        genre_counts = df['event_genre'].value_counts().head(10)
        axes[0, 2].bar(range(len(genre_counts)), genre_counts.values, color='lightgreen')
        axes[0, 2].set_title('Top 10 Generi Musicali')
        axes[0, 2].set_xlabel('Genere')
        axes[0, 2].set_ylabel('Numero Eventi')
        axes[0, 2].set_xticks(range(len(genre_counts)))
        axes[0, 2].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    
    # 4. Anomalie per genere musicale
    if 'event_genre' in df.columns:
        genre_anomaly_rate = df.groupby('event_genre')['is_anomaly_detected'].agg(['sum', 'count', 'mean'])
        genre_anomaly_rate = genre_anomaly_rate[genre_anomaly_rate['count'] >= 10]  # Solo generi con >10 eventi
        top_anomaly_genres = genre_anomaly_rate.nlargest(10, 'mean')
        
        axes[1, 0].bar(range(len(top_anomaly_genres)), top_anomaly_genres['mean'], color='orange')
        axes[1, 0].set_title('Tasso Anomalie per Genere')
        axes[1, 0].set_xlabel('Genere')
        axes[1, 0].set_ylabel('Tasso Anomalie')
        axes[1, 0].set_xticks(range(len(top_anomaly_genres)))
        axes[1, 0].set_xticklabels(top_anomaly_genres.index, rotation=45, ha='right')
    
    # 5. Numero artisti vs Revenue
    if 'n_artists' in df.columns:
        axes[1, 1].scatter(normal_events['n_artists'], normal_events['total_revenue'], 
                          alpha=0.6, s=20, label='Eventi normali')
        axes[1, 1].scatter(anomaly_events['n_artists'], anomaly_events['total_revenue'], 
                          alpha=0.8, s=30, color='red', label='Anomalie rilevate')
        axes[1, 1].set_title('Numero Artisti vs Revenue')
        axes[1, 1].set_xlabel('Numero Artisti')
        axes[1, 1].set_ylabel('Total Revenue')
        axes[1, 1].legend()
    
    # 6. Clustering venues
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    for i, cluster in enumerate(venue_features['cluster'].unique()):
        if cluster == -1:
            continue
        cluster_data = venue_features[venue_features['cluster'] == cluster]
        axes[1, 2].scatter(cluster_data['avg_attendance'], cluster_data['avg_revenue'], 
                          alpha=0.7, s=50, color=cluster_colors[i % len(cluster_colors)], 
                          label=f'Cluster {cluster}')
    
    # Venue outlier
    outlier_venues = venue_features[venue_features['cluster'] == -1]
    if not outlier_venues.empty:
        axes[1, 2].scatter(outlier_venues['avg_attendance'], outlier_venues['avg_revenue'], 
                          alpha=0.7, s=50, color='black', marker='x', label='Outlier venues')
    
    axes[1, 2].set_title('Clustering Venues')
    axes[1, 2].set_xlabel('Average Attendance')
    axes[1, 2].set_ylabel('Average Revenue')
    axes[1, 2].legend()
    
    # 7. Durata eventi vs Anomalie
    if 'estimated_duration_minutes' in df.columns:
        axes[2, 0].scatter(normal_events['estimated_duration_minutes'], normal_events['attendance'], 
                          alpha=0.6, s=20, label='Eventi normali')
        axes[2, 0].scatter(anomaly_events['estimated_duration_minutes'], anomaly_events['attendance'], 
                          alpha=0.8, s=30, color='red', label='Anomalie rilevate')
        axes[2, 0].set_title('Durata Eventi vs Attendance')
        axes[2, 0].set_xlabel('Durata (minuti)')
        axes[2, 0].set_ylabel('Attendance')
        axes[2, 0].legend()
    
    # 8. Distribuzione anomalie per ora del giorno
    hour_anomalies = df[df['is_anomaly_detected']]['hour'].value_counts().sort_index()
    axes[2, 1].bar(hour_anomalies.index, hour_anomalies.values, color='lightcoral')
    axes[2, 1].set_title('Distribuzione Anomalie per Ora')
    axes[2, 1].set_xlabel('Ora del giorno')
    axes[2, 1].set_ylabel('Numero anomalie')
    
    # 9. Heatmap matrice di confusione
    cm = confusion_matrix(df['anomaly_type'].notna(), df['is_anomaly_detected'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 2])
    axes[2, 2].set_title('Confusion Matrix')
    axes[2, 2].set_xlabel('Predicted')
    axes[2, 2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Grafico aggiuntivo per analisi FMA
    if 'event_genre' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Crea un grafico a torta per i generi musicali
        genre_counts = df['event_genre'].value_counts().head(8)
        others = df['event_genre'].value_counts().iloc[8:].sum()
        if others > 0:
            genre_counts['Altri'] = others
        
        plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribuzione Generi Musicali negli Eventi Live')
        plt.axis('equal')
        plt.savefig('genre_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Funzione principale che esegue l'intero pipeline
    """
    print("=== SIAE ANOMALY DETECTION HACKATHON ===")
    print("Track 1: Anomaly Detection in Live Events")
    print("==========================================\n")
    
    # 1. Scarica e processa metadati FMA
    fma_df = download_fma_metadata()
    music_data = process_fma_for_events(fma_df)
    
    # 2. Genera dataset con informazioni musicali FMA
    df = generate_live_events_dataset(n_events=10000, music_data=music_data)
    
    # 3. Feature engineering (include features FMA)
    df = feature_engineering(df)
    
    # 4. Applica Isolation Forest
    df, iso_forest, scaler, feature_cols = apply_isolation_forest(df, contamination=0.1)
    
    # 5. Applica DBSCAN clustering
    venue_features, dbscan = apply_dbscan_clustering(df)
    
    # 6. Valuta performance
    precision, recall, f1 = evaluate_performance(df)
    
    # 7. Crea visualizzazioni
    create_visualizations(df, venue_features)
    
    # 8. Salva risultati
    print("\nSalvataggio risultati...")
    df.to_csv('live_events_with_anomalies.csv', index=False)
    venue_features.to_csv('venue_clustering_results.csv')
    
    # Salva anche informazioni sui generi musicali
    genre_analysis = df.groupby('event_genre').agg({
        'is_anomaly_detected': ['sum', 'count'],
        'total_revenue': 'mean',
        'attendance': 'mean'
    })
    genre_analysis.to_csv('genre_analysis.csv')
    
    print("\n=== RIEPILOGO RISULTATI ===")
    print(f"Dataset generato: {len(df)} eventi")
    print(f"Anomalie vere: {df['anomaly_type'].notna().sum()}")
    print(f"Anomalie rilevate: {df['is_anomaly_detected'].sum()}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Venue clusters: {len(venue_features['cluster'].unique()) - (1 if -1 in venue_features['cluster'].unique() else 0)}")
    
    # Statistiche musicali
    print(f"\n=== STATISTICHE MUSICALI (FMA) ===")
    print(f"Generi musicali totali: {df['event_genre'].nunique()}")
    print(f"Top 5 generi per numero eventi:")
    top_genres = df['event_genre'].value_counts().head()
    for genre, count in top_genres.items():
        anomaly_rate = df[df['event_genre'] == genre]['is_anomaly_detected'].mean()
        print(f"  {genre}: {count} eventi (anomaly rate: {anomaly_rate:.1%})")
    
    print("\nFile salvati:")
    print("- live_events_with_anomalies.csv")
    print("- venue_clustering_results.csv")
    print("- genre_analysis.csv")
    print("- anomaly_detection_results.png")
    print("- genre_distribution.png")
    
    print("\n=== ANALISI COMPLETATA ===")
    
    return df, venue_features, iso_forest, dbscan

if __name__ == "__main__":
    # Installazione automatica dipendenze se necessario
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as e:
        print(f"Errore import: {e}")
        print("Installare le dipendenze con:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn")
        exit(1)
    
    # Esegui analisi
    df, venue_features, iso_forest, dbscan = main() 