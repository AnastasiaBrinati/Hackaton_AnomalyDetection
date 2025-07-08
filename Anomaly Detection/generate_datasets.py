#!/usr/bin/env python3
"""
SIAE Hackathon - Dataset Generator
Genera dataset identici per tutti i partecipanti per garantire performance comparabili

Questo script genera:
- Track 1: 50,000 eventi live con 5 tipi di anomalie
- Track 2: 5,000 documenti con 5 tipi di frodi  
- Track 3: 25,000 tracce musicali con 5 tipi di anomalie
- Track 4: 15,000 opere creative con 5 tipi di violazioni

Tutti i dataset sono salvati in formato CSV nella cartella datasets/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import hashlib
import urllib.request
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# IMPORTANTISSIMO: Seed identico per tutti i partecipanti
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def ensure_datasets_dir():
    """Crea la directory datasets se non esiste"""
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir

def download_fma_metadata():
    """Scarica i metadati FMA se non già presenti"""
    print("🎵 Verificando disponibilità metadati FMA...")
    
    fma_dir = Path('fma_metadata')
    if fma_dir.exists():
        print("✅ Metadati FMA già disponibili")
        return load_fma_metadata()
    
    print("📥 Scaricando metadati FMA (342 MB)...")
    try:
        urllib.request.urlretrieve(
            'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',
            'fma_metadata.zip'
        )
        print("✅ Download completato!")
        
        print("📂 Estraendo archivio...")
        with zipfile.ZipFile('fma_metadata.zip', 'r') as zip_ref:
            zip_ref.extractall('fma_metadata')
        print("✅ Estrazione completata!")
        
        return load_fma_metadata()
        
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        print("🔄 Generando dataset FMA sintetico...")
        return create_synthetic_fma_data()

def load_fma_metadata():
    """Carica i metadati FMA reali dal file scaricato"""
    try:
        tracks_file = Path('fma_metadata/tracks.csv')
        if tracks_file.exists():
            print("📖 Caricando metadati FMA reali...")
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
                print(f"✅ Caricati {len(tracks_simplified)} track con metadati FMA reali")
                return tracks_simplified
                
    except Exception as e:
        print(f"❌ Errore nel caricamento FMA: {e}")
    
    # Fallback al dataset sintetico
    print("🔄 Generando dataset FMA sintetico...")
    return create_synthetic_fma_data()

def create_synthetic_fma_data():
    """Crea un dataset FMA sintetico quando il download non è disponibile"""
    print("🎼 Creando dataset FMA sintetico...")
    
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

def generate_track1_dataset(n_events=50000, music_data=None):
    """Genera dataset Track 1: Live Events Anomaly Detection"""
    print(f"🎪 Generando Track 1: {n_events} eventi live...")
    
    # Reset seed per consistenza
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    venues = [f"Venue_{i}" for i in range(1, 501)]
    cities = ["Milano", "Roma", "Napoli", "Torino", "Bologna", "Firenze", 
              "Palermo", "Genova", "Bari", "Venezia"]
    
    # Usa dati musicali FMA se disponibili
    if music_data is not None:
        if 'genre_top' in music_data.columns:
            genres = music_data['genre_top'].dropna().unique().tolist()
        else:
            genres = ['Rock', 'Pop', 'Jazz', 'Electronic', 'Classical']
        
        if 'artist_name' in music_data.columns:
            artists = music_data['artist_name'].dropna().unique().tolist()[:500]
        else:
            artists = [f"Artist_{i}" for i in range(1, 100)]
    else:
        genres = ['Rock', 'Pop', 'Jazz', 'Electronic', 'Classical']
        artists = [f"Artist_{i}" for i in range(1, 100)]
    
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
        
        # Genere musicale prevalente
        genre = random.choice(genres)
        
        event = {
            'event_id': f"EVENT_{i+1:06d}",
            'venue': venue,
            'city': city,
            'event_date': event_date,
            'capacity': capacity,
            'attendance': attendance,
            'n_songs_declared': n_songs,
            'total_revenue': base_revenue,
            'genre': genre,
            'main_artist': random.choice(artists),
            'event_duration_hours': random.uniform(2, 8),
            'ticket_price_avg': base_revenue / attendance if attendance > 0 else 0,
            'anomaly_type': None,
            'is_anomaly': False
        }
        
        events.append(event)
    
    df = pd.DataFrame(events)
    
    # Genera anomalie - STESSO METODO DELLE SOLUZIONI
    print("🚨 Generando anomalie Track 1...")
    
    # Tipo 1: Duplicate declaration (stesso venue+data)
    duplicate_mask = np.random.random(len(df)) < 0.02
    df.loc[duplicate_mask, 'anomaly_type'] = 'duplicate_declaration'
    df.loc[duplicate_mask, 'is_anomaly'] = True
    
    # Tipo 2: Impossible attendance (>capacity)
    impossible_mask = np.random.random(len(df)) < 0.03
    df.loc[impossible_mask, 'anomaly_type'] = 'impossible_attendance'
    df.loc[impossible_mask, 'is_anomaly'] = True
    df.loc[impossible_mask, 'attendance'] = df.loc[impossible_mask, 'capacity'] * np.random.uniform(1.2, 2.0, sum(impossible_mask))
    
    # Tipo 3: Revenue mismatch
    revenue_mask = np.random.random(len(df)) < 0.025
    df.loc[revenue_mask, 'anomaly_type'] = 'revenue_mismatch'
    df.loc[revenue_mask, 'is_anomaly'] = True
    df.loc[revenue_mask, 'total_revenue'] *= np.random.uniform(0.1, 0.3, sum(revenue_mask))
    
    # Tipo 4: Excessive songs (>40)
    songs_mask = np.random.random(len(df)) < 0.02
    df.loc[songs_mask, 'anomaly_type'] = 'excessive_songs'
    df.loc[songs_mask, 'is_anomaly'] = True
    df.loc[songs_mask, 'n_songs_declared'] = np.random.randint(50, 100, sum(songs_mask))
    
    # Tipo 5: Suspicious timing (2-6 AM)
    timing_mask = np.random.random(len(df)) < 0.015
    df.loc[timing_mask, 'anomaly_type'] = 'suspicious_timing'
    df.loc[timing_mask, 'is_anomaly'] = True
    
    print(f"✅ Track 1 generato: {len(df)} eventi, {df['is_anomaly'].sum()} anomalie ({df['is_anomaly'].mean():.2%})")
    return df

def generate_track2_dataset(n_documents=5000):
    """Genera dataset Track 2: Document Fraud Detection"""
    print(f"📄 Generando Track 2: {n_documents} documenti...")
    
    # Reset seed per consistenza
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    document_types = [
        'Contratto_Editore', 'Licenza_Esecuzione', 'Dichiarazione_Musica_Live',
        'Cessione_Diritti', 'Registrazione_Opera', 'Richiesta_Risarcimento'
    ]
    
    documents = []
    
    for i in range(n_documents):
        doc_type = random.choice(document_types)
        
        doc_data = {
            'document_id': f"SIAE_{i+1:06d}",
            'document_type': doc_type,
            'creation_date': datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)),
            'page_count': random.randint(1, 20),
            'file_size_kb': np.random.exponential(500) + 50,
            'resolution_dpi': np.random.choice([150, 200, 300, 600], p=[0.2, 0.3, 0.4, 0.1]),
            'text_blocks_count': np.random.poisson(8) + 2,
            'signature_regions': random.randint(1, 4),
            'logo_elements': random.randint(0, 3),
            'text_confidence_avg': random.uniform(0.7, 0.99),
            'text_confidence_std': random.uniform(0.02, 0.15),
            'word_count': np.random.poisson(200) + 50,
            'siae_watermark_detected': np.random.choice([0, 1], p=[0.15, 0.85]),
            'official_seal_detected': np.random.choice([0, 1], p=[0.25, 0.75]),
            'pixel_noise_level': random.uniform(0.001, 0.05),
            'edge_sharpness': random.uniform(0.6, 1.0),
            'metadata_consistency': random.uniform(0.8, 1.0),
            'fraud_type': None,
            'is_fraudulent': False
        }
        
        documents.append(doc_data)
    
    df = pd.DataFrame(documents)
    
    # Genera frodi - STESSO METODO DELLE SOLUZIONI
    print("🚨 Generando frodi Track 2...")
    
    # Tipo 1: Alterazioni digitali
    digital_mask = np.random.random(len(df)) < 0.08
    df.loc[digital_mask, 'fraud_type'] = 'digital_alteration'
    df.loc[digital_mask, 'is_fraudulent'] = True
    df.loc[digital_mask, 'pixel_noise_level'] *= np.random.uniform(2.0, 5.0, sum(digital_mask))
    
    # Tipo 2: Firme contraffatte
    signature_mask = np.random.random(len(df)) < 0.05
    df.loc[signature_mask, 'fraud_type'] = 'signature_forgery'
    df.loc[signature_mask, 'is_fraudulent'] = True
    df.loc[signature_mask, 'signature_regions'] = 0
    
    # Tipo 3: Template fraud
    template_mask = np.random.random(len(df)) < 0.04
    df.loc[template_mask, 'fraud_type'] = 'template_fraud'
    df.loc[template_mask, 'is_fraudulent'] = True
    df.loc[template_mask, 'siae_watermark_detected'] = 0
    df.loc[template_mask, 'official_seal_detected'] = 0
    
    # Tipo 4: Manipolazione metadati
    metadata_mask = np.random.random(len(df)) < 0.03
    df.loc[metadata_mask, 'fraud_type'] = 'metadata_manipulation'
    df.loc[metadata_mask, 'is_fraudulent'] = True
    df.loc[metadata_mask, 'metadata_consistency'] *= np.random.uniform(0.3, 0.6, sum(metadata_mask))
    
    # Tipo 5: Inconsistenza qualità
    quality_mask = np.random.random(len(df)) < 0.02
    df.loc[quality_mask, 'fraud_type'] = 'quality_inconsistency'
    df.loc[quality_mask, 'is_fraudulent'] = True
    df.loc[quality_mask, 'edge_sharpness'] *= np.random.uniform(0.3, 0.6, sum(quality_mask))
    
    print(f"✅ Track 2 generato: {len(df)} documenti, {df['is_fraudulent'].sum()} frodi ({df['is_fraudulent'].mean():.2%})")
    return df

def generate_track3_dataset(n_tracks=25000):
    """Genera dataset Track 3: Music Anomaly Detection"""
    print(f"🎵 Generando Track 3: {n_tracks} tracce musicali...")
    
    # Reset seed per consistenza
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    genres = ['Electronic', 'Rock', 'Hip-Hop', 'Folk', 'Pop', 'Experimental', 
              'Jazz', 'Classical', 'Country', 'Blues', 'International', 'Ambient',
              'Metal', 'Punk', 'Reggae', 'Indie', 'Alternative', 'Techno']
    
    subgenres = {
        'Electronic': ['House', 'Trance', 'Dubstep', 'Ambient', 'Drum & Bass'],
        'Rock': ['Alternative Rock', 'Hard Rock', 'Progressive Rock', 'Indie Rock'],
        'Hip-Hop': ['Rap', 'Trap', 'Old School', 'Conscious Hip-Hop'],
        'Jazz': ['Smooth Jazz', 'Bebop', 'Jazz Fusion', 'Free Jazz'],
        'Classical': ['Baroque', 'Romantic', 'Modern Classical', 'Chamber Music']
    }
    
    tracks = []
    artists = [f"Artist_{i:04d}" for i in range(1, 2001)]
    
    for i in range(n_tracks):
        genre = random.choice(genres)
        subgenre = random.choice(subgenres.get(genre, [genre]))
        artist = random.choice(artists)
        
        track = {
            'track_id': i,
            'artist_name': artist,
            'track_title': f"Track_{i:05d}",
            'album_title': f"Album_{i//10:04d}",
            'genre_top': genre,
            'genre_sub': subgenre,
            'track_date_created': datetime(2010, 1, 1) + timedelta(days=random.randint(0, 5000)),
            'track_duration': random.randint(120, 480),
            'track_listens': random.randint(100, 1000000),
            'track_favorites': random.randint(0, 10000),
            'track_comments': random.randint(0, 500),
            'track_downloads': random.randint(0, 50000),
            'tempo': random.uniform(60, 200),
            'loudness': random.uniform(-60, 0),
            'energy': random.uniform(0, 1),
            'danceability': random.uniform(0, 1),
            'valence': random.uniform(0, 1),
            'acousticness': random.uniform(0, 1),
            'instrumentalness': random.uniform(0, 1),
            'speechiness': random.uniform(0, 1),
            'liveness': random.uniform(0, 1),
            'artist_active_year_begin': random.randint(1950, 2020),
            'artist_latitude': random.uniform(-90, 90),
            'artist_longitude': random.uniform(-180, 180),
            'artist_location': random.choice(['US', 'UK', 'DE', 'FR', 'IT', 'ES', 'CA', 'AU']),
            'bit_rate': random.choice([128, 192, 256, 320]),
            'sample_rate': random.choice([22050, 44100, 48000]),
            'file_size': random.randint(3000, 15000),
            'anomaly_type': None,
            'is_anomaly': False
        }
        
        tracks.append(track)
    
    df = pd.DataFrame(tracks)
    
    # Genera anomalie - STESSO METODO DELLE SOLUZIONI
    print("🚨 Generando anomalie Track 3...")
    
    # Tipo 1: Plagio (similarità sospetta)
    plagio_mask = np.random.random(len(df)) < 0.03
    df.loc[plagio_mask, 'anomaly_type'] = 'plagio_similarity'
    df.loc[plagio_mask, 'is_anomaly'] = True
    df.loc[plagio_mask, 'tempo'] = 120 + np.random.normal(0, 5, sum(plagio_mask))
    df.loc[plagio_mask, 'energy'] = 0.7 + np.random.normal(0, 0.1, sum(plagio_mask))
    
    # Tipo 2: Bot streaming
    bot_mask = np.random.random(len(df)) < 0.025
    df.loc[bot_mask, 'anomaly_type'] = 'bot_streaming'
    df.loc[bot_mask, 'is_anomaly'] = True
    df.loc[bot_mask, 'track_listens'] *= np.random.uniform(10, 100, sum(bot_mask))
    df.loc[bot_mask, 'track_favorites'] *= np.random.uniform(0.1, 0.3, sum(bot_mask))
    
    # Tipo 3: Manipolazione metadati
    metadata_mask = np.random.random(len(df)) < 0.02
    df.loc[metadata_mask, 'anomaly_type'] = 'metadata_manipulation'
    df.loc[metadata_mask, 'is_anomaly'] = True
    df.loc[metadata_mask, 'track_date_created'] = datetime(2030, 1, 1)
    
    # Tipo 4: Genre mismatch
    genre_mask = np.random.random(len(df)) < 0.015
    df.loc[genre_mask, 'anomaly_type'] = 'genre_mismatch'
    df.loc[genre_mask, 'is_anomaly'] = True
    df.loc[genre_mask, 'genre_top'] = 'Classical'
    df.loc[genre_mask, 'energy'] = np.random.uniform(0.8, 1.0, sum(genre_mask))
    df.loc[genre_mask, 'danceability'] = np.random.uniform(0.8, 1.0, sum(genre_mask))
    
    # Tipo 5: Audio quality fraud
    quality_mask = np.random.random(len(df)) < 0.01
    df.loc[quality_mask, 'anomaly_type'] = 'audio_quality_fraud'
    df.loc[quality_mask, 'is_anomaly'] = True
    df.loc[quality_mask, 'bit_rate'] = 320
    df.loc[quality_mask, 'file_size'] = np.random.randint(500, 1000, sum(quality_mask))
    
    print(f"✅ Track 3 generato: {len(df)} tracce, {df['is_anomaly'].sum()} anomalie ({df['is_anomaly'].mean():.2%})")
    return df

def generate_track4_dataset(n_works=15000):
    """Genera dataset Track 4: Copyright Infringement Detection"""
    print(f"🔒 Generando Track 4: {n_works} opere creative...")
    
    # Reset seed per consistenza
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    work_types = ['Music_Track', 'Audio_Recording', 'Podcast_Episode', 'Commercial_Jingle']
    genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical']
    platforms = ['Spotify', 'YouTube', 'SoundCloud', 'Apple_Music', 'TikTok', 'Instagram']
    
    works = []
    for i in range(n_works):
        work = {
            'work_id': f"SIAE_CP_{i+1:06d}",
            'work_type': random.choice(work_types),
            'genre': random.choice(genres),
            'platform': random.choice(platforms),
            'release_date': datetime(2010, 1, 1) + timedelta(days=random.randint(0, 5000)),
            'duration_seconds': random.randint(30, 600),
            'tempo': random.uniform(60, 200),
            'spectral_centroid': random.uniform(1000, 8000),
            'mfcc_1': random.uniform(-50, 50),
            'mfcc_2': random.uniform(-30, 30),
            'play_count': random.randint(1000, 1000000),
            'like_count': random.randint(10, 10000),
            'share_count': random.randint(0, 5000),
            'revenue_generated': random.uniform(0, 50000),
            'compression_ratio': random.uniform(0.1, 0.9),
            'file_hash': hashlib.md5(f"work_{i}".encode()).hexdigest()[:16],
            'infringement_type': None,
            'is_infringement': False
        }
        works.append(work)
    
    df = pd.DataFrame(works)
    
    # Genera violazioni - STESSO METODO DELLE SOLUZIONI
    print("🚨 Generando violazioni Track 4...")
    
    # Tipo 1-3: Unauthorized Sampling (3 cluster per tempo)
    # Cluster 1: Tempo Lento (70-90 BPM)
    sampling_slow_indices = np.random.choice(df.index, size=250, replace=False)
    df.loc[sampling_slow_indices, 'infringement_type'] = 'unauthorized_sampling'
    df.loc[sampling_slow_indices, 'is_infringement'] = True
    df.loc[sampling_slow_indices, 'tempo'] = 80 + np.random.normal(0, 5, 250)
    
    # Cluster 2: Tempo Medio (120-140 BPM)
    remaining_indices = df[~df['is_infringement']].index
    sampling_med_indices = np.random.choice(remaining_indices, size=200, replace=False)
    df.loc[sampling_med_indices, 'infringement_type'] = 'unauthorized_sampling'
    df.loc[sampling_med_indices, 'is_infringement'] = True
    df.loc[sampling_med_indices, 'tempo'] = 130 + np.random.normal(0, 5, 200)
    
    # Cluster 3: Tempo Veloce (160-180 BPM)
    remaining_indices = df[~df['is_infringement']].index
    sampling_fast_indices = np.random.choice(remaining_indices, size=180, replace=False)
    df.loc[sampling_fast_indices, 'infringement_type'] = 'unauthorized_sampling'
    df.loc[sampling_fast_indices, 'is_infringement'] = True
    df.loc[sampling_fast_indices, 'tempo'] = 170 + np.random.normal(0, 5, 180)
    
    # Tipo 4: Derivative Work - High Engagement
    remaining_indices = df[~df['is_infringement']].index
    derivative_high_indices = np.random.choice(remaining_indices, size=150, replace=False)
    df.loc[derivative_high_indices, 'infringement_type'] = 'derivative_work'
    df.loc[derivative_high_indices, 'is_infringement'] = True
    df.loc[derivative_high_indices, 'like_count'] *= np.random.uniform(5, 15, 150)
    df.loc[derivative_high_indices, 'share_count'] *= np.random.uniform(8, 20, 150)
    
    # Tipo 5: Metadata Manipulation
    remaining_indices = df[~df['is_infringement']].index
    metadata_indices = np.random.choice(remaining_indices, size=100, replace=False)
    df.loc[metadata_indices, 'infringement_type'] = 'metadata_manipulation'
    df.loc[metadata_indices, 'is_infringement'] = True
    df.loc[metadata_indices, 'revenue_generated'] *= np.random.uniform(10, 30, 100)
    
    # Tipo 6: Cross-Platform Violation
    remaining_indices = df[~df['is_infringement']].index
    cross_platform_indices = np.random.choice(remaining_indices, size=80, replace=False)
    df.loc[cross_platform_indices, 'infringement_type'] = 'cross_platform_violation'
    df.loc[cross_platform_indices, 'is_infringement'] = True
    df.loc[cross_platform_indices, 'revenue_generated'] *= np.random.uniform(3, 8, 80)
    
    # Tipo 7: Content ID Manipulation
    remaining_indices = df[~df['is_infringement']].index
    content_id_indices = np.random.choice(remaining_indices, size=70, replace=False)
    df.loc[content_id_indices, 'infringement_type'] = 'content_id_manipulation'
    df.loc[content_id_indices, 'is_infringement'] = True
    df.loc[content_id_indices, 'compression_ratio'] *= np.random.uniform(0.3, 0.7, 70)
    
    total_violations = df['is_infringement'].sum()
    print(f"✅ Track 4 generato: {len(df)} opere, {total_violations} violazioni ({total_violations/len(df)*100:.1f}%)")
    return df

def main():
    """Funzione principale - genera tutti i dataset"""
    print("🎯 SIAE Hackathon - Dataset Generator")
    print("=" * 50)
    print("Generando dataset identici per tutti i partecipanti...")
    print(f"🔑 Random seed: {RANDOM_SEED}")
    print()
    
    # Crea directory datasets
    datasets_dir = ensure_datasets_dir()
    
    # Scarica/genera dati FMA per Track 1 e 3
    print("📥 Preparazione dati FMA...")
    fma_data = download_fma_metadata()
    print()
    
    # Genera tutti i dataset
    datasets = {}
    
    # Track 1: Live Events
    datasets['track1'] = generate_track1_dataset(n_events=50000, music_data=fma_data)
    datasets['track1'].to_csv(datasets_dir / 'track1_live_events.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track1_live_events.csv'}")
    print()
    
    # Track 2: Document Fraud
    datasets['track2'] = generate_track2_dataset(n_documents=5000)
    datasets['track2'].to_csv(datasets_dir / 'track2_documents.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track2_documents.csv'}")
    print()
    
    # Track 3: Music Anomaly
    datasets['track3'] = generate_track3_dataset(n_tracks=25000)
    datasets['track3'].to_csv(datasets_dir / 'track3_music.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track3_music.csv'}")
    print()
    
    # Track 4: Copyright Infringement
    datasets['track4'] = generate_track4_dataset(n_works=15000)
    datasets['track4'].to_csv(datasets_dir / 'track4_copyright.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track4_copyright.csv'}")
    print()
    
    # Statistiche finali
    print("📊 STATISTICHE FINALI")
    print("=" * 50)
    total_samples = sum(len(df) for df in datasets.values())
    print(f"📋 Totale campioni: {total_samples:,}")
    print()
    
    for track, df in datasets.items():
        anomaly_col = 'is_anomaly' if 'is_anomaly' in df.columns else 'is_fraudulent' if 'is_fraudulent' in df.columns else 'is_infringement'
        if anomaly_col in df.columns:
            anomaly_count = df[anomaly_col].sum()
            anomaly_rate = anomaly_count / len(df) * 100
            print(f"🎯 {track.upper()}: {len(df):,} campioni, {anomaly_count:,} anomalie ({anomaly_rate:.1f}%)")
    
    print()
    print("✅ GENERAZIONE COMPLETATA!")
    print("📁 Tutti i dataset sono salvati in: datasets/")
    print("🎉 I dataset sono identici per tutti i partecipanti!")
    print()
    print("💡 Per usare i dataset nei tuoi script:")
    print("   df = pd.read_csv('datasets/track1_live_events.csv')")
    print("   df = pd.read_csv('datasets/track2_documents.csv')")
    print("   df = pd.read_csv('datasets/track3_music.csv')")
    print("   df = pd.read_csv('datasets/track4_copyright.csv')")

if __name__ == "__main__":
    main() 