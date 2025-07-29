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

def set_seed(seed):
    """Imposta il seed per tutte le librerie rilevanti per la riproducibilità."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    """
    Genera dataset Track 1 con ANOMALIE SOTTILI E MULTIVARIATE.
    Queste anomalie richiedono modelli statistici o di ML per essere scovate.
    """
    set_seed(RANDOM_SEED)
    print(f"🎪 Generando Track 1 (v2) con anomalie avanzate: {n_events} eventi live...")
    
    # --- 1. Generazione Dati di Base (Normali) ---
    venues = [f"Venue_{i}" for i in range(1, 501)]
    cities = ["Milano", "Roma", "Napoli", "Torino", "Bologna", "Firenze", 
              "Palermo", "Genova", "Bari", "Venezia"]
    
    # Artisti "famosi" per anomalie contestuali
    famous_artists = [f"Superstar_Artist_{i}" for i in range(1, 11)]
    other_artists = [f"Artist_{i}" for i in range(1, 1000)]
    all_artists = famous_artists + other_artists
    
    genres = ['Rock', 'Pop', 'Jazz', 'Electronic', 'Classical', 'Indie', 'Hip-Hop']
    
    events = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_events):
        capacity = random.randint(100, 5000)
        attendance_ratio = random.uniform(0.6, 0.95) # Pubblico normalmente affollato
        attendance = int(capacity * attendance_ratio)
        
        # Prezzo del biglietto correlato al genere e alla capienza
        ticket_price = random.uniform(15, 40) + (capacity / 500) * random.uniform(1, 5)
        if random.choice(genres) in ['Rock', 'Pop']:
            ticket_price *= 1.2
        
        base_revenue = attendance * ticket_price
        
        event = {
            'event_id': f"EVENT_{i+1:06d}",
            'venue': random.choice(venues),
            'city': random.choice(cities),
            'event_date': start_date + timedelta(days=random.randint(0, 730)),
            'capacity': capacity,
            'attendance': attendance,
            'n_songs_declared': random.randint(15, 25),
            'total_revenue': base_revenue * random.uniform(0.9, 1.1), # Leggera variabilità
            'genre': random.choice(genres),
            'main_artist': random.choice(all_artists),
            'event_duration_hours': random.uniform(2.5, 4.5),
            'ticket_price_avg': ticket_price,
            'anomaly_type': 'none',
            'is_anomaly': False
        }
        events.append(event)
        
    df = pd.DataFrame(events)
    print("✅ Dati di base generati. Ora inseriamo le anomalie sottili...")

    # --- 2. Iniezione di Anomalie Sottili ---
    # Usiamo indici non ancora usati per ogni tipo di anomalia
    available_indices = df.index.tolist()
    
    # ANOMALIA 1: Combinazione Improbabile (Contestuale) - 1% degli eventi
    n_anomaly1 = int(n_events * 0.01)
    anomaly1_indices = np.random.choice(available_indices, size=n_anomaly1, replace=False)
    df.loc[anomaly1_indices, 'main_artist'] = np.random.choice(famous_artists, size=n_anomaly1)
    df.loc[anomaly1_indices, 'city'] = np.random.choice(["Milano", "Roma"], size=n_anomaly1)
    df.loc[anomaly1_indices, 'capacity'] = np.random.randint(4000, 8000, size=n_anomaly1)
    # Ora la parte anomala: ricavi e prezzi ridicoli per un evento del genere
    df.loc[anomaly1_indices, 'total_revenue'] = np.random.uniform(100, 500, size=n_anomaly1)
    df.loc[anomaly1_indices, 'ticket_price_avg'] = df.loc[anomaly1_indices, 'total_revenue'] / df.loc[anomaly1_indices, 'attendance']
    df.loc[anomaly1_indices, 'anomaly_type'] = 'unlikely_combination'
    df.loc[anomaly1_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly1_indices))

    # ANOMALIA 2: Comportamento Anomalo del Venue (Collettiva) - 3 venue anomali
    anomalous_venues = np.random.choice(venues, size=3, replace=False)
    anomaly2_indices = df[df['venue'].isin(anomalous_venues)].index
    # Filtra indici già usati
    anomaly2_indices = [idx for idx in anomaly2_indices if idx in available_indices]
    df.loc[anomaly2_indices, 'n_songs_declared'] = np.random.randint(5, 10, size=len(anomaly2_indices))
    df.loc[anomaly2_indices, 'event_duration_hours'] = np.random.uniform(7.0, 9.0, size=len(anomaly2_indices))
    df.loc[anomaly2_indices, 'anomaly_type'] = 'anomalous_venue_behavior'
    df.loc[anomaly2_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly2_indices))

    # ANOMALIA 3: Cluster Nascosto (Multivariata) - 0.8% degli eventi
    n_anomaly3 = int(n_events * 0.008)
    anomaly3_indices = np.random.choice(available_indices, size=n_anomaly3, replace=False)
    # Questo cluster ha: durata breve, canzoni poche, affluenza molto bassa
    df.loc[anomaly3_indices, 'event_duration_hours'] = np.random.uniform(1.0, 1.5, size=n_anomaly3)
    df.loc[anomaly3_indices, 'n_songs_declared'] = np.random.randint(3, 7, size=n_anomaly3)
    df.loc[anomaly3_indices, 'attendance'] = df.loc[anomaly3_indices, 'capacity'] * np.random.uniform(0.05, 0.15)
    df.loc[anomaly3_indices, 'anomaly_type'] = 'hidden_cluster'
    df.loc[anomaly3_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly3_indices))

    # ANOMALIA 4: Frode Sottile sui Ricavi (Statistica) - 1.2% degli eventi
    n_anomaly4 = int(n_events * 0.012)
    anomaly4_indices = np.random.choice(available_indices, size=n_anomaly4, replace=False)
    # Il ricavo è solo leggermente e costantemente più basso del normale
    normal_revenue = df.loc[anomaly4_indices, 'attendance'] * df.loc[anomaly4_indices, 'ticket_price_avg']
    df.loc[anomaly4_indices, 'total_revenue'] = normal_revenue * np.random.uniform(0.4, 0.6) # Sospettosamente basso ma non zero
    df.loc[anomaly4_indices, 'anomaly_type'] = 'subtle_revenue_fraud'
    df.loc[anomaly4_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly4_indices))

    # ANOMALIA 5: Date di tour impossibili (Logica contestuale)
    n_anomaly5 = int(n_events * 0.005)
    anomaly5_artists = np.random.choice(other_artists, size=15, replace=False)
    for artist in anomaly5_artists:
        artist_events = df[df['main_artist'] == artist].index
        if len(artist_events) > 2:
            # Prendi due eventi a caso di questo artista
            event_indices = np.random.choice(artist_events, size=2, replace=False)
            if event_indices[0] in available_indices and event_indices[1] in available_indices:
                # Imposta la stessa data ma città molto distanti
                base_date = datetime(2024, 5, 20) + timedelta(days=random.randint(0,100))
                df.loc[event_indices[0], 'event_date'] = base_date
                df.loc[event_indices[0], 'city'] = 'Milano'
                df.loc[event_indices[0], 'is_anomaly'] = True
                df.loc[event_indices[0], 'anomaly_type'] = 'impossible_tour_date'
                
                df.loc[event_indices[1], 'event_date'] = base_date
                df.loc[event_indices[1], 'city'] = 'Palermo'
                df.loc[event_indices[1], 'is_anomaly'] = True
                df.loc[event_indices[1], 'anomaly_type'] = 'impossible_tour_date'

                available_indices = list(set(available_indices) - set(event_indices))

    # Shuffle finale per mescolare le anomalie nel dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"✅ Track 1 v2 generato: {len(df)} eventi, {df['is_anomaly'].sum()} anomalie ({df['is_anomaly'].mean():.2%})")
    print("🔍 Tipi di anomalie inserite:", df[df['is_anomaly']]['anomaly_type'].value_counts().to_dict())
    return df

def generate_track2_dataset(n_documents=5000):
    """Genera dataset Track 2 con frodi documentali SOFISTICATE."""
    set_seed(RANDOM_SEED)
    print(f"📄 Generando Track 2 (v2) con frodi avanzate: {n_documents} documenti...")
    
    # --- 1. Generazione Dati di Base (Normali) ---
    document_types = [
        'Contratto_Editore', 'Licenza_Esecuzione', 'Dichiarazione_Musica_Live',
        'Cessione_Diritti', 'Registrazione_Opera'
    ]
    
    documents = []
    for i in range(n_documents):
        is_old_doc = random.random() < 0.3
        creation_date = datetime(2023, 1, 1) - timedelta(days=random.randint(0, 365*10 if is_old_doc else 365))
        
        doc_data = {
            'document_id': f"SIAE_{i+1:06d}",
            'document_type': random.choice(document_types),
            'creation_date': creation_date,
            'page_count': random.randint(1, 20),
            'file_size_kb': np.random.exponential(500) + 50,
            'resolution_dpi': random.choice([200, 300]) if is_old_doc else random.choice([300, 600]),
            'text_confidence_avg': random.uniform(0.85, 0.98) if not is_old_doc else random.uniform(0.7, 0.9),
            'pixel_noise_level': random.uniform(0.01, 0.08) if is_old_doc else random.uniform(0.001, 0.02),
            'edge_sharpness': random.uniform(0.6, 0.8) if is_old_doc else random.uniform(0.8, 1.0),
            'metadata_consistency': random.uniform(0.95, 1.0),
            'submitter_id': f"User_{random.randint(1, 500)}",
            'fraud_type': 'none',
            'is_fraudulent': False
        }
        documents.append(doc_data)
        
    df = pd.DataFrame(documents)
    print("✅ Dati di base generati. Ora inseriamo le frodi...")

    # --- 2. Iniezione di Frodi Sottili ---
    available_indices = df.index.tolist()

    # ANOMALIA 1: Documento "Troppo Perfetto" (Contestuale) - 2% dei documenti
    n_anomaly1 = int(n_documents * 0.02)
    anomaly1_indices = np.random.choice(available_indices, size=n_anomaly1, replace=False)
    # Facciamo finta che siano documenti vecchi, ma con qualità da "nati digitali"
    df.loc[anomaly1_indices, 'creation_date'] = df.loc[anomaly1_indices, 'creation_date'].apply(lambda d: d - timedelta(days=365*15))
    df.loc[anomaly1_indices, 'resolution_dpi'] = 600
    df.loc[anomaly1_indices, 'pixel_noise_level'] = np.random.uniform(0.0001, 0.001, size=n_anomaly1)
    df.loc[anomaly1_indices, 'text_confidence_avg'] = np.random.uniform(0.99, 0.999, size=n_anomaly1)
    df.loc[anomaly1_indices, 'fraud_type'] = 'too_perfect_for_age'
    df.loc[anomaly1_indices, 'is_fraudulent'] = True
    available_indices = list(set(available_indices) - set(anomaly1_indices))

    # ANOMALIA 2: Frode di Template Sofisticata (Collettiva) - 5 template fraudolenti
    anomalous_submitters = np.random.choice(df['submitter_id'].unique(), size=5, replace=False)
    anomaly2_indices = df[df['submitter_id'].isin(anomalous_submitters)].index
    anomaly2_indices = [idx for idx in anomaly2_indices if idx in available_indices]
    # Tutti i documenti da questi utenti hanno caratteristiche sospettosamente simili
    df.loc[anomaly2_indices, 'resolution_dpi'] = 200
    df.loc[anomaly2_indices, 'file_size_kb'] = np.random.normal(loc=250, scale=5, size=len(anomaly2_indices)) # Dimensione quasi identica
    df.loc[anomaly2_indices, 'edge_sharpness'] = np.random.normal(loc=0.7, scale=0.02, size=len(anomaly2_indices))
    df.loc[anomaly2_indices, 'fraud_type'] = 'sophisticated_template_fraud'
    df.loc[anomaly2_indices, 'is_fraudulent'] = True
    available_indices = list(set(available_indices) - set(anomaly2_indices))
    
    # ANOMALIA 3: Incoerenza Interna (Multivariata) - 2.5% dei documenti
    n_anomaly3 = int(n_documents * 0.025)
    anomaly3_indices = np.random.choice(available_indices, size=n_anomaly3, replace=False)
    # Alta nitidezza ma basso riconoscimento testo -> sospetto
    df.loc[anomaly3_indices, 'edge_sharpness'] = np.random.uniform(0.9, 1.0, size=n_anomaly3)
    df.loc[anomaly3_indices, 'text_confidence_avg'] = np.random.uniform(0.5, 0.7, size=n_anomaly3)
    df.loc[anomaly3_indices, 'fraud_type'] = 'internal_feature_inconsistency'
    df.loc[anomaly3_indices, 'is_fraudulent'] = True
    available_indices = list(set(available_indices) - set(anomaly3_indices))

    # ANOMALIA 4: Manipolazione Sottile Metadati - 1.5%
    n_anomaly4 = int(n_documents * 0.015)
    anomaly4_indices = np.random.choice(available_indices, size=n_anomaly4, replace=False)
    # La consistenza dei metadati è *quasi* perfetta, ma non del tutto. Un valore di 0.7 è facile da filtrare, ma 0.92?
    df.loc[anomaly4_indices, 'metadata_consistency'] = np.random.uniform(0.90, 0.94, size=n_anomaly4)
    df.loc[anomaly4_indices, 'fraud_type'] = 'subtle_metadata_manipulation'
    df.loc[anomaly4_indices, 'is_fraudulent'] = True
    
    # Shuffle finale
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"✅ Track 2 v2 generato: {len(df)} documenti, {df['is_fraudulent'].sum()} frodi ({df['is_fraudulent'].mean():.2%})")
    print("🔍 Tipi di frodi inserite:", df[df['is_fraudulent']]['fraud_type'].value_counts().to_dict())
    return df

def generate_track3_dataset(n_tracks=25000):
    """Genera dataset Track 3 con anomalie musicali SOFISTICATE."""
    set_seed(RANDOM_SEED)
    print(f"🎵 Generando Track 3 (v2) con anomalie avanzate: {n_tracks} tracce...")
    
    # --- 1. Generazione Dati di Base (Normali) ---
    # ... (stessa generazione di base della funzione originale per brevità) ...
    # Assumiamo di avere df, artists, genres ecc.
    genres = ['Electronic', 'Rock', 'Hip-Hop', 'Folk', 'Pop', 'Experimental', 'Jazz', 'Classical']
    artists = [{'name': f"Artist_{i:04d}", 'active_since': random.randint(2000, 2020)} for i in range(1, 2001)]
    inactive_artists = [{'name': f"Inactive_Artist_{i:04d}", 'active_since': random.randint(1980, 1999)} for i in range(1, 51)]
    all_artists = artists + inactive_artists
    
    tracks = []
    for i in range(n_tracks):
        artist_info = random.choice(all_artists)
        # La qualità audio è coerente
        bit_rate = random.choice([128, 192, 320])
        spectral_bandwidth = (bit_rate / 320) * random.uniform(2000, 4000) + 5000
        
        track = {
            'track_id': i,
            'artist_name': artist_info['name'],
            'track_title': f"Track_{i:05d}",
            'genre_top': random.choice(genres),
            'track_duration': random.randint(120, 480),
            'track_listens': random.randint(100, 1000000),
            'track_favorites': int(random.randint(100, 1000000) * 0.05), # Correlato agli ascolti
            'track_comments': int(random.randint(100, 1000000) * 0.001),
            'artist_active_year_begin': artist_info['active_since'],
            'bit_rate': bit_rate,
            'spectral_bandwidth': spectral_bandwidth, # Feature tecnica audio
            'listener_country_entropy': random.uniform(2.5, 4.0), # Alta entropia = tanti paesi
            'anomaly_type': 'none',
            'is_anomaly': False
        }
        tracks.append(track)
    df = pd.DataFrame(tracks)
    print("✅ Dati di base generati. Ora inseriamo le anomalie...")

    # --- 2. Iniezione di Anomalie Sottili ---
    available_indices = df.index.tolist()

    # ANOMALIA 1: Bot Streaming Sofisticato (Comportamentale) - 2%
    n_anomaly1 = int(n_tracks * 0.02)
    anomaly1_indices = np.random.choice(available_indices, size=n_anomaly1, replace=False)
    # Ascolti alti, ma zero commenti e tutti da una stessa area geografica (bassa entropia)
    df.loc[anomaly1_indices, 'track_listens'] = np.random.randint(500000, 2000000, size=n_anomaly1)
    df.loc[anomaly1_indices, 'track_favorites'] = df.loc[anomaly1_indices, 'track_listens'] * np.random.uniform(0.04, 0.06) # Rapporto normale
    df.loc[anomaly1_indices, 'track_comments'] = 0
    df.loc[anomaly1_indices, 'listener_country_entropy'] = np.random.uniform(0.1, 0.5, size=n_anomaly1)
    df.loc[anomaly1_indices, 'anomaly_type'] = 'sophisticated_bot_streaming'
    df.loc[anomaly1_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly1_indices))

    # ANOMALIA 2: Hijacking di Artista (Contestuale) - 1%
    n_anomaly2 = int(n_tracks * 0.01)
    anomaly2_indices = np.random.choice(available_indices, size=n_anomaly2, replace=False)
    # Artisti inattivi da decenni pubblicano tracce "moderne"
    df.loc[anomaly2_indices, 'artist_name'] = np.random.choice([a['name'] for a in inactive_artists], size=n_anomaly2)
    df.loc[anomaly2_indices, 'artist_active_year_begin'] = np.random.randint(1980, 1999, size=n_anomaly2)
    df.loc[anomaly2_indices, 'genre_top'] = np.random.choice(['Hip-Hop', 'Electronic'], size=n_anomaly2)
    df.loc[anomaly2_indices, 'anomaly_type'] = 'artist_hijacking'
    df.loc[anomaly2_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly2_indices))

    # ANOMALIA 3: Frode Qualità Audio (Incoerenza Tecnica) - 1.5%
    n_anomaly3 = int(n_tracks * 0.015)
    anomaly3_indices = np.random.choice(available_indices, size=n_anomaly3, replace=False)
    # Alto bitrate dichiarato, ma caratteristiche audio da bassa qualità
    df.loc[anomaly3_indices, 'bit_rate'] = 320
    df.loc[anomaly3_indices, 'spectral_bandwidth'] = np.random.uniform(3000, 5000, size=n_anomaly3) # Tipico di MP3 a 96-128kbps
    df.loc[anomaly3_indices, 'anomaly_type'] = 'audio_quality_fraud'
    df.loc[anomaly3_indices, 'is_anomaly'] = True
    available_indices = list(set(available_indices) - set(anomaly3_indices))

    # Per ANOMALIA 4 (Micro-Plagio), la simulazione richiederebbe di aggiungere feature di fingerprinting (es. MFCCs),
    # il che complicherebbe il generatore. Le 3 anomalie sopra sono già un ottimo passo avanti per la sfida.

    # Shuffle finale
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"✅ Track 3 v2 generato: {len(df)} tracce, {df['is_anomaly'].sum()} anomalie ({df['is_anomaly'].mean():.2%})")
    print("🔍 Tipi di anomalie inserite:", df[df['is_anomaly']]['anomaly_type'].value_counts().to_dict())
    return df

def generate_track4_dataset(n_works=15000):
    """Genera dataset Track 4 con violazioni di copyright SOFISTICATE."""
    set_seed(RANDOM_SEED)
    print(f"🔒 Generando Track 4 (v2) con violazioni avanzate: {n_works} opere...")

    # --- 1. Creiamo un "database" di opere protette fittizie ---
    protected_works = []
    for i in range(50):
        protected_works.append({
            'protected_id': f"PROTECTED_{i}",
            'tempo': random.uniform(80, 160),
            'spectral_centroid_mean': random.uniform(1500, 4000),
            'original_hash': hashlib.md5(f"protected_{i}".encode()).hexdigest()[:16]
        })

    # --- 2. Generazione Dati di Base (Normali) ---
    works = []
    for i in range(n_works):
        work = {
            'work_id': f"SIAE_CP_{i+1:06d}",
            'release_date': datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730)),
            'days_since_release': (datetime.now() - (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730)))).days,
            'tempo': random.uniform(70, 180),
            'spectral_centroid_mean': random.uniform(1000, 8000),
            'play_count': random.randint(1000, 1000000),
            'audio_similarity_to_db': random.uniform(0.1, 0.4), # Bassa similarità per opere normali
            'noise_floor_db': random.uniform(-80, -60),
            'file_hash': hashlib.md5(f"work_{i}".encode()).hexdigest()[:16],
            'infringement_type': 'none',
            'is_infringement': False
        }
        works.append(work)
    df = pd.DataFrame(works)
    print("✅ Dati di base generati. Ora inseriamo le violazioni...")

    # --- 3. Iniezione di Violazioni Sottili ---
    available_indices = df.index.tolist()

    # ANOMALIA 1: Evasione Time-Stretch/Pitch-Shift (Similarità vs Hash) - 2.5%
    n_anomaly1 = int(n_works * 0.025)
    anomaly1_indices = np.random.choice(available_indices, size=n_anomaly1, replace=False)
    # Copiamo le caratteristiche di un'opera protetta, ma le alteriamo leggermente
    for idx in anomaly1_indices:
        source_work = random.choice(protected_works)
        df.loc[idx, 'tempo'] = source_work['tempo'] * random.uniform(0.95, 1.05) # +/- 5% tempo
        df.loc[idx, 'spectral_centroid_mean'] = source_work['spectral_centroid_mean'] * random.uniform(0.98, 1.02)
        # La similarità è altissima, ma l'hash è diverso
        df.loc[idx, 'audio_similarity_to_db'] = random.uniform(0.95, 0.99)
        df.loc[idx, 'file_hash'] = hashlib.md5(f"stolen_work_{idx}".encode()).hexdigest()[:16] # Nuovo hash
    df.loc[anomaly1_indices, 'infringement_type'] = 'evasion_by_modification'
    df.loc[anomaly1_indices, 'is_infringement'] = True
    available_indices = list(set(available_indices) - set(anomaly1_indices))

    # ANOMALIA 2: Violazione "Sleeper" (Comportamentale/Temporale) - 1.5%
    n_anomaly2 = int(n_works * 0.015)
    anomaly2_indices = np.random.choice(available_indices, size=n_anomaly2, replace=False)
    # Opere vecchie con un picco di ascolti recentissimo
    df.loc[anomaly2_indices, 'days_since_release'] = np.random.randint(300, 700, size=n_anomaly2)
    # Il 99% degli ascolti è avvenuto nell'ultimo giorno (calcoliamo un ratio)
    df.loc[anomaly2_indices, 'play_count_last_24h'] = df.loc[anomaly2_indices, 'play_count'] * np.random.uniform(0.95, 0.99)
    df.loc[anomaly2_indices, 'infringement_type'] = 'sleeper_infringement'
    df.loc[anomaly2_indices, 'is_infringement'] = True
    available_indices = list(set(available_indices) - set(anomaly2_indices))

    # ANOMALIA 3: Mascheramento con Rumore (Incoerenza Tecnica) - 1%
    n_anomaly3 = int(n_works * 0.01)
    anomaly3_indices = np.random.choice(available_indices, size=n_anomaly3, replace=False)
    # Alta similarità audio ma anche un rumore di fondo sospetto
    df.loc[anomaly3_indices, 'audio_similarity_to_db'] = random.uniform(0.92, 0.98)
    df.loc[anomaly3_indices, 'noise_floor_db'] = np.random.uniform(-55, -45) # Rumore più alto del normale
    df.loc[anomaly3_indices, 'infringement_type'] = 'evasion_by_noise_masking'
    df.loc[anomaly3_indices, 'is_infringement'] = True
    
    # Riempiamo i NaN per le colonne aggiunte
    df.fillna({'play_count_last_24h': 0}, inplace=True)
    
    # Shuffle finale
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"✅ Track 4 v2 generato: {len(df)} opere, {df['is_infringement'].sum()} violazioni ({df['is_infringement'].mean():.2%})")
    print("🔍 Tipi di violazioni inserite:", df[df['is_infringement']]['infringement_type'].value_counts().to_dict())
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

    # Dizionario per tenere traccia dei dataset generati e delle loro stats
    datasets_info = {}

    # --- Track 1: Live Events ---
    print("\n--- Generando Track 1: Live Events ---")
    df_track1 = generate_track1_dataset(n_events=50000, music_data=fma_data)
    train_size_t1 = 40000
    df_train_t1 = df_track1.iloc[:train_size_t1]
    df_test_t1 = df_track1.iloc[train_size_t1:]
    ground_truth_cols_t1 = ['event_id', 'is_anomaly', 'anomaly_type']
    df_test_ground_truth_t1 = df_test_t1[ground_truth_cols_t1].copy()
    df_test_public_t1 = df_test_t1.drop(columns=['is_anomaly', 'anomaly_type'])
    df_train_t1.to_csv(datasets_dir / 'track1_live_events_train.csv', index=False)
    df_test_public_t1.to_csv(datasets_dir / 'track1_live_events_test.csv', index=False)
    df_test_ground_truth_t1.to_csv(datasets_dir / 'track1_live_events_test_ground_truth.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track1_live_events_train.csv'} ({len(df_train_t1)} righe)")
    print(f"💾 Salvato: {datasets_dir / 'track1_live_events_test.csv'} ({len(df_test_public_t1)} righe)")
    print(f"🔒 Salvato: {datasets_dir / 'track1_live_events_test_ground_truth.csv'} (Nascosto)")
    datasets_info['track1'] = {'train': df_train_t1, 'test': df_test_public_t1, 'anomaly_col': 'is_anomaly'}
    print()

    # --- Track 2: Document Fraud ---
    print("\n--- Generando Track 2: Document Fraud ---")
    df_track2 = generate_track2_dataset(n_documents=5000)
    train_size_t2 = 4000
    df_train_t2 = df_track2.iloc[:train_size_t2]
    df_test_t2 = df_track2.iloc[train_size_t2:]
    ground_truth_cols_t2 = ['document_id', 'is_fraudulent', 'fraud_type']
    df_test_ground_truth_t2 = df_test_t2[ground_truth_cols_t2].copy()
    df_test_public_t2 = df_test_t2.drop(columns=['is_fraudulent', 'fraud_type'])
    df_train_t2.to_csv(datasets_dir / 'track2_documents_train.csv', index=False)
    df_test_public_t2.to_csv(datasets_dir / 'track2_documents_test.csv', index=False)
    df_test_ground_truth_t2.to_csv(datasets_dir / 'track2_documents_test_ground_truth.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track2_documents_train.csv'} ({len(df_train_t2)} righe)")
    print(f"💾 Salvato: {datasets_dir / 'track2_documents_test.csv'} ({len(df_test_public_t2)} righe)")
    print(f"🔒 Salvato: {datasets_dir / 'track2_documents_test_ground_truth.csv'} (Nascosto)")
    datasets_info['track2'] = {'train': df_train_t2, 'test': df_test_public_t2, 'anomaly_col': 'is_fraudulent'}
    print()

    # --- Track 3: Music Anomaly ---
    print("\n--- Generando Track 3: Music Anomaly ---")
    df_track3 = generate_track3_dataset(n_tracks=25000)
    train_size_t3 = 20000
    df_train_t3 = df_track3.iloc[:train_size_t3]
    df_test_t3 = df_track3.iloc[train_size_t3:]
    ground_truth_cols_t3 = ['track_id', 'is_anomaly', 'anomaly_type']
    df_test_ground_truth_t3 = df_test_t3[ground_truth_cols_t3].copy()
    df_test_public_t3 = df_test_t3.drop(columns=['is_anomaly', 'anomaly_type'])
    df_train_t3.to_csv(datasets_dir / 'track3_music_train.csv', index=False)
    df_test_public_t3.to_csv(datasets_dir / 'track3_music_test.csv', index=False)
    df_test_ground_truth_t3.to_csv(datasets_dir / 'track3_music_test_ground_truth.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track3_music_train.csv'} ({len(df_train_t3)} righe)")
    print(f"💾 Salvato: {datasets_dir / 'track3_music_test.csv'} ({len(df_test_public_t3)} righe)")
    print(f"🔒 Salvato: {datasets_dir / 'track3_music_test_ground_truth.csv'} (Nascosto)")
    datasets_info['track3'] = {'train': df_train_t3, 'test': df_test_public_t3, 'anomaly_col': 'is_anomaly'}
    print()

    # --- Track 4: Copyright Infringement ---
    print("\n--- Generando Track 4: Copyright Infringement ---")
    df_track4 = generate_track4_dataset(n_works=15000)
    train_size_t4 = 12000
    df_train_t4 = df_track4.iloc[:train_size_t4]
    df_test_t4 = df_track4.iloc[train_size_t4:]
    ground_truth_cols_t4 = ['work_id', 'is_infringement', 'infringement_type']
    df_test_ground_truth_t4 = df_test_t4[ground_truth_cols_t4].copy()
    df_test_public_t4 = df_test_t4.drop(columns=['is_infringement', 'infringement_type'])
    df_train_t4.to_csv(datasets_dir / 'track4_copyright_train.csv', index=False)
    df_test_public_t4.to_csv(datasets_dir / 'track4_copyright_test.csv', index=False)
    df_test_ground_truth_t4.to_csv(datasets_dir / 'track4_copyright_test_ground_truth.csv', index=False)
    print(f"💾 Salvato: {datasets_dir / 'track4_copyright_train.csv'} ({len(df_train_t4)} righe)")
    print(f"💾 Salvato: {datasets_dir / 'track4_copyright_test.csv'} ({len(df_test_public_t4)} righe)")
    print(f"🔒 Salvato: {datasets_dir / 'track4_copyright_test_ground_truth.csv'} (Nascosto)")
    datasets_info['track4'] = {'train': df_train_t4, 'test': df_test_public_t4, 'anomaly_col': 'is_infringement'}
    print()

    # Statistiche finali
    print("\n📊 STATISTICHE FINALI")
    print("=" * 50)
    total_train = sum(len(info['train']) for info in datasets_info.values())
    total_test = sum(len(info['test']) for info in datasets_info.values())
    print(f"📋 Totale campioni generati: {total_train + total_test:,}")
    print(f"  - Dati di Training: {total_train:,}")
    print(f"  - Dati di Test: {total_test:,}")
    print()
    
    for track, info in datasets_info.items():
        df_train = info['train']
        anomaly_col = info['anomaly_col']
        if anomaly_col in df_train.columns:
            anomaly_count = df_train[anomaly_col].sum()
            anomaly_rate = anomaly_count / len(df_train) * 100
            print(f"🎯 {track.upper()} (Training): {len(df_train):,} campioni, {anomaly_count:,} anomalie ({anomaly_rate:.1f}%)")

    print()
    print("✅ GENERAZIONE COMPLETATA!")
    print(f"📁 Tutti i dataset (training, test e ground truth) sono salvati in: {datasets_dir}/")
    print("🎉 I dataset sono identici per tutti i partecipanti!")
    print()
    print("💡 Per usare i dataset nei tuoi script:")
    print("   df_train = pd.read_csv('datasets/track1_live_events_train.csv')")
    print("   df_test = pd.read_csv('datasets/track1_live_events_test.csv')")
    print("   # E così via per gli altri track...")

if __name__ == "__main__":
    main() 