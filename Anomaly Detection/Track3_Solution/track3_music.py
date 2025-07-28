#!/usr/bin/env python3
"""
SIAE Anomaly Detection Hackathon - Track 3: Music Anomaly Detection
Music Anomaly Detection using a multi-model unsupervised approach.

This script implements the complete pipeline for Track 3:
1. Loads the pre-generated training and test datasets.
2. Performs feature engineering tailored for music data anomalies.
3. Trains and compares several unsupervised models:
    - Autoencoder (using PyTorch, for reconstruction-based anomalies)
    - DBSCAN (for density-based anomaly detection)
    - Isolation Forest
    - Local Outlier Factor (LOF)
4. Visualizes the performance comparison of the models.
5. Automatically selects the best model based on F1-Score on the training set.
6. Applies the best model to the test set to generate predictions.
7. Creates the submission file in the specified format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import warnings
import os
import json
import time
from pathlib import Path
import sys

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch non è installato. L'Autoencoder non sarà disponibile.", file=sys.stderr)
    print("Per installarlo, esegui: pip install torch", file=sys.stderr)
    torch = None

warnings.filterwarnings('ignore')
np.random.seed(42)
if torch:
    torch.manual_seed(42)

# --- 1. Data Loading ---

def load_train_test_datasets():
    """
    Loads the training and test datasets for Track 3.
    """
    print("📥 Caricando dataset di training e test per Track 3...")
    
    train_path = '../datasets/track3_music_train.csv'
    if not os.path.exists(train_path):
        print(f"❌ File di training non trovato: {train_path}", file=sys.stderr)
        print("💡 Assicurati di aver eseguito 'python generate_datasets.py' nella directory principale.", file=sys.stderr)
        sys.exit(1)
    
    df_train = pd.read_csv(train_path)
    print(f"✅ Dataset di train caricato: {len(df_train)} tracce")
    
    test_path = '../datasets/track3_music_test.csv'
    if not os.path.exists(test_path):
        print(f"❌ File di test non trovato: {test_path}", file=sys.stderr)
        sys.exit(1)
    
    df_test = pd.read_csv(test_path)
    print(f"✅ Dataset di test caricato: {len(df_test)} tracce")
    
    return df_train, df_test

# --- 2. Feature Engineering ---

def feature_engineering_music(df):
    """
    Advanced feature engineering for music anomaly detection, corrected for actual dataset columns.
    """
    print("🔧 Eseguendo feature engineering per dati musicali (versione corretta)...")
    
    df = df.copy()

    # Engagement Ratios
    df['favorites_per_listen'] = df['track_favorites'] / (df['track_listens'] + 1)

    # Audio & File Features
    df['bitrate_per_second'] = df['bit_rate'] / (df['track_duration'] + 1e-6)
    df['size_per_listen'] = df['file_size'] / (df['track_listens'] + 1)
    
    # Tempo & Energy interactions
    df['tempo_x_energy'] = df['tempo'] * df['energy']

    # Encode categorical features
    for col in ['artist_name', 'genre_top']:
        le = LabelEncoder()
        # Ensure all potential values from both train and test are considered if run separately
        # For simplicity here, we fit on the current dataframe's unique values
        df[f'{col}_encoded'] = le.fit_transform(df[col])

    # Stats per artist
    artist_stats = df.groupby('artist_name').agg(
        artist_avg_listens=('track_listens', 'mean'),
        artist_avg_duration=('track_duration', 'mean'),
        artist_track_count=('track_id', 'count')
    ).fillna(0)
    
    df = df.merge(artist_stats, on='artist_name', how='left')
    
    # Deviation from artist's norm
    df['listens_dev_from_artist_avg'] = (df['track_listens'] - df['artist_avg_listens']) / (df['artist_avg_listens'] + 1e-6)
    df['duration_dev_from_artist_avg'] = (df['track_duration'] - df['artist_avg_duration']) / (df['artist_avg_duration'] + 1e-6)

    # Simple outlier flags
    df['is_long_duration'] = (df['track_duration'] > df['track_duration'].quantile(0.95)).astype(int)
    df['is_high_tempo'] = (df['tempo'] > df['tempo'].quantile(0.95)).astype(int)

    print(f"✅ Feature engineering completato: {df.shape[1]} colonne totali")
    return df

# --- 3. Model Implementations ---

# Autoencoder (from Track 2, it's generic enough)
if torch:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim), nn.Sigmoid())
        def forward(self, x): return self.decoder(self.encoder(x))

    def train_autoencoder(model, data_loader, epochs=50):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(epochs):
            for data, _ in data_loader:
                outputs = model(data)
                loss = criterion(outputs, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0: print(f"AE Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        return model

    def get_autoencoder_scores(model, data_loader):
        model.eval()
        errors = []
        criterion = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for data, _ in data_loader:
                errors.extend(criterion(model(data), data).mean(dim=1).numpy())
        return np.array(errors)

def train_and_evaluate_models(df_train, feature_cols):
    """
    Trains, evaluates, and compares different unsupervised models.
    """
    print("\n🤖 Addestrando e confrontando modelli unsupervised per Track 3...")
    
    X_train = df_train[feature_cols].fillna(0).values
    y_true = df_train['is_anomaly'].values
    
    scaler_std = StandardScaler()
    X_train_scaled_std = scaler_std.fit_transform(X_train)
    
    scaler_mm = MinMaxScaler()
    X_train_scaled_mm = scaler_mm.fit_transform(X_train)

    models = {}
    results = {}
    
    contamination_level = df_train['is_anomaly'].mean() if df_train['is_anomaly'].any() else 0.05

    # --- Isolation Forest ---
    print("\n--- 🌲 Isolation Forest ---")
    iso_forest = IsolationForest(contamination=contamination_level, n_estimators=200, random_state=42)
    iso_forest.fit(X_train_scaled_std)
    models['Isolation Forest'] = {'model': iso_forest, 'scaler': scaler_std, 'type': 'sklearn'}

    # --- Local Outlier Factor (LOF) ---
    print("\n--- 🌐 Local Outlier Factor ---")
    lof = LocalOutlierFactor(n_neighbors=30, contamination=contamination_level, novelty=True)
    lof.fit(X_train_scaled_std)
    models['LOF'] = {'model': lof, 'scaler': scaler_std, 'type': 'sklearn'}
    
    # --- DBSCAN ---
    print("\n--- Cluster DBSCAN ---")
    dbscan = DBSCAN(eps=1.5, min_samples=10) # Params may need tuning
    dbscan.fit(X_train_scaled_std)
    models['DBSCAN'] = {'model': dbscan, 'scaler': scaler_std, 'type': 'clustering'}

    # --- Autoencoder (PyTorch) ---
    if torch:
        print("\n--- 🔥 Autoencoder (PyTorch) ---")
        input_dim = X_train_scaled_mm.shape[1]
        autoencoder = Autoencoder(input_dim)
        X_normal_mm = X_train_scaled_mm[y_true == 0]
        train_loader_ae = DataLoader(TensorDataset(torch.FloatTensor(X_normal_mm), torch.FloatTensor(X_normal_mm)), batch_size=128, shuffle=True)
        autoencoder = train_autoencoder(autoencoder, train_loader_ae)
        full_loader_ae = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled_mm), torch.FloatTensor(X_train_scaled_mm)), batch_size=256, shuffle=False)
        models['Autoencoder'] = {'model': autoencoder, 'scaler': scaler_mm, 'type': 'pytorch', 'loader': full_loader_ae}

    # Evaluate and compare
    print("\n📊 Valutando performance sul training set...")
    best_model_name, best_f1 = None, -1

    for name, model_info in models.items():
        scaler, model_type = model_info['scaler'], model_info['type']
        X_scaled = scaler.transform(X_train)
        
        if model_type == 'sklearn':
            scores = -model_info['model'].score_samples(X_scaled) if name != 'LOF' else model_info['model'].negative_outlier_factor_
            y_pred = (model_info['model'].predict(X_scaled) == -1).astype(int)
        elif model_type == 'clustering':
            scores = np.ones(len(X_scaled)) # DBSCAN doesn't provide scores directly
            y_pred = (model_info['model'].fit_predict(X_scaled) == -1).astype(int)
        elif model_type == 'pytorch':
            scores = get_autoencoder_scores(model_info['model'], model_info['loader'])
            threshold = np.quantile(scores[y_true==0], 0.95)
            y_pred = (scores > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.5 # Cannot compute for DBSCAN with uniform scores
            
        results[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
        print(f"  - {name:20s} | F1: {f1:.3f} | AUC: {auc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

        if f1 > best_f1:
            best_f1, best_model_name = f1, name
            models[name]['predictions'] = y_pred # Salva le predizioni del modello migliore

    print(f"\n🏆 Modello migliore: {best_model_name} (F1-Score: {best_f1:.3f})")
    create_comparison_chart(results, "Track 3")
    
    # Matrice di confusione
    if best_model_name and 'predictions' in models[best_model_name]:
        create_confusion_matrix(y_true, models[best_model_name]['predictions'], best_model_name, "Track 3")
    
    return models[best_model_name], best_model_name

def create_confusion_matrix(y_true, y_pred, model_name, track_name):
    """Crea e salva la matrice di confusione."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normale', 'Anomalia'], 
                yticklabels=['Normale', 'Anomalia'])
    plt.title(f'Matrice di Confusione - {model_name} ({track_name})', fontsize=16, fontweight='bold')
    plt.ylabel('Label Reale')
    plt.xlabel('Label Predetta')
    plt.tight_layout()
    filename = f'{track_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"✅ Matrice di confusione salvata in: {filename}")


def create_comparison_chart(results, track_name):
    """Creates a bar chart to compare model performances."""
    df_results = pd.DataFrame(results).T.sort_values(by='f1', ascending=False)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    df_results[['f1', 'auc', 'precision', 'recall']].plot(kind='bar', ax=ax, colormap='plasma')
    ax.set_title(f'Confronto Performance Modelli - {track_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Modello', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(df_results.index, rotation=45, ha='right')
    ax.legend(title='Metriche')
    for i, f1_val in enumerate(df_results['f1']):
        ax.text(i, f1_val + 0.01, f'{f1_val:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{track_name.lower().replace(" ", "_")}_model_comparison.png', dpi=300)
    plt.show()
    print(f"✅ Grafico di confronto salvato in: {track_name.lower().replace(' ', '_')}_model_comparison.png")

# --- 4. Submission Generation ---

def generate_submission(df_test, predictions, scores, team_name, members, model_name, feature_cols, metrics):
    """
    Generates the submission file in the required JSON format.
    """
    print(f"\n🚀 Generando submission per team: {team_name}")
    
    submission = {
        "team_info": {"team_name": team_name, "members": members, "track": "Track3", "submission_time": datetime.now().isoformat() + "Z"},
        "model_info": {"algorithm": model_name, "features_used": feature_cols},
        "results": {"total_test_samples": len(df_test), "anomalies_detected": int(predictions.sum()), "predictions": predictions.tolist(), "scores": scores.tolist()},
        "metrics": {"precision": round(metrics['precision'], 4), "recall": round(metrics['recall'], 4), "f1_score": round(metrics['f1'], 4), "auc_roc": round(metrics['auc'], 4)}
    }
    
    submission_filename = f"../submissions/submission_{team_name.lower().replace(' ', '_')}_track3.json"
    os.makedirs("../submissions", exist_ok=True)
    with open(submission_filename, 'w') as f: json.dump(submission, f, indent=2)
    print(f"✅ Submission salvata: {submission_filename}")
    return submission_filename

# --- 5. Main Pipeline ---

def main():
    """Main pipeline for Track 3."""
    print("="*60 + f"\n🎵 SIAE Hackathon - Track 3: Music Anomaly Detection\n" + "="*60)
    
    df_train, df_test = load_train_test_datasets()
    
    df_train = feature_engineering_music(df_train)
    df_test = feature_engineering_music(df_test)
    
    exclude_cols = ['track_id', 'artist_name', 'genre_top', 'is_anomaly', 'anomaly_type']
    # Correctly build feature_cols list
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    for col in feature_cols:
        if col not in df_test.columns: df_test[col] = 0
    df_test = df_test[df_train.columns.drop(['is_anomaly', 'anomaly_type'], errors='ignore')]

    best_model_info, best_model_name = train_and_evaluate_models(df_train, feature_cols)
    
    print(f"\n🔮 Applicando il modello migliore ({best_model_name}) al test set...")
    model, scaler, model_type = best_model_info['model'], best_model_info['scaler'], best_model_info['type']
    
    X_test = df_test[feature_cols].fillna(0).values
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'sklearn':
        test_scores = -model.score_samples(X_test_scaled) if best_model_name != 'LOF' else model.negative_outlier_factor_
        test_predictions = (model.predict(X_test_scaled) == -1).astype(int)
    elif model_type == 'clustering':
        test_scores = np.ones(len(X_test_scaled))
        test_predictions = (model.fit_predict(X_test_scaled) == -1).astype(int)
    elif model_type == 'pytorch':
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(X_test_scaled)), batch_size=256, shuffle=False)
        test_scores = get_autoencoder_scores(model, test_loader)
        # Re-calculate train scores with the final scaler to determine threshold
        X_train_scaled_mm = scaler.transform(df_train[feature_cols].fillna(0).values)
        train_loader_ae_full = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled_mm), torch.FloatTensor(X_train_scaled_mm)), batch_size=256, shuffle=False)
        train_scores_all = get_autoencoder_scores(model, train_loader_ae_full)
        threshold = np.quantile(train_scores_all[df_train['is_anomaly']==0], 0.95)
        test_predictions = (test_scores > threshold).astype(int)
    
    y_true_train = df_train['is_anomaly'].values
    X_train_final_scaled = scaler.transform(df_train[feature_cols].fillna(0).values)

    if model_type == 'sklearn':
        train_pred = (model.predict(X_train_final_scaled) == -1).astype(int)
        train_scores = -model.score_samples(X_train_final_scaled) if best_model_name != 'LOF' else model.negative_outlier_factor_
    elif model_type == 'clustering':
        train_pred = (model.fit_predict(X_train_final_scaled) == -1).astype(int)
        train_scores = np.ones(len(df_train))
    else: # PyTorch
        train_loader_full = DataLoader(TensorDataset(torch.FloatTensor(X_train_final_scaled), torch.FloatTensor(X_train_final_scaled)), batch_size=256, shuffle=False)
        train_scores = get_autoencoder_scores(model, train_loader_full)
        train_pred = (train_scores > threshold).astype(int)
    
    final_metrics = {
        'precision': precision_recall_fscore_support(y_true_train, train_pred, average='binary', zero_division=0)[0],
        'recall': precision_recall_fscore_support(y_true_train, train_pred, average='binary', zero_division=0)[1],
        'f1': precision_recall_fscore_support(y_true_train, train_pred, average='binary', zero_division=0)[2],
        'auc': roc_auc_score(y_true_train, train_scores) if model_type != 'clustering' else 0.5
    }

    generate_submission(df_test, test_predictions, test_scores, "DataPizzaGang", ["Mirko", "Giorgio"], best_model_name, feature_cols, final_metrics)
    
    print("\n🎉 PIPELINE TRACK 3 COMPLETATO!")

if __name__ == "__main__":
    main() 