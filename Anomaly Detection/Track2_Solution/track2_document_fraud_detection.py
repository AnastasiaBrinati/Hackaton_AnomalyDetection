#!/usr/bin/env python3
"""
SIAE Hackathon - Track 2: Document Fraud Detection
Rilevamento di documenti fraudolenti con approccio multi-modello unsupervised.

Questo script implementa un pipeline avanzato per il Track 2:
1. Carica i dataset di training e test.
2. Esegue un feature engineering specifico per la frode documentale.
3. Addestra e confronta diversi modelli unsupervised:
    - Autoencoder (con PyTorch)
    - One-Class SVM
    - Local Outlier Factor (LOF)
    - Isolation Forest
4. Visualizza le performance dei modelli a confronto.
5. Seleziona automaticamente il modello migliore in base all'F1-Score sul training set.
6. Applica il modello migliore al test set per generare le predizioni.
7. Crea il file di submission nel formato corretto.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
import time

# Importazioni per i modelli di ML
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Importazioni per PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch non è installato. L'Autoencoder non sarà disponibile.")
    print("Per installarlo, esegui: pip install torch")
    torch = None

warnings.filterwarnings('ignore')
np.random.seed(42)
if torch:
    torch.manual_seed(42)

# --- 1. Caricamento Dati ---

def load_train_test_datasets():
    """
    Carica i dataset di train e test per Track 2.
    """
    print("📥 Caricando dataset di training e test...")
    
    train_path = '../datasets/track2_documents_train.csv'
    if not os.path.exists(train_path):
        print(f"❌ File di training non trovato: {train_path}", file=sys.stderr)
        print("💡 Assicurati di aver eseguito 'python generate_datasets.py' nella directory principale.", file=sys.stderr)
        sys.exit(1)
    
    df_train = pd.read_csv(train_path)
    print(f"✅ Dataset di train caricato: {len(df_train)} documenti")
    
    test_path = '../datasets/track2_documents_test.csv'
    if not os.path.exists(test_path):
        print(f"❌ File di test non trovato: {test_path}", file=sys.stderr)
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    print(f"✅ Dataset di test caricato: {len(df_test)} documenti")
    
    return df_train, df_test

# --- 2. Feature Engineering ---

def feature_engineering_documents(df):
    """
    Feature engineering specifico per il rilevamento di frodi documentali.
    Utilizza le colonne reali presenti nel dataset generato.
    """
    print("🔧 Eseguendo feature engineering per documenti (versione corretta)...")
    
    df = df.copy()

    # Rinomina le colonne per coerenza e pulizia
    df = df.rename(columns={
        'doc_id': 'document_id',
        'signature_similarity': 'signature_similarity_score',
        'metadata_validity': 'metadata_validity_score',
        'quality_score': 'doc_quality_score'
    })

    # Feature di rapporto e interazione
    df['images_per_page'] = df['num_images'] / (df['num_pages'] + 1e-6)
    df['quality_to_signature_ratio'] = df['doc_quality_score'] / (df['signature_similarity_score'] + 1e-6)
    df['metadata_x_quality'] = df['metadata_validity_score'] * df['doc_quality_score']
    
    # Feature polinomiali per catturare relazioni non lineari
    df['pages_squared'] = df['num_pages']**2
    df['quality_squared'] = df['doc_quality_score']**2

    # Feature semplici ma potenzialmente utili
    df['has_images'] = (df['num_images'] > 0).astype(int)
    
    # Rilevamento di valori anomali/estremi
    df['is_high_pages'] = (df['num_pages'] > df['num_pages'].quantile(0.95)).astype(int)
    df['is_low_quality'] = (df['doc_quality_score'] < df['doc_quality_score'].quantile(0.05)).astype(int)

    print(f"✅ Feature engineering completato: {df.shape[1]} colonne totali")
    return df

# --- 3. Implementazione Modelli ---

# Autoencoder con PyTorch
if torch:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid() # Uscita tra 0 e 1, richiede dati scalati in questo range
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def train_autoencoder(model, data_loader, epochs=50, learning_rate=1e-3):
    """Funzione per addestrare l'autoencoder."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    start_time = time.time()
    for epoch in range(epochs):
        for data in data_loader:
            inputs, _ = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    end_time = time.time()
    print(f"⏱️ Tempo di addestramento Autoencoder: {end_time - start_time:.2f} secondi")
    return model

def get_autoencoder_scores(model, data_loader):
    """Calcola l'errore di ricostruzione per ogni campione."""
    model.eval()
    reconstruction_errors = []
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for data in data_loader:
            inputs, _ = data
            outputs = model(inputs)
            errors = criterion(outputs, inputs).mean(dim=1)
            reconstruction_errors.extend(errors.numpy())
    return np.array(reconstruction_errors)

def train_and_evaluate_models(df_train, feature_cols):
    """
    Addestra, valuta e confronta diversi modelli unsupervised.
    """
    print("\n🤖 Addestrando e confrontando modelli unsupervised...")
    
    # Prepara i dati
    X_train = df_train[feature_cols].fillna(0).values
    y_true = df_train['is_fraudulent'].values
    
    # Scaler standard per la maggior parte dei modelli
    scaler_std = StandardScaler()
    X_train_scaled_std = scaler_std.fit_transform(X_train)
    
    # Scaler MinMax per l'Autoencoder (richiede input tra 0 e 1)
    scaler_mm = MinMaxScaler()
    X_train_scaled_mm = scaler_mm.fit_transform(X_train)

    models = {}
    results = {}

    # --- Isolation Forest ---
    print("\n--- 🌲 Isolation Forest ---")
    iso_forest = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    iso_forest.fit(X_train_scaled_std)
    models['Isolation Forest'] = {'model': iso_forest, 'scaler': scaler_std, 'type': 'sklearn'}

    # --- One-Class SVM ---
    print("\n--- 🧠 One-Class SVM ---")
    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    oc_svm.fit(X_train_scaled_std)
    models['One-Class SVM'] = {'model': oc_svm, 'scaler': scaler_std, 'type': 'sklearn'}

    # --- Local Outlier Factor (LOF) ---
    print("\n--- 🌐 Local Outlier Factor ---")
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.05, novelty=True)
    lof.fit(X_train_scaled_std)
    models['LOF'] = {'model': lof, 'scaler': scaler_std, 'type': 'sklearn'}

    # --- Autoencoder (PyTorch) ---
    if torch:
        print("\n--- 🔥 Autoencoder (PyTorch) ---")
        input_dim = X_train_scaled_mm.shape[1]
        autoencoder = Autoencoder(input_dim)
        
        # Prepara DataLoader solo con dati non-fraudolenti per l'addestramento
        # Questo è un approccio semi-supervised
        X_normal = X_train_scaled_mm[y_true == 0]
        train_dataset_ae = TensorDataset(torch.FloatTensor(X_normal), torch.FloatTensor(X_normal))
        train_loader_ae = DataLoader(train_dataset_ae, batch_size=64, shuffle=True)
        
        autoencoder = train_autoencoder(autoencoder, train_loader_ae, epochs=50)
        
        # Calcola gli score su tutto il dataset
        full_dataset_ae = TensorDataset(torch.FloatTensor(X_train_scaled_mm), torch.FloatTensor(X_train_scaled_mm))
        full_loader_ae = DataLoader(full_dataset_ae, batch_size=256, shuffle=False)
        
        models['Autoencoder'] = {'model': autoencoder, 'scaler': scaler_mm, 'type': 'pytorch', 'loader': full_loader_ae}

    # Valutazione e confronto
    print("\n📊 Valutando performance sul training set...")
    best_model_name = None
    best_f1 = -1

    for name, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        model_type = model_info['type']
        
        X_scaled = scaler.transform(X_train)

        if model_type == 'sklearn':
            scores = model.score_samples(X_scaled)
            # Isolation Forest e LOF hanno score invertiti (più basso = più anomalo)
            if name in ['Isolation Forest', 'LOF']:
                scores = -scores
            predictions_raw = model.predict(X_scaled)
            y_pred = (predictions_raw == -1).astype(int)
        
        elif model_type == 'pytorch':
            full_loader = model_info['loader']
            scores = get_autoencoder_scores(model, full_loader)
            # Per Autoencoder, più alto è l'errore, più è anomalo
            threshold = np.quantile(scores[y_true==0], 0.95) # Soglia basata sui dati normali
            y_pred = (scores > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        auc_roc = roc_auc_score(y_true, scores)
        
        results[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc_roc}
        print(f"  - {name:20s} | F1: {f1:.3f} | AUC: {auc_roc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n🏆 Modello migliore: {best_model_name} (F1-Score: {best_f1:.3f})")
    
    # Grafico di confronto
    create_comparison_chart(results)
    
    # Matrice di confusione per il modello migliore
    if best_model_name:
        y_pred_best = models[best_model_name]['predictions']
        create_confusion_matrix(y_true, y_pred_best, best_model_name, "Track 2")
        
    return models[best_model_name], best_model_name, results[best_model_name]

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


def create_comparison_chart(results):
    """Crea un grafico a barre per confrontare le performance dei modelli."""
    df_results = pd.DataFrame(results).T.sort_values(by='f1', ascending=False)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_results[['f1', 'auc', 'precision', 'recall']].plot(kind='bar', ax=ax, colormap='viridis')
    
    ax.set_title('Confronto Performance Modelli - Track 2', fontsize=16, fontweight='bold')
    ax.set_xlabel('Modello', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(df_results.index, rotation=45, ha='right')
    ax.legend(title='Metriche')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Aggiungi valori sulle barre F1
    for i, f1_val in enumerate(df_results['f1']):
        ax.text(i, f1_val + 0.01, f'{f1_val:.3f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('track2_model_comparison.png', dpi=300)
    plt.show()
    print("✅ Grafico di confronto salvato in: track2_model_comparison.png")

# --- 4. Generazione Submission ---

def generate_submission(df_test, predictions, scores, team_name, members, model_name, feature_cols, metrics):
    """
    Genera il file di submission nel formato JSON richiesto.
    """
    print(f"\n🚀 Generando submission per team: {team_name}")
    
    submission = {
        "team_info": {
            "team_name": team_name,
            "members": members,
            "track": "Track2",
            "submission_time": datetime.now().isoformat() + "Z"
        },
        "model_info": {
            "algorithm": model_name,
            "features_used": feature_cols,
        },
        "results": {
            "total_test_samples": len(df_test),
            "anomalies_detected": int(predictions.sum()),
            "predictions": predictions.tolist(),
            "scores": scores.tolist()
        },
        "metrics": {
             "precision": round(metrics['precision'], 4),
             "recall": round(metrics['recall'], 4),
             "f1_score": round(metrics['f1'], 4),
             "auc_roc": round(metrics['auc'], 4)
        }
    }
    
    submission_filename = f"../submissions/submission_{team_name.lower().replace(' ', '_')}_track2.json"
    os.makedirs("../submissions", exist_ok=True)
    
    with open(submission_filename, 'w') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Submission salvata: {submission_filename}")
    return submission_filename

# --- 5. Pipeline Principale ---

def main():
    """Pipeline principale per Track 2."""
    print("="*60)
    print("🕵️  SIAE Hackathon - Track 2: Document Fraud Detection")
    print("="*60)
    
    # 1. Carica dati
    df_train, df_test = load_train_test_datasets()
    
    # 2. Feature engineering
    df_train = feature_engineering_documents(df_train)
    df_test = feature_engineering_documents(df_test)
    
    # Definisci le feature da usare (escludendo ID e target)
    exclude_cols = ['document_id', 'is_fraudulent', 'fraud_type']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    # Allinea colonne tra train e test
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    df_test = df_test[df_train.columns.drop(['is_fraudulent', 'fraud_type'])]

    # 3. Addestra e seleziona il modello migliore
    best_model_info, best_model_name, final_metrics = train_and_evaluate_models(df_train, feature_cols)
    
    # 4. Applica il modello migliore al test set
    print(f"\n🔮 Applicando il modello migliore ({best_model_name}) al test set...")
    model = best_model_info['model']
    scaler = best_model_info['scaler']
    model_type = best_model_info['type']
    
    X_test = df_test[feature_cols].fillna(0).values
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'sklearn':
        test_scores = model.score_samples(X_test_scaled)
        if best_model_name in ['Isolation Forest', 'LOF']:
            test_scores = -test_scores
        test_predictions_raw = model.predict(X_test_scaled)
        test_predictions = (test_predictions_raw == -1).astype(int)
    
    elif model_type == 'pytorch':
        test_dataset_ae = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(X_test_scaled))
        test_loader_ae = DataLoader(test_dataset_ae, batch_size=256, shuffle=False)
        test_scores = get_autoencoder_scores(model, test_loader_ae)
        
        # Usa la stessa soglia calcolata sui dati di training normali
        y_true_train = df_train['is_fraudulent'].values
        X_train_scaled_mm = scaler.transform(df_train[feature_cols].fillna(0).values)
        train_scores_dataset = TensorDataset(torch.FloatTensor(X_train_scaled_mm), torch.FloatTensor(X_train_scaled_mm))
        train_scores = get_autoencoder_scores(model, DataLoader(train_scores_dataset, batch_size=256, shuffle=False))
        threshold = np.quantile(train_scores[y_true_train==0], 0.95)
        test_predictions = (test_scores > threshold).astype(int)

    # 5. Genera submission
    team_name = "DataPizzaGang"
    members = ["Mirko", "Giorgio"]
    
    # Recupera le metriche del modello migliore dal training per il report
    y_true_train = df_train['is_fraudulent'].values
    X_train_scaled = scaler.transform(df_train[feature_cols].fillna(0).values)
    
    if model_type == 'sklearn':
        train_pred_raw = model.predict(X_train_scaled)
        train_pred = (train_pred_raw == -1).astype(int)
        train_scores = model.score_samples(X_train_scaled)
        if best_model_name in ['Isolation Forest', 'LOF']:
            train_scores = -train_scores
    elif model_type == 'pytorch':
         train_dataset_final = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(X_train_scaled))
         train_loader = DataLoader(train_dataset_final, batch_size=256, shuffle=False)
         train_scores = get_autoencoder_scores(model, train_loader)
         train_pred = (train_scores > threshold).astype(int)

    # Ricalcola le metriche finali per sicurezza (anche se già presenti)
    final_metrics_recalc = {
        'precision': precision_recall_fscore_support(y_true_train, train_pred, average='binary')[0],
        'recall': precision_recall_fscore_support(y_true_train, train_pred, average='binary')[1],
        'f1': precision_recall_fscore_support(y_true_train, train_pred, average='binary')[2],
        'auc': roc_auc_score(y_true_train, train_scores)
    }

    generate_submission(
        df_test, test_predictions, test_scores, 
        team_name, members, best_model_name, feature_cols, final_metrics_recalc
    )
    
    print("\n🎉 PIPELINE TRACK 2 COMPLETATO!")

if __name__ == "__main__":
    main()
