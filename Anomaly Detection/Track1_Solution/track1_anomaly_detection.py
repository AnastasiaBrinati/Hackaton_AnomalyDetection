#!/usr/bin/env python3
"""
SIAE Anomaly Detection Hackathon - Track 1: Live Events
Strategia: Model Bake-Off & Ensemble Ibrido con Selezione Automatica

Questo script implementa un pipeline State-of-the-Art:
1.  Addestra una suite di modelli di anomaly detection (classici e SOTA da PyOD).
2.  Normalizza gli anomaly scores di ogni modello per un confronto equo.
3.  Crea un "Super Modello" tramite l'ensemble (media) degli score di tutti i modelli.
4.  Usa le etichette del training set per calibrare il threshold ottimale sia per i modelli individuali che per l'ensemble.
5.  Seleziona il performer migliore (singolo modello o ensemble) in base all'F1-Score.
6.  Usa il "campione" per generare le predizioni finali sul test set.
7.  Crea visualizzazioni di confronto per analizzare i risultati della competizione.
"""

# --- DIPENDENZE ---
# Assicurati di aver installato pyod: pip install pyod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, confusion_matrix

# Importa i modelli da PyOD e Sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder as AutoEncoderTorch

import warnings
import os
import json
from datetime import datetime
import sys

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- CONFIGURAZIONE ---
TEAM_NAME = "DatapizzaTheBest"
MEMBERS = ["Mirko", "Giorgio"]
GUARANTEED_COLUMNS = [
    'event_id', 'venue', 'city', 'event_date', 'attendance', 
    'capacity', 'n_songs', 'total_revenue', 'is_anomaly'
]

# (Le funzioni load_and_validate_datasets e create_features rimangono identiche alla versione precedente)
def load_and_validate_datasets():
    print("📥 Caricando e validando i dataset...")
    train_path = '../datasets/track1_live_events_train.csv'
    test_path = '../datasets/track1_live_events_test.csv'
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ File non trovati! Esegui 'python generate_datasets.py' prima.")
        sys.exit(1)
    df_train = pd.read_csv(train_path, parse_dates=['event_date'])
    df_test = pd.read_csv(test_path, parse_dates=['event_date'])
    for col in GUARANTEED_COLUMNS:
        if col not in df_train.columns:
            print(f"❌ Colonna '{col}' MANCANTE! Impossibile procedere.")
            sys.exit(1)
    print(f"✅ Dati caricati e validati: {len(df_train)} train, {len(df_test)} test.")
    return df_train, df_test

def create_features(df, training_df=None):
    print(f"🔧 Creando feature per { 'test' if training_df is not None else 'training'} set...")
    df = df.copy()
    # Feature esistenti
    df['occupancy_rate'] = (df['attendance'] / (df['capacity'] + 1e-6)).clip(0, 5)
    df['revenue_per_attendee'] = df['total_revenue'] / (df['attendance'] + 1e-6)
    df['day_of_week'] = df['event_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # NUOVE FEATURE
    df['songs_per_revenue'] = df['n_songs'] / (df['total_revenue'] + 1e-6)
    df['revenue_per_capacity'] = df['total_revenue'] / (df['capacity'] + 1e-6)
    df['is_high_n_songs'] = (df['n_songs'] > df['n_songs'].quantile(0.95)).astype(int)
    
    source_df = training_df if training_df is not None else df
    venue_stats = source_df.groupby('venue').agg(
        venue_avg_revenue=('total_revenue', 'mean'),
        venue_avg_attendance=('attendance', 'mean'),
    ).reset_index()
    df = df.merge(venue_stats, on='venue', how='left')
    df['dev_revenue_from_venue_avg'] = df['total_revenue'] / (df['venue_avg_revenue'] + 1e-6)
    df['dev_attendance_from_venue_avg'] = df['attendance'] / (df['venue_avg_attendance'] + 1e-6)
    for col in ['venue', 'city']:
        le = LabelEncoder()
        all_values = pd.concat([df[col], source_df[col]]).unique()
        le.fit(all_values)
        df[col + '_encoded'] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df

def create_models_to_train():
    """Definisce il dizionario di modelli che parteciperanno alla competizione."""
    return {
        'IsolationForest': IsolationForest(n_estimators=200, random_state=42, n_jobs=-1),
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=-1),
        'HBOS': HBOS(n_bins=50),
        'COPOD': COPOD(),
        'AutoEncoderTorch': AutoEncoderTorch(hidden_neuron_list=[64, 32, 32, 64], epoch_num=20, verbose=0)
    }

def train_and_evaluate_models(df_train, feature_cols, models_dict):
    """Addestra tutti i modelli, crea un ensemble, e seleziona il miglior performer."""
    print("\n" + "="*50)
    print("🏆 Inizio della Competizione tra Modelli (e l'Ensemble) 🏆")
    print("="*50)
    
    X_train = df_train[feature_cols].fillna(0)
    y_true = df_train['is_anomaly']
    
    trained_models = {}
    evaluation_results = {}
    all_normalized_scores = []

    for name, model in models_dict.items():
        try:
            print(f"🤖 Addestrando {name}...")
            model.fit(X_train)
            trained_models[name] = model

            anomaly_scores = model.decision_function(X_train)
            
            scaler = MinMaxScaler()
            normalized_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
            all_normalized_scores.append(normalized_scores)

            precision, recall, thresholds = precision_recall_curve(y_true, normalized_scores)
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
            
            best_f1_idx = np.argmax(f1_scores)
            best_f1 = f1_scores[best_f1_idx]
            best_threshold = thresholds[best_f1_idx]

            evaluation_results[name] = {
                'f1_score': best_f1,
                'threshold': best_threshold,
                'scaler': scaler
            }
            print(f"✅ {name} - F1-Score: {best_f1:.4f}\n")

        except Exception as e:
            print(f"❌ Errore durante l'addestramento di {name}: {e}\n")

    if not trained_models:
        print("❌ Nessun modello è stato addestrato con successo.")
        sys.exit(1)

    # --- ENSEMBLE STRATEGY ---
    print("-" * 20 + " 🤝 Creando l'Ensemble 🤝 " + "-" * 20)
    ensemble_scores = np.mean(np.array(all_normalized_scores), axis=0)
    
    precision_ens, recall_ens, thresholds_ens = precision_recall_curve(y_true, ensemble_scores)
    f1_scores_ens = (2 * precision_ens * recall_ens) / (precision_ens + recall_ens + 1e-9)
    best_f1_idx_ens = np.argmax(f1_scores_ens)
    best_f1_ens = f1_scores_ens[best_f1_idx_ens]
    best_threshold_ens = thresholds_ens[best_f1_idx_ens]

    print(f"Ensemble F1-Score: {best_f1_ens:.4f}\n")
    evaluation_results['Ensemble'] = {
        'f1_score': best_f1_ens,
        'threshold': best_threshold_ens,
        'scaler': None # L'ensemble non ha uno scaler, lavora su score già normalizzati
    }
        
    winner_name = max(evaluation_results, key=lambda k: evaluation_results[k]['f1_score'])
    print(f"🥇 Il modello vincitore è: {winner_name} con F1-Score = {evaluation_results[winner_name]['f1_score']:.4f}")
    
    return trained_models, evaluation_results, winner_name

def create_visualizations(evaluation_results, df_train_winner, y_true, y_pred, winner_name):
    """Crea visualizzazioni di confronto e di analisi del modello vincitore."""
    print("🎨 Creando visualizzazioni di confronto...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f'Analisi del Modello Vincitore: {winner_name}', fontsize=22, fontweight='bold')

    # 1. Grafico a Barre delle Performance (F1-Score)
    models = list(evaluation_results.keys())
    f1_scores = [res['f1_score'] for res in evaluation_results.values()]
    
    colors = []
    for model in models:
        if model == winner_name:
            colors.append('gold') # Colore per il vincitore
        elif model == 'Ensemble':
            colors.append('darkorange') # Colore speciale per l'Ensemble se non è il vincitore
        else:
            colors.append('skyblue')

    bars = axes[0, 0].bar(models, f1_scores, color=colors)
    axes[0, 0].set_ylabel('F1-Score Ottimale su Training Set')
    axes[0, 0].set_title('Confronto Performance Modelli', fontweight='bold', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    # 2. Precision-Recall Curve del modello VINCITORE
    precision, recall, _ = precision_recall_curve(y_true, df_train_winner['normalized_score'])
    pr_auc = auc(recall, precision)
    
    axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    best_f1 = evaluation_results[winner_name]['f1_score']
    axes[0, 1].plot([], [], ' ', label=f'Best F1-Score = {best_f1:.3f}') # Aggiungi F1 in legenda
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'Precision-Recall Curve per {winner_name}', fontweight='bold', fontsize=14)
    axes[0, 1].legend()

    # 3. Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Normale (Pred)', 'Anomalia (Pred)'],
                yticklabels=['Normale (True)', 'Anomalia (True)'])
    axes[1, 0].set_title('Matrice di Confusione sul Training Set', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Predizione')
    axes[1, 0].set_ylabel('Reale')

    # 4. Riepilogo Metriche
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred)
    axes[1, 1].axis('off')
    metrics_text = (
        f"**Metriche sul Training Set ({winner_name})**\n\n"
        f"F1-Score: {f1_val:.4f}\n"
        f"Precision: {precision_val:.4f}\n"
        f"Recall: {recall_val:.4f}\n"
        f"AUC-ROC: {pr_auc:.4f}\n\n"
        f"**Valori Matrice di Confusione:**\n"
        f"True Negatives: {cm[0, 0]}\n"
        f"False Positives: {cm[0, 1]}\n"
        f"False Negatives: {cm[1, 0]}\n"
        f"True Positives: {cm[1, 1]}"
    )
    axes[1, 1].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5))
    axes[1, 1].set_title('Riepilogo Performance', fontweight='bold', fontsize=14)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('track1_model_comparison.png', dpi=150)
    plt.show()
    print("✅ Grafici di confronto salvati in: track1_model_comparison.png")


def generate_submission(df_test, feature_cols, winner_name, training_metrics):
    # Simile alla versione precedente, ma adattato per prendere il nome del vincitore
    print(f"\n🚀 Generando submission per team: {TEAM_NAME} con il modello vincitore: {winner_name}")
    
    # Calcola precision e recall dal training set per il report
    y_true_train = training_metrics['y_true']
    y_pred_train = training_metrics['y_pred']
    precision_train = precision_score(y_true_train, y_pred_train)
    recall_train = recall_score(y_true_train, y_pred_train)
    auc_roc_train = roc_auc_score(y_true_train, training_metrics['scores'])

    submission = {
        "team_info": {"team_name": TEAM_NAME, "members": MEMBERS, "track": "Track1"},
        "model_info": {
            "algorithm": winner_name,
            "features_used": feature_cols,
        },
        "results": {
            "total_test_samples": len(df_test),
            "anomalies_detected": int(df_test['is_anomaly_predicted'].sum()),
            "predictions": df_test['is_anomaly_predicted'].tolist(),
            "scores": df_test['anomaly_score'].tolist()
        },
        "metrics": {
            "f1_score": round(training_metrics['f1_score'], 4),
            "precision": round(precision_train, 4),
            "recall": round(recall_train, 4),
            "auc_roc": round(auc_roc_train, 4)
        }
    }
    submission_filename = f"../submissions/submission_{TEAM_NAME.lower().replace(' ', '_')}_track1.json"
    os.makedirs("../submissions", exist_ok=True)
    with open(submission_filename, 'w') as f: json.dump(submission, f, indent=2)
    print(f"✅ Submission salvata: {submission_filename}")

def main():
    df_train, df_test = load_and_validate_datasets()

    df_train_feat = create_features(df_train)
    df_test_feat = create_features(df_test, training_df=df_train)

    feature_cols = [col for col in df_train_feat.columns if col not in GUARANTEED_COLUMNS + ['event_date', 'anomaly_type']]
    
    models_dict = create_models_to_train()
    
    trained_models, eval_results, winner_name = train_and_evaluate_models(df_train_feat, feature_cols, models_dict)

    X_test = df_test_feat[feature_cols].fillna(0)
    
    # Usa il modello/ensemble vincitore per le predizioni finali
    if winner_name == 'Ensemble':
        print("🚀 Usando l'Ensemble per le predizioni finali...")
        all_test_scores = []
        for name, model in trained_models.items():
            scaler = eval_results[name]['scaler']
            test_scores_raw = model.decision_function(X_test)
            normalized_scores_single = scaler.transform(test_scores_raw.reshape(-1, 1)).flatten()
            all_test_scores.append(normalized_scores_single)
        
        final_test_scores_normalized = np.mean(np.array(all_test_scores), axis=0)
        winner_threshold = eval_results['Ensemble']['threshold']
        test_predictions = (final_test_scores_normalized >= winner_threshold).astype(int)
        
        df_test_feat['anomaly_score'] = final_test_scores_normalized # Score già normalizzato
        df_test_feat['normalized_score'] = final_test_scores_normalized
        df_test_feat['is_anomaly_predicted'] = test_predictions

    else: # Un modello singolo ha vinto
        print(f"🚀 Usando il modello vincitore '{winner_name}' per le predizioni finali...")
        winner_model = trained_models[winner_name]
        winner_threshold = eval_results[winner_name]['threshold']
        winner_scaler = eval_results[winner_name]['scaler']

        test_scores_raw = winner_model.decision_function(X_test)
        normalized_test_scores = winner_scaler.transform(test_scores_raw.reshape(-1, 1)).flatten()
        test_predictions = (normalized_test_scores >= winner_threshold).astype(int)
        
        df_test_feat['anomaly_score'] = test_scores_raw
        df_test_feat['normalized_score'] = normalized_test_scores
        df_test_feat['is_anomaly_predicted'] = test_predictions

    # Prepara le metriche complete dal training set per la submission
    y_true_train = df_train_feat['is_anomaly']
    if winner_name == 'Ensemble':
        all_train_scores = []
        for name, model in trained_models.items():
            scaler = eval_results[name]['scaler']
            train_scores_raw = model.decision_function(df_train_feat[feature_cols].fillna(0))
            all_train_scores.append(scaler.transform(train_scores_raw.reshape(-1, 1)).flatten())
        
        final_train_scores_normalized = np.mean(np.array(all_train_scores), axis=0)
        y_pred_train = (final_train_scores_normalized >= eval_results['Ensemble']['threshold']).astype(int)
        final_scores_for_auc = final_train_scores_normalized
    else:
        winner_model = trained_models[winner_name]
        winner_scaler = eval_results[winner_name]['scaler']
        train_scores_raw = winner_model.decision_function(df_train_feat[feature_cols].fillna(0))
        final_train_scores_normalized = winner_scaler.transform(train_scores_raw.reshape(-1, 1)).flatten()
        y_pred_train = (final_train_scores_normalized >= eval_results[winner_name]['threshold']).astype(int)
        final_scores_for_auc = final_train_scores_normalized

    df_train_feat['normalized_score'] = final_train_scores_normalized

    training_metrics = {
        "f1_score": eval_results[winner_name]['f1_score'],
        "y_true": y_true_train,
        "y_pred": y_pred_train,
        "scores": final_scores_for_auc
    }
    
    generate_submission(df_test_feat, feature_cols, winner_name, training_metrics)
    
    create_visualizations(eval_results, df_train_feat, y_true_train, y_pred_train, winner_name)

    print("\n🎉 PIPELINE COMPLETATO! 🎉")

if __name__ == "__main__":
    try:
        import pyod
    except ImportError:
        print("❌ Libreria PyOD non trovata. Per favore, installala con:")
        print("pip install pyod")
        sys.exit(1)
    main()