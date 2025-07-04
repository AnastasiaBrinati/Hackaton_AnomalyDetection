# ��️ Track 2: Document Fraud Detection

## 🎯 Challenge Overview

**Obiettivo**: Sviluppare un sistema di rilevamento frodi per documenti digitali SIAE utilizzando tecniche di computer vision, OCR e machine learning.

## 📋 Problema da Risolvere

La SIAE gestisce migliaia di documenti digitali ogni giorno:
- **Contratti editoriali**
- **Licenze di esecuzione**
- **Dichiarazioni di musica live**
- **Cessioni di diritti**
- **Registrazioni di opere**
- **Richieste di risarcimento**

### 🚨 Tipi di Frodi da Rilevare

1. **Alterazioni Digitali**: Modifiche ai testi originali, manipolazioni delle immagini
2. **Firme Contraffatte**: Firme mancanti o sospette, inconsistenze nella grafia
3. **Template Fraud**: Utilizzo di template non autorizzati, assenza di watermark SIAE
4. **Manipolazioni OCR**: Testi alterati post-OCR, caratteri speciali sospetti

## 🔧 Tecnologie Utilizzate

### Computer Vision
- **OpenCV**: Elaborazione immagini
- **PIL**: Manipolazione immagini
- **Edge Detection**: Rilevamento artefatti

### Machine Learning
- **Isolation Forest**: Rilevamento anomalie
- **DBSCAN**: Clustering documenti sospetti
- **Feature Engineering**: Caratteristiche specifiche per documenti

## 📊 Dataset Sintetico

- **5,000 documenti** generati sinteticamente
- **~17% documenti fraudolenti** (850 frodi)
- **6 tipi di documento** SIAE
- **Multiple tipologie di frodi** per training realistico

## 🚀 Come Eseguire la Soluzione

### 1. Setup Environment
```bash
cd Track2_Solution
pip install -r requirements.txt
```

### 2. Esecuzione Script
```bash
python track2_document_fraud_detection.py
```

### 3. Output Generati
- **CSV**: `documents_fraud_detection.csv`
- **Visualizzazioni**: `document_fraud_detection_results.png`
- **Submission**: `../submissions/submission_me&giorgio_track2.json`

## 📈 Features Utilizzate

### Features Base Documento
- `page_count`: Numero pagine
- `file_size_kb`: Dimensione file
- `resolution_dpi`: Risoluzione immagine
- `text_confidence_avg`: Confidenza media OCR
- `signature_regions`: Regioni firme rilevate

### Features Autenticità SIAE
- `siae_watermark_detected`: Watermark SIAE presente
- `official_seal_detected`: Sigillo ufficiale presente
- `metadata_consistency`: Consistenza metadati

### Features Engineered
- `text_quality_score`: Score qualità testo
- `visual_integrity_score`: Score integrità visiva
- `siae_authenticity_score`: Score autenticità SIAE

## 🎯 Modelli Applicati

### 1. Isolation Forest
- **Contamination**: 0.12 (12% frodi attese)
- **N_estimators**: 200
- **Obiettivo**: Rilevamento anomalie generali

### 2. DBSCAN Clustering
- **Eps**: 0.8
- **Min_samples**: 5
- **Obiettivo**: Clustering documenti sospetti

## 📊 Performance Attese

### Metriche Target
- **Precision**: ~0.82 (82% delle frodi rilevate sono vere)
- **Recall**: ~0.76 (76% delle frodi vengono rilevate)
- **F1-Score**: ~0.79 (bilanciamento precision-recall)
- **AUC-ROC**: ~0.87 (ottima separazione classi)

## 🏆 Submissione Track 2

Il sistema genera automaticamente un file JSON con:
- Informazioni del team
- Metriche di performance
- Risultati specifici per Track 2
- Breakdown per tipo di frode rilevata

---

**🚀 Buona fortuna con il Track 2! Rileva le frodi e proteggi i diritti d'autore! 🔍**
