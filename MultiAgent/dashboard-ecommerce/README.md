# 📊 Dashboard Interattiva Vendite E-commerce

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-ff69b4.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**[📍 Visualizza la Dashboard Online](URL_DELLA_TUA_APP_STREAMLIT_QUI)**

Una dashboard web interattiva per l'analisi delle vendite di un e-commerce fittizio, costruita con Streamlit e Plotly. L'applicazione permette di esplorare i dati di vendita annuali, applicare filtri dinamici e visualizzare insight chiave attraverso grafici interattivi.


*(Nota: Sostituisci questo link con uno screenshot del tuo progetto una volta online)*

## 🚀 Funzionalità Principali

*   **📈 Visualizzazioni Dinamiche**: Grafici interattivi che si aggiornano in tempo reale in base ai filtri selezionati.
    *   **Andamento Mensile**: Un line chart per monitorare i ricavi nel tempo.
    *   **Distribuzione Categorie**: Un pie chart per analizzare il peso di ogni categoria sui ricavi.
    *   **Clienti per Categoria**: Un bar chart per confrontare il numero di clienti e il loro ticket medio.
    *   **Heatmap Stagionalità**: Una heatmap per identificare i mesi di picco per ogni categoria.
*   **🔍 Filtri Interattivi**: Controlli nella sidebar per filtrare i dati per:
    *   **Categoria di Prodotto** (Elettronica, Abbigliamento, Casa).
    *   **Periodo Temporale** (intervallo di mesi).
*   **💡 KPI in Tempo Reale**: Metriche chiave come Ricavi Totali, Clienti Totali e Ticket Medio calcolate dinamicamente.
*   **📊 Dati Fittizi Realistici**: Il dataset è generato programmaticamente con pattern stagionali per simulare uno scenario di e-commerce reale.
*   **📥 Download Dati**: Funzionalità per scaricare il dataset completo in formato CSV.

## 🛠️ Stack Tecnologico

*   **Linguaggio**: Python 3.9
*   **Framework Web**: [Streamlit](https://streamlit.io/)
*   **Librerie Dati**: [Pandas](https://pandas.pydata.org/) & [Numpy](https://numpy.org/)
*   **Visualizzazione Dati**: [Plotly](https://plotly.com/python/)
*   **Deployment**: Streamlit Community Cloud

## ⚙️ Installazione e Avvio in Locale

Segui questi passaggi per eseguire la dashboard sul tuo computer.

### Prerequisiti

*   Avere [Anaconda o Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installato.

### 1. Clona o Scarica il Repository

```bash
git clone https://github.com/TUO_USERNAME/dashboard-ecommerce-streamlit.git
cd dashboard-ecommerce-streamlit
```

### 2. Crea e Attiva l'Ambiente Conda

Crea un ambiente virtuale per isolare le dipendenze del progetto.

```bash
# Crea l'ambiente
conda create --name dashboard_env python=3.9

# Attiva l'ambiente
conda activate dashboard_env
```

### 3. Installa le Dipendenze

Installa tutte le librerie necessarie tramite il file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Esegui l'Applicazione

Lancia la dashboard con il comando di Streamlit.

```bash
streamlit run ecommerce-dashboard.py
```

L'applicazione si aprirà automaticamente nel tuo browser all'indirizzo `http://localhost:8501`.

## 📁 Struttura del Progetto

```
dashboard-ecommerce-streamlit/
├── ecommerce-dashboard.py    # Script principale dell'applicazione Streamlit
├── requirements.txt          # Elenco delle dipendenze Python
└── README.md                 # Questo file di documentazione
```

## ☁️ Deployment

Questa applicazione è stata pensata per essere distribuita su **Streamlit Community Cloud**. È sufficiente collegare il proprio account GitHub, selezionare questo repository e il file `ecommerce-dashboard.py` per pubblicare la dashboard online gratuitamente.

## 📜 Licenza

Questo progetto è rilasciato sotto la Licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.

---
_Dashboard sviluppata come progetto dimostrativo. I dati sono simulati e non rappresentano vendite reali._
