"""
Dashboard Interattiva Vendite E-commerce
Autore: Assistant
Data: 2025
Descrizione: Dashboard per l'analisi delle vendite mensili con grafici interattivi
"""

# === IMPORTAZIONE LIBRERIE ===
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# === CONFIGURAZIONE PAGINA ===
st.set_page_config(
    page_title="Dashboard Vendite E-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STILI CSS PERSONALIZZATI ===
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:16px !important;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === GENERAZIONE DATASET FITTIZIO ===
@st.cache_data
def genera_dataset():
    """
    Genera un dataset simulato di vendite mensili per l'anno 2024
    con pattern stagionali realistici
    """
    np.random.seed(42)  # Per riproducibilità
    
    mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
            'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
    
    # Pattern stagionale per i ricavi (picco estivo e natalizio)
    pattern_ricavi = [0.8, 0.85, 0.9, 0.95, 1.0, 1.2, 
                      1.3, 1.25, 1.1, 1.0, 1.15, 1.4]
    
    # Categorie e loro peso sui ricavi totali
    categorie = ['Elettronica', 'Abbigliamento', 'Casa']
    pesi_categorie = [0.45, 0.35, 0.20]
    
    data = []
    
    for i, mese in enumerate(mesi):
        # Ricavi base con variazione stagionale
        ricavi_base = 50000 * pattern_ricavi[i]
        
        for j, categoria in enumerate(categorie):
            # Aggiungere variazione per categoria
            variazione = np.random.uniform(0.9, 1.1)
            ricavi = ricavi_base * pesi_categorie[j] * variazione
            
            # Numero clienti proporzionale ai ricavi con rumore
            numero_clienti = int(ricavi / 150 * np.random.uniform(0.8, 1.2))
            
            data.append({
                'mese': mese,
                'mese_num': i + 1,
                'ricavi': round(ricavi, 2),
                'numero_clienti': numero_clienti,
                'categoria_prodotto': categoria
            })
    
    return pd.DataFrame(data)

# Caricamento dati
df = genera_dataset()

# === SIDEBAR CON FILTRI ===
st.sidebar.header("🔍 Filtri")

# Filtro categoria con opzione "Tutte"
categorie_disponibili = ['Tutte'] + list(df['categoria_prodotto'].unique())
categoria_selezionata = st.sidebar.selectbox(
    "Seleziona Categoria:",
    categorie_disponibili,
    help="Filtra i dati per categoria di prodotto"
)

# Filtro periodo
mesi_disponibili = df['mese'].unique()
periodo_selezionato = st.sidebar.select_slider(
    "Seleziona Periodo:",
    options=list(mesi_disponibili),
    value=(mesi_disponibili[0], mesi_disponibili[-1]),
    help="Seleziona l'intervallo di mesi da visualizzare"
)

# === APPLICAZIONE FILTRI ===
df_filtrato = df.copy()

# Filtro categoria
if categoria_selezionata != 'Tutte':
    df_filtrato = df_filtrato[df_filtrato['categoria_prodotto'] == categoria_selezionata]

# Filtro periodo
mese_inizio = list(mesi_disponibili).index(periodo_selezionato[0]) + 1
mese_fine = list(mesi_disponibili).index(periodo_selezionato[1]) + 1
df_filtrato = df_filtrato[(df_filtrato['mese_num'] >= mese_inizio) & 
                          (df_filtrato['mese_num'] <= mese_fine)]

# === HEADER E METRICHE PRINCIPALI ===
st.title("📊 Dashboard Vendite E-commerce 2024")
st.markdown("---")

# Calcolo metriche
ricavi_totali = df_filtrato['ricavi'].sum()
clienti_totali = df_filtrato['numero_clienti'].sum()
ticket_medio = ricavi_totali / clienti_totali if clienti_totali > 0 else 0

# Display metriche in colonne
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 Ricavi Totali",
        value=f"€ {ricavi_totali:,.0f}",
        delta=f"+{(ricavi_totali/12):,.0f} medio/mese"
    )

with col2:
    st.metric(
        label="👥 Clienti Totali",
        value=f"{clienti_totali:,}",
        delta=f"~{(clienti_totali/12):.0f} medio/mese"
    )

with col3:
    st.metric(
        label="🛒 Ticket Medio",
        value=f"€ {ticket_medio:.2f}"
    )

with col4:
    st.metric(
        label="📦 Categorie",
        value=len(df_filtrato['categoria_prodotto'].unique())
    )

st.markdown("---")

# === LAYOUT A DUE COLONNE PER I GRAFICI ===
col_sx, col_dx = st.columns(2)

# === COLONNA SINISTRA ===
with col_sx:
    # 1. LINE CHART - Ricavi totali per mese
    st.subheader("📈 Andamento Ricavi Mensili")
    
    # Aggregazione dati per mese
    ricavi_mensili = df_filtrato.groupby(['mese', 'mese_num'])['ricavi'].sum().reset_index()
    ricavi_mensili = ricavi_mensili.sort_values('mese_num')
    
    # Creazione grafico
    fig_line = px.line(
        ricavi_mensili, 
        x='mese', 
        y='ricavi',
        title='Trend Ricavi Totali',
        markers=True,
        line_shape='spline'
    )
    
    # Personalizzazione
    fig_line.update_traces(
        line_color='#1f77b4',
        line_width=3,
        marker_size=8
    )
    
    fig_line.update_layout(
        xaxis_title="Mese",
        yaxis_title="Ricavi (€)",
        hovermode='x unified',
        showlegend=False,
        yaxis_tickformat=',.0f'
    )
    
    # Aggiunta annotazioni per valori massimi e minimi
    max_ricavo = ricavi_mensili.loc[ricavi_mensili['ricavi'].idxmax()]
    min_ricavo = ricavi_mensili.loc[ricavi_mensili['ricavi'].idxmin()]
    
    fig_line.add_annotation(
        x=max_ricavo['mese'],
        y=max_ricavo['ricavi'],
        text=f"Max: €{max_ricavo['ricavi']:,.0f}",
        showarrow=True,
        arrowhead=2
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Descrizione
    st.markdown("""
    <div class='medium-font'>
    📊 <b>Analisi:</b> Il grafico mostra l'andamento dei ricavi durante l'anno. 
    Si notano picchi stagionali in estate (luglio-agosto) e durante il periodo natalizio (dicembre), 
    tipici del settore e-commerce.
    </div>
    """, unsafe_allow_html=True)
    
    # 3. PIE CHART - Distribuzione ricavi per categoria
    st.subheader("🥧 Distribuzione Ricavi per Categoria")
    
    # Aggregazione per categoria
    ricavi_categoria = df_filtrato.groupby('categoria_prodotto')['ricavi'].sum().reset_index()
    
    # Creazione grafico
    fig_pie = px.pie(
        ricavi_categoria,
        values='ricavi',
        names='categoria_prodotto',
        title='Percentuale Ricavi per Categoria',
        hole=0.4  # Donut chart
    )
    
    # Personalizzazione
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>' +
                      'Ricavi: €%{value:,.0f}<br>' +
                      'Percentuale: %{percent}<br>' +
                      '<extra></extra>'
    )
    
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Descrizione
    st.markdown("""
    <div class='medium-font'>
    📊 <b>Analisi:</b> L'Elettronica domina le vendite (~45%), seguita dall'Abbigliamento (~35%). 
    La categoria Casa rappresenta una quota minore ma stabile (~20%).
    </div>
    """, unsafe_allow_html=True)

# === COLONNA DESTRA ===
with col_dx:
    # 2. BAR CHART - Numero clienti per categoria
    st.subheader("📊 Numero Clienti per Categoria")
    
    # Aggregazione per categoria
    clienti_categoria = df_filtrato.groupby('categoria_prodotto').agg({
        'numero_clienti': 'sum',
        'ricavi': 'sum'
    }).reset_index()
    
    # Calcolo ticket medio per categoria
    clienti_categoria['ticket_medio'] = (clienti_categoria['ricavi'] / 
                                         clienti_categoria['numero_clienti'])
    
    # Creazione grafico
    fig_bar = px.bar(
        clienti_categoria,
        x='categoria_prodotto',
        y='numero_clienti',
        title='Distribuzione Clienti per Categoria',
        text='numero_clienti',
        color='ticket_medio',
        color_continuous_scale='Blues'
    )
    
    # Personalizzazione
    fig_bar.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside'
    )
    
    fig_bar.update_layout(
        xaxis_title="Categoria Prodotto",
        yaxis_title="Numero Clienti",
        showlegend=False,
        coloraxis_colorbar_title="Ticket<br>Medio (€)"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Descrizione
    st.markdown("""
    <div class='medium-font'>
    📊 <b>Analisi:</b> L'Elettronica attrae il maggior numero di clienti, 
    con un ticket medio elevato. L'Abbigliamento ha un buon volume di clienti con acquisti frequenti.
    </div>
    """, unsafe_allow_html=True)
    
    # GRAFICO AGGIUNTIVO - Heatmap stagionalità
    st.subheader("🗓️ Heatmap Stagionalità Vendite")
    
    # Preparazione dati per heatmap
    heatmap_data = df_filtrato.pivot_table(
        values='ricavi',
        index='categoria_prodotto',
        columns='mese',
        aggfunc='sum'
    )
    
    # Riordina colonne per mese
    mesi_ordinati = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                     'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
    heatmap_data = heatmap_data.reindex(columns=[m for m in mesi_ordinati if m in heatmap_data.columns])
    
    # Creazione heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd',
        text=heatmap_data.values.round(0),
        texttemplate='€%{text:,.0f}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title='Ricavi per Categoria e Mese',
        xaxis_title="Mese",
        yaxis_title="Categoria",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Descrizione
    st.markdown("""
    <div class='medium-font'>
    📊 <b>Analisi:</b> La heatmap evidenzia i periodi di picco per ogni categoria. 
    L'Elettronica ha vendite costanti con picchi a dicembre (regali natalizi), 
    mentre l'Abbigliamento mostra stagionalità più marcata.
    </div>
    """, unsafe_allow_html=True)

# === SEZIONE INSIGHTS E RACCOMANDAZIONI ===
st.markdown("---")
st.header("💡 Insights e Raccomandazioni")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.markdown("""
    ### 📈 Trend Identificati:
    - **Stagionalità marcata**: Picchi in estate e periodo natalizio
    - **Elettronica leader**: 45% dei ricavi totali
    - **Crescita progressiva**: +75% da gennaio a dicembre
    
    ### 🎯 Opportunità:
    - Potenziare stock elettronica pre-natale
    - Campagne mirate abbigliamento in primavera/estate
    - Sviluppare categoria Casa (potenziale non sfruttato)
    """)

with col_insight2:
    st.markdown("""
    ### 📊 KPI Principali:
    - **Ticket medio**: €{:.2f}
    - **Clienti medi/mese**: {:,.0f}
    - **Tasso crescita**: +{:.1f}% mensile
    
    ### 🚀 Azioni Suggerite:
    - Programmi fedeltà per aumentare retention
    - Cross-selling tra categorie
    - Ottimizzazione prezzi in bassa stagione
    """.format(
        ticket_medio,
        clienti_totali / 12,
        ((ricavi_mensili['ricavi'].iloc[-1] / ricavi_mensili['ricavi'].iloc[0]) ** (1/11) - 1) * 100
    ))

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    Dashboard Vendite E-commerce | Dati simulati per dimostrazione | 
    Creato con il ❤️ usando Streamlit
</div>
""", unsafe_allow_html=True)

# === SIDEBAR INFO ===
with st.sidebar:
    st.markdown("---")
    st.info("""
    **ℹ️ Info Dashboard**
    
    Questa dashboard analizza le vendite di un e-commerce fittizio nel 2024.
    
    **Funzionalità:**
    - Filtri interattivi per categoria e periodo
    - 4 visualizzazioni principali
    - Metriche in tempo reale
    - Analisi stagionalità
    
    **Tecnologie:**
    - Streamlit
    - Plotly
    - Pandas
    """)
    
    # Download dati
    st.markdown("---")
    st.download_button(
        label="📥 Scarica Dataset CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='vendite_ecommerce_2024.csv',
        mime='text/csv'
    )