# Esercizio Multi-Agent

Oggi costruiremo un sistema di agenti AI piuttosto avanzato. L'idea non è quella di creare un solo agente che sappia fare tutto, ma di costruire un **"Manager" intelligente** che gestisce un piccolo team di **"Specialisti"**.

## **L'obiettivo finale: Il "Grand Agent"**

Vogliamo creare un agente principale, che chiameremo **Grand Agent**. Il suo unico compito è ricevere una domanda dall'utente e capire quale dei suoi specialisti è il più adatto per risolverla. Una volta capito, gli passa il lavoro e attende la risposta.

Questo è un pattern molto potente, chiamato **Agente Router** o Agente Gerarchico.



### What do we need?

Per costruire questo sistema, ci servono tre componenti principali:

1.  **Lo Specialista #1: L'Agente Python (`Python Agent`)**
    *   **Cosa sa fare?** È un esperto programmatore Python. Può scrivere ed eseguire codice per rispondere a domande generiche, fare calcoli, manipolare testo e, nel nostro caso, persino creare file come i QR code.
    *   **Di che strumento ha bisogno?** Un interprete Python a cui può inviare il suo codice. In LangChain, questo si chiama `PythonREPLTool`.

2.  **Lo Specialista #2: L'Analista di Dati (`CSV Agent`)**
    *   **Cosa sa fare?** È un data-scientist specializzato nell'analisi di file di dati. Può leggere un file CSV, capire la struttura dei dati e rispondere a domande specifiche su di essi (es. "qual è la media?", "filtra per...").
    *   **Di che strumento ha bisogno?** Un modo per interagire con i dati usando la libreria `pandas`. LangChain ci fornisce un creatore di agenti apposta per questo: `create_pandas_dataframe_agent`.

3.  **Il Manager: Il "Grand Agent"**
    *   **Come fa a decidere a chi affidare il compito?** Questa è la parte più magica! Il nostro Manager non analizza la richiesta nel dettaglio. Invece, legge la **"descrizione del lavoro"** di ogni specialista. Noi scriveremo una descrizione chiara per l'Agente Python e per l'Agente CSV. Il Grand Agent leggerà la domanda dell'utente e la confronterà con le descrizioni per trovare la corrispondenza migliore.
    *   **Di che strumenti ha bisogno?** I suoi "strumenti" non sono strumenti normali, ma sono i nostri due agenti specialisti!

Ora che abbiamo il nostro piano, vediamo come trasformarlo in codice.

---

## **Spiegazione del codice, passo passo**

Spieghiamo ora ogni riga di codice per capire come abbiamo costruito il nostro sistema.

### Installazione delle Dipendenze

```python
!pip install langchain langchain-openai langchain-experimental pandas python-dotenv langchainhub qrcode pillow
```

**Perché questa riga?**
Questa è la prima cosa da fare: installare la nostra "cassetta degli attrezzi". Ogni pacchetto ha uno scopo preciso:
*   `langchain`, `langchain-openai`, `langchain-experimental`: Il cuore del nostro progetto. Ci forniscono gli strumenti per creare, collegare e orchestrare gli agenti e i loro componenti. `experimental` contiene tool più nuovi, come quello per Pandas.
*   `pandas`: La libreria fondamentale per l'analisi dei dati in Python. Sarà usata dal nostro `CSV Agent`.
*   `python-dotenv`: Una piccola utilità per gestire le nostre "chiavi segrete" (API key) in modo sicuro, caricandole da un file `.env`.
*   `langchainhub`: Un "hub" online da cui possiamo scaricare facilmente componenti pronti all'uso, come i template dei prompt.
*   `qrcode`, `pillow`: Queste sono le librerie che il nostro `Python Agent` userà per generare e salvare le immagini dei QR code. Devono essere installate nell'ambiente affinché l'agente le possa usare.

---

#### **2. Importazioni e Configurazione dell'Ambiente**

```python
# ... (codice di gestione API key) ...

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import pandas as pd

load_dotenv()
# ... (verifica caricamento API key) ...
```

**Perché questo blocco?**
Qui importiamo tutti i "mattoni" che ci serviranno per costruire i nostri agenti.
*   **Gestione API Key**: Il codice iniziale (con `os` e `getpass`) è una procedura robusta per assicurarci di avere la chiave API di OpenAI, senza scriverla direttamente nel codice.
*   `load_dotenv()`: Esegue la funzione della libreria `python-dotenv` per caricare le variabili dal file `.env`.
*   `Tool`: È la classe che useremo per "impacchettare" i nostri agenti specialisti e darli in pasto al manager.
*   `ChatOpenAI`: È il nostro accesso al "cervello" dei modelli OpenAI (come GPT-4).
*   `create_react_agent`: È una "funzione fabbrica" di LangChain. Prende un modello, un prompt e degli strumenti e assembla la logica di pensiero di un agente (il "ragionamento").
*   `AgentExecutor`: È l'esecutore. Prende l'agente "pensante" creato sopra e gli dà la capacità di agire, eseguendo veramente gli strumenti.
*   `PythonREPLTool`: Lo strumento specifico che dà all'agente Python l'accesso a un interprete.
*   `create_pandas_dataframe_agent`: La funzione "scorciatoia" che ci costruisce un agente specializzato per analizzare un DataFrame Pandas.

---

#### **3. Creazione del File CSV di Esempio**

```python
data = {'season': [...], 'episode_num': [...], 'title': [...], 'viewers_millions': [...]}
df_sample = pd.DataFrame(data)
df_sample.to_csv("episode_info.csv", index=False)
```

**Perché questo blocco?**
Il nostro `CSV Agent` ha bisogno di dati su cui lavorare! Per rendere il notebook indipendente e funzionante per chiunque, qui creiamo al volo un file `episode_info.csv` con dati di esempio. In un caso reale, questo file esisterebbe già.

---

#### **4. Definizione del Sotto-Agente Python**

```python
python_agent_instructions = """...""" # Istruzioni per l'agente

# Carichiamo un template di prompt standard
base_prompt_template = hub.pull("langchain-ai/react-agent-template")
# Inseriamo le nostre istruzioni specifiche nel template
python_agent_prompt = base_prompt_template.partial(instructions=python_agent_instructions)

# Definiamo gli strumenti dello specialista
python_tools = [PythonREPLTool()]

# Creiamo l'agente e il suo esecutore
python_agent_llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
python_agent = create_react_agent(...)
python_agent_executor = AgentExecutor(...)
```

**Perché questo blocco?**
Qui stiamo assemblando il nostro primo specialista, il programmatore Python.
*   `python_agent_instructions`: Questa è l'anima dell'agente. Stiamo scrivendo il suo "manuale d'istruzioni" in linguaggio naturale. Gli diciamo chi è, cosa può fare (`'qrcode' package installed`), come deve comportarsi e cosa fare in caso di errore.
*   `hub.pull(...)`: Per non reinventare la ruota, scarichiamo un prompt "ReAct" (Reasoning and Acting) già testato ed efficace da LangChain Hub.
*   `.partial(instructions=...)`: Inseriamo le nostre istruzioni personalizzate nel template standard.
*   `python_tools = [PythonREPLTool()]`: Dichiariamo esplicitamente che l'unico strumento che questo agente può usare è l'interprete Python.
*   `ChatOpenAI(temperature=0, ...)`: Scegliamo il "cervello" (GPT-4 Turbo) e impostiamo la `temperature` a 0 per avere risposte precise e non troppo creative, ideali per scrivere codice.
*   `create_react_agent` e `AgentExecutor`: Usiamo le "fabbriche" di LangChain per costruire prima la logica dell'agente e poi il suo esecutore, combinando istruzioni, cervello e strumenti.

---

#### **5. Definizione del Sotto-Agente CSV**

```python
df = pd.read_csv("episode_info.csv")

csv_agent_executor: AgentExecutor = create_pandas_dataframe_agent(
    llm=csv_agent_llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True
)
```

**Perché questo blocco?**
Qui creiamo il nostro secondo specialista, l'analista di dati.
*   `df = pd.read_csv(...)`: Per prima cosa, carichiamo i dati dal file CSV in un DataFrame Pandas.
*   `create_pandas_dataframe_agent(...)`: Questa è una funzione di alto livello che fa tutto il lavoro pesante per noi. Crea un agente ottimizzato per interagire con Pandas.
*   `llm=...`, `df=...`: Gli passiamo il cervello (LLM) e i dati su cui deve lavorare (il DataFrame `df`).
*   `allow_dangerous_code=True`: **Importante!** Lo impostiamo a `True` perché l'agente, per rispondere, genera ed esegue codice Python che interagisce con il DataFrame. Dobbiamo dargli esplicitamente il permesso di farlo.

---

#### **6. Definizione dell'Agente Router (Grand Agent)**

```python
# ... (funzione wrapper) ...

grand_agent_tools = [
    Tool(
        name="Python_Code_Executor",
        func=python_agent_executor_wrapper,
        description="""... L'input per questo tool deve essere la descrizione completa ...""",
    ),
    Tool(
        name="CSV_Data_Analyzer",
        func=csv_agent_executor.invoke,
        description="""... Utile quando devi rispondere a domande basate sui dati ...""",
    ),
]

# ... (creazione del Grand Agent e del suo executor) ...
```

**Perché questo blocco?**
Questo è il cuore del nostro sistema: la creazione del Manager.
*   `python_agent_executor_wrapper`: Una piccola funzione "collante". L'oggetto `Tool` si aspetta una funzione che accetti una semplice stringa, mentre l'`AgentExecutor` si aspetta un dizionario. Questa funzione fa da ponte tra i due.
*   `grand_agent_tools = [...]`: Qui definiamo la lista degli strumenti del nostro Manager. Ma attenzione: i suoi strumenti sono i nostri agenti specialisti!
*   `Tool(...)`: Per ogni specialista, creiamo un `Tool`. Analizziamone i parametri:
    *   `name`: Un nome semplice che l'agente userà nei suoi "pensieri".
    *   `func`: La funzione che deve essere eseguita. Nota come passiamo gli `executor` dei nostri sotto-agenti.
    *   `description`: **Questa è la parte più importante!** La descrizione è il "curriculum" dello specialista. Il Grand Agent leggerà queste descrizioni per decidere chi è il più qualificato per la domanda dell'utente. Una descrizione chiara e dettagliata è FONDAMENTALE per un buon routing.
*   Infine, creiamo il `grand_agent` e il suo `grand_agent_executor` esattamente come abbiamo fatto per l'agente Python, ma passandogli la lista di `grand_agent_tools`.

---

#### **7. Esecuzione dell'Agente Router con Esempi**

```python
query_csv = "which season has the most episodes?"
response_csv = grand_agent_executor.invoke({"input": query_csv})

query_python_qrcode = "Generate and save ... 3 qrcodes ..."
response_python_qrcode = grand_agent_executor.invoke({"input": query_python_qrcode})
```

**Perché questo blocco?**
È il momento del test!
1.  **Prima Query (`query_csv`)**: Poniamo una domanda relativa ai dati del CSV. Ci aspettiamo che il `Grand Agent` legga le descrizioni, capisca che `"CSV_Data_Analyzer"` è lo strumento giusto e gli passi la richiesta.
2.  **Seconda Query (`query_python_qrcode`)**: Poniamo una richiesta che richiede di scrivere codice e creare file. Ci aspettiamo che il `Grand Agent` scelga `"Python_Code_Executor"`.

L'opzione `verbose=True` negli executor è essenziale qui, perché ci mostrerà la "catena di pensieri" dell'agente: vedremo come analizza la domanda e quale strumento sceglie, confermando che il nostro router sta funzionando come previsto.
