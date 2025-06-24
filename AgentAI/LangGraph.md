# Guida Completa a LangGraph per la Creazione di Agenti

## Indice
1. [Introduzione a LangGraph](#-introduzione)
2. [Concetti Fondamentali](#-concetti-fondamentali)
3. [Installazione e Setup](#-installazione)
4. [Il Primo Agente Semplice](#-primo-agente)
5. [Stati e Memoria](#-stati-memoria)
6. [Agenti con Tools](#-agenti-tools)
7. [Multi-Agenti e Collaborazione](#-multi-agenti)
8. [Pattern Avanzati](#-pattern-avanzati)
9. [Best Practices](#-best-practices)
10. [Esempi Pratici](#-esempi-pratici)

---

## Introduzione a LangGraph

**LangGraph** è un framework sviluppato da LangChain per costruire applicazioni **stateful** e **multi-agente** usando grafi diretti aciclici (DAG). È progettato per creare agenti intelligenti che possono:

- Mantenere stato persistente tra le interazioni
- Collaborare tra loro in sistemi complessi
- Gestire flussi di lavoro complessi con logica condizionale
- Integrarsi facilmente con LLM e tools esterni

### Perché LangGraph?

**Vantaggi rispetto ad approcci tradizionali:**
- **Controllo granulare** del flusso di esecuzione
- **Stato persistente** tra le chiamate
- **Debugging avanzato** con visualizzazione del grafo
- **Scalabilità** per sistemi multi-agente complessi
- **Integrazione nativa** con l'ecosistema LangChain

---

## Concetti Fondamentali

### 1. Grafo (Graph)
Il **grafo** è la struttura principale che definisce il flusso di esecuzione dell'agente.

```python
from langgraph.graph import StateGraph

# Creazione di un grafo
graph = StateGraph(StateSchema)
```

### 2. Nodi (Nodes)
I **nodi** sono le unità di elaborazione del grafo. Ogni nodo rappresenta una funzione o un'azione.

```python
def my_node(state):
    # Logica del nodo
    return {"key": "value"}

# Aggiungere un nodo al grafo
graph.add_node("node_name", my_node)
```

### 3. Edges (Connessioni)
Gli **edges** definiscono il flusso tra i nodi.

```python
# Edge semplice
graph.add_edge("node_a", "node_b")

# Edge condizionale
graph.add_conditional_edges(
    "node_a",
    condition_function,
    {
        "path_1": "node_b",
        "path_2": "node_c"
    }
)
```

### 4. Stato (State)
Lo **stato** è condiviso tra tutti i nodi e persiste durante l'esecuzione.

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    current_step: str
    tools_used: list
```

### 5. Checkpoints
I **checkpoints** permettono di salvare e ripristinare lo stato dell'agente.

---

## Installazione e Setup

```bash
# Installazione base
pip install langgraph langchain

# Con supporto per checkpoints
pip install langgraph[sqlite]

# Per visualizzazione grafi
pip install langgraph[viz]
```

### Setup base del progetto

```python
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

# Configurazione LLM
os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = ChatOpenAI(model="gpt-4")

# Definizione dello stato base
class AgentState(TypedDict):
    messages: List[str]
    current_step: str
    iteration_count: int
```

---

## Il Primo Agente Semplice

Creiamo un agente base che elabora messaggi e risponde.

```python
from langgraph.graph import StateGraph, END

# Definizione dello stato
class SimpleAgentState(TypedDict):
    input: str
    output: str
    step_count: int

# Nodo di elaborazione
def process_input(state: SimpleAgentState):
    user_input = state["input"]
    response = f"Ho ricevuto: '{user_input}'. Elaborando..."
    
    return {
        "output": response,
        "step_count": state.get("step_count", 0) + 1
    }

# Nodo di finalizzazione
def finalize_response(state: SimpleAgentState):
    final_response = f"{state['output']} Completato in {state['step_count']} passi."
    return {"output": final_response}

# Costruzione del grafo
def create_simple_agent():
    graph = StateGraph(SimpleAgentState)
    
    # Aggiunta nodi
    graph.add_node("process", process_input)
    graph.add_node("finalize", finalize_response)
    
    # Definizione del flusso
    graph.set_entry_point("process")
    graph.add_edge("process", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()

# Utilizzo
agent = create_simple_agent()
result = agent.invoke({
    "input": "Ciao, come stai?",
    "step_count": 0
})

print(result["output"])
```

---

## Stati e Memoria

### Gestione dello Stato Avanzata

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AdvancedAgentState(TypedDict):
    # Lista di messaggi che si accumula
    messages: Annotated[List, add_messages]
    # Memoria a lungo termine
    memory: dict
    # Contesto corrente
    context: str
    # Tools disponibili
    available_tools: List[str]

def memory_node(state: AdvancedAgentState):
    """Nodo che gestisce la memoria dell'agente"""
    
    # Recupera memoria esistente
    memory = state.get("memory", {})
    messages = state.get("messages", [])
    
    # Analizza i messaggi per estrarre informazioni
    if messages:
        last_message = messages[-1]
        # Salva informazioni importanti in memoria
        if "preferenza" in last_message.lower():
            memory["user_preferences"] = memory.get("user_preferences", [])
            memory["user_preferences"].append(last_message)
    
    return {"memory": memory}

def context_analyzer(state: AdvancedAgentState):
    """Analizza il contesto della conversazione"""
    
    messages = state.get("messages", [])
    memory = state.get("memory", {})
    
    # Determina il contesto basandosi sui messaggi recenti
    if len(messages) > 0:
        recent_topics = [msg for msg in messages[-3:]]
        context = "conversazione_generale"
        
        # Logica per determinare il contesto
        if any("aiuto" in msg.lower() for msg in recent_topics):
            context = "richiesta_assistenza"
        elif any("tecnico" in msg.lower() for msg in recent_topics):
            context = "supporto_tecnico"
    
    return {"context": context}
```

### Checkpoints per Persistenza

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Configurazione checkpoint
checkpointer = SqliteSaver.from_conn_string("agent_memory.db")

def create_persistent_agent():
    graph = StateGraph(AdvancedAgentState)
    
    # Aggiunta nodi
    graph.add_node("memory", memory_node)
    graph.add_node("context", context_analyzer)
    graph.add_node("respond", response_node)
    
    # Flusso
    graph.set_entry_point("memory")
    graph.add_edge("memory", "context")
    graph.add_edge("context", "respond")
    graph.add_edge("respond", END)
    
    # Compilazione con checkpoints
    return graph.compile(checkpointer=checkpointer)

# Utilizzo con sessioni persistenti
agent = create_persistent_agent()

# Thread ID per mantenere la sessione
thread_id = "user_123"
config = {"configurable": {"thread_id": thread_id}}

# Prima interazione
result1 = agent.invoke({
    "messages": ["Ciao, mi piace il caffè"],
    "memory": {},
    "context": ""
}, config=config)

# Seconda interazione (mantiene memoria)
result2 = agent.invoke({
    "messages": ["Che bevande consigli?"],
}, config=config)
```

---

## Agenti con Tools

### Integrazione di Tools Esterni

```python
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Literal

# Definizione tools
search_tool = DuckDuckGoSearchRun()

def calculate_tool(expression: str) -> str:
    """Calcolatrice semplice"""
    try:
        result = eval(expression)  # In produzione usare ast.literal_eval
        return f"Risultato: {result}"
    except:
        return "Errore nel calcolo"

# Tools disponibili
tools = [
    Tool(
        name="search",
        description="Cerca informazioni su internet",
        func=search_tool.run
    ),
    Tool(
        name="calculate",
        description="Esegue calcoli matematici",
        func=calculate_tool
    )
]

class ToolAgentState(TypedDict):
    messages: List[str]
    tool_results: List[dict]
    next_action: str

def tool_selector(state: ToolAgentState):
    """Seleziona quale tool utilizzare"""
    
    last_message = state["messages"][-1] if state["messages"] else ""
    
    if any(word in last_message.lower() for word in ["cerca", "search", "trova"]):
        return {"next_action": "search"}
    elif any(word in last_message.lower() for word in ["calcola", "matematica", "+"]):
        return {"next_action": "calculate"}
    else:
        return {"next_action": "respond"}

def execute_search(state: ToolAgentState):
    """Esegue ricerca web"""
    query = state["messages"][-1]
    result = search_tool.run(query)
    
    tool_results = state.get("tool_results", [])
    tool_results.append({
        "tool": "search",
        "query": query,
        "result": result
    })
    
    return {"tool_results": tool_results}

def execute_calculation(state: ToolAgentState):
    """Esegue calcoli"""
    expression = state["messages"][-1]
    result = calculate_tool(expression)
    
    tool_results = state.get("tool_results", [])
    tool_results.append({
        "tool": "calculate",
        "expression": expression,
        "result": result
    })
    
    return {"tool_results": tool_results}

def route_tools(state: ToolAgentState) -> Literal["search", "calculate", "respond"]:
    """Router condizionale per i tools"""
    return state["next_action"]

def create_tool_agent():
    graph = StateGraph(ToolAgentState)
    
    # Nodi
    graph.add_node("selector", tool_selector)
    graph.add_node("search", execute_search)
    graph.add_node("calculate", execute_calculation)
    graph.add_node("respond", final_response)
    
    # Flusso
    graph.set_entry_point("selector")
    
    # Router condizionale
    graph.add_conditional_edges(
        "selector",
        route_tools,
        {
            "search": "search",
            "calculate": "calculate",
            "respond": "respond"
        }
    )
    
    graph.add_edge("search", "respond")
    graph.add_edge("calculate", "respond")
    graph.add_edge("respond", END)
    
    return graph.compile()
```

---

## Multi-Agenti e Collaborazione

### Sistema di Agenti Collaborativi

```python
class MultiAgentState(TypedDict):
    task: str
    research_results: List[str]
    analysis_results: List[str]
    final_report: str
    current_agent: str

# Agente Ricercatore
def researcher_agent(state: MultiAgentState):
    """Agente specializzato nella ricerca"""
    task = state["task"]
    
    # Simula ricerca (in produzione userebbe tools reali)
    research_results = [
        f"Ricerca 1 per: {task}",
        f"Ricerca 2 per: {task}",
        f"Dati statistici per: {task}"
    ]
    
    return {
        "research_results": research_results,
        "current_agent": "researcher"
    }

# Agente Analista
def analyst_agent(state: MultiAgentState):
    """Agente specializzato nell'analisi"""
    research_results = state.get("research_results", [])
    
    analysis_results = [
        f"Analisi di: {result}" for result in research_results
    ]
    
    return {
        "analysis_results": analysis_results,
        "current_agent": "analyst"
    }

# Agente Reporter
def reporter_agent(state: MultiAgentState):
    """Agente specializzato nella sintesi finale"""
    analysis_results = state.get("analysis_results", [])
    task = state["task"]
    
    final_report = f"""
    REPORT FINALE per: {task}
    
    Analisi completate: {len(analysis_results)}
    
    Sommario:
    {' '.join(analysis_results)}
    
    Conclusioni: Task completato con successo.
    """
    
    return {
        "final_report": final_report,
        "current_agent": "reporter"
    }

def create_multi_agent_system():
    graph = StateGraph(MultiAgentState)
    
    # Agenti specializzati
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("reporter", reporter_agent)
    
    # Flusso sequenziale
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "reporter")
    graph.add_edge("reporter", END)
    
    return graph.compile()

# Utilizzo del sistema multi-agente
multi_agent = create_multi_agent_system()
result = multi_agent.invoke({
    "task": "Analisi del mercato AI 2024",
    "research_results": [],
    "analysis_results": [],
    "final_report": "",
    "current_agent": ""
})

print(result["final_report"])
```

### Coordinatore di Agenti

```python
class CoordinatedAgentState(TypedDict):
    main_task: str
    subtasks: List[dict]
    completed_tasks: List[dict]
    active_agents: List[str]
    coordinator_decision: str

def coordinator_agent(state: CoordinatedAgentState):
    """Agente coordinatore che assegna compiti"""
    
    main_task = state["main_task"]
    completed_tasks = state.get("completed_tasks", [])
    
    # Logica di coordinamento
    if len(completed_tasks) == 0:
        # Prima fase: ricerca
        decision = "assign_research"
        subtasks = [
            {"type": "research", "topic": "background", "assigned_to": "researcher"},
            {"type": "research", "topic": "current_state", "assigned_to": "researcher"}
        ]
    elif len(completed_tasks) < 3:
        # Seconda fase: analisi
        decision = "assign_analysis"
        subtasks = state.get("subtasks", [])
    else:
        # Fase finale: report
        decision = "assign_reporting"
        subtasks = [{"type": "report", "data": completed_tasks, "assigned_to": "reporter"}]
    
    return {
        "subtasks": subtasks,
        "coordinator_decision": decision
    }

def route_coordinator(state: CoordinatedAgentState) -> str:
    """Router per le decisioni del coordinatore"""
    decision = state.get("coordinator_decision", "")
    
    if decision == "assign_research":
        return "researcher"
    elif decision == "assign_analysis":
        return "analyst"
    elif decision == "assign_reporting":
        return "reporter"
    else:
        return END
```

---

## Pattern Avanzati

### 1. Human-in-the-Loop

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

class HumanLoopState(TypedDict):
    task: str
    ai_proposal: str
    human_feedback: str
    approved: bool
    iteration: int

def ai_propose(state: HumanLoopState):
    """AI propone una soluzione"""
    task = state["task"]
    iteration = state.get("iteration", 0)
    
    # Simula proposta AI
    proposal = f"Proposta AI (iterazione {iteration + 1}) per: {task}"
    
    return {
        "ai_proposal": proposal,
        "iteration": iteration + 1,
        "approved": False
    }

def wait_for_human(state: HumanLoopState):
    """Interruzione per input umano"""
    # Questo nodo viene interrotto automaticamente
    # L'umano può continuare fornendo feedback
    return state

def process_feedback(state: HumanLoopState):
    """Elabora il feedback umano"""
    feedback = state.get("human_feedback", "")
    
    if "approvato" in feedback.lower() or "ok" in feedback.lower():
        return {"approved": True}
    else:
        # Incorpora feedback per la prossima iterazione
        return {"approved": False}

def needs_approval(state: HumanLoopState) -> str:
    """Controlla se serve approvazione"""
    if state.get("approved", False):
        return "complete"
    else:
        return "revise"

def create_human_loop_agent():
    graph = StateGraph(HumanLoopState)
    
    graph.add_node("propose", ai_propose)
    graph.add_node("human_review", wait_for_human)
    graph.add_node("process_feedback", process_feedback)
    
    graph.set_entry_point("propose")
    graph.add_edge("propose", "human_review")
    graph.add_edge("human_review", "process_feedback")
    
    graph.add_conditional_edges(
        "process_feedback",
        needs_approval,
        {
            "complete": END,
            "revise": "propose"
        }
    )
    
    # Interruzione per input umano
    graph.add_interrupt("human_review")
    
    return graph.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))
```

### 2. Error Handling e Retry Logic

```python
class RobustAgentState(TypedDict):
    task: str
    attempts: int
    max_attempts: int
    last_error: str
    success: bool
    result: str

def risky_operation(state: RobustAgentState):
    """Operazione che può fallire"""
    import random
    
    attempts = state.get("attempts", 0) + 1
    
    # Simula operazione che può fallire
    if random.random() < 0.7:  # 70% di probabilità di successo
        return {
            "attempts": attempts,
            "success": True,
            "result": f"Operazione completata al tentativo {attempts}"
        }
    else:
        return {
            "attempts": attempts,
            "success": False,
            "last_error": f"Errore al tentativo {attempts}"
        }

def should_retry(state: RobustAgentState) -> str:
    """Logica di retry"""
    max_attempts = state.get("max_attempts", 3)
    attempts = state.get("attempts", 0)
    success = state.get("success", False)
    
    if success:
        return "success"
    elif attempts >= max_attempts:
        return "failure"
    else:
        return "retry"

def handle_success(state: RobustAgentState):
    return {"result": f"Completato con successo: {state['result']}"}

def handle_failure(state: RobustAgentState):
    return {"result": f"Fallito dopo {state['attempts']} tentativi. Ultimo errore: {state['last_error']}"}

def create_robust_agent():
    graph = StateGraph(RobustAgentState)
    
    graph.add_node("operation", risky_operation)
    graph.add_node("success_handler", handle_success)
    graph.add_node("failure_handler", handle_failure)
    
    graph.set_entry_point("operation")
    
    graph.add_conditional_edges(
        "operation",
        should_retry,
        {
            "success": "success_handler",
            "failure": "failure_handler",
            "retry": "operation"
        }
    )
    
    graph.add_edge("success_handler", END)
    graph.add_edge("failure_handler", END)
    
    return graph.compile()
```

---

## Best Practices

### 1. Struttura del Codice

```python
# Organizzazione raccomandata dei file

# agents/
#   ├── __init__.py
#   ├── base_agent.py      # Classe base per agenti
#   ├── research_agent.py  # Agenti specializzati
#   ├── analysis_agent.py
#   └── coordinator.py
#
# states/
#   ├── __init__.py
#   └── agent_states.py    # Definizioni degli stati
#
# tools/
#   ├── __init__.py
#   ├── search_tools.py    # Tools personalizzati
#   └── calculation_tools.py
#
# graphs/
#   ├── __init__.py
#   └── agent_graphs.py    # Configurazioni dei grafi

# Esempio di struttura base
class BaseAgent:
    def __init__(self, name: str, tools: List = None):
        self.name = name
        self.tools = tools or []
        self.graph = None
    
    def create_graph(self):
        """Metodo da implementare nelle sottoclassi"""
        raise NotImplementedError
    
    def compile_graph(self, checkpointer=None):
        """Compila il grafo dell'agente"""
        if not self.graph:
            self.graph = self.create_graph()
        return self.graph.compile(checkpointer=checkpointer)
```

### 2. Testing e Debugging

```python
import pytest
from langgraph.graph import StateGraph

def test_agent_flow():
    """Test del flusso base dell'agente"""
    
    # Setup
    agent = create_simple_agent()
    test_input = {
        "input": "test message",
        "step_count": 0
    }
    
    # Esecuzione
    result = agent.invoke(test_input)
    
    # Assertions
    assert "output" in result
    assert result["step_count"] > 0
    assert "test message" in result["output"]

def test_conditional_routing():
    """Test del routing condizionale"""
    
    agent = create_tool_agent()
    
    # Test ricerca
    search_input = {"messages": ["cerca informazioni su Python"]}
    result = agent.invoke(search_input)
    assert any("search" in str(tool) for tool in result.get("tool_results", []))
    
    # Test calcolo
    calc_input = {"messages": ["calcola 2 + 2"]}
    result = agent.invoke(calc_input)
    assert any("calculate" in str(tool) for tool in result.get("tool_results", []))
```

### 3. Monitoraggio e Logging

```python
import logging
from datetime import datetime

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_node(func):
    """Decorator per logging dei nodi"""
    def wrapper(state):
        logger.info(f"Executing node: {func.__name__}")
        logger.info(f"Input state keys: {list(state.keys())}")
        
        start_time = datetime.now()
        result = func(state)
        end_time = datetime.now()
        
        logger.info(f"Node {func.__name__} completed in {end_time - start_time}")
        logger.info(f"Output state keys: {list(result.keys())}")
        
        return result
    return wrapper

# Utilizzo
@logged_node
def my_agent_node(state):
    # Logica del nodo
    return {"result": "processed"}
```

### 4. Gestione degli Errori

```python
def safe_node(func):
    """Decorator per gestione errori sicura"""
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            logger.error(f"Error in node {func.__name__}: {str(e)}")
            return {
                "error": str(e),
                "error_node": func.__name__,
                "error_occurred": True
            }
    return wrapper

def error_recovery_node(state):
    """Nodo per il recupero da errori"""
    if state.get("error_occurred", False):
        error = state.get("error", "Unknown error")
        logger.info(f"Recovering from error: {error}")
        
        # Logica di recupero
        return {
            "error_occurred": False,
            "recovery_attempted": True,
            "original_error": error
        }
    
    return state
```

---

## Esempi Pratici {#esempi-pratici}

### Esempio 1: Assistente Ricerca Scientifica

```python
class ResearchAssistantState(TypedDict):
    query: str
    papers_found: List[dict]
    summary: str
    recommendations: List[str]

def search_papers(state: ResearchAssistantState):
    """Cerca papers scientifici"""
    query = state["query"]
    
    # Simula ricerca (in produzione: arXiv API, PubMed, etc.)
    papers = [
        {"title": f"Paper 1 su {query}", "authors": ["Autore A"], "year": 2024},
        {"title": f"Paper 2 su {query}", "authors": ["Autore B"], "year": 2023}
    ]
    
    return {"papers_found": papers}

def analyze_papers(state: ResearchAssistantState):
    """Analizza i papers trovati"""
    papers = state["papers_found"]
    
    summary = f"Trovati {len(papers)} papers rilevanti. "
    summary += "Temi principali: machine learning, deep learning, neural networks."
    
    return {"summary": summary}

def generate_recommendations(state: ResearchAssistantState):
    """Genera raccomandazioni"""
    papers = state["papers_found"]
    
    recommendations = [
        "Approfondire gli approcci di deep learning",
        "Considerare l'applicazione a domini specifici",
        "Investigare i dataset utilizzati"
    ]
    
    return {"recommendations": recommendations}

def create_research_assistant():
    graph = StateGraph(ResearchAssistantState)
    
    graph.add_node("search", search_papers)
    graph.add_node("analyze", analyze_papers)
    graph.add_node("recommend", generate_recommendations)
    
    graph.set_entry_point("search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", END)
    
    return graph.compile()

# Utilizzo
research_agent = create_research_assistant()
result = research_agent.invoke({
    "query": "neural networks for image classification",
    "papers_found": [],
    "summary": "",
    "recommendations": []
})

print(f"Summary: {result['summary']}")
print(f"Recommendations: {result['recommendations']}")
```

### Esempio 2: Sistema di Customer Support

```python
class SupportTicketState(TypedDict):
    ticket_id: str
    customer_message: str
    category: str
    priority: str
    assigned_agent: str
    resolution_steps: List[str]
    status: str

def categorize_ticket(state: SupportTicketState):
    """Categorizza automaticamente il ticket"""
    message = state["customer_message"].lower()
    
    if "password" in message or "login" in message:
        category = "authentication"
        priority = "medium"
    elif "bug" in message or "errore" in message:
        category = "technical"
        priority = "high"
    elif "fattura" in message or "pagamento" in message:
        category = "billing"
        priority = "low"
    else:
        category = "general"
        priority = "medium"
    
    return {"category": category, "priority": priority}

def assign_agent(state: SupportTicketState):
    """Assegna l'agente appropriato"""
    category = state["category"]
    
    agent_mapping = {
        "authentication": "auth_specialist",
        "technical": "tech_support",
        "billing": "billing_team",
        "general": "general_support"
    }
    
    assigned_agent = agent_mapping.get(category, "general_support")
    return {"assigned_agent": assigned_agent}

def generate_resolution_steps(state: SupportTicketState):
    """Genera passi per la risoluzione"""
    category = state["category"]
    
    steps_mapping = {
        "authentication": [
            "Verificare email utente",
            "Inviare link reset password",
            "Confermare accesso ripristinato"
        ],
        "technical": [
            "Replicare il problema",
            "Identificare la causa",
            "Implementare fix",
            "Verificare risoluzione"
        ],
        "billing": [
            "Verificare account",
            "Controllare transazioni",
            "Risolvere discrepanze"
        ]
    }
    
    steps = steps_mapping.get(category, ["Analizzare richiesta", "Fornire assistenza"])
    return {"resolution_steps": steps, "status": "in_progress"}

def create_support_system():
    graph = StateGraph(SupportTicketState)
    
    graph.add_node("categorize", categorize_ticket)
    graph.add_node("assign", assign_agent)
    graph.add_node("resolve", generate_resolution_steps)
    
    graph.set_entry_point("categorize")
    graph.add_edge("categorize", "assign")
    graph.add_edge("assign", "resolve")
    graph.add_edge("resolve", END)
    
    return graph.compile()

# Utilizzo
support_system = create_support_system()
ticket = support_system.invoke({
    "ticket_id": "TICK-001",
    "customer_message": "Non riesco ad accedere al mio account, ho dimenticato la password",
    "category": "",
    "priority": "",
    "assigned_agent": "",
    "resolution_steps": [],
    "status": "new"
})

print(f"Categoria: {ticket['category']}")
print(f"Priorità: {ticket['priority']}")
print(f"Agente assegnato: {ticket['assigned_agent']}")
print(f"Passi risoluzione: {ticket['resolution_steps']}")
```

---

## Conclusioni

LangGraph offre un framework potente e flessibile per costruire agenti intelligenti con le seguenti caratteristiche chiave:

**Vantaggi principali:**
- **Controllo granulare** del flusso di esecuzione
- **Stato persistente** tra le interazioni
- **Scalabilità** per sistemi complessi
- **Debugging avanzato** con visualizzazione
- **Integrazione ecosistema** LangChain

**Casi d'uso ideali:**
- Assistenti virtuali complessi
- Sistemi multi-agente
- Workflow automatizzati
- Applicazioni che richiedono stato persistente
- Processi che necessitano di approvazione umana

**Prossimi passi:**
1. Sperimentare con gli esempi forniti
2. Integrare tools specifici per il tuo dominio
3. Implementare sistemi di monitoraggio
4. Ottimizzare le performance per casi d'uso reali
5. Esplorare pattern avanzati come reinforcement learning

LangGraph rappresenta il futuro dello sviluppo di agenti AI, combinando la potenza dei LLM con la flessibilità dei grafi di stato per creare applicazioni intelligenti davvero sofisticate.
