# Guida Completa a LangChain per Applicazioni AI

## Indice
1. [Introduzione a LangChain](#introduzione-a-langchain)
2. [Installazione e Setup](#installazione-e-setup)
3. [LLM e Chat Models](#llm-e-chat-models)
4. [Prompts e Templates](#prompts-e-templates)
5. [Chains - Catene di Elaborazione](#chains)
6. [Memory - Gestione della Memoria](#memory)
7. [Tools e Agents](#tools-e-agents)
8. [RAG (Retrieval Augmented Generation)](#rag-retrieval-augmented-generation)
9. [Output Parsers](#output-parsers)
10. [Callbacks e Monitoring](#callbacks-e-monitoring)
11. [Pattern Avanzati](#pattern-avanzati)
12. [Esempi Pratici](#esempi-pratici)

---

## Introduzione a LangChain

**LangChain** è un framework open-source per sviluppare applicazioni powered by language models. È progettato per creare applicazioni che sono:

- **Data-aware**: connesse a fonti di dati
- **Agentic**: permettono ai LLM di interagire con l'ambiente
- **Composable**: costruite combinando componenti modulari

### Componenti principali

**🔗 Chains**: Sequenze di chiamate a LLM o altri componenti
**💭 Prompts**: Template e gestione di prompt
**🧠 Memory**: Persistenza di stato tra interazioni
**🛠️ Tools**: Integrazione con servizi esterni
**🤖 Agents**: LLM che possono decidere azioni da intraprendere
**📚 Retrievers**: Sistemi per recuperare informazioni rilevanti

### Architettura generale

```python
# Flusso tipico di una applicazione LangChain
Input → Prompt Template → LLM → Output Parser → Response
   ↓
Memory ← → Tools/Retrievers
```

---

## Installazione e Setup

### Installazione Base

```bash
# Core LangChain
pip install langchain

# Integrazioni LLM
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-google-genai

# Integrazioni Vector Stores
pip install langchain-chroma
pip install langchain-pinecone
pip install faiss-cpu

# Tools comuni
pip install langchain-community
pip install wikipedia duckduckgo-search

# Utilities
pip install python-dotenv
```

### Setup Iniziale

```python
import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# Setup API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Import principali
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
```

---

## LLM e Chat Models

### Configurazione Base degli LLM

```python
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic

# Chat Models (raccomandati per applicazioni conversazionali)
chat_openai = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

chat_anthropic = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7
)

# LLM tradizionali (per completion)
llm_openai = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.5
)

# Test base
response = chat_openai.invoke("Ciao! Come stai?")
print(response.content)
```

### Streaming e Callbacks

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LLM con streaming
streaming_llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Utilizzo con streaming
for chunk in streaming_llm.stream("Racconta una storia breve"):
    print(chunk.content, end="", flush=True)
```

### Gestione di Modelli Multipli

```python
class ModelManager:
    def __init__(self):
        self.models = {
            "fast": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
            "smart": ChatOpenAI(model="gpt-4", temperature=0.7),
            "creative": ChatOpenAI(model="gpt-4", temperature=0.9)
        }
    
    def get_model(self, task_type: str):
        """Seleziona il modello appropriato per il task"""
        if task_type in ["summary", "extraction"]:
            return self.models["fast"]
        elif task_type in ["analysis", "reasoning"]:
            return self.models["smart"]
        elif task_type in ["creative", "writing"]:
            return self.models["creative"]
        else:
            return self.models["fast"]

# Utilizzo
manager = ModelManager()
creative_model = manager.get_model("creative")
response = creative_model.invoke("Scrivi una poesia sul machine learning")
```

---

## Prompts e Templates

### Prompt Templates Base

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

# Template semplice
simple_template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Spiega {topic} in modo adatto a {audience}"
)

# Chat Template con ruoli
chat_template = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto {domain} che aiuta gli utenti."),
    ("human", "Ho bisogno di aiuto con: {question}"),
    ("ai", "Sarò felice di aiutarti! Dimmi di più su {question}."),
    ("human", "{follow_up}")
])

# Utilizzo
formatted_prompt = simple_template.format(
    topic="machine learning", 
    audience="principianti"
)
print(formatted_prompt)
```

### Few-Shot Prompting

```python
# Esempi per few-shot learning
examples = [
    {
        "question": "Cos'è Python?",
        "answer": "Python è un linguaggio di programmazione ad alto livello, interpretato e general-purpose."
    },
    {
        "question": "Cos'è JavaScript?", 
        "answer": "JavaScript è un linguaggio di programmazione principalmente usato per lo sviluppo web."
    }
]

# Template per gli esempi
example_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="Domanda: {question}\nRisposta: {answer}"
)

# Few-shot template
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Rispondi alle domande sui linguaggi di programmazione:",
    suffix="Domanda: {input}\nRisposta:",
    input_variables=["input"]
)

# Utilizzo
prompt = few_shot_template.format(input="Cos'è Java?")
print(prompt)
```

### Template dinamici e condizionali

```python
from langchain_core.prompts import PromptTemplate

class DynamicPromptBuilder:
    def __init__(self):
        self.base_templates = {
            "technical": """
            Sei un esperto tecnico. Fornisci una spiegazione dettagliata e precisa di {topic}.
            Includi esempi pratici e considera il livello {level}.
            """,
            "casual": """
            Spiega {topic} in modo semplice e conversazionale.
            Usa analogie e esempi di vita quotidiana per il livello {level}.
            """,
            "academic": """
            Fornisci un'analisi accademica di {topic} appropriata per {level}.
            Includi riferimenti teorici e struttura la risposta formalmente.
            """
        }
    
    def build_prompt(self, style: str, topic: str, level: str, 
                    include_examples: bool = True) -> PromptTemplate:
        """Costruisce un prompt dinamico basato sui parametri"""
        
        base = self.base_templates.get(style, self.base_templates["casual"])
        
        if include_examples:
            base += "\nFornisci almeno 2 esempi concreti."
        
        return PromptTemplate(
            input_variables=["topic", "level"],
            template=base
        )

# Utilizzo
builder = DynamicPromptBuilder()
prompt = builder.build_prompt("technical", "neural networks", "intermediate")
formatted = prompt.format(topic="neural networks", level="intermediate")
```

---

## Chains
In questa sezione trattiamo le catene di elaborazione

### Simple Chain

```python
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

# Chain base
simple_chain = LLMChain(
    llm=chat_openai,
    prompt=simple_template,
    output_parser=StrOutputParser()
)

# Esecuzione
result = simple_chain.invoke({
    "topic": "blockchain",
    "audience": "studenti universitari"
})
print(result["text"])
```

### Sequential Chains

```python
from langchain.chains import SequentialChain

# Prima chain: genera un sommario
summary_template = PromptTemplate(
    input_variables=["text"],
    template="Riassumi il seguente testo in 3 punti principali:\n\n{text}"
)

summary_chain = LLMChain(
    llm=chat_openai,
    prompt=summary_template,
    output_key="summary"
)

# Seconda chain: genera domande
questions_template = PromptTemplate(
    input_variables=["summary"],
    template="Basandoti su questo sommario, genera 3 domande per testare la comprensione:\n\n{summary}"
)

questions_chain = LLMChain(
    llm=chat_openai,
    prompt=questions_template,
    output_key="questions"
)

# Sequential chain
overall_chain = SequentialChain(
    chains=[summary_chain, questions_chain],
    input_variables=["text"],
    output_variables=["summary", "questions"],
    verbose=True
)

# Utilizzo
result = overall_chain.invoke({
    "text": """
    L'intelligenza artificiale è una branca dell'informatica che si occupa 
    di creare sistemi in grado di eseguire compiti che tipicamente richiedono 
    intelligenza umana. Include machine learning, deep learning, e natural 
    language processing.
    """
})

print("Sommario:", result["summary"])
print("Domande:", result["questions"])
```

### LCEL - LangChain Expression Language

```python
from langchain_core.runnables import RunnablePassthrough

# Chain con LCEL (approccio moderno raccomandato)
prompt = ChatPromptTemplate.from_template("Traduci '{text}' in {language}")
output_parser = StrOutputParser()

# Composizione con pipe operator
translation_chain = prompt | chat_openai | output_parser

# Utilizzo
result = translation_chain.invoke({
    "text": "Hello, how are you?",
    "language": "italiano"
})
print(result)

# Chain più complessa con parallel processing
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    sentiment=ChatPromptTemplate.from_template("Analizza il sentiment di: {text}") | chat_openai | StrOutputParser(),
    summary=ChatPromptTemplate.from_template("Riassumi in una frase: {text}") | chat_openai | StrOutputParser(),
    keywords=ChatPromptTemplate.from_template("Estrai 3 parole chiave da: {text}") | chat_openai | StrOutputParser()
)

# Esecuzione parallela
result = analysis_chain.invoke({"text": "Sono molto felice di imparare LangChain!"})
print("Sentiment:", result["sentiment"])
print("Summary:", result["summary"])
print("Keywords:", result["keywords"])
```

### Router Chains

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Definizione prompt specializzati
prompts = {
    "physics": PromptTemplate(
        template="Sei un fisico esperto. Spiega {input} usando principi fisici.",
        input_variables=["input"]
    ),
    "biology": PromptTemplate(
        template="Sei un biologo esperto. Spiega {input} da una prospettiva biologica.",
        input_variables=["input"]
    ),
    "computer_science": PromptTemplate(
        template="Sei un esperto di informatica. Spiega {input} con esempi di programmazione.",
        input_variables=["input"]
    )
}

# Chains specializzate
destination_chains = {}
for name, prompt in prompts.items():
    chain = LLMChain(llm=chat_openai, prompt=prompt)
    destination_chains[name] = chain

# Router template
router_template = """
Sei un router intelligente. Dato l'input dell'utente, scegli la categoria più appropriata:

{destinations}

<< INPUT >>
{input}

<< OUTPUT (deve essere solo il nome della categoria) >>
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    partial_variables={
        "destinations": "\n".join([f"{k}: {v.prompt.template}" for k, v in destination_chains.items()])
    }
)

# Router chain
router_chain = LLMRouterChain.from_llm(
    chat_openai, 
    router_prompt,
    router_output_parser=RouterOutputParser()
)

# Multi-prompt chain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["computer_science"],
    verbose=True
)

# Test
result = chain.invoke({"input": "Come funzionano le reti neurali?"})
print(result["text"])
```

---

## Memory
In questa sezione analizziamo la gestione della memoria

### Tipi di memoria

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)

# 1. Buffer Memory - mantiene tutto
buffer_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 2. Window Memory - mantiene solo le ultime K interazioni
window_memory = ConversationBufferWindowMemory(
    k=3,  # mantiene ultime 3 interazioni
    return_messages=True,
    memory_key="chat_history"
)

# 3. Summary Memory - riassume la conversazione
summary_memory = ConversationSummaryMemory(
    llm=chat_openai,
    return_messages=True,
    memory_key="chat_history"
)

# 4. Summary Buffer Memory - combina summary e buffer
summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=chat_openai,
    max_token_limit=1000,
    return_messages=True,
    memory_key="chat_history"
)
```

### Conversational Chain con memoria

```python
from langchain.chains import ConversationChain

# Template per conversazione
conversation_template = """
La seguente è una conversazione amichevole tra un umano e un AI.
L'AI è loquace e fornisce molti dettagli specifici dal suo contesto.
Se l'AI non sa la risposta a una domanda, dice onestamente che non lo sa.

Conversazione corrente:
{chat_history}
Umano: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=conversation_template
)

# Conversational chain
conversation = ConversationChain(
    llm=chat_openai,
    memory=buffer_memory,
    prompt=prompt,
    verbose=True
)

# Simulazione conversazione
print("=== Conversazione Esempio ===")
response1 = conversation.predict(input="Ciao! Mi chiamo Marco.")
print(f"AI: {response1}")

response2 = conversation.predict(input="Qual è il mio nome?")
print(f"AI: {response2}")

response3 = conversation.predict(input="Parlami del machine learning")
print(f"AI: {response3}")

# Visualizza la memoria
print("\n=== Contenuto Memoria ===")
print(buffer_memory.buffer)
```

### Custom Memory implementation

```python
from langchain.schema import BaseMemory
from typing import Dict, List, Any

class CustomMemory(BaseMemory):
    """Memory personalizzata con categorizzazione dei messaggi"""
    
    def __init__(self):
        self.messages = []
        self.categories = {
            "personal": [],
            "technical": [],
            "general": []
        }
    
    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history", "personal_info", "technical_context"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Categorizza i messaggi
        personal_msgs = "\n".join(self.categories["personal"][-3:])
        technical_msgs = "\n".join(self.categories["technical"][-3:])
        general_msgs = "\n".join(self.messages[-5:])
        
        return {
            "chat_history": general_msgs,
            "personal_info": personal_msgs,
            "technical_context": technical_msgs
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        human_msg = inputs.get("input", "")
        ai_msg = outputs.get("response", "")
        
        # Categorizza basandosi sul contenuto
        if any(word in human_msg.lower() for word in ["nome", "età", "lavoro", "famiglia"]):
            self.categories["personal"].append(f"Human: {human_msg}")
            self.categories["personal"].append(f"AI: {ai_msg}")
        elif any(word in human_msg.lower() for word in ["python", "code", "algoritmo", "tecnico"]):
            self.categories["technical"].append(f"Human: {human_msg}")
            self.categories["technical"].append(f"AI: {ai_msg}")
        
        # Salva tutto nei messaggi generali
        self.messages.append(f"Human: {human_msg}")
        self.messages.append(f"AI: {ai_msg}")
    
    def clear(self) -> None:
        self.messages.clear()
        for category in self.categories.values():
            category.clear()

# Utilizzo della memoria personalizzata
custom_memory = CustomMemory()

custom_conversation = ConversationChain(
    llm=chat_openai,
    memory=custom_memory,
    verbose=True
)
```

---

## Tools e agents

### Definizione di Tools

```python
from langchain.tools import Tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Optional, Type
from pydantic import BaseModel, Field

# Tool semplice da funzione
def get_word_length(word: str) -> str:
    """Restituisce la lunghezza di una parola."""
    return f"La parola '{word}' ha {len(word)} caratteri."

word_length_tool = Tool(
    name="WordLength",
    description="Utile per sapere quanti caratteri ha una parola",
    func=get_word_length
)

# Tool predefinito
search_tool = DuckDuckGoSearchRun()

# Custom tool con Pydantic
class CalculatorInput(BaseModel):
    expression: str = Field(description="Espressione matematica da calcolare")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Utile per fare calcoli matematici. Input: espressione matematica come stringa."
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)  # In produzione usare ast.literal_eval
            return f"Risultato: {result}"
        except Exception as e:
            return f"Errore nel calcolo: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        # Versione asincrona
        return self._run(expression)

calculator_tool = CalculatorTool()

# Lista di tools disponibili
tools = [word_length_tool, search_tool, calculator_tool]
```

### Agent base con ReAct

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Template per ReAct agent
react_template = """
Rispondi alla seguente domanda nel modo più accurato possibile. Hai accesso ai seguenti tools:

{tools}

Usa il seguente formato:

Question: la domanda di input
Thought: dovresti sempre pensare a cosa fare
Action: l'azione da intraprendere, dovrebbe essere una di [{tool_names}]
Action Input: l'input per l'azione
Observation: il risultato dell'azione
... (questo Thought/Action/Action Input/Observation può ripetersi N volte)
Thought: ora so la risposta finale
Final Answer: la risposta finale alla domanda originale

Inizia!

Question: {input}
Thought: {agent_scratchpad}
"""

react_prompt = PromptTemplate.from_template(react_template)

# Creazione agent
agent = create_react_agent(
    llm=chat_openai,
    tools=tools,
    prompt=react_prompt
)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    return_intermediate_steps=True
)

# Test dell'agent
result = agent_executor.invoke({
    "input": "Qual è la lunghezza della parola 'intelligenza' e quanto fa 15 + 27?"
})

print("Risposta finale:", result["output"])
print("Passi intermedi:", result["intermediate_steps"])
```

### Structured Tool Agent

```python
from langchain.agents import create_structured_chat_agent

# Template per structured chat agent
structured_template = """
Rispondi alla domanda dell'utente nel modo migliore possibile. Hai accesso ai seguenti tools:

{tools}

Per utilizzare un tool, usa questo formato JSON:
{{
    "action": "nome_del_tool",
    "action_input": "input_del_tool"
}}


Quando hai la risposta finale, rispondi direttamente senza utilizzare tools.

Conversazione:
{chat_history}

Question: {input}
{agent_scratchpad}
"""
structured_prompt = ChatPromptTemplate.from_template(structured_template)

# Creazione structured agent
structured_agent = create_structured_chat_agent(
    llm=chat_openai,
    tools=tools,
    prompt=structured_prompt
)

structured_executor = AgentExecutor(
    agent=structured_agent,
    tools=tools,
    verbose=True,
    memory=buffer_memory,
    return_intermediate_steps=True
)
```

### Multi-Agent System

```python
class SpecializedAgent:
    def __init__(self, name: str, role: str, tools: List[Tool]):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Template specifico per il ruolo
        template = f"""
        Sei {name}, un {role}. 
        Il tuo compito è fornire assistenza specializzata nel tuo dominio.
        
        Tools disponibili: {[tool.name for tool in tools]}
        
        Conversazione:
        {{chat_history}}
        
        Domanda: {{input}}
        {{agent_scratchpad}}
        """
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(
            llm=chat_openai,
            tools=tools,
            prompt=prompt
        )
        
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
    
    def process(self, query: str) -> str:
        return self.executor.invoke({"input": query})["output"]

# Creazione agenti specializzati
research_agent = SpecializedAgent(
    name="ResearchBot",
    role="ricercatore specializzato",
    tools=[search_tool]
)

calc_agent = SpecializedAgent(
    name="MathBot", 
    role="matematico esperto",
    tools=[calculator_tool]
)

text_agent = SpecializedAgent(
    name="TextBot",
    role="analista di testo",
    tools=[word_length_tool]
)

# Coordinator per routing
class AgentCoordinator:
    def __init__(self, agents: Dict[str, SpecializedAgent]):
        self.agents = agents
    
    def route_query(self, query: str) -> str:
        """Determina quale agente dovrebbe gestire la query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["cerca", "ricerca", "trova", "google"]):
            return "research"
        elif any(word in query_lower for word in ["calcola", "matematica", "+", "-", "*", "/"]):
            return "math"
        elif any(word in query_lower for word in ["parola", "caratteri", "lunghezza", "testo"]):
            return "text"
        else:
            return "research"  # default
    
    def process_query(self, query: str) -> str:
        agent_type = self.route_query(query)
        agent_mapping = {
            "research": research_agent,
            "math": calc_agent,
            "text": text_agent
        }
        
        selected_agent = agent_mapping[agent_type]
        return f"[{selected_agent.name}]: {selected_agent.process(query)}"

# Utilizzo del sistema multi-agent
coordinator = AgentCoordinator({
    "research": research_agent,
    "math": calc_agent,
    "text": text_agent
})

# Test
print(coordinator.process_query("Quanto fa 25 * 3?"))
print(coordinator.process_query("Cerca informazioni su Python"))
print(coordinator.process_query("Quanti caratteri ha la parola 'langchain'?"))
```

---

## RAG (Retrieval Augmented Generation)

### Setup Base per RAG

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Caricamento documenti
def load_documents(directory_path: str):
    """Carica documenti da una directory"""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    return documents

# 2. Splitting documenti
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def split_documents(documents):
    """Divide i documenti in chunks"""
    return text_splitter.split_documents(documents)

# 3. Creazione embeddings e vector store
embeddings = OpenAIEmbeddings()

def create_vector_store(documents, persist_directory="./chroma_db"):
    """Crea un vector store da documenti"""
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

# 4. Retriever
def create_retriever(vectorstore, search_type="similarity", k=4):
    """Crea un retriever dal vector store"""
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
```

### RAG Chain Completa

```python
from langchain_core.prompts import PromptTemplate

# Template per RAG
rag_template = """
Usa il seguente contesto per rispondere alla domanda. Se non trovi informazioni 
rilevanti nel contesto, dillo chiaramente e fornisci una risposta generale.

Contesto:
{context}

Domanda: {question}

Risposta utile e dettagliata:"""

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question"]
)

class RAGSystem:
    def __init__(self, documents_path: str = None, vectorstore_path: str = "./chroma_db"):
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings()
        
        if documents_path:
            # Carica e processa nuovi documenti
            documents = load_documents(documents_path)
            split_docs = split_documents(documents)
            self.vectorstore = create_vector_store(split_docs, vectorstore_path)
        else:
            # Carica vector store esistente
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings
            )
        
        self.retriever = create_retriever(self.vectorstore)
        
        # Creazione RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=chat_openai,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )
    
    def query(self, question: str):
        """Esegue una query RAG"""
        result = self.rag_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }
    
    def add_documents(self, new_docs_path: str):
        """Aggiunge nuovi documenti al vector store"""
        documents = load_documents(new_docs_path)
        split_docs = split_documents(documents)
        self.vectorstore.add_documents(split_docs)

# Esempio di utilizzo
# rag_system = RAGSystem(documents_path="./documents/")
# result = rag_system.query("Che cos'è il machine learning?")
# print("Risposta:", result["answer"])
# print("Fonti:", result["sources"])
```

### RAG avanzato con Self-Query

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Metadati per self-query
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Il file sorgente del documento",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="Il numero di pagina del documento",
        type="integer"
    ),
    AttributeInfo(
        name="topic",
        description="L'argomento principale del documento",
        type="string"
    )
]

document_content_description = "Documenti tecnici su AI e machine learning"

# Self-query retriever
def create_self_query_retriever(vectorstore):
    return SelfQueryRetriever.from_llm(
        llm=chat_openai,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )

# RAG con conversational memory
from langchain.chains import ConversationalRetrievalChain

class ConversationalRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_openai,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def chat(self, question: str):
        result = self.qa_chain.invoke({"question": question})
        return {
            "answer": result["answer"],
            "sources": [doc.page_content[:200] for doc in result.get("source_documents", [])]
        }
```

---

## Output Parsers

### Parser Base

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# String parser (default)
string_parser = StrOutputParser()

# JSON parser
json_parser = JsonOutputParser()

# Test
prompt = ChatPromptTemplate.from_template(
    "Elenca 3 benefici di {topic} in formato JSON con chiavi 'benefici'"
)

json_chain = prompt | chat_openai | json_parser
result = json_chain.invoke({"topic": "machine learning"})
print(result)
```

### Structured Output con Pydantic

```python
from langchain_core.output_parsers import PydanticOutputParser

# Definizione modello Pydantic
class PersonInfo(BaseModel):
    name: str = Field(description="Nome della persona")
    age: int = Field(description="Età della persona")
    occupation: str = Field(description="Professione")
    skills: List[str] = Field(description="Lista delle competenze")

class BookReview(BaseModel):
    title: str = Field(description="Titolo del libro")
    author: str = Field(description="Autore del libro")
    rating: int = Field(description="Valutazione da 1 a 5")
    summary: str = Field(description="Riassunto breve")
    pros: List[str] = Field(description="Aspetti positivi")
    cons: List[str] = Field(description="Aspetti negativi")

# Parser
person_parser = PydanticOutputParser(pydantic_object=PersonInfo)
book_parser = PydanticOutputParser(pydantic_object=BookReview)

# Template con format instructions
person_template = """
Estrai le informazioni sulla persona dal seguente testo:

{text}

{format_instructions}
"""

person_prompt = PromptTemplate(
    template=person_template,
    input_variables=["text"],
    partial_variables={"format_instructions": person_parser.get_format_instructions()}
)

# Chain completa
person_chain = person_prompt | chat_openai | person_parser

# Test
text = """
Marco Rossi ha 35 anni e lavora come data scientist. 
È esperto in Python, machine learning e data visualization.
"""

result = person_chain.invoke({"text": text})
print(f"Nome: {result.name}")
print(f"Età: {result.age}")
print(f"Professione: {result.occupation}")
print(f"Competenze: {result.skills}")
```

### Custom Output Parser

```python
from langchain.schema import BaseOutputParser
import re

class CustomListParser(BaseOutputParser):
    """Parser personalizzato per liste numerate"""
    
    def parse(self, text: str) -> List[str]:
        # Trova pattern come "1. item", "2. item", etc.
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        # Pulisce e restituisce gli items
        return [match.strip() for match in matches if match.strip()]
    
    @property
    def _type(self) -> str:
        return "custom_list"

class EmailParser(BaseOutputParser):
    """Parser per estrarre componenti email"""
    
    def parse(self, text: str) -> dict:
        # Estrae subject
        subject_match = re.search(r'Subject:\s*(.+)', text)
        subject = subject_match.group(1).strip() if subject_match else ""
        
        # Estrae recipient
        to_match = re.search(r'To:\s*(.+)', text)
        recipient = to_match.group(1).strip() if to_match else ""
        
        # Estrae body (tutto dopo "Body:")
        body_match = re.search(r'Body:\s*(.+)', text, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""
        
        return {
            "subject": subject,
            "recipient": recipient,
            "body": body
        }

# Utilizzo parser personalizzati
list_parser = CustomListParser()
email_parser = EmailParser()

# Template per lista
list_template = """
Genera una lista di 5 consigli per {topic}:

Formato:
1. Primo consiglio
2. Secondo consiglio
...
"""

list_prompt = PromptTemplate(
    template=list_template,
    input_variables=["topic"]
)

list_chain = list_prompt | chat_openai | list_parser

# Test
tips = list_chain.invoke({"topic": "imparare Python"})
print("Consigli:", tips)

# Template per email
email_template = """
Componi un'email professionale per {purpose}:

To: {recipient}
Subject: [Genera subject appropriato]
Body: [Genera corpo email professionale]
"""

email_prompt = PromptTemplate(
    template=email_template,
    input_variables=["purpose", "recipient"]
)

email_chain = email_prompt | chat_openai | email_parser

# Test
email = email_chain.invoke({
    "purpose": "richiedere informazioni su un corso", 
    "recipient": "info@corsopython.it"
})
print("Email generata:", email)
```

---

## Callbacks e Monitoring

### Callback Base

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List
import time

class CustomCallbackHandler(BaseCallbackHandler):
    """Callback personalizzato per monitoring"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
        self.costs = 0.0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Chiamato all'inizio di una chiamata LLM"""
        self.start_time = time.time()
        print(f"🚀 Inizio chiamata LLM con {len(prompts)} prompt(s)")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Chiamato alla fine di una chiamata LLM"""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Calcola token usage se disponibile
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            total_tokens = token_usage.get('total_tokens', 0)
            self.tokens_used += total_tokens
            
            # Stima costi (esempio per GPT-4)
            cost = total_tokens * 0.03 / 1000  # $0.03 per 1K tokens
            self.costs += cost
            
            print(f"✅ Chiamata completata in {duration:.2f}s")
            print(f"   Tokens: {total_tokens} (Totale: {self.tokens_used})")
            print(f"   Costo stimato: ${cost:.4f} (Totale: ${self.costs:.4f})")
        else:
            print(f"✅ Chiamata completata in {duration:.2f}s")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Chiamato in caso di errore LLM"""
        print(f"❌ Errore LLM: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Chiamato all'inizio di una chain"""
        chain_name = serialized.get('name', 'UnknownChain')
        print(f"🔗 Inizio chain: {chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Chiamato alla fine di una chain"""
        print(f"🔗 Chain completata")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Chiamato all'inizio dell'uso di un tool"""
        tool_name = serialized.get('name', 'UnknownTool')
        print(f"🛠️  Utilizzo tool: {tool_name} con input: {input_str[:50]}...")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Chiamato alla fine dell'uso di un tool"""
        print(f"🛠️  Tool completato. Output: {output[:50]}...")

# Utilizzo del callback
callback_handler = CustomCallbackHandler()

# Chain con callback
monitored_chain = prompt | chat_openai.with_config(callbacks=[callback_handler]) | StrOutputParser()

result = monitored_chain.invoke({"text": "Ciao", "language": "inglese"})
```

### Logging strutturato

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, log_file: str = "langchain_app.log"):
        self.logger = logging.getLogger("LangChainApp")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter per JSON
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_interaction(self, interaction_type: str, data: dict):
        """Log di un'interazione strutturata"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "data": data
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback che logga tutto in formato strutturato"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.session_id = f"session_{int(time.time())}"
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.logger.log_interaction("llm_start", {
            "session_id": self.session_id,
            "model": serialized.get('name', 'unknown'),
            "prompt_count": len(prompts),
            "prompts": prompts[:2]  # Solo primi 2 per privacy
        })
    
    def on_llm_end(self, response, **kwargs) -> None:
        # Estrae informazioni dalla response
        token_info = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            token_info = response.llm_output.get('token_usage', {})
        
        self.logger.log_interaction("llm_end", {
            "session_id": self.session_id,
            "token_usage": token_info,
            "response_length": len(str(response)) if response else 0
        })

# Setup logging
structured_logger = StructuredLogger()
logging_callback = LoggingCallbackHandler(structured_logger)

# Chain con logging
logged_chain = prompt | chat_openai.with_config(callbacks=[logging_callback]) | StrOutputParser()
```

### Performance Monitoring

```python
import psutil
import threading
from collections import defaultdict

class PerformanceMonitor:
    """Monitor delle performance dell'applicazione"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Inizia il monitoring delle performance"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Ferma il monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Loop principale di monitoring"""
        while self.monitoring:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            self.metrics['memory_available_gb'].append(memory.available / (1024**3))
            
            # Process info
            process = psutil.Process()
            self.metrics['process_memory_mb'].append(process.memory_info().rss / (1024**2))
            
            time.sleep(interval)
    
    def get_summary(self) -> dict:
        """Restituisce un sommario delle metriche"""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'avg': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'current': values[-1]
                }
        return summary
    
    def reset_metrics(self):
        """Reset delle metriche"""
        self.metrics.clear()

# Callback con performance monitoring
class PerformanceCallbackHandler(BaseCallbackHandler):
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.chain_metrics = {}
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        chain_id = kwargs.get('run_id', 'unknown')
        self.chain_metrics[chain_id] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / (1024**2)
        }
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        chain_id = kwargs.get('run_id', 'unknown')
        if chain_id in self.chain_metrics:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            
            duration = end_time - self.chain_metrics[chain_id]['start_time']
            memory_delta = end_memory - self.chain_metrics[chain_id]['start_memory']
            
            print(f"⏱️  Chain duration: {duration:.2f}s")
            print(f"💾 Memory delta: {memory_delta:.2f}MB")
            
            del self.chain_metrics[chain_id]

# Utilizzo completo del monitoring
monitor = PerformanceMonitor()
perf_callback = PerformanceCallbackHandler(monitor)

# Avvia monitoring
monitor.start_monitoring()

# Esegui operazioni
monitored_chain = prompt | chat_openai.with_config(callbacks=[perf_callback]) | StrOutputParser()

# Test multiple chiamate
for i in range(3):
    result = monitored_chain.invoke({"text": f"Messaggio {i}", "language": "inglese"})
    time.sleep(1)

# Visualizza summary
print("\n📊 Performance Summary:")
summary = monitor.get_summary()
for metric, stats in summary.items():
    print(f"{metric}: avg={stats['avg']:.2f}, max={stats['max']:.2f}")

monitor.stop_monitoring()
```

---

## Pattern avanzati

### 1. Caching Intelligente

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache
import hashlib

# Setup cache globale
set_llm_cache(InMemoryCache())

# Custom cache con TTL
class TTLCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

class CachingChain:
    """Chain con caching personalizzato"""
    
    def __init__(self, chain, cache_ttl: int = 3600):
        self.chain = chain
        self.cache = TTLCache(cache_ttl)
    
    def _make_cache_key(self, inputs: dict) -> str:
        """Crea una chiave cache from inputs"""
        key_string = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def invoke(self, inputs: dict):
        cache_key = self._make_cache_key(inputs)
        
        # Controlla cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            print("🎯 Cache hit!")
            return cached_result
        
        # Esegui chain
        print("🔄 Executing chain...")
        result = self.chain.invoke(inputs)
        
        # Salva in cache
        self.cache.set(cache_key, result)
        return result

# Utilizzo
cached_chain = CachingChain(translation_chain, cache_ttl=1800)

# Prima chiamata
result1 = cached_chain.invoke({"text": "Hello", "language": "italiano"})

# Seconda chiamata (da cache)
result2 = cached_chain.invoke({"text": "Hello", "language": "italiano"})
```

### 2. Retry e fallback logic

```python
from functools import wraps
import random

def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator per retry con exponential backoff"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Tentativo {attempt + 1} fallito: {e}. Retry in {delay:.2f}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator

class FallbackLLMChain:
    """Chain con fallback tra LLM diversi"""
    
    def __init__(self, primary_llm, fallback_llms: List):
        self.primary_llm = primary_llm
        self.fallback_llms = fallback_llms
        self.prompt = ChatPromptTemplate.from_template("{input}")
    
    @retry_with_exponential_backoff(max_retries=2)
    def _try_llm(self, llm, input_text: str):
        """Prova un LLM specifico"""
        chain = self.prompt | llm | StrOutputParser()
        return chain.invoke({"input": input_text})
    
    def invoke(self, input_text: str):
        """Prova primary LLM, poi fallback in ordine"""
        
        # Prova primary LLM
        try:
            print("🎯 Trying primary LLM...")
            return self._try_llm(self.primary_llm, input_text)
        except Exception as e:
            print(f"❌ Primary LLM failed: {e}")
        
        # Prova fallback LLMs
        for i, fallback_llm in enumerate(self.fallback_llms):
            try:
                print(f"🔄 Trying fallback LLM {i+1}...")
                return self._try_llm(fallback_llm, input_text)
            except Exception as e:
                print(f"❌ Fallback LLM {i+1} failed: {e}")
                continue
        
        raise Exception("All LLMs failed")

# Setup fallback chain
primary = ChatOpenAI(model="gpt-4", temperature=0.7)
fallbacks = [
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    # ChatAnthropic(model="claude-3-sonnet-20240229")  # se disponibile
]

fallback_chain = FallbackLLMChain(primary, fallbacks)

# Test
try:
    result = fallback_chain.invoke("Spiegami il quantum computing")
    print("✅ Risultato:", result)
except Exception as e:
    print("❌ Tutti i tentativi falliti:", e)
```

### 3. Load Balancing

```python
import random
from typing import List, Dict
from datetime import datetime, timedelta

class LLMLoadBalancer:
    """Load balancer per distribuire richieste tra LLM"""
    
    def __init__(self, llms: List[tuple], strategy: str = "round_robin"):
        # llms: lista di (name, llm_instance, weight)
        self.llms = llms
        self.strategy = strategy
        self.current_index = 0
        self.request_counts = {name: 0 for name, _, _ in llms}
        self.error_counts = {name: 0 for name, _, _ in llms}
        self.last_request_time = {name: datetime.now() for name, _, _ in llms}
    
    def get_llm(self):
        """Seleziona LLM basandosi sulla strategia"""
        
        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "weighted":
            return self._weighted_selection()
        elif self.strategy == "least_loaded":
            return self._least_loaded()
        elif self.strategy == "health_based":
            return self._health_based()
        else:
            return self._round_robin()
    
    def _round_robin(self):
        """Strategia round-robin semplice"""
        llm_info = self.llms[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.llms)
        return llm_info
    
    def _weighted_selection(self):
        """Selezione basata sui pesi"""
        weights = [weight for _, _, weight in self.llms]
        return random.choices(self.llms, weights=weights)[0]
    
    def _least_loaded(self):
        """Seleziona l'LLM con meno richieste"""
        min_requests = min(self.request_counts.values())
        candidates = [
            llm_info for llm_info in self.llms
            if self.request_counts[llm_info[0]] == min_requests
        ]
        return random.choice(candidates)
    
    def _health_based(self):
        """Seleziona basandosi sulla salute (error rate)"""
        healthy_llms = []
        for name, llm, weight in self.llms:
            total_requests = self.request_counts[name]
            if total_requests == 0:
                error_rate = 0
            else:
                error_rate = self.error_counts[name] / total_requests
            
            # Considera "sano" se error rate < 10%
            if error_rate < 0.1:
                healthy_llms.append((name, llm, weight))
        
        if healthy_llms:
            return random.choice(healthy_llms)
        else:
            # Se tutti hanno problemi, usa round-robin
            return self._round_robin()
    
    def record_request(self, llm_name: str, success: bool):
        """Registra il risultato di una richiesta"""
        self.request_counts[llm_name] += 1
        if not success:
            self.error_counts[llm_name] += 1
        self.last_request_time[llm_name] = datetime.now()
    
    def get_stats(self) -> Dict:
        """Statistiche del load balancer"""
        stats = {}
        for name, _, _ in self.llms:
            total = self.request_counts[name]
            errors = self.error_counts[name]
            error_rate = (errors / total * 100) if total > 0 else 0
            
            stats[name] = {
                "total_requests": total,
                "errors": errors,
                "error_rate": f"{error_rate:.1f}%",
                "last_request": self.last_request_time[name].strftime("%H:%M:%S")
            }
        return stats

class LoadBalancedChain:
    """Chain che usa load balancing"""
    
    def __init__(self, llm_configs: List[tuple], strategy: str = "round_robin"):
        self.load_balancer = LLMLoadBalancer(llm_configs, strategy)
        self.prompt = ChatPromptTemplate.from_template("{input}")
    
    def invoke(self, input_text: str):
        """Esegue la richiesta con load balancing"""
        
        llm_name, llm, _ = self.load_balancer.get_llm()
        print(f"🔄 Using {llm_name}")
        
        try:
            chain = self.prompt | llm | StrOutputParser()
            result = chain.invoke({"input": input_text})
            
            # Registra successo
            self.load_balancer.record_request(llm_name, success=True)
            return result
            
        except Exception as e:
            # Registra errore
            self.load_balancer.record_request(llm_name, success=False)
            print(f"❌ Error with {llm_name}: {e}")
            raise e
    
    def get_stats(self):
        return self.load_balancer.get_stats()

# Setup load balancer
llm_configs = [
    ("gpt-4", ChatOpenAI(model="gpt-4", temperature=0.7), 3),
    ("gpt-3.5", ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7), 5),
    ("gpt-3.5-fast", ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3), 2)
]

balanced_chain = LoadBalancedChain(llm_configs, strategy="weighted")

# Test multiple requests
for i in range(5):
    try:
        result = balanced_chain.invoke(f"Domanda numero {i+1}: Cos'è l'AI?")
        print(f"✅ Risposta {i+1} ricevuta")
    except Exception as e:
        print(f"❌ Errore nella richiesta {i+1}: {e}")

# Visualizza statistiche
print("\n📊 Load Balancer Stats:")
stats = balanced_chain.get_stats()
for llm_name, llm_stats in stats.items():
    print(f"{llm_name}: {llm_stats}")
```

---

## Esempi Pratici 

### Esempio 1: Sistema di Analisi Documenti

```python
class DocumentAnalysisSystem:
    """Sistema completo per l'analisi di documenti"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        
        # Memory per mantenere contesto
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Output parsers specializzati
        self.summary_parser = PydanticOutputParser(pydantic_object=DocumentSummary)
        self.entities_parser = PydanticOutputParser(pydantic_object=EntityList)
        
    def analyze_document(self, document_text: str) -> Dict:
        """Analisi completa di un documento"""
        
        results = {}
        
        # 1. Sommario
        summary_prompt = ChatPromptTemplate.from_template(
            """Analizza il seguente documento e fornisci un sommario strutturato:

{document}

{format_instructions}"""
        )
        
        summary_chain = (
            summary_prompt 
            | self.llm 
            | self.summary_parser
        )
        
        results["summary"] = summary_chain.invoke({
            "document": document_text,
            "format_instructions": self.summary_parser.get_format_instructions()
        })
        
        # 2. Estrazione entità
        entities_prompt = ChatPromptTemplate.from_template(
            """Estrai le entità principali (persone, luoghi, organizzazioni, date) dal documento:

{document}

{format_instructions}"""
        )
        
        entities_chain = (
            entities_prompt 
            | self.llm 
            | self.entities_parser
        )
        
        results["entities"] = entities_chain.invoke({
            "document": document_text,
            "format_instructions": self.entities_parser.get_format_instructions()
        })
        
        # 3. Analisi sentiment
        sentiment_prompt = ChatPromptTemplate.from_template(
            "Analizza il sentiment generale del documento. Rispondi con: positivo, negativo, neutro, misto.\n\nDocumento: {document}"
        )
        
        sentiment_chain = sentiment_prompt | self.llm | StrOutputParser()
        results["sentiment"] = sentiment_chain.invoke({"document": document_text})
        
        # 4. Parole chiave
        keywords_prompt = ChatPromptTemplate.from_template(
            "Estrai le 10 parole chiave più importanti dal documento, separate da virgole:\n\nDocumento: {document}"
        )
        
        keywords_chain = keywords_prompt | self.llm | StrOutputParser()
        keywords_result = keywords_chain.invoke({"document": document_text})
        results["keywords"] = [kw.strip() for kw in keywords_result.split(",")]
        
        return results

# Modelli Pydantic per output strutturato
class DocumentSummary(BaseModel):
    title: str = Field(description="Titolo o argomento principale")
    main_points: List[str] = Field(description="Punti principali (3-5)")
    conclusion: str = Field(description="Conclusione o riassunto finale")
    document_type: str = Field(description="Tipo di documento (articolo, report, etc.)")

class EntityList(BaseModel):
    people: List[str] = Field(description="Nomi di persone")
    places: List[str] = Field(description="Luoghi")
    organizations: List[str] = Field(description="Organizzazioni")
    dates: List[str] = Field(description="Date importanti")

# Utilizzo del sistema
analyzer = DocumentAnalysisSystem()

document_sample = """
Il 15 marzo 2024, la società TechCorp ha annunciato una partnership strategica con 
OpenAI per sviluppare nuove soluzioni di intelligenza artificiale. L'accordo, 
firmato a San Francisco dal CEO Marco Rossi, prevede un investimento di 50 milioni 
di dollari nei prossimi tre anni. La collaborazione si concentrerà sullo sviluppo 
di chatbot avanzati per il customer service e sistemi di raccomandazione personalizzati.
"""

# Analisi completa
results = analyzer.analyze_document(document_sample)

print("📄 ANALISI DOCUMENTO")
print("=" * 50)
print(f"Titolo: {results['summary'].title}")
print(f"Tipo: {results['summary'].document_type}")
print(f"Sentiment: {results['sentiment']}")
print(f"\nPunti principali:")
for point in results['summary'].main_points:
    print(f"  • {point}")

print(f"\nEntità estratte:")
print(f"  Persone: {', '.join(results['entities'].people)}")
print(f"  Luoghi: {', '.join(results['entities'].places)}")
print(f"  Organizzazioni: {', '.join(results['entities'].organizations)}")
print(f"  Date: {', '.join(results['entities'].dates)}")

print(f"\nParole chiave: {', '.join(results['keywords'][:5])}")
```

### Esempio 2: Assistente per content creation

```python
class ContentCreationAssistant:
    """Assistente AI per la creazione di contenuti"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.8)
        self.tools = self._setup_tools()
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Agent per ricerca e creatività
        self.agent = self._create_agent()
    
    def _setup_tools(self):
        """Setup tools per ricerca e analisi"""
        
        search_tool = DuckDuckGoSearchRun()
        
        def trend_analyzer(topic: str) -> str:
            """Analizza trend per un topic"""
            # Simula analisi trend (in produzione: Google Trends API)
            trends = [
                f"{topic} - crescita del 15% negli ultimi 3 mesi",
                f"Keyword correlate: {topic} tutorial, {topic} guide, {topic} 2024",
                f"Picco di interesse: lunedì e mercoledì"
            ]
            return "\n".join(trends)
        
        def seo_optimizer(content: str) -> str:
            """Ottimizza contenuto per SEO"""
            word_count = len(content.split())
            
            suggestions = [
                f"Lunghezza attuale: {word_count} parole",
                "Suggerimenti SEO:",
                "- Aggiungi 2-3 sottotitoli H2",
                "- Includi parole chiave long-tail",
                "- Aggiungi call-to-action",
                "- Ottimizza per snippet in evidenza"
            ]
            return "\n".join(suggestions)
        
        return [
            Tool(
                name="web_search",
                description="Cerca informazioni aggiornate su internet",
                func=search_tool.run
            ),
            Tool(
                name="trend_analysis",
                description="Analizza trend e popolarità di un argomento",
                func=trend_analyzer
            ),
            Tool(
                name="seo_optimization",
                description="Fornisce suggerimenti SEO per un contenuto",
                func=seo_optimizer
            )
        ]
    
    def _create_agent(self):
        """Crea agent per content creation"""
        
        template = """
Sei un esperto content creator che aiuta a creare contenuti di alta qualità per il web.
Puoi usare questi tools per ricerche e ottimizzazioni:

{tools}

Usa il seguente formato:
Question: la richiesta dell'utente
Thought: cosa devo fare per aiutare l'utente
Action: [tool da usare]
Action Input: input per il tool
Observation: risultato del tool
... (puoi ripetere Thought/Action/Input/Observation)
Thought: ora posso rispondere
Final Answer: la mia risposta finale con il contenuto creato

Conversazione precedente:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )
    
    def create_blog_post(self, topic: str, target_audience: str, word_count: int = 800):
        """Crea un blog post completo"""
        
        prompt = f"""
Crea un blog post completo su "{topic}" per {target_audience}.
Il post deve essere di circa {word_count} parole e includere:

1. Titolo accattivante
2. Introduzione coinvolgente  
3. 3-4 sezioni principali con sottotitoli
4. Conclusione con call-to-action
5. Meta description per SEO

Prima di scrivere, fai una ricerca per assicurarti che le informazioni siano aggiornate.
Poi ottimizza il contenuto per SEO.
"""
        
        return self.agent.invoke({"input": prompt})
    
    def create_social_media_content(self, topic: str, platforms: List[str]):
        """Crea contenuti per social media"""
        
        platform_specs = {
            "instagram": "Post Instagram: max 2200 caratteri, hashtag rilevanti, tone informale",
            "linkedin": "Post LinkedIn: tono professionale, 1-3 paragrafi, focus su insights",
            "twitter": "Tweet: max 280 caratteri, hashtag strategici, call-to-action",
            "facebook": "Post Facebook: storytelling, 1-2 paragrafi, coinvolgente"
        }
        
        results = {}
        for platform in platforms:
            if platform in platform_specs:
                prompt = f"""
Crea un post per {platform} su "{topic}".
Specifiche: {platform_specs[platform]}

Cerca prima informazioni aggiornate sull'argomento, poi crea il contenuto ottimizzato per la piattaforma.
"""
                results[platform] = self.agent.invoke({"input": prompt})
        
        return results

# Utilizzo del content creation assistant
assistant = ContentCreationAssistant()

# Creazione blog post
print("📝 CREAZIONE BLOG POST")
print("=" * 50)

blog_result = assistant.create_blog_post(
    topic="Intelligenza Artificiale nel 2024",
    target_audience="professionisti del marketing",
    word_count=1000
)

print("Blog post creato:")
print(blog_result["output"])

# Creazione contenuti social
print("\n📱 CONTENUTI SOCIAL MEDIA")
print("=" * 50)

social_results = assistant.create_social_media_content(
    topic="Trends AI 2024",
    platforms=["instagram", "linkedin", "twitter"]
)

for platform, result in social_results.items():
    print(f"\n{platform.upper()}:")
    print(result["output"])
```

### Esempio 3: Sistema di customer support intelligente

```python
class IntelligentCustomerSupport:
    """Sistema di customer support con AI"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.knowledge_base = self._setup_knowledge_base()
        self.classifier = self._setup_classifier()
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
    
    def _setup_knowledge_base(self):
        """Setup knowledge base con FAQ e procedure"""
        
        # Simula knowledge base (in produzione: da database o files)
        kb_data = [
            {
                "category": "account",
                "question": "Come resetto la password?",
                "answer": "Vai su 'Password dimenticata' nella pagina di login, inserisci la tua email e segui le istruzioni."
            },
            {
                "category": "billing",
                "question": "Come posso vedere le mie fatture?",
                "answer": "Accedi al tuo account, vai in 'Fatturazione' > 'Storico fatture' per vedere tutte le fatture."
            },
            {
                "category": "technical",
                "question": "L'app non si apre, cosa faccio?",
                "answer": "Prova a: 1) Riavviare l'app 2) Aggiornare alla versione più recente 3) Riavviare il dispositivo"
            }
        ]
        
        # Crea embeddings per similarity search
        texts = [f"{item['question']} {item['answer']}" for item in kb_data]
        embeddings = OpenAIEmbeddings()
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=kb_data
        )
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def _setup_classifier(self):
        """Setup classificatore per intent recognition"""
        
        classification_template = """
Classifica la seguente richiesta del cliente in una di queste categorie:

Categorie disponibili:
- account: problemi di login, password, registrazione
- billing: fatture, pagamenti, abbonamenti
- technical: problemi tecnici, bug, performance
- product: domande sui prodotti, funzionalità
- complaint: reclami, insoddisfazione
- general: informazioni generali, altro

Richiesta del cliente: {customer_message}

Rispondi solo con il nome della categoria.
"""
        
        prompt = PromptTemplate.from_template(classification_template)
        return prompt | self.llm | StrOutputParser()
    
    def process_customer_request(self, customer_message: str, customer_id: str = "unknown"):
        """Elabora richiesta del cliente"""
        
        # 1. Classifica intent
        category = self.classifier.invoke({"customer_message": customer_message}).strip().lower()
        print(f"🏷️  Categoria identificata: {category}")
        
        # 2. Cerca nella knowledge base
        relevant_docs = self.knowledge_base.get_relevant_documents(customer_message)
        kb_context = "\n".join([doc.page_content for doc in relevant_docs[:2]])
        
        # 3. Genera risposta personalizzata
        response_template = """
Sei un assistente customer support professionale e utile.

Informazioni dal nostro knowledge base:
{kb_context}

Categoria della richiesta: {category}
Richiesta del cliente: {customer_message}

Fornisci una risposta professionale, utile e empatica. Se le informazioni della knowledge base 
non sono sufficienti, spiega chiaramente come il cliente può ottenere ulteriore assistenza.

Include sempre:
1. Conferma di aver compreso il problema
2. Soluzione o prossimi passi chiari
3. Offerta di assistenza aggiuntiva se necessario
"""
        
        prompt = PromptTemplate.from_template(response_template)
        response_chain = prompt | self.llm | StrOutputParser()
        
        response = response_chain.invoke({
            "kb_context": kb_context,
            "category": category,
            "customer_message": customer_message
        })
        
        # 4. Determina se serve escalation
        escalation_needed = self._check_escalation_needed(customer_message, category)
        
        # 5. Log dell'interazione
        interaction_log = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "message": customer_message,
            "category": category,
            "response": response,
            "escalation_needed": escalation_needed,
            "kb_docs_used": len(relevant_docs)
        }
        
        return {
            "response": response,
            "category": category,
            "escalation_needed": escalation_needed,
            "log": interaction_log
        }
    
    def _check_escalation_needed(self, message: str, category: str) -> bool:
        """Determina se serve escalation a operatore umano"""
        
        escalation_keywords = [
            "insoddisfatto", "arrabbiato", "terribile", "chiudere account",
            "avvocato", "rimborso", "reclamo", "inaccettabile", "scandaloso"
        ]
        
        # Escalation per parole chiave critiche
        if any(keyword in message.lower() for keyword in escalation_keywords):
            return True
        
        # Escalation per categorie complesse
        if category in ["complaint", "billing"]:
            return True
            
        return False
    
    def generate_summary_report(self, interactions: List[dict]):
        """Genera report riassuntivo delle interazioni"""
        
        summary_template = """
Analizza le seguenti interazioni di customer support e genera un report riassuntivo:

Interazioni: {interactions}

Il report deve includere:
1. Numero totale di interazioni
2. Categorie più comuni
3. Tasso di escalation
4. Problemi ricorrenti identificati
5. Raccomandazioni per migliorare il servizio
"""
        
        prompt = PromptTemplate.from_template(summary_template)
        summary_chain = prompt | self.llm | StrOutputParser()
        
        return summary_chain.invoke({
            "interactions": json.dumps(interactions, indent=2, ensure_ascii=False)
        })

# Utilizzo del sistema di customer support
support_system = IntelligentCustomerSupport()

# Simulazione richieste clienti
customer_requests = [
    "Non riesco ad accedere al mio account, ho dimenticato la password",
    "La vostra app continua a crashare, è inaccettabile!",
    "Dove posso vedere le mie fatture del mese scorso?",
    "Vorrei informazioni sui vostri piani di abbonamento"
]

print("🎧 SISTEMA CUSTOMER SUPPORT")
print("=" * 60)

interaction_logs = []

for i, request in enumerate(customer_requests, 1):
    print(f"\n--- RICHIESTA {i} ---")
    print(f"Cliente: {request}")
    
    result = support_system.process_customer_request(
        customer_message=request,
        customer_id=f"customer_{i}"
    )
    
    print(f"💬 Risposta: {result['response']}")
    print(f"🏷️  Categoria: {result['category']}")
    
    if result['escalation_needed']:
        print("⚠️  ESCALATION NECESSARIA - Trasferimento a operatore umano")
    
    interaction_logs.append(result['log'])

# Genera report riassuntivo
print("\n" + "=" * 60)
print("📊 REPORT RIASSUNTIVO")
print("=" * 60)

summary_report = support_system.generate_summary_report(interaction_logs)
print(summary_report)
```

---

## Conclusioni

LangChain è un framework estremamente potente e versatile per costruire applicazioni AI avanzate. I suoi punti di forza includono:

* **🔧 Modularità**: Componenti riutilizzabili e intercambiabili
* **🔗 Integrazioni**: Ampio ecosistema di connettori
* **💾 Memoria**: Gestione sofisticata dello stato conversazionale  
* **🛠️ Tools**: Facile integrazione con servizi esterni
* **🤖 Agents**: Capacità di ragionamento e decisione autonoma
* **📚 RAG**: Supporto nativo per retrieval-augmented generation

**Casi d'uso principali:**
- Chatbot e assistenti conversazionali
- Sistemi di Q&A su documenti
- Automazione di workflow complessi
- Analisi e generazione di contenuti
- Customer support intelligente
- Sistemi di raccomandazione

**Best practices da seguire:**
1. **Inizia semplice**: Usa LCEL per chain moderne e pulite
2. **Gestisci la memoria**: Scegli il tipo appropriato per il tuo caso d'uso
3. **Monitora le performance**: Implementa logging e callbacks
4. **Testa intensivamente**: Valida comportamenti edge case
5. **Ottimizza i costi**: Usa caching e fallback intelligenti
6. **Struttura l'output**: Usa Pydantic per dati consistenti

LangChain continua ad evolversi rapidamente, quindi mantieniti aggiornato con la documentazione ufficiale e sperimenta con i nuovi componenti man mano che vengono rilasciati.

**Prossimi passi consigliati:**
1. Implementa un progetto pilota con i pattern base
2. Esplora integrazioni specifiche per il tuo dominio  
3. Sperimentare con LangGraph per workflow complessi
4. Implementa monitoring e analytics in produzione
5. Contribuisci alla community open-source

Il futuro delle applicazioni AI è nelle mani di framework come LangChain che democratizzano lo sviluppo di soluzioni intelligenti e sofisticate!
