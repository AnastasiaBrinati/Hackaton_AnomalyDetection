import React, { useState, useEffect } from 'react';
import { Brain, MessageSquare, Eye, Zap, ChevronRight, Play, Pause, RefreshCw, Settings, Code, BookOpen, Lightbulb, Target, Search, Database, Globe, FileText, AlertCircle, CheckCircle, Terminal, Cpu, Link2, Loader, ArrowRight, FileSearch, Calculator, GitBranch } from 'lucide-react';

const AIAgentDashboard = () => {
  const [activeTab, setActiveTab] = useState('agent-flow');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [showOutput, setShowOutput] = useState({});
  const [userQuery, setUserQuery] = useState("Rileva anomalie nelle transazioni bancarie del file transactions.csv");
  const [systemPrompt, setSystemPrompt] = useState(`Sei un esperto di anomaly detection specializzato in frodi finanziarie.
Usa sempre tool esterni quando necessario per analizzare dati.
Spiega il tuo ragionamento passo dopo passo.
Fornisci risultati concreti e actionable.`);
  const [executedSteps, setExecutedSteps] = useState([]);
  const [selectedTool, setSelectedTool] = useState(null);

  // Tool disponibili
  const availableTools = [
    {
      id: 'web_search',
      name: 'Web Search',
      icon: <Globe className="w-4 h-4" />,
      description: 'Cerca informazioni aggiornate sul web',
      color: 'blue'
    },
    {
      id: 'file_reader',
      name: 'File Reader',
      icon: <FileText className="w-4 h-4" />,
      description: 'Legge e analizza file CSV, JSON, TXT',
      color: 'green'
    },
    {
      id: 'code_executor',
      name: 'Code Executor',
      icon: <Terminal className="w-4 h-4" />,
      description: 'Esegue codice Python per analisi dati',
      color: 'purple'
    },
    {
      id: 'database_query',
      name: 'Database Query',
      icon: <Database className="w-4 h-4" />,
      description: 'Interroga database SQL',
      color: 'orange'
    },
    {
      id: 'ml_model',
      name: 'ML Model',
      icon: <Cpu className="w-4 h-4" />,
      description: 'Applica modelli ML pre-addestrati',
      color: 'red'
    }
  ];

  // Processo completo dell'agente con tool calls
  const agentSteps = [
    {
      id: 1,
      type: 'thought',
      title: 'Analisi della richiesta',
      content: 'L\'utente vuole rilevare anomalie in un file di transazioni bancarie. Devo prima leggere il file per capire la struttura dei dati.',
      output: 'Query parsed: anomaly_detection(target="transactions.csv", domain="banking")',
      tools: [],
      duration: '120ms'
    },
    {
      id: 2,
      type: 'action',
      title: 'Lettura del file',
      content: 'Utilizzo il File Reader per caricare e ispezionare il dataset delle transazioni.',
      tools: ['file_reader'],
      toolCall: {
        tool: 'file_reader',
        params: { file: 'transactions.csv', preview_rows: 5 },
        result: `
Loaded: transactions.csv
Shape: (10000, 8)
Columns: [transaction_id, user_id, amount, merchant, timestamp, location, category, risk_score]

Preview:
| transaction_id | user_id | amount  | merchant      | timestamp           | location | risk_score |
|----------------|---------|---------|---------------|---------------------|----------|------------|
| TRX001         | U1234   | 1250.00 | Amazon        | 2025-01-15 10:30:00 | IT       | 0.12       |
| TRX002         | U1234   | 35.50   | Starbucks     | 2025-01-15 11:45:00 | IT       | 0.08       |
| TRX003         | U5678   | 8750.00 | Wire Transfer | 2025-01-15 02:15:00 | RU       | 0.89       |`
      },
      output: 'File caricato: 10,000 transazioni, 8 features identificate',
      duration: '450ms'
    },
    {
      id: 3,
      type: 'observation',
      title: 'Analisi preliminare dei dati',
      content: 'Ho identificato pattern sospetti: transazione TRX003 con importo elevato, orario insolito (02:15) e location anomala (RU) con risk_score alto.',
      output: 'Anomalie potenziali identificate: 3.2% delle transazioni hanno risk_score > 0.7',
      tools: [],
      duration: '80ms'
    },
    {
      id: 4,
      type: 'thought',
      title: 'Selezione strategia di detection',
      content: 'Basandomi sulla struttura dei dati, applicherò multiple tecniche: Isolation Forest per pattern generali, LSTM per anomalie temporali, e analisi statistica per outlier.',
      output: 'Strategy selected: ensemble_approach(methods=["isolation_forest", "lstm", "statistical"])',
      tools: [],
      duration: '95ms'
    },
    {
      id: 5,
      type: 'action',
      title: 'Ricerca best practices',
      content: 'Cerco informazioni aggiornate sulle migliori tecniche di fraud detection nel banking.',
      tools: ['web_search'],
      toolCall: {
        tool: 'web_search',
        params: { query: 'bank fraud detection isolation forest 2025 best practices' },
        result: `
Found 3 relevant results:
1. "2025 Guide: ML for Banking Fraud" - Isolation Forest achieves 94% accuracy
2. "Real-time Anomaly Detection in Financial Services" - Ensemble methods recommended
3. "New EU Regulations on AI in Banking" - Explainability requirements for fraud detection`
      },
      output: 'Best practices identificate: ensemble methods con focus su explainability',
      duration: '1200ms'
    },
    {
      id: 6,
      type: 'action',
      title: 'Esecuzione codice di analisi',
      content: 'Implemento Isolation Forest sul dataset per identificare transazioni anomale.',
      tools: ['code_executor'],
      toolCall: {
        tool: 'code_executor',
        params: { 
          language: 'python',
          code: `
from sklearn.ensemble import IsolationForest
import pandas as pd

# Preparo features per il modello
features = ['amount', 'hour_of_day', 'location_risk', 'merchant_frequency']
X = prepare_features(df)

# Addestro Isolation Forest
clf = IsolationForest(contamination=0.03, random_state=42)
predictions = clf.fit_predict(X)

# Identifico anomalie
anomalies = df[predictions == -1]
print(f"Anomalie trovate: {len(anomalies)}")
print(f"Top anomalie per risk score: {anomalies.nlargest(5, 'risk_score')}")`
        },
        result: `
Executing...
Anomalie trovate: 312
Top anomalie per risk score:
| transaction_id | amount   | risk_score | reason                    |
|----------------|----------|------------|---------------------------|
| TRX003         | 8750.00  | 0.89       | High amount + unusual time |
| TRX1847        | 12500.00 | 0.85       | New merchant + high amount |
| TRX5692        | 6300.00  | 0.82       | Unusual location pattern   |`
      },
      output: '312 transazioni anomale identificate (3.12% del totale)',
      duration: '2500ms'
    },
    {
      id: 7,
      type: 'action',
      title: 'Query database per pattern storici',
      content: 'Verifico se questi pattern anomali corrispondono a frodi confermate nel database storico.',
      tools: ['database_query'],
      toolCall: {
        tool: 'database_query',
        params: { 
          query: `SELECT pattern_type, fraud_confirmed, COUNT(*) 
                  FROM historical_frauds 
                  WHERE pattern_type IN ('high_amount_unusual_time', 'new_merchant_high_value')
                  GROUP BY pattern_type, fraud_confirmed`
        },
        result: `
Query executed successfully:
| pattern_type              | fraud_confirmed | count |
|---------------------------|-----------------|-------|
| high_amount_unusual_time  | true           | 847   |
| high_amount_unusual_time  | false          | 123   |
| new_merchant_high_value   | true           | 592   |
| new_merchant_high_value   | false          | 201   |

Precision rate: 82.4% for these patterns`
      },
      output: 'Pattern validati: 82.4% precision rate su frodi storiche',
      duration: '680ms'
    },
    {
      id: 8,
      type: 'observation',
      title: 'Sintesi risultati e raccomandazioni',
      content: 'Ho identificato 312 transazioni sospette usando Isolation Forest. I pattern principali (importi elevati in orari insoliti, nuovi merchant) hanno un tasso di precisione dell\'82.4% basato sui dati storici.',
      output: `
REPORT FINALE:
- Transazioni analizzate: 10,000
- Anomalie rilevate: 312 (3.12%)
- Top risk patterns: unusual time + high amount
- Precision stimata: 82.4%
- Azione consigliata: Review manuale immediata per TRX003, TRX1847, TRX5692`,
      tools: [],
      duration: '150ms'
    }
  ];

  // TAO steps più dettagliati
  const taoSteps = [
    {
      thought: "Devo analizzare transazioni bancarie per anomalie. Prima verifico struttura dati e identifico features rilevanti.",
      action: "file_reader.read('transactions.csv') → Carico dataset e analizzo schema",
      observation: "Dataset con 10k transazioni, 8 features. Risk score presente suggerisce pre-analisi.",
      output: "✓ Dati caricati correttamente"
    },
    {
      thought: "Pattern anomali potrebbero includere: importi elevati, orari insoliti, location sospette. Serve approccio multi-metodo.",
      action: "code_executor.run(isolation_forest) → Applico algoritmo ML per anomaly detection",
      observation: "312 anomalie rilevate. Pattern principale: transazioni >5000€ tra 00:00-06:00 da location inusuali.",
      output: "⚠️ 3.12% transazioni flaggate come anomale"
    },
    {
      thought: "Devo validare questi risultati contro dati storici per evitare falsi positivi.",
      action: "database_query.execute('SELECT fraud_patterns...') → Confronto con frodi confermate",
      observation: "82.4% delle anomalie simili sono risultate frodi reali. Alta confidenza nei risultati.",
      output: "✓ Modello validato con 82.4% precision"
    }
  ];

  // Animation effect
  useEffect(() => {
    if (isPlaying && currentStep < agentSteps.length) {
      const timer = setTimeout(() => {
        setExecutedSteps([...executedSteps, agentSteps[currentStep]]);
        setCurrentStep(currentStep + 1);
      }, 2000);
      return () => clearTimeout(timer);
    } else if (isPlaying && currentStep === agentSteps.length) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep]);

  const resetAnimation = () => {
    setCurrentStep(0);
    setExecutedSteps([]);
    setIsPlaying(false);
    setShowOutput({});
  };

  const getStepIcon = (type) => {
    switch(type) {
      case 'thought': return <Brain className="w-5 h-5" />;
      case 'action': return <Zap className="w-5 h-5" />;
      case 'observation': return <Eye className="w-5 h-5" />;
      default: return <AlertCircle className="w-5 h-5" />;
    }
  };

  const getStepColor = (type) => {
    switch(type) {
      case 'thought': return 'blue';
      case 'action': return 'purple';
      case 'observation': return 'green';
      default: return 'gray';
    }
  };

  const TabButton = ({ id, label, icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
        activeTab === id 
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg' 
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Come Funziona un Agente AI
          </h1>
          <p className="text-gray-600 text-lg">Dashboard interattiva con tool esterni e output reali</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center gap-4 mb-8 flex-wrap">
          <TabButton id="agent-flow" label="Agent Flow Completo" icon={<GitBranch className="w-5 h-5" />} />
          <TabButton id="tools" label="Tool Esterni" icon={<Link2 className="w-5 h-5" />} />
          <TabButton id="tao" label="TAO in Azione" icon={<Eye className="w-5 h-5" />} />
          <TabButton id="system-prompt" label="System Prompt" icon={<Settings className="w-5 h-5" />} />
        </div>

        {/* Content Area */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Agent Flow Tab */}
          {activeTab === 'agent-flow' && (
            <div>
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <GitBranch className="w-6 h-6 text-blue-600" />
                Flusso Completo dell'Agente AI
              </h2>
              
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-blue-800">
                  Osserva come un agente AI moderno elabora una richiesta complessa usando reasoning, tool esterni e iterazioni TAO.
                  Ogni step produce output concreti e verificabili.
                </p>
              </div>

              {/* User Query Input */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Query Utente:</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={userQuery}
                    onChange={(e) => setUserQuery(e.target.value)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Inserisci una richiesta complessa..."
                  />
                  <button className="px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors">
                    <FileSearch className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Animation Controls */}
              <div className="flex items-center gap-4 mb-6">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? 'Pausa' : 'Avvia'} Esecuzione
                </button>
                <button
                  onClick={resetAnimation}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Reset
                </button>
                <div className="ml-auto flex items-center gap-2 text-sm text-gray-600">
                  <Cpu className="w-4 h-4" />
                  Steps: {currentStep}/{agentSteps.length}
                </div>
              </div>

              {/* Agent Steps Execution */}
              <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                {executedSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className={`border-2 rounded-lg overflow-hidden transition-all duration-500 ${
                      index === executedSteps.length - 1 ? 'border-blue-500 shadow-lg' : 'border-gray-300'
                    }`}
                  >
                    <div className={`flex items-center justify-between px-4 py-3 bg-gradient-to-r ${
                      step.type === 'thought' ? 'from-blue-50 to-blue-100' :
                      step.type === 'action' ? 'from-purple-50 to-purple-100' :
                      'from-green-50 to-green-100'
                    }`}>
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-full ${
                          step.type === 'thought' ? 'bg-blue-600 text-white' :
                          step.type === 'action' ? 'bg-purple-600 text-white' :
                          'bg-green-600 text-white'
                        }`}>
                          {getStepIcon(step.type)}
                        </div>
                        <div>
                          <h3 className="font-semibold">{step.title}</h3>
                          <p className="text-sm text-gray-600">{step.type.toUpperCase()} • Step {step.id}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {step.tools.map(tool => {
                          const toolInfo = availableTools.find(t => t.id === tool);
                          return toolInfo ? (
                            <span key={tool} className="flex items-center gap-1 px-2 py-1 bg-white rounded text-xs">
                              {toolInfo.icon}
                              {toolInfo.name}
                            </span>
                          ) : null;
                        })}
                        <span className="text-xs text-gray-500">{step.duration}</span>
                      </div>
                    </div>
                    
                    <div className="p-4">
                      <p className="text-gray-700 mb-3">{step.content}</p>
                      
                      {/* Tool Call Details */}
                      {step.toolCall && (
                        <div className="mb-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                          <div className="flex items-center gap-2 mb-2">
                            <Terminal className="w-4 h-4 text-gray-600" />
                            <span className="font-mono text-sm text-gray-700">
                              {step.toolCall.tool}({JSON.stringify(step.toolCall.params)})
                            </span>
                          </div>
                          <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded overflow-x-auto">
{step.toolCall.result}
                          </pre>
                        </div>
                      )}
                      
                      {/* Output */}
                      <div className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                        <div className="flex-1">
                          <span className="text-sm font-medium text-gray-600">Output:</span>
                          <p className="text-sm text-gray-800 font-mono bg-gray-100 px-2 py-1 rounded mt-1">
                            {step.output}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                
                {isPlaying && currentStep < agentSteps.length && (
                  <div className="flex items-center justify-center py-8">
                    <Loader className="w-8 h-8 text-blue-600 animate-spin" />
                    <span className="ml-3 text-gray-600">Processing step {currentStep + 1}...</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div>
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Link2 className="w-6 h-6 text-purple-600" />
                Tool Esterni dell'Agente
              </h2>
              
              <div className="mb-6 p-4 bg-purple-50 rounded-lg">
                <p className="text-purple-800">
                  Gli agenti AI moderni non sono isolati - utilizzano tool esterni per accedere a dati, eseguire codice, 
                  interrogare database e molto altro. Questo li rende estremamente potenti e versatili.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                {availableTools.map(tool => (
                  <div
                    key={tool.id}
                    onClick={() => setSelectedTool(tool)}
                    className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                      selectedTool?.id === tool.id 
                        ? 'border-purple-500 bg-purple-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg bg-${tool.color}-100 text-${tool.color}-600`}>
                        {tool.icon}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold">{tool.name}</h3>
                        <p className="text-sm text-gray-600 mt-1">{tool.description}</p>
                        <code className="text-xs font-mono text-gray-500 mt-2 block">
                          {tool.id}()
                        </code>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {selectedTool && (
                <div className="border-2 border-purple-200 rounded-lg p-6 bg-purple-50">
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    {selectedTool.icon}
                    {selectedTool.name} - Esempio di Utilizzo
                  </h3>
                  
                  {selectedTool.id === 'web_search' && (
                    <div className="space-y-3">
                      <div className="bg-white rounded p-3">
                        <p className="text-sm font-mono mb-2">web_search(query="anomaly detection banking 2025")</p>
                        <p className="text-sm text-gray-600">→ Cerca best practices aggiornate per fraud detection</p>
                      </div>
                      <div className="text-sm">
                        <strong>Quando usarlo:</strong> Per informazioni real-time, trend attuali, normative recenti
                      </div>
                    </div>
                  )}
                  
                  {selectedTool.id === 'file_reader' && (
                    <div className="space-y-3">
                      <div className="bg-white rounded p-3">
                        <p className="text-sm font-mono mb-2">file_reader(file="transactions.csv", preview=True)</p>
                        <p className="text-sm text-gray-600">→ Carica e analizza struttura del dataset</p>
                      </div>
                      <div className="text-sm">
                        <strong>Formati supportati:</strong> CSV, JSON, Excel, TXT, PDF
                      </div>
                    </div>
                  )}
                  
                  {selectedTool.id === 'code_executor' && (
                    <div className="space-y-3">
                      <div className="bg-white rounded p-3">
                        <p className="text-sm font-mono mb-2">code_executor(lang="python", code="...")</p>
                        <p className="text-sm text-gray-600">→ Esegue algoritmi ML, analisi statistiche, visualizzazioni</p>
                      </div>
                      <div className="text-sm">
                        <strong>Librerie disponibili:</strong> sklearn, pandas, numpy, matplotlib
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tool Connection Flow */}
              <div className="mt-8">
                <h3 className="font-semibold mb-4">Flusso di Connessione ai Tool</h3>
                <div className="bg-gray-50 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white mb-2 mx-auto">
                        <Brain className="w-8 h-8" />
                      </div>
                      <p className="text-sm font-medium">Agent Core</p>
                    </div>
                    
                    <ArrowRight className="w-8 h-8 text-gray-400" />
                    
                    <div className="text-center">
                      <div className="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center text-white mb-2 mx-auto">
                        <GitBranch className="w-8 h-8" />
                      </div>
                      <p className="text-sm font-medium">Tool Router</p>
                    </div>
                    
                    <ArrowRight className="w-8 h-8 text-gray-400" />
                    
                    <div className="grid grid-cols-2 gap-2">
                      {availableTools.slice(0, 4).map(tool => (
                        <div key={tool.id} className="w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center">
                          {tool.icon}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-4 text-center text-sm text-gray-600">
                    L'agente decide autonomamente quali tool utilizzare basandosi sul contesto e sulla query
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAO Tab */}
          {activeTab === 'tao' && (
            <div>
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Eye className="w-6 h-6 text-green-600" />
                Paradigma TAO con Output Reali
              </h2>
              
              <div className="mb-6 p-4 bg-green-50 rounded-lg">
                <p className="text-green-800">
                  Thought-Action-Observation (TAO) è il ciclo fondamentale degli agenti AI. 
                  Qui vediamo come ogni fase produce output concreti e misurabili.
                </p>
              </div>

              {/* TAO Cycle Visualization */}
              <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 via-purple-50 to-green-50 rounded-xl">
                <div className="flex items-center justify-center gap-8">
                  <div className="text-center">
                    <div className="w-24 h-24 bg-white border-4 border-blue-500 rounded-full flex items-center justify-center mb-3 shadow-lg">
                      <Brain className="w-12 h-12 text-blue-600" />
                    </div>
                    <h4 className="font-bold text-blue-600">THOUGHT</h4>
                    <p className="text-sm text-gray-600 mt-1">Analizza e pianifica</p>
                  </div>
                  
                  <div className="flex flex-col items-center">
                    <ArrowRight className="w-10 h-10 text-gray-400" />
                    <span className="text-xs text-gray-500 mt-1">decide</span>
                  </div>
                  
                  <div className="text-center">
                    <div className="w-24 h-24 bg-white border-4 border-purple-500 rounded-full flex items-center justify-center mb-3 shadow-lg">
                      <Zap className="w-12 h-12 text-purple-600" />
                    </div>
                    <h4 className="font-bold text-purple-600">ACTION</h4>
                    <p className="text-sm text-gray-600 mt-1">Esegue operazioni</p>
                  </div>
                  
                  <div className="flex flex-col items-center">
                    <ArrowRight className="w-10 h-10 text-gray-400" />
                    <span className="text-xs text-gray-500 mt-1">produce</span>
                  </div>
                  
                  <div className="text-center">
                    <div className="w-24 h-24 bg-white border-4 border-green-500 rounded-full flex items-center justify-center mb-3 shadow-lg">
                      <Eye className="w-12 h-12 text-green-600" />
                    </div>
                    <h4 className="font-bold text-green-600">OBSERVATION</h4>
                    <p className="text-sm text-gray-600 mt-1">Valuta risultati</p>
                  </div>
                </div>
              </div>

              {/* TAO Steps with Real Output */}
              <div className="space-y-4">
                <h3 className="font-semibold text-lg mb-3">Esempio Reale: Anomaly Detection su Transazioni</h3>
                
                {taoSteps.map((step, index) => (
                  <div key={index} className="border-2 border-gray-200 rounded-lg overflow-hidden">
                    <div className="bg-gray-50 px-4 py-2 font-semibold text-gray-700 flex items-center justify-between">
                      <span>Ciclo TAO #{index + 1}</span>
                      <span className="text-sm font-normal text-gray-500">
                        {index === 0 ? 'Inizializzazione' : index === 1 ? 'Elaborazione' : 'Validazione'}
                      </span>
                    </div>
                    
                    <div className="divide-y divide-gray-200">
                      {/* Thought */}
                      <div className="p-4 bg-blue-50">
                        <div className="flex items-start gap-3">
                          <Brain className="w-5 h-5 text-blue-600 mt-1" />
                          <div className="flex-1">
                            <p className="font-medium text-blue-800 mb-1">Thought</p>
                            <p className="text-sm text-blue-700 italic">"{step.thought}"</p>
                          </div>
                        </div>
                      </div>
                      
                      {/* Action */}
                      <div className="p-4 bg-purple-50">
                        <div className="flex items-start gap-3">
                          <Zap className="w-5 h-5 text-purple-600 mt-1" />
                          <div className="flex-1">
                            <p className="font-medium text-purple-800 mb-1">Action</p>
                            <code className="text-sm text-purple-700 font-mono block bg-purple-100 px-2 py-1 rounded">
                              {step.action}
                            </code>
                          </div>
                        </div>
                      </div>
                      
                      {/* Observation */}
                      <div className="p-4 bg-green-50">
                        <div className="flex items-start gap-3">
                          <Eye className="w-5 h-5 text-green-600 mt-1" />
                          <div className="flex-1">
                            <p className="font-medium text-green-800 mb-1">Observation</p>
                            <p className="text-sm text-green-700">{step.observation}</p>
                          </div>
                        </div>
                      </div>
                      
                      {/* Output */}
                      <div className="p-4 bg-yellow-50">
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-5 h-5 text-yellow-600 mt-1" />
                          <div className="flex-1">
                            <p className="font-medium text-yellow-800 mb-1">Output Prodotto</p>
                            <p className="text-sm font-mono bg-yellow-100 px-3 py-2 rounded">
                              {step.output}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* TAO Metrics */}
              <div className="mt-8 grid grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-blue-100 to-blue-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-blue-700">3</div>
                  <p className="text-sm text-blue-600 mt-1">Cicli TAO completati</p>
                </div>
                <div className="bg-gradient-to-br from-purple-100 to-purple-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-purple-700">5</div>
                  <p className="text-sm text-purple-600 mt-1">Tool utilizzati</p>
                </div>
                <div className="bg-gradient-to-br from-green-100 to-green-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-green-700">312</div>
                  <p className="text-sm text-green-600 mt-1">Anomalie rilevate</p>
                </div>
              </div>
            </div>
          )}

          {/* System Prompt Tab */}
          {activeTab === 'system-prompt' && (
            <div>
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Settings className="w-6 h-6 text-orange-600" />
                System Prompt: Configurazione dell'Agente
              </h2>
              
              <div className="mb-6 p-4 bg-orange-50 rounded-lg">
                <p className="text-orange-800">
                  Il system prompt definisce il comportamento, le capacità e i limiti dell'agente. 
                  È fondamentale per garantire output consistenti e affidabili.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* System Prompt Editor */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Code className="w-5 h-5" />
                    Editor System Prompt
                  </h3>
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    className="w-full h-80 px-4 py-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  />
                  
                  <div className="mt-4 space-y-2">
                    <h4 className="font-medium text-sm text-gray-700">Template predefiniti:</h4>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
                      <option>Fraud Detection Expert</option>
                      <option>Data Analyst</option>
                      <option>Security Auditor</option>
                      <option>Financial Advisor</option>
                    </select>
                  </div>
                </div>

                {/* System Prompt Analysis */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Lightbulb className="w-5 h-5" />
                    Analisi del System Prompt
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <h4 className="font-medium text-blue-800 mb-2">🎯 Ruolo Definito</h4>
                      <p className="text-sm text-blue-700">
                        "Esperto di anomaly detection" → L'agente userà terminologia tecnica e approcci specializzati
                      </p>
                    </div>
                    
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <h4 className="font-medium text-green-800 mb-2">🔧 Tool Usage</h4>
                      <p className="text-sm text-green-700">
                        "Usa sempre tool esterni" → L'agente preferirà analisi basate su dati reali
                      </p>
                    </div>
                    
                    <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                      <h4 className="font-medium text-purple-800 mb-2">📊 Output Format</h4>
                      <p className="text-sm text-purple-700">
                        "Risultati concreti e actionable" → Output con metriche, percentuali e raccomandazioni
                      </p>
                    </div>
                    
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <h4 className="font-medium text-yellow-800 mb-2">🧠 Reasoning Style</h4>
                      <p className="text-sm text-yellow-700">
                        "Spiega passo dopo passo" → Trasparenza nel processo decisionale
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-4 bg-gray-100 rounded-lg">
                    <h4 className="font-medium mb-2">Impatto Stimato:</h4>
                    <div className="space-y-1 text-sm">
                      <div className="flex items-center justify-between">
                        <span>Precisione risposte:</span>
                        <div className="flex gap-1">
                          {[1,2,3,4,5].map(i => (
                            <div key={i} className={`w-4 h-4 rounded ${i <= 4 ? 'bg-green-500' : 'bg-gray-300'}`} />
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Uso tool esterni:</span>
                        <div className="flex gap-1">
                          {[1,2,3,4,5].map(i => (
                            <div key={i} className={`w-4 h-4 rounded ${i <= 5 ? 'bg-blue-500' : 'bg-gray-300'}`} />
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Dettaglio tecnico:</span>
                        <div className="flex gap-1">
                          {[1,2,3,4,5].map(i => (
                            <div key={i} className={`w-4 h-4 rounded ${i <= 4 ? 'bg-purple-500' : 'bg-gray-300'}`} />
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Interactive Summary */}
        <div className="mt-8 bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Calculator className="w-6 h-6 text-indigo-600" />
            Riepilogo: Come l'Agente Ha Processato la Query
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-100 to-blue-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold text-blue-800">Reasoning Steps</h3>
              </div>
              <p className="text-2xl font-bold text-blue-700">8</p>
              <p className="text-sm text-blue-600">Thought + Analysis</p>
            </div>
            
            <div className="bg-gradient-to-br from-purple-100 to-purple-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Link2 className="w-5 h-5 text-purple-600" />
                <h3 className="font-semibold text-purple-800">Tool Calls</h3>
              </div>
              <p className="text-2xl font-bold text-purple-700">4</p>
              <p className="text-sm text-purple-600">External connections</p>
            </div>
            
            <div className="bg-gradient-to-br from-green-100 to-green-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Eye className="w-5 h-5 text-green-600" />
                <h3 className="font-semibold text-green-800">TAO Cycles</h3>
              </div>
              <p className="text-2xl font-bold text-green-700">3</p>
              <p className="text-sm text-green-600">Complete iterations</p>
            </div>
            
            <div className="bg-gradient-to-br from-orange-100 to-orange-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-orange-600" />
                <h3 className="font-semibold text-orange-800">Final Output</h3>
              </div>
              <p className="text-2xl font-bold text-orange-700">312</p>
              <p className="text-sm text-orange-600">Anomalies detected</p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-indigo-50 rounded-lg">
            <p className="text-indigo-800 text-center">
              <strong>Tempo totale di elaborazione:</strong> 7.235 secondi • 
              <strong>Token utilizzati:</strong> ~2,400 • 
              <strong>Confidence score:</strong> 94.2%
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-600">
          <p className="mb-2">
            Dashboard educativa per comprendere il funzionamento degli agenti AI moderni
          </p>
          <p className="text-sm">
            Con esempi pratici di anomaly detection basati su tecniche reali di ML
          </p>
        </div>
      </div>
    </div>
  );
};

export default AIAgentDashboard;