import React, { useState, useEffect } from 'react';
import { Brain, Calculator, Search, Palette, BarChart3, MessageSquare, Sparkles, Users, Activity } from 'lucide-react';

const MultiAgentDashboard = () => {
  const [userQuery, setUserQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [messages, setMessages] = useState([]);
  const [activeAgents, setActiveAgents] = useState(new Set());
  const [agentResponses, setAgentResponses] = useState({});
  const [finalAnswer, setFinalAnswer] = useState('');
  const [showTutorial, setShowTutorial] = useState(true);

  // Definizione degli agent specializzati
  const agents = {
    math: {
      id: 'math',
      name: 'Agent Matematico',
      icon: Calculator,
      color: 'bg-blue-500',
      description: 'Specializzato in calcoli e analisi numeriche',
      examples: ['calcoli complessi', 'statistiche', 'conversioni']
    },
    search: {
      id: 'search',
      name: 'Agent Ricerca',
      icon: Search,
      color: 'bg-green-500',
      description: 'Cerca informazioni aggiornate e fatti',
      examples: ['notizie recenti', 'dati storici', 'informazioni generali']
    },
    creative: {
      id: 'creative',
      name: 'Agent Creativo',
      icon: Palette,
      color: 'bg-purple-500',
      description: 'Genera contenuti creativi e idee',
      examples: ['storie', 'idee marketing', 'brainstorming']
    },
    analyst: {
      id: 'analyst',
      name: 'Agent Analista',
      icon: BarChart3,
      color: 'bg-orange-500',
      description: 'Analizza dati e fornisce insights',
      examples: ['trend analysis', 'pattern recognition', 'reporting']
    }
  };

  // Esempi di query predefinite
  const exampleQueries = [
    "Qual è la popolazione di Milano e quanto rappresenta in percentuale rispetto all'Italia?",
    "Crea una storia breve su un robot che impara a cucinare e poi analizza i suoi ingredienti preferiti",
    "Trova le ultime notizie su AI e calcola quante aziende sono menzionate",
    "Analizza i trend del mercato tech e suggerisci 3 idee innovative per una startup"
  ];

  // Simulazione del processo multi-agent
  const processQuery = async () => {
    if (!userQuery.trim()) return;
    
    setIsProcessing(true);
    setCurrentStep(1);
    setMessages([]);
    setActiveAgents(new Set());
    setAgentResponses({});
    setFinalAnswer('');

    // Step 1: Manager riceve la query
    await simulateStep(
      "🎯 Manager LLM sta analizzando la tua richiesta...",
      1500
    );

    // Step 2: Decomposizione del task
    setCurrentStep(2);
    const decomposition = decomposeTask(userQuery);
    await simulateStep(
      `📋 Ho identificato ${decomposition.length} sotto-task da assegnare agli agent specializzati:`,
      1000
    );
    
    decomposition.forEach((task, index) => {
      setTimeout(() => {
        addMessage(`${index + 1}. ${task.description}`, 'decomposition');
      }, 500 * (index + 1));
    });

    await new Promise(resolve => setTimeout(resolve, 500 * decomposition.length + 1000));

    // Step 3: Assegnazione agli agent
    setCurrentStep(3);
    const assignedAgents = new Set(decomposition.map(t => t.agent));
    setActiveAgents(assignedAgents);
    
    await simulateStep(
      `🤝 Sto assegnando i task a ${assignedAgents.size} agent specializzati...`,
      1500
    );

    // Step 4: Esecuzione parallela degli agent
    setCurrentStep(4);
    const responses = {};
    
    for (const agentId of assignedAgents) {
      const agent = agents[agentId];
      const agentTasks = decomposition.filter(t => t.agent === agentId);
      
      setTimeout(async () => {
        addMessage(
          `${agent.name} sta lavorando su: ${agentTasks.map(t => t.description).join(', ')}`,
          'agent-working',
          agentId
        );
        
        // Simula risposta dell'agent
        const response = await simulateAgentResponse(agentId, agentTasks);
        responses[agentId] = response;
        setAgentResponses(prev => ({ ...prev, [agentId]: response }));
        
        addMessage(
          `✅ ${agent.name} ha completato il suo task!`,
          'agent-complete',
          agentId
        );
      }, 2000 + Math.random() * 2000);
    }

    // Attendi che tutti gli agent finiscano
    await new Promise(resolve => setTimeout(resolve, 6000));

    // Step 5: Sintesi finale
    setCurrentStep(5);
    await simulateStep(
      "🎭 Manager LLM sta combinando tutte le risposte...",
      2000
    );

    const final = synthesizeResponses(userQuery, responses);
    setFinalAnswer(final);
    setCurrentStep(6);
    setIsProcessing(false);
  };

  // Funzioni di supporto per la simulazione
  const decomposeTask = (query) => {
    const tasks = [];
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('calcol') || lowerQuery.includes('percent') || lowerQuery.includes('quant')) {
      tasks.push({ agent: 'math', description: 'Eseguire calcoli numerici e percentuali' });
    }
    if (lowerQuery.includes('popolazione') || lowerQuery.includes('notizie') || lowerQuery.includes('trova') || lowerQuery.includes('cerca')) {
      tasks.push({ agent: 'search', description: 'Cercare informazioni e dati aggiornati' });
    }
    if (lowerQuery.includes('storia') || lowerQuery.includes('crea') || lowerQuery.includes('idee') || lowerQuery.includes('suggerisci')) {
      tasks.push({ agent: 'creative', description: 'Generare contenuti creativi e idee' });
    }
    if (lowerQuery.includes('analizza') || lowerQuery.includes('trend') || lowerQuery.includes('pattern')) {
      tasks.push({ agent: 'analyst', description: 'Analizzare dati e identificare pattern' });
    }
    
    if (tasks.length === 0) {
      tasks.push({ agent: 'search', description: 'Gestire la richiesta generale' });
    }
    
    return tasks;
  };

  const simulateAgentResponse = async (agentId, tasks) => {
    // Simula risposte realistiche per ogni agent
    const responses = {
      math: "Ho calcolato che Milano ha circa 1.4 milioni di abitanti, che rappresenta il 2.3% della popolazione italiana (60 milioni). Ho anche preparato un grafico comparativo con altre città italiane.",
      search: "Ho trovato che Milano ha una popolazione di 1,396,059 abitanti (dati ISTAT 2023). L'Italia ha 58,997,201 abitanti. Ho anche recuperato trend demografici degli ultimi 10 anni.",
      creative: "Ho creato una storia affascinante su 'Byte', un robot chef che scopre la passione per la cucina italiana. La storia include elementi di humor e apprendimento automatico applicato alla gastronomia.",
      analyst: "Ho analizzato i trend: 1) AI generativa in crescita del 250%, 2) Sostenibilità tech +180%, 3) Quantum computing +150%. Suggerisco: AI per personalizzazione educativa, piattaforma carbon tracking, o servizi quantum-as-a-service."
    };
    
    return responses[agentId] || "Task completato con successo.";
  };

  const synthesizeResponses = (query, responses) => {
    const responseValues = Object.values(responses).join(' ');
    return `Basandomi sull'analisi coordinata di ${Object.keys(responses).length} agent specializzati:\n\n${responseValues}\n\nIn sintesi, ho combinato ricerca dati, analisi numerica e insights creativi per fornirti una risposta completa e accurata alla tua domanda.`;
  };

  const simulateStep = (message, delay) => {
    return new Promise(resolve => {
      addMessage(message, 'system');
      setTimeout(resolve, delay);
    });
  };

  const addMessage = (text, type = 'user', agentId = null) => {
    setMessages(prev => [...prev, { text, type, agentId, timestamp: Date.now() }]);
  };

  // Componente Agent Card
  const AgentCard = ({ agent, isActive, hasResponse }) => {
    const Icon = agent.icon;
    const pulseAnimation = isActive && !hasResponse ? 'animate-pulse' : '';
    const scaleAnimation = hasResponse ? 'scale-110' : 'scale-100';
    
    return (
      <div className={`relative transition-all duration-500 ${scaleAnimation}`}>
        <div className={`
          ${agent.color} rounded-xl p-4 shadow-lg transform transition-all duration-300
          ${isActive ? 'ring-4 ring-white ring-opacity-60' : ''}
          ${hasResponse ? 'opacity-100' : isActive ? 'opacity-90' : 'opacity-60'}
          ${pulseAnimation}
        `}>
          <Icon className="w-8 h-8 text-white mb-2" />
          <h3 className="text-white font-bold text-sm">{agent.name}</h3>
          <p className="text-white text-xs opacity-90 mt-1">{agent.description}</p>
          
          {hasResponse && (
            <div className="absolute -top-2 -right-2 bg-green-500 rounded-full p-1">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
          )}
        </div>
        
        {isActive && !hasResponse && (
          <div className="absolute inset-0 rounded-xl bg-white opacity-20 animate-ping"></div>
        )}
      </div>
    );
  };

  // Tutorial Modal
  const TutorialModal = () => (
    <div className={`fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 transition-opacity duration-300 ${showTutorial ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
      <div className="bg-white rounded-2xl p-8 max-w-2xl mx-4 shadow-2xl">
        <h2 className="text-2xl font-bold mb-4 text-gray-800 flex items-center gap-2">
          <Users className="w-8 h-8 text-purple-600" />
          Benvenuto nell'Orchestra Multi-Agent LLM! 
        </h2>
        
        <div className="space-y-4 text-gray-600">
          <p className="text-lg">
            Questa dashboard interattiva mostra come un <span className="font-semibold text-purple-600">LLM Manager</span> coordina 
            diversi <span className="font-semibold text-blue-600">LLM Agent specializzati</span> per risolvere task complessi.
          </p>
          
          <div className="bg-gray-50 rounded-lg p-4 space-y-2">
            <h3 className="font-semibold text-gray-800">Come funziona:</h3>
            <ol className="list-decimal list-inside space-y-1 text-sm">
              <li>Scrivi una domanda complessa nel box di input</li>
              <li>Il Manager LLM analizza e decompone la richiesta</li>
              <li>Assegna sotto-task agli agent specializzati appropriati</li>
              <li>Gli agent lavorano in parallelo sui loro task</li>
              <li>Il Manager combina le risposte in un risultato finale</li>
            </ol>
          </div>
          
          <p className="text-sm italic">
            💡 Prova con domande che richiedono diverse competenze: ricerca, calcoli, creatività e analisi!
          </p>
        </div>
        
        <button
          onClick={() => setShowTutorial(false)}
          className="mt-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-full font-semibold hover:shadow-lg transition-all duration-300 transform hover:scale-105"
        >
          Inizia l'Esperienza
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900 p-4">
      <TutorialModal />
      
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Brain className="w-10 h-10 text-purple-400" />
            LLM Multi-Agent Orchestra
            <Activity className="w-10 h-10 text-blue-400" />
          </h1>
          <p className="text-gray-300 text-lg">
            Scopri come un LLM Manager coordina agent specializzati per risolvere task complessi
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Agent Orchestra Visualization */}
          <div className="lg:col-span-2 bg-gray-800 bg-opacity-50 backdrop-blur rounded-2xl p-6">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Users className="w-6 h-6" />
              Agent Orchestra
            </h2>
            
            {/* Manager LLM al centro */}
            <div className="relative h-96">
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className={`
                  bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-6 shadow-2xl
                  ${isProcessing ? 'animate-pulse ring-8 ring-purple-400 ring-opacity-30' : ''}
                  transition-all duration-500
                `}>
                  <Brain className="w-12 h-12 text-white mb-2 mx-auto" />
                  <h3 className="text-white font-bold text-lg">Manager LLM</h3>
                  <p className="text-white text-sm opacity-90">Orchestratore Principale</p>
                  
                  {currentStep > 0 && currentStep < 6 && (
                    <div className="mt-3 text-xs text-white bg-white bg-opacity-20 rounded-full px-3 py-1">
                      Step {currentStep}/5
                    </div>
                  )}
                </div>
              </div>
              
              {/* Agent specializzati intorno */}
              <div className="absolute top-4 left-4">
                <AgentCard 
                  agent={agents.math} 
                  isActive={activeAgents.has('math')} 
                  hasResponse={!!agentResponses.math}
                />
              </div>
              
              <div className="absolute top-4 right-4">
                <AgentCard 
                  agent={agents.search} 
                  isActive={activeAgents.has('search')} 
                  hasResponse={!!agentResponses.search}
                />
              </div>
              
              <div className="absolute bottom-4 left-4">
                <AgentCard 
                  agent={agents.creative} 
                  isActive={activeAgents.has('creative')} 
                  hasResponse={!!agentResponses.creative}
                />
              </div>
              
              <div className="absolute bottom-4 right-4">
                <AgentCard 
                  agent={agents.analyst} 
                  isActive={activeAgents.has('analyst')} 
                  hasResponse={!!agentResponses.analyst}
                />
              </div>
              
              {/* Linee di connessione animate */}
              {isProcessing && activeAgents.size > 0 && (
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {Array.from(activeAgents).map((agentId, index) => (
                    <line
                      key={agentId}
                      x1="50%"
                      y1="50%"
                      x2={agentId === 'math' || agentId === 'creative' ? '20%' : '80%'}
                      y2={agentId === 'math' || agentId === 'search' ? '20%' : '80%'}
                      stroke="white"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                      opacity="0.3"
                      className="animate-pulse"
                    >
                      <animate
                        attributeName="stroke-dashoffset"
                        from="0"
                        to="10"
                        dur="1s"
                        repeatCount="indefinite"
                      />
                    </line>
                  ))}
                </svg>
              )}
            </div>
            
            {/* Progress Steps */}
            <div className="mt-6 flex justify-between items-center px-4">
              {['Ricezione', 'Analisi', 'Assegnazione', 'Esecuzione', 'Sintesi'].map((step, index) => (
                <div key={index} className="flex flex-col items-center">
                  <div className={`
                    w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold
                    transition-all duration-500
                    ${currentStep > index + 1 ? 'bg-green-500 text-white' : 
                      currentStep === index + 1 ? 'bg-blue-500 text-white animate-pulse' : 
                      'bg-gray-600 text-gray-400'}
                  `}>
                    {currentStep > index + 1 ? '✓' : index + 1}
                  </div>
                  <span className="text-xs text-gray-400 mt-1">{step}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Message Log */}
          <div className="bg-gray-800 bg-opacity-50 backdrop-blur rounded-2xl p-6">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <MessageSquare className="w-6 h-6" />
              Log Comunicazioni
            </h2>
            
            <div className="h-96 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`
                    p-3 rounded-lg text-sm animate-fadeIn
                    ${msg.type === 'system' ? 'bg-purple-900 bg-opacity-50 text-purple-200' :
                      msg.type === 'decomposition' ? 'bg-blue-900 bg-opacity-50 text-blue-200' :
                      msg.type === 'agent-working' ? 'bg-yellow-900 bg-opacity-50 text-yellow-200' :
                      msg.type === 'agent-complete' ? 'bg-green-900 bg-opacity-50 text-green-200' :
                      'bg-gray-700 bg-opacity-50 text-gray-200'}
                  `}
                >
                  {msg.agentId && (
                    <span className="font-semibold mr-2">
                      [{agents[msg.agentId]?.name}]
                    </span>
                  )}
                  {msg.text}
                </div>
              ))}
              
              {messages.length === 0 && (
                <div className="text-gray-500 text-center mt-8">
                  I messaggi appariranno qui durante l'elaborazione...
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-gray-800 bg-opacity-50 backdrop-blur rounded-2xl p-6 mb-6">
          <h2 className="text-xl font-bold text-white mb-4">Prova il Sistema Multi-Agent</h2>
          
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isProcessing && processQuery()}
              placeholder="Inserisci una domanda complessa che richiede diverse competenze..."
              className="flex-1 px-4 py-3 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={isProcessing}
            />
            
            <button
              onClick={processQuery}
              disabled={isProcessing || !userQuery.trim()}
              className={`
                px-6 py-3 rounded-lg font-semibold transition-all duration-300
                ${isProcessing || !userQuery.trim() 
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:shadow-lg transform hover:scale-105'}
              `}
            >
              {isProcessing ? 'Elaborazione...' : 'Invia Query'}
            </button>
          </div>
          
          {/* Example Queries */}
          <div className="flex flex-wrap gap-2">
            <span className="text-gray-400 text-sm">Esempi:</span>
            {exampleQueries.map((example, index) => (
              <button
                key={index}
                onClick={() => setUserQuery(example)}
                disabled={isProcessing}
                className="text-xs bg-gray-700 text-gray-300 px-3 py-1 rounded-full hover:bg-gray-600 transition-colors"
              >
                {example.substring(0, 40)}...
              </button>
            ))}
          </div>
        </div>

        {/* Final Answer */}
        {finalAnswer && (
          <div className="bg-gradient-to-r from-green-900 to-blue-900 bg-opacity-50 backdrop-blur rounded-2xl p-6 animate-fadeIn">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-6 h-6" />
              Risposta Finale Orchestrata
            </h2>
            <div className="text-white whitespace-pre-wrap leading-relaxed">
              {finalAnswer}
            </div>
          </div>
        )}
      </div>
      
      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        
        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }
        
        .scrollbar-thumb-gray-600::-webkit-scrollbar-thumb {
          background-color: #4B5563;
          border-radius: 3px;
        }
        
        .scrollbar-track-transparent::-webkit-scrollbar-track {
          background-color: transparent;
        }
      `}</style>
    </div>
  );
};

export default MultiAgentDashboard;