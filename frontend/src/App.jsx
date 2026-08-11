import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as Icons from 'lucide-react';
import { UploadCloud, Database, Activity, Code, FileSpreadsheet, RefreshCw, Download, ShieldCheck, CheckCircle2, Sliders, Brain, FileText, BarChart3, ChevronRight, AlertCircle } from 'lucide-react';
import './index.css';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

const PIPELINE_STAGES = [
  {
    id: 'STAGE_INGEST',
    name: 'Ingestion & Profiling',
    description: 'Data load, type inference & structured evidence computation',
    icon: 'Database',
    agentIds: ['INGESTION', 'PROFILER']
  },
  {
    id: 'STAGE_SPECIALISTS',
    name: 'Specialist Interpretation',
    icon: 'Brain',
    description: 'Quality audit, shape/distribution analysis, and relationship testing',
    agentIds: ['DATA_QUALITY_AGENT', 'DISTRIBUTION_AGENT', 'RELATIONSHIP_AGENT']
  },
  {
    id: 'STAGE_TRANSFORM',
    name: 'Data Processing & Stats',
    icon: 'Sliders',
    description: 'Missingness contract, transformations, FDR testing, ML readiness',
    agentIds: ['CLEANER', 'FEATURE_ENGINEER', 'ANALYST', 'ADVANCED_ANALYST', 'TIMESERIES_ANALYST']
  },
  {
    id: 'STAGE_VERIFY',
    name: 'Verification & Charts',
    icon: 'ShieldCheck',
    description: 'Plotly visual generation, claim extraction, and evidence verification gate',
    agentIds: ['VISUALIZER', 'CLAIM_GENERATOR', 'VALIDATOR']
  },
  {
    id: 'STAGE_REPORT',
    name: 'Executive Synthesis',
    icon: 'FileText',
    description: 'Synthesizing strictly verified findings into executive report',
    agentIds: ['REPORTER']
  }
];

const AGENTS = [
  { id: 'INGESTION', name: 'Ingestion', role: 'Load & Validate', category: 'core', icon: 'Database' },
  { id: 'PROFILER', name: 'Profiler', role: 'Schema & Evidence Engine', category: 'core', icon: 'FileSearch' },
  { id: 'DATA_QUALITY_AGENT', name: 'Data Quality Specialist', role: 'Quality & PII Audit', category: 'specialists', icon: 'ShieldCheck' },
  { id: 'DISTRIBUTION_AGENT', name: 'Distribution Specialist', role: 'Shape, Normality & Outliers', category: 'specialists', icon: 'PieChart' },
  { id: 'RELATIONSHIP_AGENT', name: 'Relationship Specialist', role: 'Correlations & Hypothesis Tests', category: 'specialists', icon: 'GitMerge' },
  { id: 'CLEANER', name: 'Cleaner', role: 'Imputation & Capping Contract', category: 'processing', icon: 'Wand2' },
  { id: 'FEATURE_ENGINEER', name: 'Feature Engineer', role: 'Non-linear Transformations', category: 'processing', icon: 'Cpu' },
  { id: 'ANALYST', name: 'Statistical Analyst', role: 'FDR Controlled Tests', category: 'processing', icon: 'Activity' },
  { id: 'ADVANCED_ANALYST', name: 'Adv. Analyst', role: 'ML Readiness & Leakage', category: 'processing', icon: 'Network' },
  { id: 'TIMESERIES_ANALYST', name: 'Time-Series Analyst', role: 'Temporal Drift Check', category: 'processing', icon: 'TrendingUp' },
  { id: 'VISUALIZER', name: 'Visualizer', role: 'Interactive Plotly Charts', category: 'verification', icon: 'BarChart3' },
  { id: 'CLAIM_GENERATOR', name: 'Claims Generator', role: 'Structured Evidence Claims', category: 'verification', icon: 'FileCode' },
  { id: 'VALIDATOR', name: 'Validator Gate', role: 'Claim Grounding Verifier', category: 'verification', icon: 'CheckCircle2' },
  { id: 'REPORTER', name: 'Report Synthesizer', role: 'Executive Markdown Report', category: 'core', icon: 'FileText' }
];

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState('');
  const [runId, setRunId] = useState('');
  const [status, setStatus] = useState('idle'); // idle, uploading, running, complete, error
  const [reportContent, setReportContent] = useState('');
  const [chartPaths, setChartPaths] = useState([]);
  const [nodeStatuses, setNodeStatuses] = useState({});
  const [activeCategory, setActiveCategory] = useState('all');
  const [activeNodeDetails, setActiveNodeDetails] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const startPipeline = async () => {
    if (!file) return;
    
    // PURGE PREVIOUS RESULTS FROM STATE ON NEXT RUN
    setReportContent('');
    setChartPaths([]);
    setNodeStatuses({ INGESTION: 'running' });
    setActiveNodeDetails('Starting dataset ingestion...');
    setErrorMessage('');
    setStatus('uploading');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const uploadRes = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadRes.ok) {
        const errData = await uploadRes.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${uploadRes.status}`);
      }
      
      const uploadData = await uploadRes.json();
      console.log('[EDA Agent] Upload response:', JSON.stringify(uploadData));
      
      const uploadedFilename = uploadData.filename || uploadData.original_filename || "dataset.csv";
      let newRunId = uploadData.run_id || uploadData.runId || uploadData.id;
      
      if (!newRunId || newRunId === 'undefined') {
        console.warn('[EDA Agent] Upload response missing run_id, generating client fallback run_id');
        newRunId = Array.from({length: 32}, () => Math.floor(Math.random() * 16).toString(16)).join('');
      }
      
      setFilename(uploadedFilename);
      setRunId(newRunId);
      setStatus('running');
      
      const eventSource = new EventSource(`${API_URL}/api/run?run_id=${encodeURIComponent(newRunId)}&filename=${encodeURIComponent(uploadedFilename)}`);
      
      eventSource.onmessage = (event) => {
        if (!event.data) return;
        const data = JSON.parse(event.data);
        
        if (data.node) {
          const nodeUpper = data.node;
          if (nodeUpper === 'SYSTEM') {
            if (data.status.includes('Starting')) return;
            if (data.status.includes('Error') || data.status.includes('Failed')) {
              setErrorMessage(data.details || data.status);
              setStatus('error');
              return;
            }
          }
          
          if (data.details) {
            setActiveNodeDetails(data.details);
          }

          setNodeStatuses(prev => {
            const newStatuses = { ...prev };
            const isFailed = data.state === 'failed' || data.status.toLowerCase().includes('error') || data.status.toLowerCase().includes('failed') || (data.details && data.details.startsWith('ERROR:'));
            const isSuccess = data.state === 'success' || data.status.toLowerCase().includes('finished') || data.status.toLowerCase().includes('complete') || data.status.toLowerCase().includes('done') || data.status.toLowerCase().includes('success');

            if (isFailed) {
              newStatuses[nodeUpper] = 'failed';
            } else if (isSuccess) {
              newStatuses[nodeUpper] = 'success';
              const currentIndex = AGENTS.findIndex(a => a.id === nodeUpper);
              if (currentIndex >= 0 && currentIndex < AGENTS.length - 1) {
                const nextAgent = AGENTS[currentIndex + 1];
                if (newStatuses[nextAgent.id] !== 'success' && newStatuses[nextAgent.id] !== 'failed') {
                  newStatuses[nextAgent.id] = 'running';
                }
              }
            } else {
              newStatuses[nodeUpper] = 'running';
            }
            return newStatuses;
          });
        }
        
        if (data.charts && data.charts.length > 0) {
          setChartPaths(prev => [...new Set([...prev, ...data.charts])]);
        }
        
        if (data.node === 'SYSTEM' && data.status.includes('Complete')) {
          eventSource.close();
          fetchReport(newRunId);
        }
      };
      
      eventSource.onerror = (err) => {
        console.warn("SSE stream finished/closed:", err);
        eventSource.close();
        fetchReport(newRunId);
      };
      
    } catch (err) {
      console.error("Pipeline start error:", err);
      setErrorMessage(err.message || "Failed to start pipeline.");
      setStatus('error');
      setNodeStatuses(prev => ({ ...prev, INGESTION: 'failed' }));
    }
  };

  const fetchReport = async (targetRunId) => {
    const idToUse = targetRunId || runId;
    if (!idToUse) return;
    try {
      const res = await fetch(`${API_URL}/api/report/${idToUse}`);
      if (res.ok) {
        const data = await res.json();
        const cleanedContent = (data.content || '').replace(/<iframe.*?><\/iframe>/g, '');
        setReportContent(cleanedContent);
        setStatus('complete');
      } else {
        console.warn("Report fetch returned status:", res.status);
      }
    } catch (err) {
       console.error("Report fetch error:", err);
       setErrorMessage(err.message || "Failed to fetch report from backend server.");
       setStatus('error');
    }
  };

  const resetState = () => {
    setFile(null);
    setFilename('');
    setRunId('');
    setStatus('idle');
    setReportContent('');
    setChartPaths([]);
    setNodeStatuses({});
    setActiveNodeDetails('');
    setErrorMessage('');
  };

  // Helper to determine stage state
  const getStageStatus = (stage) => {
    const statuses = stage.agentIds.map(id => nodeStatuses[id] || 'pending');
    if (statuses.some(s => s === 'failed')) return 'failed';
    if (statuses.some(s => s === 'running')) return 'running';
    if (statuses.every(s => s === 'success')) return 'success';
    return 'pending';
  };

  const completedAgents = Object.keys(nodeStatuses).filter(k => k !== 'SYSTEM' && nodeStatuses[k] === 'success' && AGENTS.some(a => a.id === k));
  const progress = Math.min(1, completedAgents.length / AGENTS.length);

  const filteredAgents = activeCategory === 'all' 
    ? AGENTS 
    : AGENTS.filter(a => a.category === activeCategory);

  const activeAgent = AGENTS.find(a => nodeStatuses[a.id] === 'running');

  return (
    <>
      <div className="bg-grid" />
      <main className="relative z-10 flex flex-col min-h-screen px-4 sm:px-6 py-8 max-w-7xl mx-auto w-full">
        
        {/* HERO SECTION */}
        <section className="py-6 flex flex-col items-center text-center animate-in fade-in duration-700">
          <h1 className="font-display text-4xl md:text-5xl font-bold tracking-tight mb-3 mt-4">
            Automated <span className="text-gradient">EDA Agent</span>
          </h1>
          
          <p className="text-muted-foreground text-sm md:text-base max-w-2xl mb-6 font-sans">
            Deterministic evidence-backed multi-agent architecture powered by LangGraph, Specialist Interpreters, and Claim Verifiers.
          </p>
          
          <div className="flex flex-wrap justify-center gap-3">
            <div className="flex items-center gap-2 bg-card px-3 py-1.5 rounded-lg border border-border">
              <Database size={15} className="text-mint" />
              <span className="text-xs font-medium">13 Specialist Agents</span>
            </div>
            <div className="flex items-center gap-2 bg-card px-3 py-1.5 rounded-lg border border-border">
              <CheckCircle2 size={15} className="text-mint" />
              <span className="text-xs font-medium">Claim Verification Gate</span>
            </div>
            <div className="flex items-center gap-2 bg-card px-3 py-1.5 rounded-lg border border-border">
              <Activity size={15} className="text-mint" />
              <span className="text-xs font-medium">Deterministic Evidence Engine</span>
            </div>
          </div>
        </section>

        {/* ERROR DISPLAY SECTION */}
        {status === 'error' && (
          <section className="mb-10 animate-in slide-in-from-bottom-4 duration-500">
            <div className="bg-card border border-red-500/50 rounded-xl p-8 flex flex-col items-center text-center max-w-2xl mx-auto shadow-lg">
              <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center mb-4 text-red-500">
                <AlertCircle size={28} />
              </div>
              <h3 className="text-xl font-display font-bold text-red-400 mb-2">Pipeline Execution Alert</h3>
              <p className="text-sm text-muted-foreground mb-6 max-w-md">
                {errorMessage || "An error occurred during pipeline execution. Please verify server connection and try again."}
              </p>
              <button 
                onClick={resetState}
                className="bg-primary hover:bg-mint-bright text-primary-foreground font-semibold py-2.5 px-6 rounded-lg transition-all flex items-center gap-2 text-sm shadow-glow"
              >
                <RefreshCw size={16} />
                Reset & Select Dataset
              </button>
            </div>
          </section>
        )}

        {/* UPLOAD ZONE */}
        {status === 'idle' && (
          <section className="mb-10 animate-in slide-in-from-bottom-4 duration-500">
            <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
              <UploadCloud className="text-primary" /> Select Dataset
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                className="border-2 border-dashed border-border rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer bg-card hover:border-primary/50 transition-colors group"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <UploadCloud size={44} className="text-muted-foreground group-hover:text-primary transition-colors mb-3" />
                <p className="text-base font-medium">{file ? file.name : "Click or drag dataset here"}</p>
                <p className="text-xs text-muted-foreground mt-1.5">Supports CSV & Excel files (Up to 70 MB)</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileChange}
                />
              </div>

              <div className="flex flex-col justify-center items-center p-8 bg-card border border-border rounded-xl">
                <button 
                  className="bg-primary hover:bg-mint-bright text-primary-foreground font-semibold py-3 px-8 rounded-lg shadow-glow transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
                  disabled={!file} 
                  onClick={startPipeline}
                >
                  <Activity size={18} />
                  Run Auto EDA Pipeline
                </button>
              </div>
            </div>
          </section>
        )}

        {/* CLEAN 5-STAGE PIPELINE STEPPER */}
        {(status === 'running' || status === 'complete' || status === 'uploading' || status === 'error') && (
          <section className="mb-10 animate-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                <Activity size={18} className="text-primary" /> Execution Pipeline
              </h2>
              <span className="text-xs font-mono text-primary bg-primary/10 px-3 py-1 rounded-full border border-primary/20">
                {status === 'complete' ? 100 : Math.round(progress * 100)}% Complete
              </span>
            </div>
            
            <div className="relative h-2 bg-card rounded-full overflow-hidden mb-6 border border-border">
              <div 
                className="absolute top-0 left-0 h-full bg-primary transition-all duration-500 ease-out"
                style={{ width: `${status === 'complete' ? 100 : progress * 100}%` }}
              />
            </div>

            {/* 5 MASTER STAGES */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative">
              {PIPELINE_STAGES.map((stage, idx) => {
                const stageState = getStageStatus(stage);
                const isCompleted = stageState === 'success';
                const isActive = stageState === 'running' || (idx === 0 && status === 'uploading');
                const isFailed = stageState === 'failed';
                
                const StageIcon = Icons[stage.icon] || Icons.Activity;

                return (
                  <div 
                    key={stage.id} 
                    className={`relative flex flex-col p-4 rounded-xl border transition-all duration-300 ${
                      isActive 
                        ? 'bg-card border-primary shadow-glow scale-[1.01]' 
                        : isCompleted
                        ? 'bg-card border-accent/40'
                        : isFailed
                        ? 'bg-card border-red-500'
                        : 'bg-card/50 border-border opacity-70'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`p-1.5 rounded-lg ${isActive ? 'bg-primary/20 text-primary' : isCompleted ? 'bg-mint/20 text-mint' : 'bg-muted text-muted-foreground'}`}>
                          <StageIcon size={16} />
                        </div>
                        <span className="text-xs font-mono text-muted-foreground">0{idx + 1}</span>
                      </div>
                      
                      {isCompleted && <CheckCircle2 size={16} className="text-mint animate-in zoom-in" />}
                      {isActive && (
                        <span className="flex h-2 w-2 relative">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                        </span>
                      )}
                    </div>
                    
                    <h3 className="text-xs font-bold font-display mb-1">{stage.name}</h3>
                    <p className="text-[10px] text-muted-foreground line-clamp-2 leading-relaxed">{stage.description}</p>
                  </div>
                );
              })}
            </div>

            {/* LIVE ACTIVITY DRAWER */}
            {activeNodeDetails && (
              <div className="mt-4 bg-card/80 border border-primary/20 rounded-lg p-3 text-xs flex items-center gap-3 animate-in fade-in">
                <div className="flex items-center gap-2 text-primary font-medium shrink-0">
                  <Activity size={14} className="animate-spin text-primary" />
                  <span>{activeAgent ? activeAgent.name : 'Processing'}:</span>
                </div>
                <span className="text-muted-foreground font-mono truncate">{activeNodeDetails}</span>
              </div>
            )}
          </section>
        )}

        {/* AGENT FLEET BENTO GRID */}
        {(status === 'running' || status === 'complete' || status === 'uploading' || status === 'error') && (
          <section className="mb-10 animate-in slide-in-from-bottom-6 duration-700">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
              <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                <Brain size={18} className="text-primary" /> Specialist Agents ({AGENTS.length})
              </h2>
              
              {/* CATEGORY TABS */}
              <div className="flex items-center bg-card border border-border rounded-lg p-1 text-xs gap-1">
                {[
                  { id: 'all', label: 'All (13)' },
                  { id: 'specialists', label: 'Specialists (3)' },
                  { id: 'processing', label: 'Processing (5)' },
                  { id: 'verification', label: 'Verification (3)' },
                  { id: 'core', label: 'Core (2)' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveCategory(tab.id)}
                    className={`px-2.5 py-1 rounded-md transition-all font-medium ${
                      activeCategory === tab.id 
                        ? 'bg-primary text-primary-foreground font-semibold' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* BENTO GRID */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {filteredAgents.map((agent) => {
                const nodeState = nodeStatuses[agent.id] || 'pending';
                const isCompleted = nodeState === 'success';
                const isActive = nodeState === 'running';
                const isFailed = nodeState === 'failed';
                
                const AgentIcon = Icons[agent.icon] || Icons.Code;

                return (
                  <div 
                    key={agent.id}
                    className={`bg-card border rounded-xl p-3.5 flex flex-col justify-between transition-all duration-300 ${
                      isActive 
                        ? 'border-primary shadow-glow scale-[1.02] bg-card' 
                        : isCompleted
                        ? 'border-accent/40 bg-card'
                        : isFailed
                        ? 'border-red-500 bg-red-500/5'
                        : 'border-border opacity-75'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`p-2 rounded-lg ${isActive ? 'bg-primary/20 text-primary' : isCompleted ? 'bg-mint/20 text-mint' : 'bg-muted text-muted-foreground'}`}>
                          <AgentIcon size={16} />
                        </div>
                        <div>
                          <h3 className="text-xs font-bold">{agent.name}</h3>
                          <p className="text-[10px] text-muted-foreground">{agent.role}</p>
                        </div>
                      </div>
                      
                      {isCompleted && (
                        <span className="text-[10px] font-medium text-mint bg-mint/10 border border-mint/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                          <CheckCircle2 size={10} /> Done
                        </span>
                      )}
                      {isActive && (
                        <span className="text-[10px] font-medium text-primary bg-primary/10 border border-primary/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                          <Activity size={10} className="animate-spin" /> Active
                        </span>
                      )}
                      {isFailed && (
                        <span className="text-[10px] font-medium text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-full">
                          Failed
                        </span>
                      )}
                      {nodeState === 'pending' && (
                        <span className="text-[10px] text-muted-foreground bg-muted/40 px-2 py-0.5 rounded-full">
                          Waiting
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* FINAL EXECUTIVE REPORT SECTION */}
        {reportContent && (
          <section className="mb-10 animate-in slide-in-from-bottom-8 duration-700">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
              <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                <FileText className="text-primary" /> Verified Executive Report
              </h2>
              
              <div className="flex items-center gap-2">
                <button 
                  onClick={resetState} 
                  className="bg-card hover:bg-muted text-foreground border border-border font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2 text-xs"
                >
                  <RefreshCw size={14} />
                  Analyze Another Dataset
                </button>
                
                <a 
                  href={`${API_URL}/api/download/${runId}`} 
                  download 
                  className="bg-primary hover:bg-mint-bright text-primary-foreground font-semibold py-2 px-4 rounded-lg transition-all flex items-center gap-2 text-xs shadow-glow"
                >
                  <Download size={14} />
                  Download Report (.md)
                </a>
              </div>
            </div>
            
            <div className="bg-card border border-border rounded-xl p-6 sm:p-8 shadow-xl">
              <div className="prose prose-invert prose-mint max-w-none prose-headings:font-display prose-headings:font-bold prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg prose-p:text-sm prose-p:leading-relaxed prose-li:text-sm prose-td:text-xs prose-th:text-xs prose-table:border prose-table:border-border prose-th:bg-muted/50 prose-td:border-b prose-td:border-border/50">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {reportContent}
                </ReactMarkdown>
              </div>
            </div>
          </section>
        )}

        {/* EMBEDDED PLOTLY VISUALIZATIONS SECTION */}
        {chartPaths.length > 0 && (
          <section className="mb-10 animate-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="text-primary" /> Generated Visualizations ({chartPaths.length})
            </h2>
            <div className="flex flex-col gap-6">
              {chartPaths.map((path, i) => {
                const fname = path.split('/').pop().replace('\\', '');
                let chartUrl;
                if (path.startsWith('http')) {
                  chartUrl = path;
                } else if (path.startsWith('/api/')) {
                  chartUrl = `${API_URL}${path}`;
                } else {
                  chartUrl = runId ? `${API_URL}/api/plots/${runId}/plots/${fname}` : `${API_URL}/api/sandbox/plots/${fname}`;
                }
                return (
                  <div key={i} className="bg-card border border-border rounded-xl p-4 flex flex-col h-[550px] w-full shadow-lg">
                    <h3 className="text-xs font-medium mb-2 text-muted-foreground truncate">{fname}</h3>
                    <iframe 
                      src={chartUrl}
                      className="w-full h-full border-none rounded-lg bg-white"
                      title={`Chart ${i}`}
                    />
                  </div>
                );
              })}
            </div>
          </section>
        )}

      </main>
    </>
  );
}

export default App;
