import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as Icons from 'lucide-react';
import { UploadCloud, Database, Activity, Code, FileSpreadsheet, RefreshCw, Download, ShieldCheck, CheckCircle2, Sliders, Brain, FileText, BarChart3, ChevronRight } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
        throw new Error(errData.detail || "Upload failed");
      }
      
      const uploadData = await uploadRes.json();
      const uploadedFilename = uploadData.filename;
      const newRunId = uploadData.run_id;
      
      setFilename(uploadedFilename);
      setRunId(newRunId);
      setStatus('running');
      
      const eventSource = new EventSource(`${API_URL}/api/run?run_id=${encodeURIComponent(newRunId)}&filename=${encodeURIComponent(uploadedFilename)}`);
      
      eventSource.onmessage = (event) => {
        if (!event.data) return;
        const data = JSON.parse(event.data);
        
        if (data.node) {
          const nodeUpper = data.node;
          if (nodeUpper === 'SYSTEM' && data.status.includes('Starting')) return;
          
          if (data.details) {
            setActiveNodeDetails(data.details);
          }

          setNodeStatuses(prev => {
            const newStatuses = { ...prev };
            if (data.status.toLowerCase().includes('error') || data.status.toLowerCase().includes('failed')) {
              newStatuses[nodeUpper] = 'failed';
            } else if (data.status.toLowerCase().includes('finished') || data.status.toLowerCase().includes('complete')) {
              newStatuses[nodeUpper] = 'success';
              
              const currentIndex = AGENTS.findIndex(a => a.id === nodeUpper);
              if (currentIndex >= 0 && currentIndex < AGENTS.length - 1) {
                const nextAgent = AGENTS[currentIndex + 1];
                newStatuses[nextAgent.id] = 'running';
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
        {(status === 'running' || status === 'complete' || status === 'uploading') && (
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
                      <div className={`p-2 rounded-lg ${
                        isActive ? 'bg-primary/20 text-primary animate-pulse' :
                        isCompleted ? 'bg-accent/20 text-accent' :
                        isFailed ? 'bg-red-500/20 text-red-500' :
                        'bg-background text-muted-foreground'
                      }`}>
                        <StageIcon size={18} />
                      </div>
                      <span className={`text-[9px] uppercase font-mono tracking-wider font-bold px-2 py-0.5 rounded-full border ${
                        isActive ? 'border-primary/50 text-primary bg-primary/10 animate-pulse' :
                        isCompleted ? 'border-accent/50 text-accent bg-accent/10' :
                        isFailed ? 'border-red-500/50 text-red-500 bg-red-500/10' :
                        'border-border text-muted-foreground'
                      }`}>
                        {isActive ? 'Active' : isCompleted ? 'Passed' : isFailed ? 'Failed' : 'Standby'}
                      </span>
                    </div>

                    <h3 className="font-display font-semibold text-xs text-foreground mb-1">{stage.name}</h3>
                    <p className="text-[10px] text-muted-foreground leading-snug">{stage.description}</p>
                  </div>
                );
              })}
            </div>

            {/* LIVE ACTIVITY STATUS DRAWER */}
            {activeAgent && (
              <div className="mt-4 p-3 bg-primary/10 border border-primary/30 rounded-xl flex items-center justify-between animate-pulse">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-primary animate-spin" />
                  <span className="text-xs font-semibold text-foreground">Executing Node: {activeAgent.name} ({activeAgent.role})</span>
                </div>
                {activeNodeDetails && (
                  <span className="text-[11px] text-muted-foreground truncate max-w-md hidden sm:inline">{activeNodeDetails.slice(0, 100)}...</span>
                )}
              </div>
            )}
          </section>
        )}

        {/* CATEGORIZED AGENT FLEET BENTO GRID */}
        {(status === 'running' || status === 'complete') && (
          <div className="mb-10 animate-in slide-in-from-bottom-8 duration-700">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-3">
              <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                <Brain size={18} className="text-primary" /> Specialist Agents & Verifiers
              </h2>

              {/* CATEGORY TABS */}
              <div className="flex flex-wrap gap-1.5 bg-card p-1 rounded-xl border border-border">
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
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                      activeCategory === tab.id
                        ? 'bg-primary text-primary-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* FLEET CARDS GRID */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {filteredAgents.map((agent) => {
                const nodeStatus = nodeStatuses[agent.id] || 'pending';
                const isCompleted = nodeStatus === 'success';
                const isActive = nodeStatus === 'running' || nodeStatus === 'uploading';
                const isFailed = nodeStatus === 'failed';
                
                const IconComponent = Icons[agent.icon] || Icons.Bot;

                return (
                  <div 
                    key={agent.id} 
                    className={`relative overflow-hidden bg-card border rounded-xl p-4 transition-all duration-300 ${
                      isActive ? 'border-primary shadow-glow scale-[1.01] z-10' : 
                      isFailed ? 'border-red-500' : 'border-border/80'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div className={`p-2 rounded-lg ${isActive ? 'bg-primary/20 text-primary' : isFailed ? 'bg-red-500/20 text-red-500' : 'bg-background text-muted-foreground'}`}>
                        <IconComponent size={20} />
                      </div>
                      
                      <div className={`text-[9px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full border ${
                        isActive ? 'border-primary/50 text-primary bg-primary/10 animate-pulse' :
                        isCompleted ? 'border-accent text-accent bg-accent/10' :
                        isFailed ? 'border-red-500/50 text-red-500 bg-red-500/10' :
                        'border-border text-muted-foreground'
                      }`}>
                        {isActive ? 'Running' : isCompleted ? 'Verified' : isFailed ? 'Failed' : 'Standby'}
                      </div>
                    </div>
                    
                    <h3 className="font-display font-semibold text-sm text-foreground mb-0.5">{agent.name}</h3>
                    <p className="text-[11px] text-muted-foreground">{agent.role}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* REPORT */}
        {status === 'complete' && (
          <section className="mb-10 animate-in slide-in-from-bottom-8 duration-700 relative">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                <FileSpreadsheet className="text-primary" /> Verified Executive Report
              </h2>
              <div className="flex gap-2">
                <button 
                  onClick={() => window.open(`${API_URL}/api/download/${runId}`, '_blank')}
                  className="flex items-center gap-1.5 bg-card hover:bg-card/80 text-foreground border border-border px-3 py-1.5 rounded-lg transition-colors font-medium text-xs"
                >
                  <Download size={14} /> Download Zip
                </button>
                <button 
                  onClick={resetState}
                  className="flex items-center gap-1.5 bg-primary/20 hover:bg-primary/30 text-primary border border-primary px-3 py-1.5 rounded-lg transition-colors font-medium text-xs shadow-glow"
                >
                  <RefreshCw size={14} /> Run Another Analysis
                </button>
              </div>
            </div>
            
            <div className="bg-card border border-border rounded-xl p-8 prose prose-invert prose-primary prose-headings:font-bold prose-headings:text-foreground max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent}</ReactMarkdown>
            </div>
          </section>
        )}

        {/* CHARTS */}
        {chartPaths.length > 0 && (
          <section className="mb-10 animate-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="text-primary" /> Generated Visualizations ({chartPaths.length})
            </h2>
            <div className="flex flex-col gap-6">
              {chartPaths.map((path, i) => {
                const fname = path.split('/').pop().replace('\\', '');
                const chartUrl = path.startsWith('http') ? path : `${API_URL}${path.startsWith('/') ? '' : '/'}${path}`;
                return (
                  <div key={i} className="bg-card border border-border rounded-xl p-4 flex flex-col h-[550px] w-full">
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
