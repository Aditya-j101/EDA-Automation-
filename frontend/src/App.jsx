import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as Icons from 'lucide-react';
import { UploadCloud, Zap, Database, Activity, Code, FileSpreadsheet, RefreshCw, Download } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const AGENTS = [
  { id: 'INGESTION', name: 'Ingestion', role: 'Load & Validate', kpiLabel: 'Status', icon: 'Database' },
  { id: 'PROFILER', name: 'Profiler', role: 'Schema Analysis', kpiLabel: 'Status', icon: 'FileSearch' },
  { id: 'CLEANER', name: 'Cleaner', role: 'Impute & Cap', kpiLabel: 'Status', icon: 'Wand2' },
  { id: 'FEATURE_ENGINEER', name: 'Feature Engineer', role: 'Transform', kpiLabel: 'Status', icon: 'Cpu' },
  { id: 'ANALYST', name: 'Analyst', role: 'Stat Tests', kpiLabel: 'Status', icon: 'Activity' },
  { id: 'ADVANCED_ANALYST', name: 'Adv. Analyst', role: 'ML Prep', kpiLabel: 'Status', icon: 'Network' },
  { id: 'TIMESERIES_ANALYST', name: 'Time-Series', role: 'Drift Check', kpiLabel: 'Status', icon: 'TrendingUp' },
  { id: 'VISUALIZER', name: 'Visualizer', role: 'Plotly Charts', kpiLabel: 'Status', icon: 'BarChart3' },
  { id: 'REPORTER', name: 'Reporter', role: 'Markdown Gen', kpiLabel: 'Status', icon: 'FileText' }
];

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState('');
  const [status, setStatus] = useState('idle'); // idle, uploading, running, complete, error
  const [reportContent, setReportContent] = useState('');
  const [chartPaths, setChartPaths] = useState([]);
  const [nodeStatuses, setNodeStatuses] = useState({});
  
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
    
    setStatus('uploading');
    setNodeStatuses({ INGESTION: 'running' });
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const uploadRes = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadRes.ok) throw new Error("Upload failed");
      
      const uploadData = await uploadRes.json();
      const uploadedFilename = uploadData.filename;
      setFilename(uploadedFilename);
      
      setStatus('running');
      
      const eventSource = new EventSource(`${API_URL}/api/run?filename=${encodeURIComponent(uploadedFilename)}`);
      
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.node) {
          const nodeUpper = data.node;
          if (nodeUpper === 'SYSTEM' && data.status.includes('Starting')) return; // Ignore initial SYSTEM message
          
          setNodeStatuses(prev => {
            const newStatuses = { ...prev };
            if (data.status.toLowerCase().includes('error')) {
              newStatuses[nodeUpper] = 'failed';
            } else if (data.status.toLowerCase().includes('finished') || data.status.toLowerCase().includes('complete')) {
              newStatuses[nodeUpper] = 'success';
              
              // Automatically move to next agent
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
          fetchReport();
        }
      };
      
      eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        setNodeStatuses(prev => ({ ...prev, INGESTION: 'failed' }));
        eventSource.close();
        fetchReport();
      };
      
    } catch (err) {
      console.error(err);
      setStatus('error');
      setNodeStatuses(prev => ({ ...prev, INGESTION: 'failed' }));
    }
  };

  const fetchReport = async () => {
    try {
      const res = await fetch(`${API_URL}/api/report`);
      if (res.ok) {
        const data = await res.json();
        const cleanedContent = data.content.replace(/<iframe.*?><\/iframe>/g, '');
        setReportContent(cleanedContent);
        setStatus('complete');
      }
    } catch (err) {
       console.error("Report fetch error:", err);
       setStatus('error');
    }
  };

  const resetState = () => {
    setFile(null);
    setFilename('');
    setStatus('idle');
    setReportContent('');
    setChartPaths([]);
    setNodeStatuses({});
  };

  const activeAgent = Object.keys(nodeStatuses).find(k => k !== 'SYSTEM' && nodeStatuses[k] === 'running') || null;
  const completedAgents = Object.keys(nodeStatuses).filter(k => k !== 'SYSTEM' && nodeStatuses[k] === 'success' && AGENTS.some(a => a.id === k));
  const progress = Math.min(1, completedAgents.length / AGENTS.length);

  return (
    <>
      <div className="bg-grid" />
      <main className="relative z-10 flex flex-col min-h-screen px-4 sm:px-6 py-8 max-w-7xl mx-auto w-full">
        
        {/* HERO SECTION */}
        <section className="py-8 flex flex-col items-center text-center animate-in fade-in duration-700">
          <h1 className="font-display text-4xl md:text-6xl font-bold tracking-tight mb-4 mt-6">
            Automated <span className="text-gradient">EDA Agent</span>
          </h1>
          
          <p className="text-muted-foreground text-base md:text-lg max-w-2xl mb-8 font-sans">
            An intelligent, multi-agent system powered by LangGraph that automates end-to-end Exploratory Data Analysis.
          </p>
          
          <div className="flex flex-wrap justify-center gap-4">
            <div className="flex items-center gap-2 bg-card px-4 py-2 rounded-lg border border-border">
              <Database size={16} className="text-mint" />
              <span className="text-xs font-medium">8 Autonomous Agents</span>
            </div>
            <div className="flex items-center gap-2 bg-card px-4 py-2 rounded-lg border border-border">
              <Activity size={16} className="text-mint" />
              <span className="text-xs font-medium">Interactive Plotly Charts</span>
            </div>
            <div className="flex items-center gap-2 bg-card px-4 py-2 rounded-lg border border-border">
              <Code size={16} className="text-mint" />
              <span className="text-xs font-medium">Safe Python Execution</span>
            </div>
          </div>
        </section>

        {/* UPLOAD ZONE */}
        {status === 'idle' && (
          <section className="mb-12 animate-in slide-in-from-bottom-4 duration-500">
            <h2 className="text-xl font-display font-semibold mb-6 flex items-center gap-2">
              <UploadCloud className="text-primary" /> Select Dataset
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                className="border-2 border-dashed border-border rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer bg-card hover:border-primary/50 transition-colors group"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <UploadCloud size={48} className="text-muted-foreground group-hover:text-primary transition-colors mb-4" />
                <p className="text-lg font-medium">{file ? file.name : "Click or drag dataset here"}</p>
                <p className="text-sm text-muted-foreground mt-2">Supports CSV & Excel files</p>
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
                  className="bg-primary hover:bg-mint-bright text-primary-foreground font-semibold py-3 px-8 rounded-lg shadow-glow transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  disabled={!file} 
                  onClick={startPipeline}
                >
                  <Activity size={20} />
                  Run Auto EDA Pipeline
                </button>
              </div>
            </div>
          </section>
        )}

        {/* PIPELINE FLOW */}
        {(status === 'running' || status === 'complete' || status === 'uploading') && (
          <section className="mb-12 animate-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-display font-semibold">Execution Pipeline</h2>
              <span className="text-sm font-mono text-primary">{status === 'complete' ? 100 : Math.round(progress * 100)}% Complete</span>
            </div>
            
            <div className="relative h-2 bg-card rounded-full overflow-hidden mb-8 border border-border">
              <div 
                className="absolute top-0 left-0 h-full bg-primary transition-all duration-500 ease-out"
                style={{ width: `${status === 'complete' ? 100 : progress * 100}%` }}
              />
            </div>

            <div className="flex flex-wrap justify-center md:justify-between items-center gap-y-8 gap-x-2 relative">
              <div className="hidden md:block absolute top-6 left-6 right-6 h-0.5 bg-border -z-10" />
              
              {AGENTS.map((agent) => {
                const nodeStatus = nodeStatuses[agent.id] || 'pending';
                const isCompleted = nodeStatus === 'success';
                const isActive = nodeStatus === 'running' || nodeStatus === 'uploading';
                
                const IconComponent = Icons[agent.icon] || Icons.Activity;

                return (
                  <div key={agent.id} className="flex flex-col items-center gap-2 z-10 w-24">
                    <div 
                      className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ${
                        isCompleted 
                          ? 'bg-primary/20 text-primary border-2 border-primary shadow-glow' 
                          : isActive
                          ? 'bg-primary/10 text-primary border-2 border-primary animate-pulse shadow-glow'
                          : 'bg-card text-muted-foreground border-2 border-border'
                      }`}
                    >
                      {isCompleted ? <Icons.Check size={20} strokeWidth={3} /> : <IconComponent size={20} />}
                    </div>
                    <span className={`text-[10px] font-semibold text-center uppercase tracking-wider ${isActive || isCompleted ? 'text-foreground' : 'text-muted-foreground'}`}>
                      {agent.name}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* BENTO GRID (Terminal Removed) */}
        {(status === 'running' || status === 'complete') && (
          <div className="mb-12 animate-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-xl font-display font-semibold mb-6">Agent Fleet</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {AGENTS.map((agent, i) => {
                const nodeStatus = nodeStatuses[agent.id] || 'pending';
                const isCompleted = nodeStatus === 'success';
                const isActive = nodeStatus === 'running' || nodeStatus === 'uploading';
                const isFailed = nodeStatus === 'failed';
                
                const IconComponent = Icons[agent.icon] || Icons.Bot;

                return (
                  <div 
                    key={agent.id} 
                    className={`relative overflow-hidden bg-card border rounded-2xl p-5 transition-all duration-300 col-span-1 ${
                      isActive ? 'border-primary shadow-glow scale-[1.02] z-10' : 
                      isFailed ? 'border-red-500' : 'border-border'
                    }`}
                  >
                    {(isActive || isCompleted) && (
                      <div className={`absolute -bottom-8 -right-8 w-32 h-32 rounded-full blur-3xl opacity-20 ${isActive ? 'bg-primary' : 'bg-accent'}`} />
                    )}
                    
                    <div className="flex justify-between items-start mb-4">
                      <div className={`p-2 rounded-lg ${isActive ? 'bg-primary/20 text-primary' : isFailed ? 'bg-red-500/20 text-red-500' : 'bg-background text-muted-foreground'}`}>
                        <IconComponent size={24} />
                      </div>
                      
                      <div className={`text-[10px] uppercase tracking-wider font-bold px-2 py-1 rounded-full border ${
                        isActive ? 'border-primary/50 text-primary bg-primary/10 animate-pulse' :
                        isCompleted ? 'border-accent text-accent bg-accent/10' :
                        isFailed ? 'border-red-500/50 text-red-500 bg-red-500/10' :
                        'border-border text-muted-foreground'
                      }`}>
                        {isActive ? 'Running' : isCompleted ? 'Success' : isFailed ? 'Failed' : 'Standby'}
                      </div>
                    </div>
                    
                    <h3 className="font-display font-semibold text-lg text-foreground mb-1">{agent.name}</h3>
                    <p className="text-xs text-muted-foreground mb-4">{agent.role}</p>
                    
                    <div className="flex items-end justify-between mt-auto">
                      <div className="flex flex-col">
                        <span className="text-[10px] text-muted-foreground uppercase">{agent.kpiLabel}</span>
                        <span className={`font-mono text-lg ${isCompleted || isActive ? 'text-foreground' : 'text-muted-foreground/30'}`}>
                          {isCompleted ? '100%' : isActive ? '...' : '-'}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* REPORT */}
        {status === 'complete' && (
          <section className="mb-12 animate-in slide-in-from-bottom-8 duration-700 relative">
            <div className="absolute -top-14 right-0 flex gap-2">
              <button 
                onClick={() => window.open(`${API_URL}/api/download`, '_blank')}
                className="flex items-center gap-2 bg-card hover:bg-card/80 text-foreground border border-border px-4 py-2 rounded-lg transition-colors font-medium text-sm"
              >
                <Download size={16} /> Download
              </button>
              <button 
                onClick={resetState}
                className="flex items-center gap-2 bg-primary/20 hover:bg-primary/30 text-primary border border-primary px-4 py-2 rounded-lg transition-colors font-medium text-sm shadow-glow"
              >
                <RefreshCw size={16} /> Run Another Analysis
              </button>
            </div>

            <h2 className="text-xl font-display font-semibold mb-6 flex items-center gap-2">
              <FileSpreadsheet className="text-primary" /> Final Report
            </h2>
            
            <div className="bg-card border border-border rounded-xl p-8 prose prose-invert prose-primary prose-headings:font-bold prose-headings:text-foreground max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent}</ReactMarkdown>
            </div>
          </section>
        )}

        {/* CHARTS */}
        {chartPaths.length > 0 && (
          <section className="mb-12 animate-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-xl font-display font-semibold mb-6 flex items-center gap-2">
              <Activity className="text-primary" /> Generated Visualizations
            </h2>
            <div className="flex flex-col gap-8">
              {chartPaths.map((path, i) => {
                const fname = path.split('/').pop().replace('\\', '');
                return (
                  <div key={i} className="bg-card border border-border rounded-xl p-4 flex flex-col h-[600px] w-full">
                    <h3 className="text-sm font-medium mb-2 text-muted-foreground truncate">{fname}</h3>
                    <iframe 
                      src={`${API_URL}/api/plots/${fname}`}
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
