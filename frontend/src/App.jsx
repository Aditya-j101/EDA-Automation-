import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { UploadCloud, Activity, CheckCircle, AlertCircle, FileText, BarChart2, Server, Download, RefreshCw, Eye, Settings, Database, Cpu, PieChart, Check } from 'lucide-react';
import './index.css';

const PIPELINE_NODES = [
  'SYSTEM',
  'PROFILER',
  'CLEANER',
  'FEATURE_ENGINEER',
  'ANALYST',
  'TIMESERIES_ANALYST',
  'VISUALIZER',
  'REPORTER'
];

const getNodeDetails = (node) => {
  switch (node) {
    case 'SYSTEM': return { label: 'Upload', icon: <UploadCloud size={24} /> };
    case 'PROFILER': return { label: 'Profiler', icon: <Eye size={24} /> };
    case 'CLEANER': return { label: 'Cleaner', icon: <Settings size={24} /> };
    case 'FEATURE_ENGINEER': return { label: 'Features', icon: <Database size={24} /> };
    case 'ANALYST': return { label: 'Analyst', icon: <Cpu size={24} /> };
    case 'TIMESERIES_ANALYST': return { label: 'Time Series', icon: <Activity size={24} /> };
    case 'VISUALIZER': return { label: 'Visuals', icon: <PieChart size={24} /> };
    case 'REPORTER': return { label: 'Reporter', icon: <FileText size={24} /> };
    default: return { label: node, icon: <Activity size={24} /> };
  }
};

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState('');
  const [status, setStatus] = useState('idle'); // idle, uploading, running, complete, error
  const [logs, setLogs] = useState([]);
  const [reportContent, setReportContent] = useState('');
  const [chartPaths, setChartPaths] = useState([]);
  const [nodeStatuses, setNodeStatuses] = useState({});
  
  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const startPipeline = async () => {
    if (!file) return;
    
    setStatus('uploading');
    setNodeStatuses({ SYSTEM: 'uploading' });
    
    // 1. Upload File
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
      setNodeStatuses({ SYSTEM: 'success' });
      
      // 2. Start SSE Stream
      const eventSource = new EventSource(`${API_URL}/api/run?filename=${encodeURIComponent(uploadedFilename)}`);
      
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.node) {
          setNodeStatuses(prev => {
            const newStatuses = { ...prev };
            // If it's an error
            if (data.status.toLowerCase().includes('error')) {
              newStatuses[data.node] = 'failed';
            } else if (data.status.toLowerCase().includes('finished') || data.status.toLowerCase().includes('complete')) {
              newStatuses[data.node] = 'success';
            } else {
              newStatuses[data.node] = 'running';
            }
            return newStatuses;
          });
        }
        
        if (data.charts && data.charts.length > 0) {
          // Accumulate chart paths properly
          setChartPaths(prev => [...new Set([...prev, ...data.charts])]);
        }
        
        if (data.node === 'SYSTEM' && data.status.includes('Complete')) {
          eventSource.close();
          fetchReport();
        }
      };
      
      eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        setNodeStatuses(prev => ({ ...prev, SYSTEM: 'failed' }));
        eventSource.close();
        // Even if connection drops, try to fetch report in case it actually finished
        fetchReport();
      };
      
    } catch (err) {
      console.error(err);
      setStatus('error');
      setNodeStatuses(prev => ({ ...prev, SYSTEM: 'failed' }));
    }
  };

  const fetchReport = async () => {
    try {
      setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), node: 'SYSTEM', msg: 'Fetching final report...', type: 'info' }]);
      const res = await fetch(`${API_URL}/api/report`);
      if (res.ok) {
        const data = await res.json();
        
        // The reporter adds iframe tags to the markdown for charts. 
        // We will strip those out of the raw markdown because react-markdown doesn't render raw iframes safely by default.
        // Instead, we will render the markdown text, and display the charts separately below it using the chartPaths we collected from SSE.
        const cleanedContent = data.content.replace(/<iframe.*?><\/iframe>/g, '');
        
        setReportContent(cleanedContent);
        setStatus('complete');
      } else {
        throw new Error("Report not found");
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
    setLogs([]);
    setReportContent('');
    setChartPaths([]);
    setNodeStatuses({});
  };

  const handleDownload = () => {
    window.open(`${API_URL}/api/download`, '_blank');
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>EDA AGENT</h1>
      </header>

      {status === 'idle' && (
        <section className="glass-panel">
          <div 
            className="upload-zone"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <UploadCloud className="upload-icon" size={48} />
            <div className="upload-text">
              {file ? file.name : "Click or drag dataset here"}
            </div>
            <div className="upload-subtext">
              Supports CSV & Excel files
            </div>
            <input 
              type="file" 
              ref={fileInputRef} 
              className="file-input" 
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
            />
          </div>
          
          <div style={{ padding: '0 2rem 2rem', textAlign: 'center' }}>
            <button 
              className="btn" 
              disabled={!file} 
              onClick={startPipeline}
            >
              <Server size={20} />
              Run Auto EDA Pipeline
            </button>
          </div>
        </section>
      )}

      {(status === 'running' || status === 'uploading' || status === 'error' || status === 'complete') && (
        <section className="glass-panel pipeline-monitor">
          <div className="monitor-header">
            <h2><Activity size={24} /> Pipeline Monitor</h2>
            <div className={`status-badge ${status === 'complete' ? 'complete' : ''}`}>
              {status === 'running' && <span className="pulse">●</span>}
              {status === 'complete' && <CheckCircle size={16} />}
              {status === 'error' && <AlertCircle size={16} />}
              {status.toUpperCase()}
            </div>
          </div>
          
          <div className="pipeline-graph">
            {PIPELINE_NODES.map((node, index) => {
              const nodeStatus = nodeStatuses[node] || 'pending'; 
              const details = getNodeDetails(node);
              const isLast = index === PIPELINE_NODES.length - 1;
              const connectorActive = nodeStatus === 'success';

              return (
                <div key={node} className="pipeline-node-container">
                  <div className={`pipeline-node status-${nodeStatus}`}>
                    <div className="node-icon">
                      {nodeStatus === 'success' ? <Check size={28} /> : details.icon}
                    </div>
                    <div className="node-label">{details.label}</div>
                  </div>
                  {!isLast && (
                     <div className={`pipeline-connector ${connectorActive ? 'active' : ''}`} />
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {status === 'complete' && (
        <section className="glass-panel report-viewer">
          <div className="monitor-header">
            <h2><FileText size={24} /> Final Report</h2>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button className="btn" style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }} onClick={handleDownload}>
                <Download size={16} /> Download All Files
              </button>
              <button className="btn" style={{ padding: '0.5rem 1rem', fontSize: '0.9rem', backgroundColor: 'var(--bg-lighter)', color: 'var(--text-main)', border: '1px solid var(--border-color)' }} onClick={resetState}>
                <RefreshCw size={16} /> Run Another Analysis
              </button>
            </div>
          </div>
          <div className="report-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent}</ReactMarkdown>
          </div>
          
          {chartPaths.length > 0 && (
            <>
              <div className="monitor-header" style={{ marginTop: '3rem' }}>
                <h2><BarChart2 size={24} /> Interactive Visualizations</h2>
              </div>
              <div className="chart-grid">
                {/* 
                  chartPaths usually look like "sandbox/plots/chart_name.html".
                  Our FastAPI backend mounts this at "/api/plots/".
                  We need to extract the filename and point the iframe to the API.
                */}
                {chartPaths.map((path, i) => {
                  const filename = path.split('/').pop().replace('\\', '');
                  return (
                    <div key={i} className="chart-card">
                      <iframe 
                        src={`${API_URL}/api/plots/${filename}`}
                        className="chart-frame"
                        title={`Chart ${i}`}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
