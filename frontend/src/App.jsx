import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  ShieldCheck, 
  Upload, 
  Layers, 
  Cpu, 
  FileText, 
  CheckCircle2, 
  AlertOctagon, 
  Sliders, 
  BarChart3, 
  Sparkles,
  RefreshCw,
  Dna,
  Microscope,
  Award,
  Zap,
  Check,
  ChevronRight,
  Printer,
  Users,
  Grid,
  Eye,
  AlertTriangle,
  Info,
  Filter,
  ArrowUpDown,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('screening'); // 'screening', 'staging', 'comparison'
  
  // File Selection State (Accepts 1 or multiple files seamlessly)
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  // Results State
  const [singleResult, setSingleResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [activeBatchCell, setActiveBatchCell] = useState(null);

  // Gallery Filter & Sorting in Multi-Cell Mode
  const [cellFilter, setCellFilter] = useState('all'); // 'all', 'blasts', 'borderline', 'normal'
  const [cellSort, setCellSort] = useState('default'); // 'default', 'malignancy_desc', 'nc_desc'

  // Explainability Studio Controls
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [saliencyViewMode, setSaliencyViewMode] = useState('3way'); // '3way', 'overlay', 'heatmap'

  const fileInputRef = useRef(null);
  const inspectorRef = useRef(null);

  // --- File Selection Handlers ---
  const handleFilesSelected = (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setError(null);
    setSingleResult(null);
    setBatchResult(null);
    setActiveBatchCell(null);

    const filesArray = Array.from(fileList);
    setSelectedFiles(filesArray);

    const urls = [];
    let loadedCount = 0;
    filesArray.forEach((file, idx) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        urls[idx] = e.target.result;
        loadedCount++;
        if (loadedCount === filesArray.length) {
          setPreviewUrls([...urls]);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  // Client-side downscaling and compression helper (max 1024px, < 1MB)
  const downscaleImageIfNeeded = (file, maxDim = 1024, quality = 0.9) => {
    return new Promise((resolve) => {
      if (!file.type || !file.type.startsWith('image/')) {
        resolve(file);
        return;
      }
      const img = new Image();
      const reader = new FileReader();
      reader.onload = (e) => {
        img.onload = () => {
          let width = img.width;
          let height = img.height;
          if (width <= maxDim && height <= maxDim && file.size < 1024 * 1024) {
            resolve(file);
            return;
          }
          if (width > height) {
            if (width > maxDim) {
              height = Math.round((height * maxDim) / width);
              width = maxDim;
            }
          } else {
            if (height > maxDim) {
              width = Math.round((width * maxDim) / height);
              height = maxDim;
            }
          }
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          canvas.toBlob((blob) => {
            if (blob) {
              const downscaledFile = new File([blob], file.name.replace(/\.[^/.]+$/, "") + ".jpg", {
                type: 'image/jpeg',
                lastModified: Date.now()
              });
              resolve(downscaledFile);
            } else {
              resolve(file);
            }
          }, 'image/jpeg', quality);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  };

  // --- Run Unified Analysis ---
  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      // Client-side downscale and sanitize files to stay under Vercel 4.5MB request limit
      const processedFiles = await Promise.all(selectedFiles.map(f => downscaleImageIfNeeded(f)));

      if (processedFiles.length === 1) {
        // Single Cell Analysis
        const formData = new FormData();
        formData.append('file', processedFiles[0]);

        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error(`Server returned error ${res.status}`);
        const data = await res.json();
        if (data.is_widefield_smear) {
          setBatchResult(data);
          setSingleResult(null);
          if (data.cells && data.cells.length > 0) {
            setActiveBatchCell(data.cells[0]);
          }
        } else {
          setSingleResult(data);
          setBatchResult(null);
          setActiveBatchCell(null);
        }
      } else {
        // Multi-Cell Slide Batch Analysis
        const formData = new FormData();
        processedFiles.forEach((file) => {
          formData.append('files', file);
        });

        const res = await fetch(`${API_BASE}/predict-batch`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error(`Server returned error ${res.status}`);
        const data = await res.json();
        setBatchResult(data);
        setSingleResult(null);
        if (data.cells && data.cells.length > 0) {
          setActiveBatchCell(data.cells[0]);
        }
      }
    } catch (err) {
      console.warn("Analysis request failed:", err);
      setError("Analysis failed. Please verify that the backend server is running and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Helper for determining current active cell details
  const currentCellDetails = singleResult || activeBatchCell;
  const currentPreviewUrl = singleResult 
    ? previewUrls[0] 
    : (activeBatchCell?.thumbnail_b64 || previewUrls[0]);

  // Filter and sort cells for the batch population gallery
  const getFilteredAndSortedCells = () => {
    if (!batchResult?.cells) return [];
    
    let list = [...batchResult.cells];

    // Filter
    if (cellFilter === 'blasts') {
      list = list.filter(c => c.probabilities.ALL_malignant >= 0.60);
    } else if (cellFilter === 'borderline') {
      list = list.filter(c => c.probabilities.ALL_malignant > 0.38 && c.probabilities.ALL_malignant < 0.60);
    } else if (cellFilter === 'normal') {
      list = list.filter(c => c.probabilities.ALL_malignant <= 0.38);
    }

    // Sort
    if (cellSort === 'malignancy_desc') {
      list.sort((a, b) => b.probabilities.ALL_malignant - a.probabilities.ALL_malignant);
    } else if (cellSort === 'nc_desc') {
      list.sort((a, b) => (b.morphology_metrics?.nc_ratio || 0) - (a.morphology_metrics?.nc_ratio || 0));
    }

    return list;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#050a12', color: '#f8fafc', fontFamily: 'Inter, system-ui, sans-serif' }}>
      
      {/* Top Clinical Precision Header */}
      <header style={{
        background: 'rgba(9, 14, 26, 0.95)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        padding: '16px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #6366f1 100%)',
            borderRadius: '12px',
            width: '44px',
            height: '44px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(56, 189, 248, 0.35)'
          }}>
            <Microscope size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '20px', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(90deg, #ffffff, #93c5fd)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', margin: 0 }}>
                LeukemiaVision Pro
              </h1>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#38bdf8', padding: '2px 8px', background: 'rgba(56, 189, 248, 0.12)', borderRadius: '6px', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
                CLINICAL AI WORKSTATION
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', margin: 0 }}>
              Department of AI & ML • Supervisor: <strong>Mr. Balika Mahesh</strong> • ISBI C-NMC 2019
            </p>
          </div>
        </div>

        {/* Global Benchmark Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '6px 14px', borderRadius: '10px', fontSize: '12px' }}>
            <span style={{ color: '#64748b', display: 'block', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Our Model Acc</span>
            <strong style={{ color: '#10b981', fontSize: '14px' }}>96.8%</strong>
          </div>
          <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '6px 14px', borderRadius: '10px', fontSize: '12px' }}>
            <span style={{ color: '#64748b', display: 'block', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Base Paper Acc</span>
            <strong style={{ color: '#94a3b8', fontSize: '14px' }}>90.7%</strong>
          </div>
          <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '6px 14px', borderRadius: '10px', fontSize: '12px' }}>
            <span style={{ color: '#10b981', display: 'block', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Accuracy Gain</span>
            <strong style={{ color: '#34d399', fontSize: '14px' }}>+6.1% Higher</strong>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
          <button 
            onClick={() => setActiveTab('screening')}
            style={{
              background: activeTab === 'screening' ? 'linear-gradient(135deg, #0284c7, #4f46e5)' : 'transparent',
              color: '#ffffff',
              border: 'none',
              padding: '8px 18px',
              borderRadius: '8px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Microscope size={15} /> Clinical Screening & Analysis
          </button>
          <button 
            onClick={() => setActiveTab('staging')}
            style={{
              background: activeTab === 'staging' ? 'linear-gradient(135deg, #0284c7, #4f46e5)' : 'transparent',
              color: '#ffffff',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            FAB Staging Matrix
          </button>
          <button 
            onClick={() => setActiveTab('comparison')}
            style={{
              background: activeTab === 'comparison' ? 'linear-gradient(135deg, #0284c7, #4f46e5)' : 'transparent',
              color: '#ffffff',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Base Paper Comparison
          </button>
        </div>
      </header>

      {/* Main Workspace Area */}
      <main style={{ flex: 1, padding: '24px 32px', maxWidth: '1680px', margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>


        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '10px', padding: '12px 18px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px', color: '#fca5a5' }}>
            <AlertTriangle size={18} />
            <span style={{ fontSize: '13px' }}>{error}</span>
          </div>
        )}

        {/* ========================================================
            TAB 1: UNIFIED CLINICAL SCREENING & WORKSTATION
           ======================================================== */}
        {activeTab === 'screening' && (
          <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '28px', alignItems: 'start' }}>
            
            {/* Left Column: Clean, Streamlined Ingestion Engine */}
            <div className="glass-panel" style={{ padding: '24px', background: 'rgba(15, 23, 42, 0.85)', borderRadius: '14px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                  <Upload size={18} color="#38bdf8" /> Smear Ingestion Station
                </h2>
                <span style={{ fontSize: '11px', color: '#38bdf8', fontWeight: 600 }}>C-NMC 2019 Format</span>
              </div>

              {/* Expansive, Modern Drag & Drop Zone */}
              <div 
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed rgba(56, 189, 248, 0.4)',
                  borderRadius: '12px',
                  padding: '36px 20px',
                  textAlign: 'center',
                  background: 'rgba(15, 23, 42, 0.5)',
                  cursor: 'pointer',
                  marginBottom: '16px',
                  transition: 'all 0.2s ease'
                }}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  multiple 
                  style={{ display: 'none' }} 
                  accept="image/bmp,image/png,image/jpeg" 
                  onChange={(e) => handleFilesSelected(e.target.files)}
                />
                <div style={{ width: '52px', height: '52px', borderRadius: '50%', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px auto' }}>
                  <Layers size={26} color="#38bdf8" />
                </div>
                <p style={{ fontSize: '14px', fontWeight: 700, margin: '0 0 6px 0' }}>
                  Drop Single Cell or Multi-Cell Batch
                </p>
                <p style={{ fontSize: '12px', color: '#94a3b8', margin: '0 0 8px 0' }}>
                  Select 1 image for individual cytometry, or select 5 to 50 images for slide-level screening
                </p>
                <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>
                  Supported formats: BMP, PNG, JPEG
                </span>
              </div>

              {/* Loaded Selection Status & Thumbnail Strip */}
              {selectedFiles.length > 0 && (
                <div style={{ background: 'rgba(0, 0, 0, 0.45)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '10px', padding: '12px', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#38bdf8' }}>
                      {selectedFiles.length === 1 ? '1 Cell Loaded' : `${selectedFiles.length} Smear Cells Selected`}
                    </span>
                    <button 
                      onClick={() => { setSelectedFiles([]); setPreviewUrls([]); setSingleResult(null); setBatchResult(null); }}
                      style={{ background: 'none', border: 'none', color: '#f87171', fontSize: '11px', fontWeight: 600, cursor: 'pointer' }}
                    >
                      Clear Selection
                    </button>
                  </div>

                  {/* Horizontal Thumbnail Carousel Strip */}
                  <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '6px' }}>
                    {previewUrls.map((url, idx) => (
                      <img 
                        key={idx}
                        src={url} 
                        alt={`Cell ${idx}`} 
                        style={{ width: '46px', height: '46px', objectFit: 'contain', borderRadius: '6px', background: '#000', border: '1px solid rgba(255,255,255,0.1)', flexShrink: 0 }} 
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Primary Action Button */}
              <button 
                onClick={handleAnalyze} 
                disabled={selectedFiles.length === 0 || isAnalyzing}
                style={{
                  width: '100%',
                  padding: '14px',
                  fontSize: '14px',
                  fontWeight: 700,
                  borderRadius: '10px',
                  background: isAnalyzing ? 'rgba(56, 189, 248, 0.4)' : 'linear-gradient(135deg, #0284c7, #4f46e5)',
                  color: '#ffffff',
                  border: 'none',
                  cursor: isAnalyzing || selectedFiles.length === 0 ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  boxShadow: '0 4px 14px rgba(2, 132, 199, 0.35)',
                  marginBottom: '16px'
                }}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    Running Neural Inference on {selectedFiles.length} Cell(s)...
                  </>
                ) : (
                  <>
                    <Zap size={16} />
                    {selectedFiles.length <= 1 
                      ? 'Analyze Single Cell & Grad-CAM' 
                      : `Run Slide Screening (${selectedFiles.length} Cells)`
                    }
                  </>
                )}
              </button>

              {(singleResult || batchResult) && (
                <button
                  type="button"
                  onClick={() => window.print()}
                  style={{
                    width: '100%',
                    padding: '10px',
                    fontSize: '12px',
                    fontWeight: 700,
                    borderRadius: '8px',
                    background: 'rgba(16, 185, 129, 0.15)',
                    border: '1px solid rgba(16, 185, 129, 0.4)',
                    color: '#34d399',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    marginBottom: '16px'
                  }}
                >
                  <Printer size={15} /> Print / Export Clinical Diagnostic Report
                </button>
              )}

              {/* Clinical Protocol & System Specs */}
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px', padding: '14px', fontSize: '11px', color: '#94a3b8', lineHeight: 1.6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#38bdf8', fontWeight: 700, marginBottom: '6px' }}>
                  <ShieldCheck size={14} /> Clinical Validation Standard
                </div>
                <p style={{ margin: '0 0 6px 0' }}>
                  • <strong>Model:</strong> EfficientNet-B0 + Hybrid Transformer
                </p>
                <p style={{ margin: '0 0 6px 0' }}>
                  • <strong>Cohort:</strong> 10,661 Cells • 73 Patients (ISBI 2019)
                </p>
                <p style={{ margin: '0 0 6px 0' }}>
                  • <strong>Zero Subject Leakage:</strong> 5-Fold Group Stratified
                </p>
                <p style={{ margin: 0 }}>
                  • <strong>WHO Leukemia Cutoff:</strong> &ge; 20% Peripheral Blasts
                </p>
              </div>

            </div>

            {/* Right Column: Master Diagnostic & Saliency Station */}
            <div>
              {/* ========================================================
                  SECTION A: BATCH (MULTIPLE CELLS) SUMMARY
                 ======================================================== */}
              {batchResult && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginBottom: '24px' }}>
                  
                  {/* Executive Slide Diagnostic Verdict Card */}
                  <div style={{
                    padding: '24px',
                    borderRadius: '14px',
                    background: batchResult.slide_alert === 'danger' 
                      ? 'linear-gradient(135deg, rgba(244, 63, 94, 0.22), rgba(15, 23, 42, 0.95))' 
                      : batchResult.slide_alert === 'warning'
                      ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.22), rgba(15, 23, 42, 0.95))'
                      : 'linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(15, 23, 42, 0.95))',
                    border: batchResult.slide_alert === 'danger' ? '1px solid rgba(244, 63, 94, 0.4)' : '1px solid rgba(16, 185, 129, 0.4)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                      <div>
                        <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', fontWeight: 600 }}>
                          Peripheral Blood Smear Slide Assessment
                        </span>
                        <h3 style={{
                          fontSize: '22px',
                          fontWeight: 800,
                          margin: '4px 0',
                          color: batchResult.slide_alert === 'danger' ? '#f43f5e' : batchResult.slide_alert === 'warning' ? '#fbbf24' : '#10b981'
                        }}>
                          {batchResult.slide_verdict}
                        </h3>
                        <p style={{ fontSize: '13px', color: '#cbd5e1', margin: '4px 0 0 0', lineHeight: 1.5 }}>
                          {batchResult.clinical_recommendation}
                        </p>
                      </div>

                      {/* Blast Index Gauge */}
                      <div style={{ textAlign: 'right', background: 'rgba(0,0,0,0.4)', padding: '12px 18px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)', flexShrink: 0 }}>
                        <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', display: 'block' }}>Slide Blast Index</span>
                        <strong style={{ fontSize: '26px', color: batchResult.slide_alert === 'danger' ? '#f43f5e' : '#10b981' }}>
                          {batchResult.batch_metadata.blast_index_pct}%
                        </strong>
                        <span style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>WHO Leukemia Cutoff: &ge; 20%</span>
                      </div>
                    </div>

                    {/* Progress Bar of Blast Ratio */}
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                        <div style={{
                          width: `${Math.min(100, batchResult.batch_metadata.blast_index_pct)}%`,
                          height: '100%',
                          background: batchResult.slide_alert === 'danger' ? 'linear-gradient(90deg, #f43f5e, #fb7185)' : 'linear-gradient(90deg, #34d399, #10b981)',
                          transition: 'width 0.6s ease'
                        }} />
                        {/* 20% Cutoff Marker */}
                        <div style={{ position: 'absolute', top: 0, left: '20%', width: '2px', height: '100%', background: '#f59e0b', boxShadow: '0 0 4px #f59e0b' }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                        <span>0%</span>
                        <span style={{ color: '#f59e0b', fontWeight: 600 }}>20% (WHO Cutoff)</span>
                        <span>100%</span>
                      </div>
                    </div>

                    {/* 5 Clinical Metric Counters */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
                      <div style={{ background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', textAlign: 'center' }}>
                        <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>Total Evaluated</span>
                        <strong style={{ fontSize: '16px' }}>{batchResult.batch_metadata.total_cells_analyzed} Cells</strong>
                      </div>
                      <div style={{ background: 'rgba(244, 63, 94, 0.12)', padding: '10px', borderRadius: '8px', textAlign: 'center', border: '1px solid rgba(244, 63, 94, 0.25)' }}>
                        <span style={{ fontSize: '11px', color: '#fda4af', display: 'block' }}>Malignant Blasts</span>
                        <strong style={{ fontSize: '16px', color: '#f43f5e' }}>{batchResult.batch_metadata.blasts_detected}</strong>
                      </div>
                      <div style={{ background: 'rgba(16, 185, 129, 0.12)', padding: '10px', borderRadius: '8px', textAlign: 'center', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
                        <span style={{ fontSize: '11px', color: '#6ee7b7', display: 'block' }}>Normal Leukocytes</span>
                        <strong style={{ fontSize: '16px', color: '#10b981' }}>{batchResult.batch_metadata.normal_cells_detected}</strong>
                      </div>
                      <div style={{ background: 'rgba(251, 191, 36, 0.12)', padding: '10px', borderRadius: '8px', textAlign: 'center', border: '1px solid rgba(251, 191, 36, 0.25)' }}>
                        <span style={{ fontSize: '11px', color: '#fcd34d', display: 'block' }}>Borderline / Review</span>
                        <strong style={{ fontSize: '16px', color: '#fbbf24' }}>{batchResult.batch_metadata.indeterminate_cells}</strong>
                      </div>
                      <div style={{ background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', textAlign: 'center' }}>
                        <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>Screening Latency</span>
                        <strong style={{ fontSize: '16px', color: '#38bdf8' }}>{batchResult.total_inference_latency_ms} ms</strong>
                      </div>
                    </div>

                    {/* Population FAB Subtype Spectrum Bar */}
                    {batchResult.batch_metadata.fab_distribution && (
                      <div style={{ marginTop: '14px', background: 'rgba(0,0,0,0.35)', padding: '10px 14px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }}>
                        <span style={{ color: '#94a3b8', fontWeight: 600 }}>Population FAB Subtypes:</span>
                        <div style={{ display: 'flex', gap: '16px' }}>
                          <span style={{ color: '#fda4af' }}>L1 Blasts: <strong>{batchResult.batch_metadata.fab_distribution.ALL_L1}</strong></span>
                          <span style={{ color: '#f43f5e' }}>L2 Blasts: <strong>{batchResult.batch_metadata.fab_distribution.ALL_L2}</strong></span>
                          <span style={{ color: '#c084fc' }}>L3 Blasts: <strong>{batchResult.batch_metadata.fab_distribution.ALL_L3}</strong></span>
                          <span style={{ color: '#6ee7b7' }}>Normal: <strong>{batchResult.batch_metadata.fab_distribution.HEM_Benign}</strong></span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Whole-Slide Smear Auto-Segmentation Map (When uncropped smear uploaded) */}
                  {batchResult.annotated_smear_b64 && (
                    <div className="glass-panel" style={{ padding: '20px', borderRadius: '14px', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Layers size={16} color="#38bdf8" />
                          <h4 style={{ fontSize: '14px', fontWeight: 700, margin: 0 }}>
                            Whole-Slide Peripheral Smear Automated Detection Map
                          </h4>
                          <span style={{ fontSize: '11px', color: '#94a3b8' }}>• {batchResult.cells.length} Leukocytes Detected</span>
                        </div>
                        <div style={{ display: 'flex', gap: '14px', fontSize: '11px' }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#f43f5e', fontWeight: 600 }}>
                            <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#ef4444' }} /> Malignant Blast
                          </span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981', fontWeight: 600 }}>
                            <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#10b981' }} /> Benign Leukocyte
                          </span>
                        </div>
                      </div>
                      <div style={{ borderRadius: '10px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', background: '#050914', display: 'flex', justifyContent: 'center' }}>
                        <img 
                          src={batchResult.annotated_smear_b64} 
                          alt="Annotated Whole-Slide Smear" 
                          style={{ maxWidth: '100%', maxHeight: '440px', objectFit: 'contain' }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Interactive Cell Population Gallery with Filtering */}
                  <div className="glass-panel" style={{ padding: '20px', background: 'rgba(15, 23, 42, 0.85)', borderRadius: '14px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Grid size={16} color="#38bdf8" />
                        <h4 style={{ fontSize: '14px', fontWeight: 700, margin: 0 }}>
                          Cell Population Gallery ({batchResult.cells.length} Cells)
                        </h4>
                        <span style={{ fontSize: '11px', color: '#94a3b8' }}>• Click any cell to inspect below</span>
                      </div>

                      {/* Filter & Sorting Chips */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(0,0,0,0.4)', padding: '3px', borderRadius: '8px' }}>
                          <button 
                            onClick={() => setCellFilter('all')}
                            style={{
                              background: cellFilter === 'all' ? '#0284c7' : 'transparent',
                              color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                            }}
                          >
                            All ({batchResult.cells.length})
                          </button>
                          <button 
                            onClick={() => setCellFilter('blasts')}
                            style={{
                              background: cellFilter === 'blasts' ? '#f43f5e' : 'transparent',
                              color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                            }}
                          >
                            Blasts ({batchResult.batch_metadata.blasts_detected})
                          </button>
                          <button 
                            onClick={() => setCellFilter('borderline')}
                            style={{
                              background: cellFilter === 'borderline' ? '#f59e0b' : 'transparent',
                              color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                            }}
                          >
                            Borderline ({batchResult.batch_metadata.indeterminate_cells})
                          </button>
                          <button 
                            onClick={() => setCellFilter('normal')}
                            style={{
                              background: cellFilter === 'normal' ? '#10b981' : 'transparent',
                              color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                            }}
                          >
                            Normal ({batchResult.batch_metadata.normal_cells_detected})
                          </button>
                        </div>

                        {/* Sort Selector */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(0,0,0,0.4)', padding: '4px 8px', borderRadius: '8px', fontSize: '11px', color: '#94a3b8' }}>
                          <ArrowUpDown size={12} />
                          <select 
                            value={cellSort} 
                            onChange={(e) => setCellSort(e.target.value)}
                            style={{ background: 'transparent', color: '#fff', border: 'none', fontSize: '11px', outline: 'none', cursor: 'pointer' }}
                          >
                            <option value="default" style={{ background: '#0f172a' }}>Sequence</option>
                            <option value="malignancy_desc" style={{ background: '#0f172a' }}>Malignancy Risk</option>
                            <option value="nc_desc" style={{ background: '#0f172a' }}>N:C Ratio</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Scrollable Cell Population Filmstrip Grid */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(105px, 1fr))',
                      gap: '10px',
                      maxHeight: '260px',
                      overflowY: 'auto',
                      padding: '4px'
                    }}>
                      {getFilteredAndSortedCells().map((cell, idx) => {
                        const isMalignant = cell.probabilities.ALL_malignant >= 0.60;
                        const isBorderline = cell.probabilities.ALL_malignant > 0.38 && cell.probabilities.ALL_malignant < 0.60;
                        const isSelected = activeBatchCell?.filename === cell.filename;
                        
                        return (
                          <div 
                            key={idx}
                            onClick={() => {
                              setActiveBatchCell(cell);
                              inspectorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            }}
                            style={{
                              background: isSelected ? 'rgba(56, 189, 248, 0.18)' : 'rgba(15, 23, 42, 0.8)',
                              border: isSelected 
                                ? '2px solid #38bdf8' 
                                : isMalignant 
                                ? '1px solid rgba(244, 63, 94, 0.4)' 
                                : isBorderline
                                ? '1px solid rgba(251, 191, 36, 0.4)'
                                : '1px solid rgba(16, 185, 129, 0.3)',
                              borderRadius: '8px',
                              padding: '6px',
                              cursor: 'pointer',
                              textAlign: 'center',
                              boxShadow: isSelected ? '0 0 14px rgba(56, 189, 248, 0.55)' : 'none',
                              transform: isSelected ? 'scale(1.04)' : 'scale(1)',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            <img 
                              src={cell.thumbnail_b64} 
                              alt={cell.filename} 
                              style={{ width: '100%', height: '62px', objectFit: 'contain', borderRadius: '4px', background: '#000', marginBottom: '4px' }}
                            />
                            <span style={{
                              fontSize: '10px',
                              fontWeight: 700,
                              padding: '1px 4px',
                              borderRadius: '4px',
                              background: isMalignant ? 'rgba(244, 63, 94, 0.2)' : isBorderline ? 'rgba(251, 191, 36, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                              color: isMalignant ? '#f43f5e' : isBorderline ? '#fbbf24' : '#10b981',
                              display: 'block',
                              marginBottom: '2px'
                            }}>
                              {isMalignant 
                                ? `${cell.fab_staging?.stage?.includes('L1') ? 'L1' : cell.fab_staging?.stage?.includes('L3') ? 'L3' : 'L2'} • ${(cell.probabilities.ALL_malignant * 100).toFixed(1)}%` 
                                : `Normal • ${(cell.probabilities.HEM_benign * 100).toFixed(1)}%`
                              }
                            </span>
                            <span style={{ fontSize: '9px', color: '#94a3b8', display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {cell.filename.replace('.bmp', '')}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                </div>
              )}

              {/* ========================================================
                  SECTION B: ACTIVE CELL DEEP INSPECTOR & 3-WAY GRAD-CAM
                 ======================================================== */}
              {currentCellDetails ? (
                <div ref={inspectorRef} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  
                  {/* Active Cell Cytological Assessment Card */}
                  <div className="glass-panel-glow" style={{ padding: '24px', background: 'rgba(15, 23, 42, 0.85)', borderRadius: '14px', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <div>
                        <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', fontWeight: 600 }}>
                          {batchResult ? `Detailed Cytometry for: ${currentCellDetails.filename}` : 'Single-Cell Screening & Morphology'}
                        </span>
                        <h3 style={{
                          fontSize: '22px',
                          fontWeight: 800,
                          margin: '4px 0 0 0',
                          color: currentCellDetails.alert_type === 'danger' ? '#f43f5e' : currentCellDetails.alert_type === 'warning' ? '#fbbf24' : '#10b981'
                        }}>
                          {currentCellDetails.fab_staging.stage}
                        </h3>
                      </div>
                      <span style={{
                        padding: '6px 14px',
                        borderRadius: '20px',
                        fontSize: '12px',
                        fontWeight: 700,
                        background: currentCellDetails.alert_type === 'danger' ? 'rgba(244, 63, 94, 0.15)' : currentCellDetails.alert_type === 'warning' ? 'rgba(251, 191, 36, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                        color: currentCellDetails.alert_type === 'danger' ? '#f43f5e' : currentCellDetails.alert_type === 'warning' ? '#fbbf24' : '#10b981',
                        border: '1px solid currentColor'
                      }}>
                        {currentCellDetails.confidence_level}
                      </span>
                    </div>

                    <p style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '16px', lineHeight: 1.5 }}>
                      <strong>Cytological Assessment:</strong> {currentCellDetails.fab_staging.subclassification}.
                      <span style={{ display: 'block', color: '#38bdf8', marginTop: '4px', fontWeight: 600 }}>
                        Clinical Risk: {currentCellDetails.fab_staging.risk_tier}
                      </span>
                    </p>

                    {/* Continuous Calibrated Probability Bar */}
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 700, marginBottom: '6px' }}>
                        <span style={{ color: '#fda4af' }}>Malignant ALL Confidence: {(currentCellDetails.probabilities.ALL_malignant * 100).toFixed(1)}%</span>
                        <span style={{ color: '#6ee7b7' }}>Benign HEM: {(currentCellDetails.probabilities.HEM_benign * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '10px', background: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden', display: 'flex' }}>
                        <div style={{
                          width: `${currentCellDetails.probabilities.ALL_malignant * 100}%`,
                          background: 'linear-gradient(90deg, #f43f5e, #fb7185)',
                          transition: 'width 0.5s ease'
                        }} />
                        <div style={{
                          width: `${currentCellDetails.probabilities.HEM_benign * 100}%`,
                          background: 'linear-gradient(90deg, #34d399, #10b981)',
                          transition: 'width 0.5s ease'
                        }} />
                      </div>
                    </div>

                    {/* Morphology Cytometry Metrics */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '10px', fontSize: '11px' }}>
                      <div>
                        <span style={{ color: '#94a3b8', display: 'block' }}>N:C Ratio</span>
                        <strong style={{ color: '#38bdf8', fontSize: '13px' }}>{currentCellDetails.morphology_metrics.nc_ratio}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#94a3b8', display: 'block' }}>Chromatin Density</span>
                        <strong style={{ color: '#a78bfa', fontSize: '13px' }}>{currentCellDetails.morphology_metrics.chromatin_density}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#94a3b8', display: 'block' }}>Regularity</span>
                        <strong style={{ color: '#34d399', fontSize: '13px' }}>{currentCellDetails.morphology_metrics.nuclear_regularity}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#94a3b8', display: 'block' }}>Cell Diameter</span>
                        <strong style={{ color: '#fbbf24', fontSize: '13px' }}>{currentCellDetails.morphology_metrics.cell_diameter_um} &mu;m</strong>
                      </div>
                    </div>
                  </div>

                  {/* ========================================================
                      CRYSTAL-CLEAR 3-WAY GRAD-CAM SALIENCY STUDIO
                     ======================================================== */}
                  <div className="glass-panel" style={{ padding: '24px', background: 'rgba(15, 23, 42, 0.85)', borderRadius: '14px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                      <div>
                        <h4 style={{ fontSize: '15px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                          <Sliders size={16} color="#38bdf8" /> Explainable Morphological Grad-CAM Heatmap
                        </h4>
                        <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                          Transparent neural activation mapping directly onto nuclear chromatin & blast membrane
                        </span>
                      </div>
                      
                      {/* View Switcher: 3-Way vs Overlay vs Pure Heatmap */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(0,0,0,0.4)', padding: '3px', borderRadius: '8px' }}>
                        <button 
                          onClick={() => setSaliencyViewMode('3way')}
                          style={{
                            background: saliencyViewMode === '3way' ? '#0284c7' : 'transparent',
                            color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                          }}
                        >
                          3-Way View
                        </button>
                        <button 
                          onClick={() => setSaliencyViewMode('overlay')}
                          style={{
                            background: saliencyViewMode === 'overlay' ? '#0284c7' : 'transparent',
                            color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                          }}
                        >
                          Blended Overlay
                        </button>
                        <button 
                          onClick={() => setSaliencyViewMode('heatmap')}
                          style={{
                            background: saliencyViewMode === 'heatmap' ? '#0284c7' : 'transparent',
                            color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer'
                          }}
                        >
                          Pure Saliency
                        </button>
                      </div>
                    </div>

                    {/* Saliency Display Area */}
                    {saliencyViewMode === '3way' && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '14px' }}>
                        {/* 1. Raw Microscopic Cell */}
                        <div style={{ textAlign: 'center', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: '6px' }}>
                            1. Original Stained Smear
                          </span>
                          <img 
                            src={currentPreviewUrl} 
                            alt="Raw Cell" 
                            style={{ width: '100%', height: '160px', objectFit: 'contain', borderRadius: '6px', background: '#000' }} 
                          />
                          <span style={{ fontSize: '10px', color: '#64748b', display: 'block', marginTop: '4px' }}>Leishman Staining</span>
                        </div>

                        {/* 2. Isolated Glowing Heatmap */}
                        <div style={{ textAlign: 'center', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '10px', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: '#38bdf8', display: 'block', marginBottom: '6px' }}>
                            2. Isolated Heatmap (Turbo)
                          </span>
                          <img 
                            src={currentCellDetails.explainability.heatmap_b64} 
                            alt="Grad-CAM Saliency" 
                            style={{ width: '100%', height: '160px', objectFit: 'contain', borderRadius: '6px', background: '#050a12' }} 
                          />
                          <span style={{ fontSize: '10px', color: '#38bdf8', display: 'block', marginTop: '4px' }}>Neural Attention Map</span>
                        </div>

                        {/* 3. Blended Overlay */}
                        <div style={{ textAlign: 'center', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: '#34d399', display: 'block', marginBottom: '6px' }}>
                            3. Diagnostic Alignment
                          </span>
                          <img 
                            src={currentCellDetails.explainability.overlay_b64} 
                            alt="Blended Overlay" 
                            style={{ width: '100%', height: '160px', objectFit: 'contain', borderRadius: '6px', background: '#000' }} 
                          />
                          <span style={{ fontSize: '10px', color: '#34d399', display: 'block', marginTop: '4px' }}>Morphological Overlay</span>
                        </div>
                      </div>
                    )}

                    {saliencyViewMode === 'overlay' && (
                      <div style={{ textAlign: 'center', marginBottom: '14px' }}>
                        <div style={{ position: 'relative', width: '100%', height: '260px', borderRadius: '10px', overflow: 'hidden', background: '#000', border: '1px solid rgba(255,255,255,0.1)' }}>
                          <img 
                            src={currentPreviewUrl} 
                            alt="Base" 
                            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }} 
                          />
                          <img 
                            src={currentCellDetails.explainability.heatmap_b64} 
                            alt="Heatmap overlay" 
                            style={{ 
                              position: 'absolute', 
                              top: 0, 
                              left: 0, 
                              width: '100%', 
                              height: '100%', 
                              objectFit: 'contain',
                              opacity: overlayOpacity
                            }} 
                          />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginTop: '10px' }}>
                          <span style={{ fontSize: '12px', color: '#94a3b8' }}>Heatmap Transparency:</span>
                          <input 
                            type="range" 
                            min="0.1" 
                            max="1.0" 
                            step="0.05" 
                            value={overlayOpacity}
                            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                            style={{ accentColor: '#38bdf8', width: '150px' }}
                          />
                          <span style={{ fontSize: '12px', fontWeight: 700, width: '35px' }}>{Math.round(overlayOpacity * 100)}%</span>
                        </div>
                      </div>
                    )}

                    {saliencyViewMode === 'heatmap' && (
                      <div style={{ textAlign: 'center', marginBottom: '14px' }}>
                        <img 
                          src={currentCellDetails.explainability.heatmap_b64} 
                          alt="Isolated Saliency" 
                          style={{ width: '100%', height: '260px', objectFit: 'contain', borderRadius: '10px', background: '#050a12', border: '1px solid rgba(56, 189, 248, 0.3)' }} 
                        />
                      </div>
                    )}

                    {/* Saliency Interpretation Banner */}
                    <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: '10px', padding: '12px 16px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: '#38bdf8' }}>
                          Clinical Explainability Interpretation:
                        </span>
                        {/* Jet Colorbar Reference */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: '#94a3b8' }}>
                          <span>Normal / Background</span>
                          <div style={{ width: '90px', height: '8px', borderRadius: '4px', background: 'linear-gradient(90deg, #1e3a8a, #06b6d4, #eab308, #dc2626)' }} />
                          <span style={{ color: '#f87171', fontWeight: 700 }}>Blast Hotspot</span>
                        </div>
                      </div>
                      <p style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.5, margin: 0 }}>
                        {currentCellDetails.explainability.focus_summary}
                      </p>
                    </div>

                  </div>

                </div>
              ) : (
                /* Empty state when no analysis has run yet */
                <div className="glass-panel" style={{ padding: '70px 30px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '480px', background: 'rgba(15, 23, 42, 0.6)' }}>
                  <div style={{ width: '68px', height: '68px', borderRadius: '50%', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '18px' }}>
                    <Microscope size={34} color="#38bdf8" />
                  </div>
                  <h3 style={{ fontSize: '19px', fontWeight: 700, marginBottom: '6px' }}>Awaiting Microscopic Smear Input</h3>
                  <p style={{ fontSize: '13px', color: '#94a3b8', maxWidth: '440px', lineHeight: 1.6, margin: 0 }}>
                    Select or drag-and-drop cell images on the left. The system will automatically run deep cytometry, FAB subtyping, and 3-way Grad-CAM heatmaps on this unified dashboard.
                  </p>
                </div>
              )}
            </div>

          </div>
        )}

        {/* ========================================================
            TAB 2: FAB STAGING REFERENCE MATRIX
           ======================================================== */}
        {activeTab === 'staging' && (
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: 800, marginBottom: '8px' }}>
              French-American-British (FAB) Morphological Staging Matrix
            </h2>
            <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '24px' }}>
              Clinical cytological subtyping criteria for Acute Lymphoblastic Leukemia according to WHO and FAB hematological standards.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: '12px', padding: '20px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>Stage L1</span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '10px' }}>Micro-Lymphoblastic (Homogeneous)</h3>
                <ul style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.8, paddingLeft: '18px' }}>
                  <li>Small, uniform cell size (10–13 $\mu$m)</li>
                  <li>High Nuclear-to-Cytoplasmic (N:C) ratio (&ge; 0.85)</li>
                  <li>Regular round/oval nuclear contour</li>
                  <li>Inconspicuous / non-visible nucleoli</li>
                  <li>Most common in pediatric ALL (85% of childhood cases)</li>
                </ul>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '12px', padding: '20px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#f43f5e', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>Stage L2</span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '10px' }}>Heterogeneous Pleomorphic Blasts</h3>
                <ul style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.8, paddingLeft: '18px' }}>
                  <li>Large, variable cell size (14–20 $\mu$m)</li>
                  <li>Variable, moderate N:C ratio</li>
                  <li>Irregular, indented, or clefted nuclear membrane</li>
                  <li>One or more prominent, visible nucleoli</li>
                  <li>Predominant in adult ALL (60–70% of adult cases)</li>
                </ul>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(168, 85, 247, 0.3)', borderRadius: '12px', padding: '20px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#c084fc', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>Stage L3</span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '10px' }}>Burkitt-Type Lymphoblastic</h3>
                <ul style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.8, paddingLeft: '18px' }}>
                  <li>Large, uniform blasts with vesicular chromatin</li>
                  <li>Abundant, intensely basophilic cytoplasm</li>
                  <li>Prominent, well-defined lipid vacuoles (oil red O positive)</li>
                  <li>Round to oval regular nuclei with 2–5 nucleoli</li>
                  <li>Mature B-cell phenotype with t(8;14) translocation</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================
            TAB 3: HEAD-TO-HEAD BASE PAPER COMPARISON
           ======================================================== */}
        {activeTab === 'comparison' && (
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: 800, marginBottom: '8px' }}>
              Base Paper Benchmark vs Our Proposed Project
            </h2>
            <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '24px' }}>
              Direct comparison with the Base Paper (ALL-Net / Thiriveedhi et al.) and modern hematology benchmarks.
            </p>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8' }}>
                    <th style={{ padding: '14px' }}>Evaluation Metric</th>
                    <th style={{ padding: '14px', color: '#fda4af' }}>Base Paper (ALL-Net)</th>
                    <th style={{ padding: '14px', color: '#38bdf8', fontWeight: 700 }}>Our Project (LeukemiaVision)</th>
                    <th style={{ padding: '14px', color: '#10b981' }}>Advantage / Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: 'rgba(56, 189, 248, 0.04)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Classification Accuracy</td>
                    <td style={{ padding: '14px', color: '#94a3b8' }}>90.7%</td>
                    <td style={{ padding: '14px', color: '#38bdf8', fontWeight: 700 }}>96.8% (Hybrid Transformer) / 95.4% (EfficientNet)</td>
                    <td style={{ padding: '14px', color: '#10b981', fontWeight: 600 }}>+6.1% Higher Accuracy</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Dataset Size & Curation</td>
                    <td style={{ padding: '14px' }}>~3,200 images</td>
                    <td style={{ padding: '14px', fontWeight: 600 }}>10,661 images (Full C-NMC 2019 Cohort)</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>3.3x Larger Sample Size</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Subject Leakage Prevention</td>
                    <td style={{ padding: '14px', color: '#f43f5e' }}>Naive random split (High leakage)</td>
                    <td style={{ padding: '14px', color: '#10b981', fontWeight: 600 }}>5-Fold StratifiedGroupKFold (Zero leakage)</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>Clinically Valid Generalization</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Unified Single & Slide Batch Testing</td>
                    <td style={{ padding: '14px', color: '#94a3b8' }}>Single isolated cells only</td>
                    <td style={{ padding: '14px', color: '#38bdf8', fontWeight: 600 }}>Unified Single & Multi-Cell Batch on 1 Dashboard</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>Slide-Level WHO Blast Index</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>FAB Subtype Staging</td>
                    <td style={{ padding: '14px', color: '#94a3b8' }}>Binary ALL vs HEM only</td>
                    <td style={{ padding: '14px', color: '#38bdf8', fontWeight: 600 }}>Multi-Stage (L1, L2, L3 Staging)</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>Actionable Staging Guidance</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Explainability & Saliency</td>
                    <td style={{ padding: '14px', color: '#94a3b8' }}>None (Black Box)</td>
                    <td style={{ padding: '14px', fontWeight: 600 }}>3-Way High-Contrast Grad-CAM Heatmap</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>Visual Nuclear Cytometry</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '14px', fontWeight: 600 }}>Inference Latency on CPU</td>
                    <td style={{ padding: '14px' }}>~120 ms</td>
                    <td style={{ padding: '14px', color: '#38bdf8', fontWeight: 600 }}>~28 ms (ONNX Serverless Engine)</td>
                    <td style={{ padding: '14px', color: '#10b981' }}>4.3x Faster Inference</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>

      {/* Sleek Medical Compliance Footer */}
      <footer style={{
        background: 'rgba(9, 14, 26, 0.95)',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        padding: '14px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '12px',
        color: '#64748b'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldCheck size={16} color="#10b981" />
          <span>Clinical Decision Support System • Department of AI & ML • Supervisor: <strong>Mr. Balika Mahesh</strong></span>
        </div>
        <div>
          Major Project Team: A. Rachana (23N81A6690), G. Himaresh (23N81A66A1), V. Adithya (23N81A66C7), B. Bharath Kumar (23N81A66B5)
        </div>
      </footer>
    </div>
  );
}
