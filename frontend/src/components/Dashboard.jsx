import React, { useState } from 'react';
import { 
  LayoutDashboard, 
  Upload, 
  Binary, 
  History, 
  BarChart3, 
  Info, 
  LogOut, 
  ChevronDown,
  Image as ImageIcon,
  ShieldCheck,
  TrendingUp,
  User,
  Settings,
  AlertCircle,
  Search,
  CheckCircle2,
  Download,
  Printer,
  Sparkles,
  Calendar,
  Heart,
  Activity,
  Shield,
  Layers
} from 'lucide-react';

export default function Dashboard({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [predictionSubTab, setPredictionSubTab] = useState('result'); // 'result' or 'explainability'
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);
  
  // State for search & filters in history
  const [searchQuery, setSearchQuery] = useState('');
  const [classFilter, setClassFilter] = useState('All');
  
  // Holds current patient details dynamically when clicking "View" report modal
  const [selectedReport, setSelectedReport] = useState(null);

  // Default to preloaded sample smear_01.jpg matching Screen 3
  const [uploadedImage, setUploadedImage] = useState('/cells.png');
  const [imageDetails, setImageDetails] = useState({
    name: 'smear_01.jpg',
    size: '2.4 MB',
    resolution: '1024 × 768'
  });

  // History List matching Screen 8
  const [historyList, setHistoryList] = useState([
    { sNo: 1, id: 'P001', date: '20/07/2026', imageName: 'smear_01.jpg', prediction: 'Koilocytotic', confidence: '96.8%', status: 'High Risk', image: '/cells.png' },
    { sNo: 2, id: 'P002', date: '19/07/2026', imageName: 'smear_02.jpg', prediction: 'Normal', confidence: '99.1%', status: 'Low Risk', image: '/preprocessed.png' },
    { sNo: 3, id: 'P003', date: '18/07/2026', imageName: 'smear_03.jpg', prediction: 'Dyskeratotic', confidence: '94.6%', status: 'High Risk', image: '/cells.png' },
    { sNo: 4, id: 'P004', date: '17/07/2026', imageName: 'smear_04.jpg', prediction: 'Metaplastic', confidence: '93.2%', status: 'Medium', image: '/preprocessed.png' },
    { sNo: 5, id: 'P005', date: '16/07/2026', imageName: 'smear_05.jpg', prediction: 'Normal', confidence: '98.7%', status: 'Low Risk', image: '/cells.png' },
  ]);

  // Handle mock image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result);
        
        // Mocking metadata based on uploaded file
        const mockSizes = ['1.8 MB', '2.2 MB', '3.1 MB', '1.4 MB'];
        const mockResolutions = ['1024 × 768', '1280 × 960', '800 × 600'];
        
        setImageDetails({
          name: file.name,
          size: mockSizes[Math.floor(Math.random() * mockSizes.length)],
          resolution: mockResolutions[Math.floor(Math.random() * mockResolutions.length)]
        });
      };
      reader.readAsDataURL(file);
    }
  };

  // Switch to prediction tab
  const handlePredictClick = () => {
    if (!uploadedImage) {
      alert('Please upload an image first.');
      return;
    }
    setActiveTab('prediction');
    setPredictionSubTab('result');
  };

  // View Report Modal trigger
  const handleViewReport = (item) => {
    setSelectedReport(item);
    setShowReportModal(true);
  };

  // Switch to predicting report modal from the primary prediction view
  const triggerPredictReportModal = () => {
    const mockReport = {
      id: 'P001',
      date: '20/07/2026',
      imageName: imageDetails.name,
      prediction: 'Koilocytotic',
      confidence: '96.8%',
      status: 'High Risk',
      image: uploadedImage
    };
    setSelectedReport(mockReport);
    setShowReportModal(true);
  };

  // Filter history list
  const filteredHistory = historyList.filter(item => {
    const matchesSearch = item.id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesClass = classFilter === 'All' || item.prediction.toLowerCase() === classFilter.toLowerCase();
    return matchesSearch && matchesClass;
  });

  const menuItems = [
    { id: 'dashboard', name: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { id: 'upload', name: 'Upload Image', icon: <Upload size={18} /> },
    { id: 'prediction', name: 'Prediction', icon: <Binary size={18} /> },
    { id: 'history', name: 'History', icon: <History size={18} /> },
    { id: 'performance', name: 'Model Performance', icon: <BarChart3 size={18} /> },
    { id: 'about', name: 'About', icon: <Info size={18} /> },
  ];

  return (
    <div className="dashboard-container">
      {/* Top Header Bar */}
      <header className="header-bar">
        <div className="header-logo-section">
          <div className="logo-circle">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2c-2.8 0-5 2.2-5 5c0 2.5 1.5 4.5 4 6.5l-4 8.5c-.3.6 0 1.3.6 1.6c.6.3 1.3 0 1.6-.6l3.8-8c.5.2 1 .2 1.6.2s1.1 0 1.6-.2l3.8 8c.3.6 1 .9 1.6.6c.6-.3.9-1 .6-1.6l-4-8.5c2.5-2 4-4 4-6.5c0-2.8-2.2-5-5-5Z" />
            </svg>
          </div>
          <h1>Cervical Cancer Detection System</h1>
        </div>

        <div className="header-user-section">
          <div 
            className="user-profile-trigger"
            onClick={() => setShowUserDropdown(!showUserDropdown)}
          >
            <div className="avatar-circle">{user?.initials || 'DS'}</div>
            <span className="user-name">{user?.name || 'Dr. Sowmiya'}</span>
            <ChevronDown size={14} />
          </div>

          {showUserDropdown && (
            <div className="user-dropdown animate-fade">
              <button className="dropdown-item" onClick={() => { setActiveTab('about'); setShowUserDropdown(false); }}>
                <User size={14} /> Profile Info
              </button>
              <button className="dropdown-item" onClick={() => { alert('Configuration settings panel is simulated.'); setShowUserDropdown(false); }}>
                <Settings size={14} /> Settings
              </button>
              <button className="dropdown-item" onClick={onLogout}>
                <LogOut size={14} /> Logout
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="main-wrapper">
        {/* Left Navigation Sidebar */}
        <aside className="sidebar">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`sidebar-nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              <span>{item.name}</span>
            </button>
          ))}

          {/* Logout at bottom */}
          <button 
            className="sidebar-nav-item logout-item"
            onClick={onLogout}
          >
            <LogOut size={18} />
            <span>Logout</span>
          </button>
        </aside>

        {/* Right Main Content Panel */}
        <main className="content-area">
          {/* TAB 1: DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="animate-fade">
              <div className="content-header">
                <h2>Dashboard</h2>
                <p>AI-powered analysis of Pap smear images for early cervical cancer detection</p>
              </div>

              {/* Status metrics grid */}
              <div className="metric-cards-grid">
                <div className="metric-card tested">
                  <div className="metric-info">
                    <span className="metric-label">Total Images Tested</span>
                    <span className="metric-value">248</span>
                  </div>
                  <div className="metric-icon-box">
                    <ImageIcon size={22} />
                  </div>
                </div>

                <div className="metric-card normal">
                  <div className="metric-info">
                    <span className="metric-label">Normal Cases</span>
                    <span className="metric-value">102</span>
                  </div>
                  <div className="metric-icon-box">
                    <ShieldCheck size={22} />
                  </div>
                </div>

                <div className="metric-card abnormal">
                  <div className="metric-info">
                    <span className="metric-label">Abnormal Cases</span>
                    <span className="metric-value">146</span>
                  </div>
                  <div className="metric-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M12 2c-2.8 0-5 2.2-5 5c0 2.5 1.5 4.5 4 6.5l-4 8.5c-.3.6 0 1.3.6 1.6c.6.3 1.3 0 1.6-.6l3.8-8.5" />
                    </svg>
                  </div>
                </div>

                <div className="metric-card accuracy">
                  <div className="metric-info">
                    <span className="metric-label">Model Accuracy</span>
                    <span className="metric-value">98.4%</span>
                  </div>
                  <div className="metric-icon-box">
                    <TrendingUp size={22} />
                  </div>
                </div>
              </div>

              {/* Bottom Quote Banner */}
              <div className="dashboard-banner-card">
                <h3>“Detect Early... Prevent Cervical Cancer”</h3>
                <p>Advanced Deep Learning for Women's Health</p>
              </div>
            </div>
          )}

          {/* TAB 2: UPLOAD IMAGE */}
          {activeTab === 'upload' && (
            <div className="animate-fade">
              <div className="content-header">
                <h2>Upload Pap Smear Image</h2>
                <p>Upload a Pap smear image to detect and classify cervical cells</p>
              </div>

              <div className="split-grid">
                {/* Drag and Drop Zone */}
                <div 
                  className="upload-card-dashed"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const file = e.dataTransfer.files[0];
                    if (file) {
                      const reader = new FileReader();
                      reader.onloadend = () => {
                        setUploadedImage(reader.result);
                        setImageDetails({
                          name: file.name,
                          size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
                          resolution: '1024 × 768'
                        });
                      };
                      reader.readAsDataURL(file);
                    }
                  }}
                  onClick={() => document.getElementById('imageLoaderInput').click()}
                >
                  <input 
                    type="file" 
                    id="imageLoaderInput" 
                    accept="image/*" 
                    style={{ display: 'none' }} 
                    onChange={handleImageUpload}
                  />
                  
                  <div className="cloud-upload-btn">
                    <Upload size={32} />
                  </div>
                  
                  <h3 style={{ fontSize: '1.25rem', color: '#1e293b' }}>Drag & Drop your image here</h3>
                  <span style={{ fontSize: '0.9rem', color: '#64748b', fontWeight: '500' }}>or</span>
                  
                  <button className="choose-file-btn" type="button">
                    Choose File
                  </button>
                  
                  <span className="supported-footnote">
                    Supported formats: JPG, PNG | Max size: 10MB
                  </span>
                </div>

                {/* Preview Card */}
                <div className="preview-details-card">
                  <h3 style={{ fontSize: '1.1rem', color: '#0f172a' }}>Image Preview</h3>
                  
                  <div className="preview-img-wrapper">
                    <img 
                      src={uploadedImage || '/cells.png'} 
                      alt="Pap Smear Preview" 
                    />
                  </div>

                  <div className="metadata-table">
                    <div className="metadata-row">
                      <span className="meta-label">File name</span>
                      <span>:</span>
                      <span className="meta-val">{imageDetails.name}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="meta-label">Size</span>
                      <span>:</span>
                      <span className="meta-val">{imageDetails.size}</span>
                    </div>
                    <div className="metadata-row">
                      <span className="meta-label">Resolution</span>
                      <span>:</span>
                      <span className="meta-val">{imageDetails.resolution}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom centered Predict button */}
              <div className="bottom-button-bar">
                <button 
                  onClick={handlePredictClick}
                  className="login-btn"
                  style={{ width: '420px', padding: '0.85rem' }}
                >
                  <Search size={18} /> Predict
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: PREDICTION */}
          {activeTab === 'prediction' && (
            <div className="animate-fade">
              {/* Toggle Sub Tabs for Prediction (Screen 4 vs Screen 5) */}
              <div className="tab-sub-nav">
                <button 
                  className={`sub-nav-btn ${predictionSubTab === 'result' ? 'active' : ''}`}
                  onClick={() => setPredictionSubTab('result')}
                >
                  Diagnostic Result
                </button>
                <button 
                  className={`sub-nav-btn ${predictionSubTab === 'explainability' ? 'active' : ''}`}
                  onClick={() => setPredictionSubTab('explainability')}
                >
                  Explainability & Pipeline
                </button>
              </div>

              {/* VIEW 1: DIAGNOSTIC RESULT (Screen 4) */}
              {predictionSubTab === 'result' && (
                <div className="animate-fade">
                  <div className="split-grid">
                    {/* Left Card: Input Image with bounding box focus */}
                    <div className="prediction-card-left">
                      <h3 style={{ fontSize: '1.1rem', color: '#0f172a' }}>Input Image</h3>
                      <div className="input-image-relative">
                        <img src={uploadedImage || '/cells.png'} alt="Analyzed Scan" />
                        <div className="bounding-box-overlay"></div>
                      </div>
                    </div>

                    {/* Right Card: Prediction results, Circle gauge progress */}
                    <div className="predicted-class-card">
                      <span className="badge-risk">High Risk</span>
                      <h3 style={{ fontSize: '1.1rem', color: '#0f172a' }}>Predicted Class</h3>
                      
                      <div style={{ marginTop: '0.5rem' }}>
                        <div className="predicted-capsule">
                          Koilocytotic Cell
                        </div>
                      </div>

                      {/* Circular Gauge Score */}
                      <div className="circular-gauge-container">
                        <div className="gauge-svg-wrapper">
                          <svg width="120" height="120" viewBox="0 0 120 120">
                            <circle cx="60" cy="60" r="50" fill="none" stroke="#f1f5f9" strokeWidth="10" />
                            <circle 
                              cx="60" cy="60" r="50" fill="none" stroke="url(#gauge-gradient)" strokeWidth="10" 
                              strokeDasharray="314" strokeDashoffset="10" strokeLinecap="round" transform="rotate(-90 60 60)"
                            />
                            <defs>
                              <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#10b981" />
                                <stop offset="100%" stopColor="#0ea5e9" />
                              </linearGradient>
                            </defs>
                          </svg>
                          <div className="gauge-center-text">96.8%</div>
                        </div>
                        <span className="gauge-label">Confidence Score</span>
                      </div>

                      {/* Classifier specifics */}
                      <div className="predicted-model-details">
                        <div className="predicted-model-row">
                          <span className="model-label">Model Used</span>
                          <span>:</span>
                          <span className="model-val">
                            Fusion Model
                            <span>(ResNet50 + ResNet101 + ResNet152)</span>
                          </span>
                        </div>
                        <div className="predicted-model-row">
                          <span className="model-label">Processing Time</span>
                          <span>:</span>
                          <span className="model-val">1.8 seconds</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Lower split: Class Probabilities progress bars and Interpretation details */}
                  <div className="split-grid" style={{ gridTemplateColumns: '1.2fr 0.8fr' }}>
                    <div className="probabilities-box">
                      <h3 style={{ fontSize: '1.1rem', color: '#0f172a' }}>Class Probabilities</h3>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="probability-bar-row">
                          <div className="prob-info">
                            <span>Koilocytotic</span>
                            <span>96.8%</span>
                          </div>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: '96.8%', background: '#6366f1' }}></div>
                          </div>
                        </div>

                        <div className="probability-bar-row">
                          <div className="prob-info">
                            <span>Dyskeratotic</span>
                            <span>1.5%</span>
                          </div>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: '1.5%', background: '#a855f7' }}></div>
                          </div>
                        </div>

                        <div className="probability-bar-row">
                          <div className="prob-info">
                            <span>Metaplastic</span>
                            <span>1.0%</span>
                          </div>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: '1.0%', background: '#10b981' }}></div>
                          </div>
                        </div>

                        <div className="probability-bar-row">
                          <div className="prob-info">
                            <span>Superficial-Intermediate</span>
                            <span>0.5%</span>
                          </div>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: '0.5%', background: '#10b981' }}></div>
                          </div>
                        </div>

                        <div className="probability-bar-row">
                          <div className="prob-info">
                            <span>Parabasal</span>
                            <span>0.2%</span>
                          </div>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: '0.2%', background: '#10b981' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="interpretation-card">
                      <h4>Interpretation :</h4>
                      <p>
                        The uploaded Pap smear image shows characteristics of <strong>Koilocytotic</strong> cells, which are associated with early cervical abnormalities.
                      </p>
                    </div>
                  </div>

                  {/* Trigger report modal */}
                  <div className="bottom-button-bar" style={{ marginTop: '2rem' }}>
                    <button 
                      onClick={triggerPredictReportModal}
                      className="download-report-btn"
                    >
                      <Download size={18} /> Download Report
                    </button>
                  </div>
                </div>
              )}

              {/* VIEW 2: EXPLAINABILITY & PIPELINE (Screen 5) */}
              {predictionSubTab === 'explainability' && (
                <div className="animate-fade">
                  {/* Stepper indicator */}
                  <div className="stepper-container">
                    <div className="stepper-step completed">
                      <span className="stepper-number">1</span>
                      <span className="stepper-label">Input Image</span>
                    </div>
                    <div className="stepper-divider"></div>
                    <div className="stepper-step completed">
                      <span className="stepper-number">2</span>
                      <span className="stepper-label">Preprocessed</span>
                    </div>
                    <div className="stepper-divider"></div>
                    <div className="stepper-step completed">
                      <span className="stepper-number">3</span>
                      <span className="stepper-label">Feature Map</span>
                    </div>
                    <div className="stepper-divider"></div>
                    <div className="stepper-step completed">
                      <span className="stepper-number">4</span>
                      <span className="stepper-label">Grad-CAM</span>
                    </div>
                    <div className="stepper-divider"></div>
                    <div className="stepper-step active">
                      <span className="stepper-number">5</span>
                      <span className="stepper-label">Prediction</span>
                    </div>
                  </div>

                  {/* Image Visualization columns */}
                  <div className="explainability-cards-grid">
                    <div className="explainability-card">
                      <h4>Original Image</h4>
                      <div className="explainability-img-box">
                        <img src="/cells.png" alt="Original Scan" />
                      </div>
                    </div>

                    <div className="explainability-card">
                      <h4>Preprocessed</h4>
                      <div className="explainability-img-box">
                        <img src="/preprocessed.png" alt="Preprocessed Scan" />
                      </div>
                    </div>

                    <div className="explainability-card">
                      <h4>Feature Map</h4>
                      <div className="explainability-img-box">
                        <img src="/feature_map.png" alt="Feature Map Extract" />
                      </div>
                    </div>

                    <div className="explainability-card">
                      <h4>Grad-CAM Heatmap</h4>
                      <div className="explainability-img-box">
                        <img src="/grad_cam.png" alt="Grad-CAM Focus" />
                      </div>
                    </div>
                  </div>

                  {/* Architecture Node Diagram */}
                  <div className="architecture-container">
                    <h3 style={{ fontSize: '1.2rem', color: '#0f172a', borderBottom: '1px solid #f1f5f9', paddingBottom: '0.8rem', marginBottom: '1.5rem' }}>
                      Fusion Deep Learning Framework
                    </h3>

                    <div className="architecture-inner-grid">
                      <svg className="arch-connector-svg">
                        <defs>
                          <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1" />
                          </marker>
                        </defs>
                        <line x1="28%" y1="20%" x2="50%" y2="50%" stroke="#cbd5e1" strokeWidth="1.5" markerEnd="url(#arrow)" />
                        <line x1="28%" y1="50%" x2="45%" y2="50%" stroke="#cbd5e1" strokeWidth="1.5" markerEnd="url(#arrow)" />
                        <line x1="28%" y1="80%" x2="50%" y2="50%" stroke="#cbd5e1" strokeWidth="1.5" markerEnd="url(#arrow)" />
                        <line x1="68%" y1="50%" x2="78%" y2="50%" stroke="#cbd5e1" strokeWidth="1.5" markerEnd="url(#arrow)" />
                      </svg>

                      {/* Left: Input Baselines */}
                      <div className="architecture-cnn-stack">
                        <div className="cnn-model-box">ResNet50</div>
                        <div className="cnn-model-box">ResNet101</div>
                        <div className="cnn-model-box">ResNet152</div>
                      </div>

                      {/* Center: Fusion Connector */}
                      <div className="fusion-central-box">
                        <div className="fusion-icon-wrapper">
                          <Sparkles size={24} />
                        </div>
                        <h4>Feature Fusion</h4>
                      </div>

                      {/* Right: Classifier List Output */}
                      <div className="classification-output-card">
                        <h4>Classification</h4>
                        <div className="output-class-item">Normal</div>
                        <div className="output-class-item selected">
                          <span>Koilocytotic</span>
                          <CheckCircle2 size={16} style={{ color: 'var(--color-primary)' }} />
                        </div>
                        <div className="output-class-item">Dyskeratotic</div>
                        <div className="output-class-item">Metaplastic</div>
                        <div className="output-class-item">Others</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: PREDICTION HISTORY (Screen 8) */}
          {activeTab === 'history' && (
            <div className="tab-placeholder animate-fade">
              <div className="content-header" style={{ marginBottom: '1.5rem' }}>
                <h2>Prediction History</h2>
              </div>

              {/* Filter Bar with Search, Dropdown & Calendar icon */}
              <div className="filter-bar">
                <div className="filter-input-wrapper">
                  <Search size={16} className="filter-input-icon" />
                  <input 
                    type="text" 
                    placeholder="Search by Patient ID..." 
                    className="filter-input"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>

                <select 
                  className="filter-select"
                  value={classFilter}
                  onChange={(e) => setClassFilter(e.target.value)}
                >
                  <option value="All">All Classes</option>
                  <option value="Koilocytotic">Koilocytotic</option>
                  <option value="Normal">Normal</option>
                  <option value="Dyskeratotic">Dyskeratotic</option>
                  <option value="Metaplastic">Metaplastic</option>
                </select>

                <button className="calendar-btn" type="button" title="Select Date Range">
                  <Calendar size={18} />
                </button>
              </div>

              {/* Grid table representation matching Screen 8 */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
                      <th style={{ padding: '0.75rem 1rem' }}>S.No</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Date</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Patient ID</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Image</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Prediction</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Confidence</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Status</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHistory.map((item) => (
                      <tr key={item.sNo} style={{ borderBottom: '1px solid #f1f5f9', fontSize: '0.9rem' }}>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: '600', color: '#64748b' }}>{item.sNo}</td>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: '500' }}>{item.date}</td>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: '600', color: '#1e293b' }}>{item.id}</td>
                        <td style={{ padding: '0.85rem 1rem' }}>
                          <img src={item.image} alt="Cell biopsy" className="history-table-img" />
                        </td>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: '700', color: '#1e1b4b' }}>{item.prediction}</td>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: '600', color: '#0f172a' }}>{item.confidence}</td>
                        <td style={{ padding: '0.85rem 1rem' }}>
                          <span className={`badge-status ${item.status === 'High Risk' ? 'high' : item.status === 'Medium' ? 'medium' : 'low'}`}>
                            {item.status}
                          </span>
                        </td>
                        <td style={{ padding: '0.85rem 1rem' }}>
                          <button 
                            className="history-action-btn"
                            onClick={() => handleViewReport(item)}
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                    {filteredHistory.length === 0 && (
                      <tr>
                        <td colSpan="8" style={{ textAlign: 'center', padding: '2rem', color: 'var(--color-text-muted)' }}>
                          No patient records matching search parameters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination controls below */}
              <div className="pagination-container">
                <button className="page-btn" disabled>&lt; Previous</button>
                <button className="page-btn active">1</button>
                <button className="page-btn">2</button>
                <button className="page-btn">3</button>
                <button className="page-btn">4</button>
                <button className="page-btn">5</button>
                <button className="page-btn">Next</button>
              </div>
            </div>
          )}

          {/* TAB 5: MODEL PERFORMANCE (Screen 7) */}
          {activeTab === 'performance' && (
            <div className="animate-fade">
              <div className="content-header">
                <h2>Model Performance</h2>
              </div>

              {/* 4 Performance Metric Cards */}
              <div className="metric-cards-grid" style={{ marginBottom: '1.5rem' }}>
                <div className="metric-card perf-accuracy">
                  <div className="metric-info">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">98.4%</span>
                  </div>
                  <div className="metric-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
                    </svg>
                  </div>
                </div>

                <div className="metric-card perf-precision">
                  <div className="metric-info">
                    <span className="metric-label">Precision</span>
                    <span className="metric-value">97.9%</span>
                  </div>
                  <div className="metric-icon-box">
                    <User size={22} />
                  </div>
                </div>

                <div className="metric-card perf-recall">
                  <div className="metric-info">
                    <span className="metric-label">Recall</span>
                    <span className="metric-value">98.1%</span>
                  </div>
                  <div className="metric-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <circle cx="12" cy="8" r="7" />
                      <polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88" />
                    </svg>
                  </div>
                </div>

                <div className="metric-card perf-f1">
                  <div className="metric-info">
                    <span className="metric-label">F1-Score</span>
                    <span className="metric-value">98.0%</span>
                  </div>
                  <div className="metric-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                      <line x1="16" y1="13" x2="8" y2="13" />
                      <line x1="16" y1="17" x2="8" y2="17" />
                      <polyline points="10 9 9 9 8 9" />
                    </svg>
                  </div>
                </div>
              </div>

              {/* 3 Graphic cards grid */}
              <div className="performance-charts-grid">
                
                {/* 1. Confusion Matrix Grid */}
                <div className="chart-card">
                  <h3>Confusion Matrix</h3>
                  <div className="matrix-wrapper">
                    <table className="matrix-table">
                      <thead>
                        <tr>
                          <th className="header-label" style={{ border: 'none' }}></th>
                          <th className="header-label">Normal</th>
                          <th className="header-label" style={{ fontSize: '0.7rem' }}>Kolocytotic</th>
                          <th className="header-label" style={{ fontSize: '0.7rem' }}>Dyskeratotic</th>
                          <th className="header-label" style={{ fontSize: '0.7rem' }}>Metaplastic</th>
                          <th className="header-label">Others</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="row-label">Normal</td>
                          <td className="matrix-cell-diagonal">90</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-zero">0</td>
                        </tr>
                        <tr>
                          <td className="row-label" style={{ fontSize: '0.7rem' }}>Kolocytotic</td>
                          <td className="matrix-cell-low">2</td>
                          <td className="matrix-cell-diagonal">97</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-zero">0</td>
                        </tr>
                        <tr>
                          <td className="row-label" style={{ fontSize: '0.7rem' }}>Dyskeratotic</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-diagonal">96</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-low">1</td>
                        </tr>
                        <tr>
                          <td className="row-label" style={{ fontSize: '0.7rem' }}>Metaplastic</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-diagonal">97</td>
                          <td className="matrix-cell-low">1</td>
                        </tr>
                        <tr>
                          <td className="row-label">Others</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-zero">0</td>
                          <td className="matrix-cell-low">1</td>
                          <td className="matrix-cell-low">2</td>
                          <td className="matrix-cell-diagonal">97</td>
                        </tr>
                      </tbody>
                    </table>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: '0.4rem', fontWeight: '600' }}>
                      <span>Actual Class (Rows)</span>
                      <span>Predicted Class (Cols)</span>
                    </div>
                  </div>
                </div>

                {/* 2. ROC Curve Graph (SVG) */}
                <div className="chart-card">
                  <h3>ROC Curve</h3>
                  <div style={{ position: 'relative', width: '100%', aspectRatio: '1.1' }}>
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                      {/* Axes */}
                      <line x1="25" y1="160" x2="185" y2="160" stroke="#cbd5e1" strokeWidth="1.5" />
                      <line x1="25" y1="20" x2="25" y2="160" stroke="#cbd5e1" strokeWidth="1.5" />
                      
                      {/* Y-Axis labels */}
                      <text x="18" y="163" fontSize="8" fill="#64748b" textAnchor="end">0.0</text>
                      <text x="18" y="93" fontSize="8" fill="#64748b" textAnchor="end">0.5</text>
                      <text x="18" y="23" fontSize="8" fill="#64748b" textAnchor="end">1.0</text>
                      <text x="-90" y="8" fontSize="8" fill="#64748b" fontWeight="600" transform="rotate(-90)" textAnchor="middle">True Positive Rate</text>
                      
                      {/* X-Axis labels */}
                      <text x="25" y="173" fontSize="8" fill="#64748b" textAnchor="middle">0.0</text>
                      <text x="105" y="173" fontSize="8" fill="#64748b" textAnchor="middle">0.5</text>
                      <text x="185" y="173" fontSize="8" fill="#64748b" textAnchor="middle">1.0</text>
                      <text x="105" y="185" fontSize="8" fill="#64748b" fontWeight="600" textAnchor="middle">False Positive Rate</text>
                      
                      {/* Grid Lines */}
                      <line x1="25" y1="90" x2="185" y2="90" stroke="#f1f5f9" strokeDasharray="3" />
                      <line x1="105" y1="20" x2="105" y2="160" stroke="#f1f5f9" strokeDasharray="3" />

                      {/* ROC curves */}
                      <path d="M 25 160 Q 30 35 185 20" fill="none" stroke="#228be6" strokeWidth="2" />
                      <path d="M 25 160 Q 40 45 185 20" fill="none" stroke="#fd7e14" strokeWidth="1.5" />
                      <path d="M 25 160 Q 55 55 185 20" fill="none" stroke="#40c057" strokeWidth="1.5" />
                      <path d="M 25 160 Q 70 70 185 20" fill="none" stroke="#845ef7" strokeWidth="1.5" />
                    </svg>

                    {/* Chart Legend */}
                    <div style={{ position: 'absolute', bottom: '25px', right: '10px', background: 'rgba(255,255,255,0.95)', border: '1px solid #e2e8f0', borderRadius: '4px', padding: '0.4rem', fontSize: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.2rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#228be6', display: 'inline-block' }}></span><span>Fusion (AUC = 0.99)</span></div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#fd7e14', display: 'inline-block' }}></span><span>ResNet152 (AUC = 0.96)</span></div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#40c057', display: 'inline-block' }}></span><span>ResNet101 (AUC = 0.93)</span></div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#845ef7', display: 'inline-block' }}></span><span>ResNet50 (AUC = 0.90)</span></div>
                    </div>
                  </div>
                </div>

                {/* 3. Training & Validation Accuracy Graph (SVG) */}
                <div className="chart-card">
                  <h3>Training & Validation Accuracy</h3>
                  <div style={{ position: 'relative', width: '100%', aspectRatio: '1.1' }}>
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                      {/* Axes */}
                      <line x1="25" y1="160" x2="185" y2="160" stroke="#cbd5e1" strokeWidth="1.5" />
                      <line x1="25" y1="20" x2="25" y2="160" stroke="#cbd5e1" strokeWidth="1.5" />
                      
                      {/* Y-Axis labels */}
                      <text x="18" y="163" fontSize="8" fill="#64748b" textAnchor="end">0.7</text>
                      <text x="18" y="117" fontSize="8" fill="#64748b" textAnchor="end">0.8</text>
                      <text x="18" y="70" fontSize="8" fill="#64748b" textAnchor="end">0.9</text>
                      <text x="18" y="23" fontSize="8" fill="#64748b" textAnchor="end">1.0</text>
                      
                      {/* X-Axis labels */}
                      {[1,2,3,4,5,6,7,8,9,10].map((epoch, idx) => (
                        <text key={epoch} x={25 + idx * 16} y="173" fontSize="7" fill="#64748b" textAnchor="middle">{epoch}</text>
                      ))}
                      <text x="105" y="187" fontSize="8" fill="#64748b" fontWeight="600" textAnchor="middle">Epochs</text>
                      
                      {/* Grid Lines */}
                      <line x1="25" y1="114" x2="185" y2="114" stroke="#f1f5f9" strokeDasharray="3" />
                      <line x1="25" y1="67" x2="185" y2="67" stroke="#f1f5f9" strokeDasharray="3" />

                      {/* Training accuracy path */}
                      <path d="M 25 110 L 41 90 L 57 85 L 73 80 L 89 74 L 105 68 L 121 60 L 137 54 L 153 50 L 169 46 L 185 43" fill="none" stroke="#228be6" strokeWidth="2" />
                      <circle cx="25" cy="110" r="1.5" fill="#228be6" />
                      <circle cx="105" cy="68" r="1.5" fill="#228be6" />
                      <circle cx="185" cy="43" r="1.5" fill="#228be6" />
                      
                      {/* Validation accuracy path */}
                      <path d="M 25 125 L 41 110 L 57 100 L 73 93 L 89 87 L 105 82 L 121 75 L 137 70 L 153 66 L 169 62 L 185 60" fill="none" stroke="#40c057" strokeWidth="2" />
                      <circle cx="25" cy="125" r="1.5" fill="#40c057" />
                      <circle cx="105" cy="82" r="1.5" fill="#40c057" />
                      <circle cx="185" cy="60" r="1.5" fill="#40c057" />
                    </svg>

                    {/* Chart Legend */}
                    <div style={{ position: 'absolute', bottom: '40px', right: '15px', background: 'rgba(255,255,255,0.95)', border: '1px solid #e2e8f0', borderRadius: '4px', padding: '0.4rem', fontSize: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.2rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#228be6', display: 'inline-block' }}></span><span>Training Acc</span></div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><span style={{ width: '8px', height: '2px', background: '#40c057', display: 'inline-block' }}></span><span>Validation Acc</span></div>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 6: ABOUT THE SYSTEM (Screen 9) */}
          {activeTab === 'about' && (
            <div className="animate-fade">
              <div className="content-header">
                <h2>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ color: 'var(--color-primary)' }}>
                    <path d="M12 2c-2.8 0-5 2.2-5 5c0 2.5 1.5 4.5 4 6.5l-4 8.5c-.3.6 0 1.3.6 1.6c.6.3 1.3 0 1.6-.6l3.8-8" />
                  </svg>
                  About the System
                </h2>
              </div>

              {/* About card with cell illustration collage */}
              <div className="about-hero-card">
                <div className="about-image-collage">
                  <img src="/cells.png" alt="Cells Collage" />
                </div>
                <div className="about-content-right">
                  <h3>A Fusion-Based Deep Learning Framework for Enhanced Cervical Cancer Detection and Classification</h3>
                  <p>This system uses a fusion of ResNet50, ResNet101 and ResNet152 deep learning models to accurately detect and classify cervical cancer cells from Pap smear images.</p>
                </div>
              </div>

              {/* Badges Row */}
              <div className="about-badges-row">
                <div className="about-badge-item">
                  <div className="about-badge-icon">
                    <Layers size={16} />
                  </div>
                  <span>Deep Learning</span>
                </div>
                <div className="about-badge-item">
                  <div className="about-badge-icon">
                    <CheckCircle2 size={16} />
                  </div>
                  <span>High Accuracy</span>
                </div>
                <div className="about-badge-item">
                  <div className="about-badge-icon">
                    <Activity size={16} />
                  </div>
                  <span>Early Detection</span>
                </div>
                <div className="about-badge-item">
                  <div className="about-badge-icon">
                    <ShieldCheck size={16} />
                  </div>
                  <span>Better Healthcare</span>
                </div>
              </div>

              {/* Dataset Description */}
              <div className="dataset-section">
                <h3>Dataset</h3>
                <ul className="dataset-list">
                  <li>SIPaKMeD & Herlev Pap smear dataset</li>
                  <li>Trained and tested with real-time Pap smear images</li>
                </ul>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* DIAGNOSTIC REPORT MODAL */}
      {showReportModal && selectedReport && (
        <div className="modal-backdrop">
          <div className="modal-scroll-container animate-fade">
            <div className="diagnostic-pdf-container">
              
              {/* Document Header */}
              <div className="pdf-header">
                <div className="logo-circle" style={{ width: '48px', height: '48px' }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M12 2c-2.8 0-5 2.2-5 5c0 2.5 1.5 4.5 4 6.5l-4 8.5c-.3.6 0 1.3.6 1.6c.6.3 1.3 0 1.6-.6l3.8-8c.5.2 1 .2 1.6.2s1.1 0 1.6-.2l3.8 8c.3.6 1 .9 1.6.6c.6-.3.9-1 .6-1.6l-4-8.5c2.5-2 4-4 4-6.5c0-2.8-2.2-5-5-5Z" />
                  </svg>
                </div>
                <div className="pdf-header-titles">
                  <h2>Cervical Cancer Detection System</h2>
                  <p>AI-POWERED DIAGNOSTIC REPORT</p>
                </div>
              </div>

              {/* Patient details section */}
              <div className="pdf-section">
                <div className="pdf-section-title">Patient / Sample Details</div>
                <div className="pdf-details-split">
                  <div className="pdf-details-table">
                    <div className="pdf-details-row">
                      <span className="label">Patient ID</span>
                      <span>:</span>
                      <span className="value">{selectedReport.id}</span>
                    </div>
                    <div className="pdf-details-row">
                      <span className="label">Date & Time</span>
                      <span>:</span>
                      <span className="value">{selectedReport.date} | 11:30 AM</span>
                    </div>
                    <div className="pdf-details-row">
                      <span className="label">Image Name</span>
                      <span>:</span>
                      <span className="value">{selectedReport.imageName}</span>
                    </div>
                  </div>
                  <div className="pdf-thumbnail">
                    <img src={selectedReport.image} alt="Biopsy smear preview" />
                  </div>
                </div>
              </div>

              {/* Prediction details section */}
              <div className="pdf-section">
                <div className="pdf-section-title">Prediction Summary</div>
                <div className="pdf-prediction-table">
                  <div className="pdf-prediction-row">
                    <span className="label">Predicted Class</span>
                    <span>:</span>
                    <span className="value">
                      <span className="pdf-pill-purple">{selectedReport.prediction} Cell</span>
                    </span>
                  </div>
                  <div className="pdf-prediction-row">
                    <span className="label">Confidence Score</span>
                    <span>:</span>
                    <span className="value">{selectedReport.confidence}</span>
                  </div>
                  <div className="pdf-prediction-row">
                    <span className="label">Risk Level</span>
                    <span>:</span>
                    <span className="value">
                      <span className={`pdf-pill-${selectedReport.status === 'High Risk' ? 'red' : selectedReport.status === 'Medium' ? 'orange' : 'success-bg'}`} style={{ color: selectedReport.status === 'High Risk' ? 'var(--color-danger)' : selectedReport.status === 'Medium' ? 'var(--color-warning)' : 'var(--color-success)' }}>
                        {selectedReport.status}
                      </span>
                    </span>
                  </div>
                  <div className="pdf-prediction-row">
                    <span className="label">Model</span>
                    <span>:</span>
                    <span className="value" style={{ fontWeight: '600', fontSize: '0.9rem', color: '#475569' }}>
                      Fusion Model (ResNet50 + ResNet101 + ResNet152)
                    </span>
                  </div>
                </div>
              </div>

              {/* Recommendations based on risk */}
              <div className="pdf-section">
                <div className="pdf-section-title">Recommendation</div>
                <ul className="pdf-recommendation-list">
                  {selectedReport.status === 'High Risk' ? (
                    <>
                      <li>Abnormal cervical cells detected.</li>
                      <li>Follow up with a gynecologist for further clinical evaluation.</li>
                      <li>Regular screening is advised for early prevention.</li>
                    </>
                  ) : selectedReport.status === 'Medium' ? (
                    <>
                      <li>Metaplastic cell changes detected (typical of healing or repair).</li>
                      <li>Schedule a routine follow-up smear in 6 months.</li>
                      <li>General cellular health guidelines recommended.</li>
                    </>
                  ) : (
                    <>
                      <li>Normal cell structures observed. No abnormalities detected.</li>
                      <li>Continue regular annual or triennial Pap smear screenings.</li>
                      <li>Maintain general preventative wellness checks.</li>
                    </>
                  )}
                </ul>
              </div>

              {/* Document Signatory */}
              <div className="pdf-sign-area">
                <div className="pdf-sign-line">Authorized Signatory</div>
              </div>

              {/* Document Disclaimer footnote */}
              <div className="pdf-footer-text">
                This report is generated automatically by the Cervical Cancer Detection System.
              </div>

            </div>

            {/* Modal Bottom control buttons */}
            <div className="pdf-action-footer">
              <button 
                className="pdf-btn-primary"
                onClick={() => { alert('Downloading Report PDF...'); setShowReportModal(false); }}
              >
                <Download size={16} /> Download PDF
              </button>
              <button 
                className="pdf-btn-secondary"
                onClick={() => { window.print(); setShowReportModal(false); }}
              >
                <Printer size={16} /> Print Report
              </button>
              <button 
                className="pdf-btn-secondary"
                style={{ borderColor: '#64748b', color: '#64748b' }}
                onClick={() => setShowReportModal(false)}
              >
                Cancel
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
