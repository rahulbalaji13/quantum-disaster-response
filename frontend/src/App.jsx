import React, { useState, useEffect, useMemo, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [selectedRating, setSelectedRating] = useState(0);
    const [hoverRating, setHoverRating] = useState(0);
    const presentationRef = useRef(null);

    const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
    const API_URL = BASE_URL.replace(/\/$/, ''); // Remove trailing slash if present

    console.log("Using Backend API URL:", API_URL);

    const apiClient = axios.create({
        baseURL: API_URL,
        timeout: 15000
    });

    useEffect(() => {
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const fetchMetrics = async () => {
        try {
            const response = await apiClient.get('/api/metrics');
            setMetrics(response.data);
        } catch (err) {
            console.error('Metrics fetch error:', err);
        }
    };

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
        setError(null);
        setAnalysis(null);
    };

    const handleAnalyze = async () => {
        if (!selectedFile) {
            setError('Please select an image');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await apiClient.post(
                '/api/analyze',
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            setAnalysis(response.data);
            fetchMetrics();
        } catch (err) {
            const backendError = err?.response?.data?.error;
            setError(backendError || err.message || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const handleStreaming = async (action) => {
        try {
            await apiClient.post(
                '/api/stream/control',
                { action }
            );
            setIsStreaming(action === 'start');
        } catch (err) {
            const backendError = err?.response?.data?.error;
            setError(backendError || err.message || 'Streaming action failed');
        }
    };

    const riskColor = (level) => {
        switch (level) {
            case 'CRITICAL': return '#ef4444';
            case 'HIGH': return '#f97316';
            case 'MEDIUM': return '#eab308';
            default: return '#22c55e';
        }
    };

    const presentationEmbedUrl = useMemo(() => {
        const pptxPath = 'frontend/src/24MCS0071_RahulB_Project-review-II.pptx';
        if (typeof window === 'undefined') return pptxPath;
        const absoluteUrl = `${window.location.origin}${pptxPath}`;
        return `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(absoluteUrl)}`;
    }, []);

    const openPresentationFullscreen = () => {
        const element = presentationRef.current;
        if (!element || !element.requestFullscreen) return;
        element.requestFullscreen().catch(() => { });
    };

    return (
        <div className="app">
            <header className="header">
                <h1>🚀 Quantum-SwarmVLA-Edge</h1>
                <p>AI-Powered Disaster Response Dashboard</p>
            </header>

            <div className="container">
                <div className="main-content">
                    <section className="upload-section">
                        <h2>📸 Analyze Disaster Image</h2>
                        <p className="upload-helper">
                            Upload only <strong>satellite imagery</strong>. If the uploaded file is not a valid satellite image,
                            the system will respond with <em>"upload correct image"</em>.
                        </p>
                        <div className="upload-area">
                            <input
                                type="file"
                                accept="image/*"
                                onChange={handleFileChange}
                                disabled={loading}
                            />
                            <button
                                onClick={handleAnalyze}
                                disabled={loading || !selectedFile}
                                className="btn-primary"
                            >
                                {loading ? 'Analyzing...' : 'Analyze Image'}
                            </button>
                        </div>
                        {selectedFile && (
                            <p className="file-chip">Selected: {selectedFile.name}</p>
                        )}
                        {error && <div className="error-message">{error}</div>}
                    </section>

                    <section className="dataset-section">
                        <h2>🛰️ Sample Dataset Preview</h2>
                        <p>Reference examples similar to your training/testing flow for disaster satellite imagery.</p>
                        <div className="sample-grid">
                            <div className="sample-card">
                                <h3>Sample Trained Image</h3>
                                <img
                                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/ISS-38_Italy_view.jpg/640px-ISS-38_Italy_view.jpg"
                                    alt="Satellite-captured view of land and water"
                                />
                            </div>
                            <div className="sample-card">
                                <h3>Sample Tested Image</h3>
                                <img
                                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Katrina_2005-08-28.jpg/640px-Katrina_2005-08-28.jpg"
                                    alt="Satellite-captured cyclone scene"
                                />
                            </div>
                        </div>
                    </section>

                    <section className="architecture-section">
                        <h2>🏗️ Project Architecture</h2>
                        <div className="architecture-flow">
                            <div>Frontend (React Dashboard)</div>
                            <span>→</span>
                            <div>Backend API (Flask)</div>
                            <span>→</span>
                            <div>Validation + Disaster Inference</div>
                            <span>→</span>
                            <div>Metrics + Alerts + Routing</div>
                        </div>
                    </section>

                    <section className="presentation-section">
                        <h2>📑 Project Presentation Preview</h2>
                        <p>
                            Presentation file: <strong>24MCS0071_RahulB_Project-review-II.pptx</strong>
                        </p>
                        <div className="presentation-controls">
                            <a
                                className="btn-primary presentation-btn"
                                href="/24MCS0071_RahulB_Project-review-II.pptx"
                                target="_blank"
                                rel="noreferrer"
                            >
                                Open PPTX File
                            </a>
                            <button
                                className="btn-stream presentation-btn"
                                type="button"
                                onClick={openPresentationFullscreen}
                            >
                                Fullscreen Preview
                            </button>
                        </div>
                        <div className="video-wrapper" ref={presentationRef}>
                            <iframe
                                src={presentationEmbedUrl}
                                title="Project presentation preview"
                                allow="fullscreen"
                                allowFullScreen
                            />
                        </div>
                    </section>

                    {analysis && (
                        <section className="results-section">
                            <h2>📊 Analysis Results</h2>
                            <div className="result-grid">
                                <div className="result-card" style={{ borderColor: riskColor(analysis.risk_level) }}>
                                    <h3>Disaster Type</h3>
                                    <p className="result-value">{analysis.disaster_type}</p>
                                </div>
                                <div className="result-card">
                                    <h3>Confidence</h3>
                                    <p className="result-value">{(analysis.confidence * 100).toFixed(1)}%</p>
                                </div>
                                <div className="result-card" style={{ borderColor: riskColor(analysis.risk_level) }}>
                                    <h3>Risk Level</h3>
                                    <p className="result-value">{analysis.risk_level}</p>
                                </div>
                            </div>

                            {analysis.routing_optimization && (
                                <div className="routing-info">
                                    <h3>🛸 Drone Routes ({analysis.routing_optimization.routes.length} drones)</h3>
                                    <p>Optimization Time: {analysis.routing_optimization.optimization_time}s | Speedup: {analysis.routing_optimization.speedup_factor}x</p>
                                </div>
                            )}
                        </section>
                    )}
                </div>

                <div className="sidebar">
                    <section className="metrics-section">
                        <h2>📈 System Metrics</h2>
                        {metrics && (
                            <div className="metrics-grid">
                                <div className="metric-card">
                                    <p className="metric-label">Total Analyses</p>
                                    <p className="metric-value">{metrics.total_analyses}</p>
                                </div>
                                <div className="metric-card">
                                    <p className="metric-label">Disasters Detected</p>
                                    <p className="metric-value">{metrics.disaster_count}</p>
                                </div>
                                <div className="metric-card">
                                    <p className="metric-label">Detection Rate</p>
                                    <p className="metric-value">{(((metrics.disaster_count || 0) / Math.max(metrics.total_analyses || 1, 1)) * 100).toFixed(1)}%</p>
                                </div>
                            </div>
                        )}
                    </section>

                    <section className="streaming-section">
                        <h2>📡 Data Streaming</h2>
                        <button
                            onClick={() => handleStreaming(isStreaming ? 'stop' : 'start')}
                            className={`btn-stream ${isStreaming ? 'active' : ''}`}
                        >
                            {isStreaming ? 'Stop Streaming' : 'Start Streaming'}
                        </button>
                    </section>

                    <section className="recent-section">
                        <h2>📋 Recent Detections</h2>
                        {metrics && metrics.recent_detections.length > 0 ? (
                            <ul className="detection-list">
                                {metrics.recent_detections.slice(-5).map((det, i) => (
                                    <li key={i} style={{ borderColor: riskColor(det.risk) }}>
                                        <span className="detection-type">{det.type}</span>
                                        <span className="detection-risk">{det.risk}</span>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p>No detections yet</p>
                        )}
                    </section>
                </div>
            </div>

            <footer className="feedback-footer">
                <h3>⭐ Rate your dashboard experience</h3>
                <p>Your feedback helps us improve response quality and usability.</p>
                <div className="star-rating" aria-label="Star rating feedback">
                    {[1, 2, 3, 4, 5].map((star) => (
                        <button
                            key={star}
                            type="button"
                            className={`star-btn ${(hoverRating || selectedRating) >= star ? 'filled' : ''}`}
                            onClick={() => setSelectedRating(star)}
                            onMouseEnter={() => setHoverRating(star)}
                            onMouseLeave={() => setHoverRating(0)}
                            aria-label={`Rate ${star} star${star > 1 ? 's' : ''}`}
                        >
                            ★
                        </button>
                    ))}
                </div>
                <p className="rating-caption">
                    {selectedRating ? `Thanks for rating us ${selectedRating}/5!` : 'Tap a star to submit quick feedback.'}
                </p>
            </footer>
        </div>
    );
}

export default App;
