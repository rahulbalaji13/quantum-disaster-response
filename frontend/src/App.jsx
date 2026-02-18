import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isStreaming, setIsStreaming] = useState(false);

    const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

    useEffect(() => {
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchMetrics = async () => {
        try {
            const response = await axios.get(`${API_URL}/api/metrics`);
            setMetrics(response.data);
        } catch (err) {
            console.error('Metrics fetch error:', err);
        }
    };

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
        setError(null);
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

            const response = await axios.post(
                `${API_URL}/api/analyze`,
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            setAnalysis(response.data);
            fetchMetrics();
        } catch (err) {
            setError(err.message || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const handleStreaming = async (action) => {
        try {
            const response = await axios.post(
                `${API_URL}/api/stream/control`,
                { action }
            );
            setIsStreaming(action === 'start');
        } catch (err) {
            setError(err.message);
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
                        {error && <div className="error-message">{error}</div>}
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
                                    <p className="metric-value">{(metrics.detection_rate * 100).toFixed(1)}%</p>
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
        </div>
    );
}

export default App;
