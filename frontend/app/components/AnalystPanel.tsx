"use client";
import { API_BASE_URL } from '../config';
import React, { useState, useEffect } from 'react';
import { Eye, RefreshCw, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const AnalystPanel: React.FC = () => {
    const [analysis, setAnalysis] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [chartUrl, setChartUrl] = useState<string | null>(null);

    const fetchLatestAnalysis = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/api/analyst/latest`);
            const data = await res.json();
            setAnalysis(data);
            if (data.has_chart) {
                // Force refresh image by appending timestamp
                setChartUrl(`${API_BASE_URL}/api/analyst/chart?t=${Date.now()}`);
            }
        } catch (e) {
            console.error("Failed to fetch analyst data", e);
        } finally {
            setIsLoading(false);
        }
    };

    const triggerAnalysis = async () => {
        setIsLoading(true);
        try {
            await fetch(`${API_BASE_URL}/api/analyst/run`, { method: 'POST' });
            // Poll for result after a few seconds
            setTimeout(fetchLatestAnalysis, 5000);
        } catch (e) {
            console.error("Failed to trigger analysis", e);
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchLatestAnalysis();
        const interval = setInterval(fetchLatestAnalysis, 15000); // Poll every 15s
        return () => clearInterval(interval);
    }, []);

    const getStatusColor = (rec: string) => {
        switch (rec?.toUpperCase()) {
            case 'KEEP': return 'text-green-400 border-green-800 bg-green-900/20';
            case 'SWITCH': return 'text-yellow-400 border-yellow-800 bg-yellow-900/20';
            case 'STOP': return 'text-red-400 border-red-800 bg-red-900/20';
            default: return 'text-gray-400 border-gray-800 bg-gray-900/20';
        }
    };

    return (
        <div className="flex flex-col w-full h-full bg-[#1e1e1e] border border-gray-700 rounded-lg overflow-hidden">
            {/* Header */}
            <div className="p-3 bg-[#2d2d2d] border-b border-gray-700 flex justify-between items-center">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <Eye size={16} className="text-cyan-400" /> Trading Eyes (Market Analyst)
                </h3>
                <button
                    onClick={triggerAnalysis}
                    disabled={isLoading}
                    className="p-1.5 text-xs bg-cyan-900/30 text-cyan-400 border border-cyan-800 rounded hover:bg-cyan-900/50 flex items-center gap-1 transition-colors disabled:opacity-50"
                >
                    <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
                    {isLoading ? "Analyzing..." : "Run Now"}
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Recommendation Banner */}
                {analysis && (
                    <div className={`p-4 rounded-lg border-l-4 ${getStatusColor(analysis.recommendation)} bg-[#252525]`}>
                        <div className="flex justify-between items-start">
                            <div>
                                <h4 className="text-lg font-bold flex items-center gap-2">
                                    {analysis.recommendation}
                                    <span className="text-xs font-normal opacity-70 px-2 py-0.5 rounded bg-black/30">
                                        {Math.round((analysis.confidence || 0) * 100)}% Confidence
                                    </span>
                                </h4>
                                <p className="text-xs mt-1 opacity-70">
                                    Last Updated: {new Date(analysis.timestamp).toLocaleTimeString()}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Chart Image */}
                <div className="bg-black/40 rounded-lg border border-gray-800 overflow-hidden min-h-[200px] flex items-center justify-center relative">
                    {chartUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={chartUrl} alt="Latest Analysis Chart" className="w-full h-auto object-contain" />
                    ) : (
                        <div className="text-gray-500 flex flex-col items-center gap-2">
                            <TrendingUp size={32} className="opacity-20" />
                            <span className="text-xs">No chart generated yet</span>
                        </div>
                    )}
                </div>

                {/* Observations */}
                {analysis?.observations && (
                    <div className="space-y-2">
                        <h5 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Observations</h5>
                        <ul className="space-y-1">
                            {analysis.observations.map((obs: string, i: number) => (
                                <li key={i} className="text-sm text-gray-300 flex items-start gap-2 bg-[#252525] p-2 rounded">
                                    <span className="text-cyan-500 mt-1">•</span> {obs}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Reasoning */}
                {analysis?.reasoning && (
                    <div className="space-y-2">
                        <h5 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Visual Reasoning</h5>
                        <div className="text-sm text-gray-300 bg-[#252525] p-3 rounded h-40 overflow-y-auto">
                            <ReactMarkdown>{analysis.reasoning}</ReactMarkdown>
                        </div>
                    </div>
                )}

                {/* Input Context Snapshot */}
                {analysis?.inputs && (
                    <div className="space-y-3 pt-2 border-t border-gray-700">
                        <h5 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                            <TrendingUp size={12} /> Input Data Context (Snapshot)
                        </h5>

                        {/* Strategy Input */}
                        <div className="bg-[#1a1a1a] p-2 rounded border border-gray-800">
                            <span className="text-xs text-gray-500 block mb-1">Strategy Logic Feed</span>
                            <div className="text-xs text-gray-300 font-mono">
                                {analysis.inputs.strategy}
                            </div>
                        </div>

                        {/* Recent Candles Input */}
                        {analysis.inputs.market_data_summary && (
                            <div className="bg-[#1a1a1a] p-2 rounded border border-gray-800 overflow-x-auto">
                                <span className="text-xs text-gray-500 block mb-1">Numerical Feed (Last 5 Candles)</span>
                                <pre className="text-[10px] text-gray-400 font-mono">
                                    {JSON.stringify(analysis.inputs.market_data_summary, null, 2)}
                                </pre>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalystPanel;
