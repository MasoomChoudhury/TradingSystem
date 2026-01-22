"use client";

import { useState, useEffect } from "react";
import { API_BASE_URL } from "../config";

interface OpenAlgoStatus {
    connected: boolean;
    authenticated?: boolean;
    ws_url?: string;
    error?: string;
}

interface AnalyzerStatus {
    data?: {
        analyze_mode: boolean;
        mode: string;
        total_logs: number;
    };
    status: string;
}

export default function OpenAlgoControlPanel() {
    const [status, setStatus] = useState<OpenAlgoStatus | null>(null);
    const [analyzerStatus, setAnalyzerStatus] = useState<AnalyzerStatus | null>(null);
    const [isLiveMode, setIsLiveMode] = useState(false);
    const [isToggling, setIsToggling] = useState(false);

    // Fetch status on mount
    useEffect(() => {
        fetchStatus();
        fetchAnalyzerStatus();

        // Poll every 10 seconds
        const interval = setInterval(() => {
            fetchStatus();
            fetchAnalyzerStatus();
        }, 10000);

        return () => clearInterval(interval);
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/openalgo/status`);
            if (res.ok) {
                const data = await res.json();
                setStatus(data);
            }
        } catch {
            setStatus({ connected: false, error: "Backend unreachable" });
        }
    };

    const fetchAnalyzerStatus = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/openalgo/analyzer-status`);
            if (res.ok) {
                const data = await res.json();
                setAnalyzerStatus(data);
                setIsLiveMode(data?.data?.mode === "live");
            }
        } catch {
            // Analyzer status not available
        }
    };

    const toggleMode = async () => {
        setIsToggling(true);
        try {
            const newMode = !isLiveMode; // true = analyze, false = live
            const res = await fetch(`${API_BASE_URL}/api/openalgo/analyzer-toggle?mode=${!newMode}`, {
                method: "POST"
            });
            if (res.ok) {
                setIsLiveMode(newMode);
                fetchAnalyzerStatus();
            }
        } catch (e) {
            console.error("Toggle failed:", e);
        } finally {
            setIsToggling(false);
        }
    };

    return (
        <div className="bg-zinc-900/50 rounded-lg border border-zinc-800 p-3">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${status?.connected ? 'bg-green-500' : 'bg-red-500'}`} />
                    OpenAlgo
                </h3>
                <span className="text-[10px] text-zinc-600">
                    {status?.ws_url || "Not connected"}
                </span>
            </div>

            {/* Mode Toggle */}
            <div className="flex items-center justify-between bg-zinc-800/50 rounded-lg p-2">
                <div className="flex flex-col">
                    <span className="text-xs font-medium text-zinc-400">Trading Mode</span>
                    <span className={`text-[10px] ${isLiveMode ? 'text-red-400' : 'text-green-400'}`}>
                        {isLiveMode ? '⚠️ LIVE - Real Money' : '✓ ANALYZE - Simulated'}
                    </span>
                </div>

                {/* Toggle Switch */}
                <button
                    onClick={toggleMode}
                    disabled={isToggling}
                    className={`relative w-14 h-7 rounded-full transition-colors duration-200 ${isLiveMode
                        ? 'bg-red-600 hover:bg-red-500'
                        : 'bg-green-600 hover:bg-green-500'
                        } ${isToggling ? 'opacity-50 cursor-wait' : ''}`}
                >
                    <span
                        className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full shadow transition-transform duration-200 ${isLiveMode ? 'translate-x-7' : 'translate-x-0'
                            }`}
                    />
                    <span className={`absolute inset-0 flex items-center text-[9px] font-bold text-white ${isLiveMode ? 'justify-start pl-1.5' : 'justify-end pr-1.5'
                        }`}>
                        {isLiveMode ? 'LIVE' : 'SIM'}
                    </span>
                </button>
            </div>

            {/* Quick Stats */}
            {analyzerStatus?.data && (
                <div className="mt-2 flex gap-2 text-[10px] text-zinc-500">
                    <span>Logs: {analyzerStatus.data.total_logs}</span>
                </div>
            )}
        </div>
    );
}
