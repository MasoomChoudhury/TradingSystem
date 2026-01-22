"use client";
import React, { useState, useEffect } from 'react';
import { Play, Pause, RefreshCw, ArrowRightLeft, Square, TrendingUp, TrendingDown, FileText } from 'lucide-react';
import { API_BASE_URL } from '../config';

interface SessionData {
    state: string;
    session_id?: string;
    date?: string;
    market_bias?: string;
    active_strategy?: {
        name: string;
        bias: string;
        entry_condition: string;
        exit_condition: string;
    };
    realized_pnl?: number;
    unrealized_pnl?: number;
    trade_count?: number;
    risk_limits?: {
        max_lots: number;
        max_daily_loss: number;
    };
}

const SessionPanel: React.FC = () => {
    const [session, setSession] = useState<SessionData | null>(null);
    const [marketReport, setMarketReport] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showReportInput, setShowReportInput] = useState(false);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/session/status`);
            const data = await res.json();
            setSession(data);
        } catch (e) {
            console.error("Failed to fetch session status", e);
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    const startSession = async () => {
        if (!marketReport.trim()) return;
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/api/session/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ report: marketReport })
            });
            const data = await res.json();
            setSession(data.session);
            setShowReportInput(false);
            setMarketReport('');
        } catch (e) {
            console.error("Failed to start session", e);
        } finally {
            setIsLoading(false);
        }
    };

    const pauseSession = async () => {
        await fetch(`${API_BASE_URL}/api/session/pause`, { method: 'POST' });
        fetchStatus();
    };

    const resumeSession = async () => {
        await fetch(`${API_BASE_URL}/api/session/resume`, { method: 'POST' });
        fetchStatus();
    };

    const switchStrategy = async () => {
        await fetch(`${API_BASE_URL}/api/session/switch?reason=Manual%20switch`, { method: 'POST' });
        fetchStatus();
    };

    const endSession = async () => {
        await fetch(`${API_BASE_URL}/api/session/end`, { method: 'POST' });
        fetchStatus();
    };

    const getBiasColor = (bias: string) => {
        if (bias === 'bullish' || bias === 'LONG') return 'text-green-400';
        if (bias === 'bearish' || bias === 'SHORT') return 'text-red-400';
        return 'text-yellow-400';
    };

    const getStateColor = (state: string) => {
        if (state === 'active') return 'bg-green-500';
        if (state === 'paused') return 'bg-yellow-500';
        if (state === 'closed') return 'bg-gray-500';
        return 'bg-gray-700';
    };

    return (
        <div className="h-full flex flex-col bg-[#1a1a1a] rounded-lg border border-gray-800 overflow-hidden">
            {/* Header */}
            <div className="p-2 bg-[#252525] border-b border-gray-800 flex justify-between items-center">
                <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1.5">
                    <FileText size={14} className="text-purple-400" />
                    Trading Session
                </h3>
                <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${getStateColor(session?.state || 'idle')}`}></span>
                    <span className="text-[10px] text-gray-400 uppercase">{session?.state || 'idle'}</span>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {/* No Active Session */}
                {(!session || session.state === 'idle') && !showReportInput && (
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">No active session</span>
                        <button
                            onClick={() => setShowReportInput(true)}
                            className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded text-xs flex items-center gap-1"
                        >
                            <Play size={12} /> Start
                        </button>
                    </div>
                )}

                {/* Market Report Input */}
                {showReportInput && (
                    <div className="space-y-2">
                        <textarea
                            value={marketReport}
                            onChange={(e) => setMarketReport(e.target.value)}
                            placeholder="Paste your market report here..."
                            className="w-full h-32 bg-[#0f0f0f] border border-gray-700 rounded p-2 text-xs text-gray-300 resize-none"
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={startSession}
                                disabled={isLoading || !marketReport.trim()}
                                className="flex-1 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded text-xs flex items-center justify-center gap-1"
                            >
                                {isLoading ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
                                {isLoading ? 'Parsing...' : 'Start Session'}
                            </button>
                            <button
                                onClick={() => setShowReportInput(false)}
                                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* Active Session View */}
                {session && session.state !== 'idle' && (
                    <>
                        {/* Market Bias */}
                        <div className="flex items-center justify-between bg-[#202020] p-2 rounded">
                            <span className="text-xs text-gray-500">Market Bias</span>
                            <span className={`text-sm font-semibold flex items-center gap-1 ${getBiasColor(session.market_bias || '')}`}>
                                {session.market_bias === 'bullish' ? <TrendingUp size={12} /> :
                                    session.market_bias === 'bearish' ? <TrendingDown size={12} /> : null}
                                {session.market_bias?.toUpperCase()}
                            </span>
                        </div>

                        {/* Active Strategy */}
                        {session.active_strategy && (
                            <div className="bg-[#202020] p-2 rounded space-y-1">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-gray-500">Active Strategy</span>
                                    <span className={`text-xs px-2 py-0.5 rounded ${getBiasColor(session.active_strategy.bias)}`}>
                                        {session.active_strategy.bias}
                                    </span>
                                </div>
                                <div className="text-sm text-white font-mono">{session.active_strategy.name}</div>
                                <div className="text-[10px] text-gray-500">
                                    Entry: {session.active_strategy.entry_condition}
                                </div>
                            </div>
                        )}

                        {/* P&L */}
                        <div className="flex gap-2">
                            <div className="flex-1 bg-[#202020] p-2 rounded text-center">
                                <div className="text-[10px] text-gray-500">Realized P&L</div>
                                <div className={`text-sm font-bold ${(session.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    ₹{(session.realized_pnl || 0).toFixed(0)}
                                </div>
                            </div>
                            <div className="flex-1 bg-[#202020] p-2 rounded text-center">
                                <div className="text-[10px] text-gray-500">Trades</div>
                                <div className="text-sm font-bold text-white">{session.trade_count || 0}</div>
                            </div>
                        </div>

                        {/* Risk Limits */}
                        {session.risk_limits && (
                            <div className="text-[10px] text-gray-500 flex justify-between">
                                <span>Max: {session.risk_limits.max_lots} lot</span>
                                <span>Loss Limit: ₹{session.risk_limits.max_daily_loss}</span>
                            </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex gap-2 pt-2">
                            {session.state === 'active' && (
                                <button onClick={pauseSession} className="flex-1 py-1.5 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-xs flex items-center justify-center gap-1">
                                    <Pause size={12} /> Pause
                                </button>
                            )}
                            {session.state === 'paused' && (
                                <button onClick={resumeSession} className="flex-1 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded text-xs flex items-center justify-center gap-1">
                                    <Play size={12} /> Resume
                                </button>
                            )}
                            <button onClick={switchStrategy} className="flex-1 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs flex items-center justify-center gap-1">
                                <ArrowRightLeft size={12} /> Switch
                            </button>
                            <button onClick={endSession} className="py-1.5 px-3 bg-red-600 hover:bg-red-700 text-white rounded text-xs flex items-center justify-center">
                                <Square size={12} />
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default SessionPanel;
