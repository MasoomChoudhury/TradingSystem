"use client";
import React, { useState, useEffect } from 'react';
import { MessageCircle, RefreshCw, ArrowRight, CheckCircle, XCircle, AlertTriangle, Eye } from 'lucide-react';
import { API_BASE_URL } from '../config';

interface AgentMessageData {
    id: string;
    timestamp: string;
    from_agent: string;
    to_agent: string;
    message_type: string;
    content: string;
    metadata: Record<string, any>;
}

const AgentCommsPanel: React.FC = () => {
    const [messages, setMessages] = useState<AgentMessageData[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [filterAgent, setFilterAgent] = useState<string>('all');

    const fetchMessages = async () => {
        setIsLoading(true);
        try {
            const url = filterAgent === 'all'
                ? `${API_BASE_URL}/api/comms/history?limit=30`
                : `${API_BASE_URL}/api/comms/history?limit=30&agent=${filterAgent}`;
            const res = await fetch(url);
            const data = await res.json();
            setMessages(data.messages || []);
        } catch (e) {
            console.error("Failed to fetch agent comms", e);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchMessages();
        const interval = setInterval(fetchMessages, 5000); // Poll every 5s
        return () => clearInterval(interval);
    }, [filterAgent]);

    const getAgentColor = (agent: string) => {
        switch (agent) {
            case 'orchestrator': return 'text-blue-400';
            case 'analyst': return 'text-cyan-400';
            case 'supervisor': return 'text-purple-400';
            case 'executor': return 'text-amber-400';
            default: return 'text-gray-400';
        }
    };

    const getAgentIcon = (agent: string) => {
        switch (agent) {
            case 'orchestrator': return '🎯';
            case 'analyst': return '👁️';
            case 'supervisor': return '🛡️';
            case 'executor': return '⚡';
            default: return '📬';
        }
    };

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'approval': return <CheckCircle size={12} className="text-green-400" />;
            case 'denial': return <XCircle size={12} className="text-red-400" />;
            case 'advisory': return <Eye size={12} className="text-cyan-400" />;
            case 'request': return <ArrowRight size={12} className="text-blue-400" />;
            default: return <MessageCircle size={12} className="text-gray-400" />;
        }
    };

    const getTypeBadgeColor = (type: string) => {
        switch (type) {
            case 'approval': return 'bg-green-900/30 text-green-400 border-green-800';
            case 'denial': return 'bg-red-900/30 text-red-400 border-red-800';
            case 'advisory': return 'bg-cyan-900/30 text-cyan-400 border-cyan-800';
            case 'request': return 'bg-blue-900/30 text-blue-400 border-blue-800';
            default: return 'bg-gray-900/30 text-gray-400 border-gray-800';
        }
    };

    return (
        <div className="h-full flex flex-col bg-[#1a1a1a] rounded-lg border border-gray-800 overflow-hidden">
            {/* Header */}
            <div className="p-2 bg-[#252525] border-b border-gray-800 flex justify-between items-center">
                <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1.5">
                    <MessageCircle size={14} className="text-indigo-400" />
                    Agent Communications
                </h3>
                <div className="flex items-center gap-2">
                    <select
                        value={filterAgent}
                        onChange={(e) => setFilterAgent(e.target.value)}
                        className="text-[10px] bg-[#1a1a1a] border border-gray-700 rounded px-1 py-0.5 text-gray-300"
                    >
                        <option value="all">All Agents</option>
                        <option value="orchestrator">🎯 Orchestrator</option>
                        <option value="analyst">👁️ Analyst</option>
                        <option value="supervisor">🛡️ Supervisor</option>
                        <option value="executor">⚡ Executor</option>
                        <option value="market_data">📊 Market Data</option>
                        <option value="indicators">📈 Indicators</option>
                        <option value="options">📉 Options</option>
                        <option value="accounts">💰 Accounts</option>
                    </select>
                    <button
                        onClick={fetchMessages}
                        disabled={isLoading}
                        className="p-1 text-gray-400 hover:text-gray-200 disabled:opacity-50"
                    >
                        <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
                    </button>
                </div>
            </div>

            {/* Messages List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {messages.length === 0 ? (
                    <div className="text-center text-gray-500 text-xs py-4">
                        No inter-agent messages yet.
                        <br />Messages will appear here when agents communicate.
                    </div>
                ) : (
                    messages.map((msg) => (
                        <div key={msg.id} className="bg-[#202020] rounded border border-gray-800 p-2">
                            {/* Header Row */}
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-1 text-xs">
                                    <span className={getAgentColor(msg.from_agent)}>
                                        {getAgentIcon(msg.from_agent)} {msg.from_agent}
                                    </span>
                                    <ArrowRight size={10} className="text-gray-600" />
                                    <span className={getAgentColor(msg.to_agent)}>
                                        {getAgentIcon(msg.to_agent)} {msg.to_agent}
                                    </span>
                                </div>
                                <span className={`text-[9px] px-1.5 py-0.5 rounded border flex items-center gap-1 ${getTypeBadgeColor(msg.message_type)}`}>
                                    {getTypeIcon(msg.message_type)}
                                    {msg.message_type}
                                </span>
                            </div>
                            {/* Content */}
                            <div className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                                {msg.content}
                            </div>
                            {/* Timestamp */}
                            <div className="text-[9px] text-gray-600 mt-1">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default AgentCommsPanel;
