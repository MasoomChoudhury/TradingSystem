"use client";
import { API_BASE_URL } from '../config';
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, User, Zap, Play, Square, RefreshCw, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface Message {
    role: 'user' | 'ai';
    content: string;
}

interface ExecutorStatus {
    status: string;
    strategy: string;
    priority: number;
    levels: {
        entry: number;
        target: number;
        stop_loss: number;
        invalidation: number;
    };
    position: {
        side: string;
        qty: number;
        entry_price: number;
    };
    pending_order: {
        id: string;
        type: string;
        status: string;
    };
    last_command: string;
    last_updated: string;
}

const ExecutorPanel: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<ExecutorStatus | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
        fetchStatus();
    }, [messages]);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/executor/status`);
            const data = await res.json();
            setStatus(data);
        } catch (e) {
            console.error("Failed to fetch executor status");
        }
    };

    const getStatusColor = (s: string) => {
        switch (s) {
            case 'IDLE': return 'bg-gray-700 text-gray-300';
            case 'WAITING_ENTRY': return 'bg-yellow-900/50 text-yellow-300 border border-yellow-700';
            case 'IN_POSITION': return 'bg-green-900/50 text-green-300 border border-green-700';
            case 'WAITING_EXIT': return 'bg-blue-900/50 text-blue-300 border border-blue-700';
            case 'WAITING_FOR_BUILD': return 'bg-purple-900/50 text-purple-300 border border-purple-700';
            case 'STOPPED': return 'bg-red-900/50 text-red-300 border border-red-700';
            default: return 'bg-gray-800 text-gray-400';
        }
    };

    const getPositionIcon = (side: string) => {
        if (side === 'LONG') return <TrendingUp size={12} className="text-green-400" />;
        if (side === 'SHORT') return <TrendingDown size={12} className="text-red-400" />;
        return <Minus size={12} className="text-gray-400" />;
    };

    const sendCommand = async (command: string) => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/executor/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command, strategy_name: "", levels: null, execution_params: null }),
            });
            const data = await response.json();
            if (data.content) {
                setMessages(prev => [...prev, { role: 'ai', content: data.content }]);
            }
            fetchStatus();
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', content: "Error sending command." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/executor/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input }),
            });
            const data = await response.json();

            if (data.content) {
                setMessages(prev => [...prev, {
                    role: 'ai',
                    content: typeof data.content === 'string' ? data.content : JSON.stringify(data.content, null, 2)
                }]);
            }
            fetchStatus();
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', content: "Error communicating with Executor." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#1e1e1e] border border-gray-700 rounded-lg overflow-hidden">
            {/* Header with status */}
            <div className="p-3 bg-[#2d2d2d] border-b border-gray-700">
                <div className="flex justify-between items-center mb-2">
                    <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                        <Zap size={16} className="text-amber-400" /> Executor Agent
                    </h3>
                    {status && (
                        <div className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${getStatusColor(status.status)}`}>
                            {status.status}
                        </div>
                    )}
                </div>
                {status && status.strategy && (
                    <div className="text-xs text-gray-400 flex items-center gap-4">
                        <span>📊 {status.strategy}</span>
                        <span className="flex items-center gap-1">
                            {status.position && getPositionIcon(status.position.side)} {status.position?.side}
                            {status.position && status.position.qty > 0 && ` x${status.position.qty}`}
                        </span>
                    </div>
                )}
                {/* Quick action buttons */}
                <div className="flex gap-2 mt-2">
                    <button
                        onClick={() => sendCommand('STOP')}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-red-900/50 text-red-300 rounded hover:bg-red-900 transition-colors"
                    >
                        <Square size={10} /> STOP
                    </button>
                    <button
                        onClick={fetchStatus}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                    >
                        <RefreshCw size={10} /> Refresh
                    </button>
                </div>
            </div>

            {/* Levels display */}
            {status && status.levels && status.levels.entry > 0 && (
                <div className="px-3 py-2 bg-[#252525] border-b border-gray-700 grid grid-cols-4 gap-2 text-xs">
                    <div>
                        <span className="text-gray-500">Entry</span>
                        <div className="text-green-400 font-mono">{status.levels.entry}</div>
                    </div>
                    <div>
                        <span className="text-gray-500">Target</span>
                        <div className="text-blue-400 font-mono">{status.levels.target}</div>
                    </div>
                    <div>
                        <span className="text-gray-500">SL</span>
                        <div className="text-red-400 font-mono">{status.levels.stop_loss}</div>
                    </div>
                    <div>
                        <span className="text-gray-500">Invalid</span>
                        <div className="text-orange-400 font-mono">{status.levels.invalidation}</div>
                    </div>
                </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 mt-10 text-sm">
                        <p>Executor manages trade execution.</p>
                        <p className="text-xs mt-2 text-gray-600">Receives LOCK/SWITCH/STOP from Supervisor</p>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        {msg.role === 'ai' && (
                            <div className="w-7 h-7 rounded-full bg-amber-900 flex items-center justify-center shrink-0">
                                <Zap size={14} className="text-amber-200" />
                            </div>
                        )}
                        <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${msg.role === 'user'
                            ? 'bg-amber-700 text-white'
                            : 'bg-[#333333] text-gray-200'
                            }`}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {msg.content}
                            </ReactMarkdown>
                        </div>
                        {msg.role === 'user' && (
                            <div className="w-7 h-7 rounded-full bg-gray-600 flex items-center justify-center shrink-0">
                                <User size={14} className="text-gray-200" />
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div className="flex gap-3 justify-start">
                        <div className="w-7 h-7 rounded-full bg-amber-900 flex items-center justify-center shrink-0 animate-pulse">
                            <Zap size={14} className="text-amber-200" />
                        </div>
                        <div className="bg-[#333333] rounded-lg px-3 py-2 text-sm text-gray-400 flex items-center gap-2">
                            <span className="animate-spin">⚙️</span>
                            Processing...
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-3 bg-[#2d2d2d] border-t border-gray-700">
                <div className="relative">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about execution status..."
                        className="w-full bg-[#1e1e1e] text-gray-200 text-sm rounded-md pl-3 pr-10 py-2 focus:outline-none focus:ring-1 focus:ring-amber-500 resize-none h-12"
                    />
                    <button
                        onClick={sendMessage}
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 p-1 text-amber-400 hover:text-amber-300 disabled:opacity-50"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ExecutorPanel;
