"use client";
import { API_BASE_URL } from '../config';
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, User, Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface Message {
    role: 'user' | 'ai';
    content: string;
    regimeStatus?: string;
}

const SupervisorPanel: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [regimeStatus, setRegimeStatus] = useState<string>('UNKNOWN');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'VALID':
                return <CheckCircle size={14} className="text-green-400" />;
            case 'WARNING':
                return <AlertTriangle size={14} className="text-yellow-400" />;
            case 'INVALID':
                return <XCircle size={14} className="text-red-400" />;
            default:
                return <Shield size={14} className="text-gray-400" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'VALID':
                return 'bg-green-900/50 border-green-700';
            case 'WARNING':
                return 'bg-yellow-900/50 border-yellow-700';
            case 'INVALID':
                return 'bg-red-900/50 border-red-700';
            default:
                return 'bg-gray-800 border-gray-700';
        }
    };

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/supervisor/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input }),
            });
            const data = await response.json();

            if (data.regime_status) {
                setRegimeStatus(data.regime_status);
            }

            if (data.role && data.content) {
                const content = typeof data.content === 'string'
                    ? data.content
                    : JSON.stringify(data.content, null, 2);
                setMessages(prev => [...prev, {
                    role: 'ai',
                    content,
                    regimeStatus: data.regime_status
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', content: "Error communicating with Supervisor." }]);
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
        <div className="flex flex-col w-full min-h-0 h-full bg-[#1e1e1e] border border-gray-700 rounded-lg overflow-hidden">
            {/* Header with regime status */}
            <div className="p-3 bg-[#2d2d2d] border-b border-gray-700">
                <div className="flex justify-between items-center">
                    <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                        <Shield size={16} className="text-purple-400" /> Supervisor Agent
                    </h3>
                    <div className={`flex items-center gap-2 px-2 py-1 rounded border ${getStatusColor(regimeStatus)}`}>
                        {getStatusIcon(regimeStatus)}
                        <span className="text-xs font-medium">{regimeStatus}</span>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 mt-10 text-sm">
                        <p>Supervisor monitors strategy execution and regime changes.</p>
                        <p className="text-xs mt-2 text-gray-600">Ask: "Is the market still in an uptrend?"</p>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        {msg.role === 'ai' && (
                            <div className="w-7 h-7 rounded-full bg-purple-900 flex items-center justify-center shrink-0">
                                <Shield size={14} className="text-purple-200" />
                            </div>
                        )}
                        <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${msg.role === 'user'
                            ? 'bg-purple-700 text-white'
                            : 'bg-[#333333] text-gray-200'
                            }`}>
                            {msg.regimeStatus && (
                                <div className={`text-xs mb-2 flex items-center gap-1 px-2 py-0.5 w-fit rounded ${getStatusColor(msg.regimeStatus)}`}>
                                    {getStatusIcon(msg.regimeStatus)}
                                    <span>{msg.regimeStatus}</span>
                                </div>
                            )}
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
                        <div className="w-7 h-7 rounded-full bg-purple-900 flex items-center justify-center shrink-0 animate-pulse">
                            <Shield size={14} className="text-purple-200" />
                        </div>
                        <div className="bg-[#333333] rounded-lg px-3 py-2 text-sm text-gray-400 flex items-center gap-2">
                            <span className="animate-spin">⚙️</span>
                            Analyzing regime...
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
                        placeholder="Ask about regime, strategy validity..."
                        className="w-full bg-[#1e1e1e] text-gray-200 text-sm rounded-md pl-3 pr-10 py-2 focus:outline-none focus:ring-1 focus:ring-purple-500 resize-none h-12"
                    />
                    <button
                        onClick={sendMessage}
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 p-1 text-purple-400 hover:text-purple-300 disabled:opacity-50"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SupervisorPanel;
