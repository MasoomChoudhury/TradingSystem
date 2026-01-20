"use client";
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Bot, User, Wrench, CheckCircle } from 'lucide-react';

interface ToolCall {
    tool: string;
    result: string;
}

interface Message {
    role: 'user' | 'ai' | 'tool';
    content: string;
    toolCalls?: ToolCall[];
}

interface ChatInterfaceProps {
    onFileChange?: () => void;  // Callback to refresh file list
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onFileChange }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentAction, setCurrentAction] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setCurrentAction("Thinking...");

        try {
            const response = await fetch('http://127.0.0.1:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input }),
            });
            const data = await response.json();

            // Handle tool calls
            if (data.tool_calls && data.tool_calls.length > 0) {
                // Add tool call messages
                for (const tc of data.tool_calls) {
                    setMessages(prev => [...prev, {
                        role: 'tool',
                        content: `**${tc.tool}**: ${tc.result}`
                    }]);
                }

                // Refresh file list if file operations were performed
                const fileOps = data.tool_calls.filter((tc: ToolCall) =>
                    ['write_file', 'read_file', 'list_files'].includes(tc.tool)
                );
                if (fileOps.length > 0 && onFileChange) {
                    onFileChange();
                }
            }

            if (data.role && data.content) {
                setMessages(prev => [...prev, {
                    role: 'ai',
                    content: data.content,
                    toolCalls: data.tool_calls
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', content: "Error communicating with backend." }]);
        } finally {
            setIsLoading(false);
            setCurrentAction(null);
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
            <div className="p-3 bg-[#2d2d2d] border-b border-gray-700 flex justify-between items-center">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <Bot size={16} /> Orchestrator Agent
                </h3>
                <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                    gemini-3-flash
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 mt-10 text-sm">
                        <p>I can help you create, edit, and deploy trading strategies.</p>
                        <p className="text-xs mt-2 text-gray-600">Try: "Create a simple RSI strategy"</p>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        {msg.role === 'ai' && (
                            <div className="w-7 h-7 rounded-full bg-blue-900 flex items-center justify-center shrink-0">
                                <Bot size={14} className="text-blue-200" />
                            </div>
                        )}
                        {msg.role === 'tool' && (
                            <div className="w-7 h-7 rounded-full bg-green-900 flex items-center justify-center shrink-0">
                                <Wrench size={12} className="text-green-300" />
                            </div>
                        )}
                        <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${msg.role === 'user'
                            ? 'bg-blue-700 text-white'
                            : msg.role === 'tool'
                                ? 'bg-green-900/30 text-green-200 border border-green-800 text-xs font-mono'
                                : 'bg-[#333333] text-gray-200'
                            }`}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}
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
                        <div className="w-7 h-7 rounded-full bg-blue-900 flex items-center justify-center shrink-0 animate-pulse">
                            <Bot size={14} className="text-blue-200" />
                        </div>
                        <div className="bg-[#333333] rounded-lg px-3 py-2 text-sm text-gray-400 flex items-center gap-2">
                            <span className="animate-spin">⚙️</span>
                            {currentAction || "Processing..."}
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="p-3 bg-[#2d2d2d] border-t border-gray-700">
                <div className="relative">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask me to create, edit, or execute code..."
                        className="w-full bg-[#1e1e1e] text-gray-200 text-sm rounded-md pl-3 pr-10 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none h-12"
                    />
                    <button
                        onClick={sendMessage}
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 p-1 text-blue-400 hover:text-blue-300 disabled:opacity-50"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
