"use client";

import { useState, useEffect, useRef } from "react";

// API Request Log Entry
interface APILogEntry {
    id: string;
    timestamp: string;
    method: string;
    endpoint: string;
    payload?: object;
    response?: object;
    status: "pending" | "success" | "error";
    duration?: number;
}

// WebSocket Log Entry (from OpenAlgo)
interface WSLogEntry {
    timestamp: string;
    category: string; // WS, AUTH, SUB, UNSUB, LTP, QUOTE, DEPTH, ERROR
    message: string;
    data?: object;
}

type TabType = "api" | "websocket";

export default function APILogPanel() {
    const [activeTab, setActiveTab] = useState<TabType>("websocket");
    const [apiLogs, setApiLogs] = useState<APILogEntry[]>([]);
    const [wsLogs, setWsLogs] = useState<WSLogEntry[]>([]);
    const [expandedId, setExpandedId] = useState<string | null>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Fetch WebSocket logs from backend
    useEffect(() => {
        const fetchWsLogs = async () => {
            try {
                const res = await fetch("http://127.0.0.1:8000/api/openalgo/ws-logs?limit=50");
                if (res.ok) {
                    const data = await res.json();
                    setWsLogs(data.logs || []);
                }
            } catch (e) {
                console.log("Failed to fetch WS logs");
            }
        };

        // Fetch immediately and then every 2 seconds
        fetchWsLogs();
        const interval = setInterval(fetchWsLogs, 2000);
        return () => clearInterval(interval);
    }, []);

    // Subscribe to API logs from backend WebSocket
    useEffect(() => {
        const ws = new WebSocket("ws://127.0.0.1:8000/ws/api-logs");

        ws.onmessage = (event) => {
            try {
                const logEntry: APILogEntry = JSON.parse(event.data);
                setApiLogs(prev => [...prev.slice(-99), logEntry]);
            } catch (e) {
                console.error("Failed to parse API log:", e);
            }
        };

        ws.onerror = () => console.log("API Log WebSocket error");
        ws.onclose = () => console.log("API Log WebSocket closed");

        return () => ws.close();
    }, []);

    const clearLogs = async () => {
        if (activeTab === "api") {
            setApiLogs([]);
        } else {
            try {
                await fetch("http://127.0.0.1:8000/api/openalgo/ws-logs", { method: "DELETE" });
                setWsLogs([]);
            } catch (e) {
                setWsLogs([]);
            }
        }
    };

    const getStatusColor = (status: APILogEntry["status"]) => {
        switch (status) {
            case "success": return "text-green-400";
            case "error": return "text-red-400";
            default: return "text-yellow-400";
        }
    };

    const getMethodColor = (method: string) => {
        switch (method.toUpperCase()) {
            case "GET": return "bg-blue-600";
            case "POST": return "bg-green-600";
            case "PUT": return "bg-yellow-600";
            case "DELETE": return "bg-red-600";
            default: return "bg-zinc-600";
        }
    };

    const getCategoryColor = (category: string) => {
        switch (category.toUpperCase()) {
            case "WS": return "bg-purple-600";
            case "AUTH": return "bg-blue-600";
            case "SUB": return "bg-green-600";
            case "UNSUB": return "bg-orange-600";
            case "LTP": return "bg-cyan-600";
            case "QUOTE": return "bg-teal-600";
            case "DEPTH": return "bg-indigo-600";
            case "ERROR": return "bg-red-600";
            default: return "bg-zinc-600";
        }
    };

    return (
        <div className="flex flex-col h-full bg-zinc-900 rounded-lg border border-zinc-800">
            {/* Header with Tabs */}
            <div className="flex justify-between items-center px-4 py-2 border-b border-zinc-800">
                <div className="flex gap-2">
                    <button
                        onClick={() => setActiveTab("websocket")}
                        className={`text-xs px-3 py-1 rounded font-semibold transition-colors ${activeTab === "websocket"
                            ? "bg-purple-600 text-white"
                            : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
                            }`}
                    >
                        WebSocket
                    </button>
                    <button
                        onClick={() => setActiveTab("api")}
                        className={`text-xs px-3 py-1 rounded font-semibold transition-colors ${activeTab === "api"
                            ? "bg-blue-600 text-white"
                            : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
                            }`}
                    >
                        API Requests
                    </button>
                </div>
                <button
                    onClick={clearLogs}
                    className="text-xs text-zinc-500 hover:text-zinc-300 px-2 py-1 rounded hover:bg-zinc-800"
                >
                    Clear
                </button>
            </div>

            {/* Logs Content */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1 text-xs font-mono">
                {activeTab === "websocket" ? (
                    // WebSocket Logs
                    wsLogs.length === 0 ? (
                        <div className="text-zinc-600 text-center py-2 text-[11px]">
                            No WebSocket logs. Connect to OpenAlgo for real-time data.
                        </div>
                    ) : (
                        wsLogs.map((log, index) => (
                            <div
                                key={`ws-${index}`}
                                className="bg-zinc-800/50 rounded p-2 cursor-pointer hover:bg-zinc-800"
                                onClick={() => setExpandedId(expandedId === `ws-${index}` ? null : `ws-${index}`)}
                            >
                                <div className="flex items-center gap-2">
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${getCategoryColor(log.category)}`}>
                                        {log.category}
                                    </span>
                                    <span className="text-zinc-300 flex-1 truncate">
                                        {log.message}
                                    </span>
                                    <span className="text-zinc-600 text-[10px]">
                                        {new Date(log.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>

                                {expandedId === `ws-${index}` && log.data && (
                                    <div className="mt-2 text-[11px]">
                                        <span className="text-zinc-500">Data:</span>
                                        <pre className="mt-1 p-2 bg-zinc-900 rounded overflow-auto max-h-32 text-zinc-400">
                                            {JSON.stringify(log.data, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        ))
                    )
                ) : (
                    // API Request Logs
                    apiLogs.length === 0 ? (
                        <div className="text-zinc-600 text-center py-8">
                            No API requests yet. OpenAlgo REST calls will appear here.
                        </div>
                    ) : (
                        apiLogs.map((log) => (
                            <div
                                key={log.id}
                                className="bg-zinc-800/50 rounded p-2 cursor-pointer hover:bg-zinc-800"
                                onClick={() => setExpandedId(expandedId === log.id ? null : log.id)}
                            >
                                <div className="flex items-center gap-2">
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${getMethodColor(log.method)}`}>
                                        {log.method}
                                    </span>
                                    <span className="text-zinc-300 flex-1 truncate">
                                        {log.endpoint}
                                    </span>
                                    <span className={`${getStatusColor(log.status)}`}>
                                        {log.status === "pending" ? "..." : log.duration ? `${log.duration}ms` : ""}
                                    </span>
                                    <span className="text-zinc-600 text-[10px]">
                                        {log.timestamp}
                                    </span>
                                </div>

                                {expandedId === log.id && (
                                    <div className="mt-2 space-y-2 text-[11px]">
                                        {log.payload && (
                                            <div>
                                                <span className="text-zinc-500">Request:</span>
                                                <pre className="mt-1 p-2 bg-zinc-900 rounded overflow-auto max-h-32 text-zinc-400">
                                                    {JSON.stringify(log.payload, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                        {log.response && (
                                            <div>
                                                <span className="text-zinc-500">Response:</span>
                                                <pre className={`mt-1 p-2 bg-zinc-900 rounded overflow-auto max-h-32 ${getStatusColor(log.status)}`}>
                                                    {JSON.stringify(log.response, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))
                    )
                )}
                <div ref={logsEndRef} />
            </div>
        </div>
    );
}
