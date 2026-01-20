"use client";

import { useEffect, useState, useRef } from "react";

interface ConsoleProps {
    wsUrl: string;
}

export default function Console({ wsUrl }: ConsoleProps) {
    const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
    const [logs, setLogs] = useState<string[]>([]);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        const connect = () => {
            if (wsRef.current?.readyState === WebSocket.OPEN) return;

            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus('connected');
            };

            ws.onmessage = (event) => {
                setLogs((prev) => [...prev.slice(-100), event.data]);
            };

            ws.onerror = () => {
                setStatus('disconnected');
            };

            ws.onclose = () => {
                setStatus('connecting');
                setTimeout(connect, 3000);
            };

            return ws;
        };

        const socket = connect();

        return () => {
            if (socket) {
                socket.close();
            }
        };
    }, [wsUrl]);

    return (
        <div className="w-full h-full bg-black border border-zinc-800 rounded-lg p-3 overflow-y-auto font-mono text-xs">
            {logs.length === 0 ? (
                <div className="text-zinc-600">Logs will appear here...</div>
            ) : (
                logs.map((log, i) => (
                    <div key={i} className="text-green-500 whitespace-pre-wrap break-all">
                        {log}
                    </div>
                ))
            )}
        </div>
    );
}
